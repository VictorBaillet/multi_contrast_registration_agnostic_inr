import time
import os
import yaml
import pathlib

import nibabel as nib
import numpy as np
import pdb

import torch
import torch.nn as nn
import wandb

from dataset.dataset import MultiModalDataset, InferDataset
from utils.utils_visualization import show_slices_gt
from utils.utils import input_mapping, compute_metrics, dict2obj, get_string, compute_mi_hist, compute_mi, generate_NIFTIs, fast_trilinear_interpolation
from utils.utils_training import forward_iteration, compute_and_log_metrics
from utils.utils_config import create_datasets, create_model, create_losses, process_config, parse_args 
from utils.loss_functions import MILossGaussian, NMI, NCC


import time
import asyncio
import cProfile
import pstats


def main(args):

    # Init arguments 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

    # Load the config 
    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config_dict)
    
    config, config_dict = process_config(config, config_dict, args)

    # Logging run
    if args.logging:
        wandb.login()
        wandb.init(config=config_dict, project=config.SETTINGS.PROJECT_NAME)

    # Make directory for models
    weight_dir = f'runs/{config.SETTINGS.PROJECT_NAME}_weights'
    image_dir = f'runs/{config.SETTINGS.PROJECT_NAME}_images'

    pathlib.Path(weight_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(image_dir).mkdir(parents=True, exist_ok=True)

    # Seeding
    torch.manual_seed(config.TRAINING.SEED)
    np.random.seed(config.TRAINING.SEED)
    
    # Training device
    device = f'cuda:{config.SETTINGS.GPU_DEVICE}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Model configuration
    model, model_name, input_mapper = create_model(config, config_dict, device)

    print(f'Number of MLP parameters {sum(p.numel() for p in model.parameters())}')

    # Losses configuration
    lpips_loss, criterion, mi_criterion, cc_criterion, model_name = create_losses(config, config_dict, model_name, device)  
    mi_buffer = np.zeros((4,1))
    mi_mean = -1.0
    
    # optimizer
    if config.TRAINING.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAINING.LR)#, weight_decay=5e-5)
        model_name = f'{model_name}_{config.TRAINING.OPTIM}_{config.TRAINING.LR}_'    
    else:
        raise ValueError('Optim not defined!')
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max= config.TRAINING.EPOCHS)

    # Load Data
    dataset, train_dataloader, infer_dataloader, threshold = create_datasets(config)
    
    # Dimensions of the two images
    x_dim_c1, y_dim_c1, z_dim_c1 = dataset.get_contrast1_dim()
    x_dim_c2, y_dim_c2, z_dim_c2 = dataset.get_contrast2_dim()
    x_dim_c2_lr, y_dim_c2_lr, z_dim_c2_lr = dataset.get_contrast2_lr_dim()
    
    # Image to be registered
    fixed_image = dataset.get_contrast2_intensities().reshape((x_dim_c2_lr, y_dim_c2_lr, z_dim_c2_lr))
    
    # Maximum and minimum coordinates of the training points (used in fast_trilinear_interpolation)
    coord_c2 = dataset.get_contrast2_data().cpu().numpy()
    affine = dataset.get_contrast2_affine().cpu().numpy()[:3,:3]
    rev_affine = np.linalg.inv(affine)
    res = rev_affine @ coord_c2.T
    res = res.T
    max_coords = [np.max(res[:,i]) for i in range(3)]
    min_coords = [-np.max(-res[:,i]) for i in range(3)]
    rev_affine = torch.tensor(rev_affine, device=device)
    
    training_args = {'config':config,
                     'args':args,
                     'device':device,
                     'input_mapper':input_mapper,
                     'fixed_image':fixed_image,
                     'lpips_loss':lpips_loss,
                     'criterion':criterion,
                     'mi_criterion':mi_criterion,
                     'cc_criterion':cc_criterion,
                     'min_coords':min_coords,
                     'max_coords':max_coords,
                     'rev_affine':rev_affine}
    
    for epoch in range(config.TRAINING.EPOCHS):
        # Set model to train
        model.train()
        wandb_epoch_dict = {}

        model_name_epoch = f'{model_name}_e{int(epoch)}_model.pt'  
        model_path = os.path.join(weight_dir, model_name_epoch)

        loss_epoch = 0.0
        start = time.time()
        for batch_idx, (data, labels) in enumerate(train_dataloader): #(data, labels, segm) in enumerate(train_dataloader):
            loss_batch = 0
            wandb_batch_dict = {}
            data = data.requires_grad_(True)
            
            # Forward pass
            loss, wandb_batch_dict = forward_iteration(model, data, labels, wandb_batch_dict, epoch, model_name, **training_args)
                    
            # zero gradients
            optimizer.zero_grad()
            # backprop
            loss.to(torch.float)
            loss.to(device=device)
            loss.backward()
            optimizer.step()
            # epoch loss
            loss_batch += loss.item()
            loss_epoch += loss_batch
            if args.logging:
                wandb_batch_dict.update({'batch_loss': loss_batch})
                wandb.log(wandb_batch_dict)  # update logs per batch
        # collect epoch stats
        epoch_time = time.time() - start

        lr = optimizer.param_groups[0]["lr"]
        if args.logging:
            wandb_epoch_dict.update({'epoch_no': epoch})
            wandb_epoch_dict.update({'epoch_time': epoch_time})
            wandb_epoch_dict.update({'epoch_loss': loss_epoch})
            wandb_epoch_dict.update({'lr': lr})

        if epoch == (config.TRAINING.EPOCHS -1):
            torch.save(model.state_dict(), model_path)


        scheduler.step()
        ################ INFERENCE #######################

        model_inference = model
        model_inference.eval()

        # start inference
        start = time.time()

        out = np.zeros((int(x_dim_c1*y_dim_c1*z_dim_c1 + x_dim_c2*y_dim_c2*z_dim_c2), 5))
        model_inference.to(device)
        for batch_idx, (data) in enumerate(infer_dataloader):
            raw_data = data
            if torch.cuda.is_available():
                data = data.to(device)
                
            if config.MODEL.USE_FF:
                data = input_mapper(data)
            elif config.MODEL.USE_SIREN:
                data = data*np.pi
            else:
                data = data
                
            output = model_inference(data)
            """"
            registration_target = output[:,2:]
            coord_temp = torch.add(registration_target, raw_data.to(device=device))
            contrast2_interpolated = fast_trilinear_interpolation(
                fixed_image,
                coord_temp[:, 0],
                coord_temp[:, 1],
                coord_temp[:, 2],
                min_coords,
                max_coords,
                device,
                rev_affine
            )
            output[:, 1] = contrast2_interpolated
            """
            if config.MODEL.USE_SIREN or config.MODEL.USE_WIRE_REAL:
                output, _ = output

            out[batch_idx*5000:(batch_idx*5000 + len(output)),:] = output.cpu().detach().numpy() 

        model_intensities=out
        
        inference_time = time.time() - start
        if args.logging:
            wandb_epoch_dict.update({'inference_time': inference_time})

        ################ EVALUATION #######################

        print("Generating NIFTIs.")
        pred_contrast1, pred_contrast2, gt_contrast1, gt_contrast2, wandb_epoch_dict = generate_NIFTIs(dataset, 
                                                                                                       model_intensities, 
                                                                                                       image_dir, 
                                                                                                       model_name_epoch, 
                                                                                                       epoch, wandb_epoch_dict, config, args)

        mask_c1 = np.zeros_like(gt_contrast1)
        mask_c1[mask_c1 == 0] = 1
        mask_c2 = np.zeros_like(gt_contrast2)
        mask_c2[mask_c2 == 0] = 1

        wandb_epoch_dict = compute_and_log_metrics(gt_contrast1, gt_contrast2, pred_contrast1, pred_contrast2, mask_c1, mask_c2, 
                                                   lpips_loss, device, wandb_epoch_dict, args)


        wandb.log(wandb_epoch_dict)  # update logs per epoch

        #mi_buffer[1:] = mi_buffer[:-1]  # shifting buffer
        #mi_buffer[0] = metrics_mi_pred["mi"]  # update buffer
        curr_mean = np.mean(np.abs(mi_buffer[:-1]-mi_buffer[1:]))  # srtore diff of abs change
        print("Current buffer: ", mi_buffer, "mean:", curr_mean)

        if np.abs(curr_mean-mi_mean)<0.0001:
            if args.early_stopping:
                print("Early stopping training", mi_buffer)
                break

        else:
            mi_mean = curr_mean


if __name__ == '__main__':
    args = parse_args()
    pr = cProfile.Profile()

    # python 3.8
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    # stats.dump_stats(filename='code_profiling.prof')


    # python 3.6
    # pr.enable()
    main(args)
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('time', 'cumulative')
    # ps.dump_stats(filename='code_profiling_improved.prof')
