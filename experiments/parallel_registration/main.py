import os
import sys
import time

import nibabel as nib
import numpy as np

import torch
import wandb
import gc

sys.path.append(os.getcwd())

from utils.config.config import parse_args 
from utils.visualization.visualization import generate_NIFTIs, compute_and_log_metrics
from utils.utils import fast_trilinear_interpolation, center_of_mass

from experiments.parallel_registration.experiment_utils.training.training import forward_iteration, inference_iteration
from experiments.parallel_registration.experiment_utils.config.config import training_config 


import time
import asyncio
import cProfile
import pstats


def main(args):
    config, model_config, data_config, directories, training_args = training_config(args)
    
    model, model_name, optimizer, scheduler = model_config
    dataset, train_dataloader, infer_dataloader, input_mapper = data_config 
    weight_dir, image_dir = directories
    device = training_args['device']
    
    for epoch in range(config.TRAINING.EPOCHS):
        ################ TRAINING #######################
        print('-----------------')
        print("Epoch n°", epoch)
        print('Training the model...')
        # Set model to train
        model.train()
        wandb_epoch_dict = {}

        model_name_epoch = f'{model_name}_e{int(epoch)}_model.pt'  
        model_path = os.path.join(weight_dir, model_name_epoch)

        loss_epoch = 0.0
        cc_loss_registration_epoch = 0.0
        start = time.time()
        i = 0
        for batch_idx, (data, labels) in enumerate(train_dataloader): #(data, labels, segm) in enumerate(train_dataloader):
            loss_batch = 0
            wandb_batch_dict = {}
            data = data.requires_grad_(True)
            
            # Forward pass
            loss, cc_loss_registration, wandb_batch_dict = forward_iteration(model, data, labels, wandb_batch_dict, epoch, model_name, **training_args)
                    
            # zero gradients
            optimizer.zero_grad()
            # backprop
            loss = loss.to(torch.float)
            loss = loss.to(device=device)
            loss.backward()
            optimizer.step()
            # epoch loss
            loss_batch += loss.detach().cpu().item()
            loss_epoch += loss_batch
            cc_loss_registration_epoch += cc_loss_registration.detach().item()
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
            wandb_epoch_dict.update({'cc_loss_registration_epoch': cc_loss_registration_epoch})
            wandb_epoch_dict.update({'lr': lr})

        if epoch == (config.TRAINING.EPOCHS -1):
            torch.save(model.state_dict(), model_path)

        scheduler.step()
        
        
        ################ INFERENCE #######################
        print('Inference...')
        model_inference = model
        model_inference.eval()

        # start inference
        start = time.time()
        
        x_dim_c1, y_dim_c1, z_dim_c1 = dataset.get_contrast1_dim()
        x_dim_c2, y_dim_c2, z_dim_c2 = dataset.get_contrast2_dim()

        out = np.zeros((int(x_dim_c1*y_dim_c1*z_dim_c1 + x_dim_c2*y_dim_c2*z_dim_c2), 8))
        model_inference.to(device)
        batch_size = 10000
        for batch_idx, (data) in enumerate(infer_dataloader):
            
            data.requires_grad_()            
            out[batch_idx*batch_size:(batch_idx*batch_size + len(data))] = inference_iteration(model_inference, data, **training_args)

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
                                                   training_args['lpips_loss'], device, wandb_epoch_dict, args)


        wandb.log(wandb_epoch_dict)  # update logs per epoch


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
