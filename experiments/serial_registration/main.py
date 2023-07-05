import os
import sys
import time
import pathlib

import nibabel as nib
import numpy as np
import pdb

import torch
import torch.nn as nn
import wandb

sys.path.append(os.getcwd())

from utils.utils import fast_trilinear_interpolation, center_of_mass
from utils.visualization.visualization import generate_NIFTIs, compute_and_log_metrics
from utils.config.config import create_losses, process_config, parse_args, create_input_mapper 

from experiments.serial_registration.experiment_utils.training.training import forward_iteration, inference_iteration_contrast2
from experiments.serial_registration.experiment_utils.config.config import training_config 


import time
import asyncio
import cProfile
import pstats


def main(args):
    config, model_config, model_registration_config, data_config, directories, training_args = training_config(args)
    
    model, model_name, optimizer, scheduler = model_config
    model_registration, optimizer_registration, scheduler_registration = model_registration_config
    dataset, train_dataloader, infer_dataloader, infer_dataloader_contrast1, infer_dataloader_contrast2, input_mapper = data_config 
    weight_dir, image_dir = directories
    device = training_args['device']
    
    for epoch in range(config.TRAINING.EPOCHS):
        ################ TRAINING #######################
        print('-----------------')
        print("Epoch n°", epoch)
        print('Training the model...')
        # Set model to train
        model.train()
        model_registration.train()
        
        wandb_epoch_dict = {}

        model_name_epoch = f'{model_name}_e{int(epoch)}_model.pt'  
        model_registration_name_epoch = f'{model_name}_e{int(epoch)}_model_registration.pt'  
        model_path = os.path.join(weight_dir, model_name_epoch)
        model_path_reg = os.path.join(weight_dir, model_registration_name_epoch)

        loss_epoch = 0.0
        start = time.time()
        i = 0
        for batch_idx, (data, labels) in enumerate(train_dataloader): #(data, labels, segm) in enumerate(train_dataloader):
            loss_batch = 0
            wandb_batch_dict = {}
            data = data.requires_grad_(True)
            
            # Forward pass
            registration_loss, mse_loss, wandb_batch_dict = forward_iteration(model, model_registration, data, labels, wandb_batch_dict, epoch, **training_args)
                    
            # zero gradients
            optimizer.zero_grad()
            optimizer_registration.zero_grad()
            # backprop
            loss= registration_loss + mse_loss
    
            registration_loss.to(torch.float)
            registration_loss.to(device=device)
            mse_loss.to(torch.float)
            mse_loss.to(device=device)
            
            model.requires_grad_(False)
            registration_loss.backward(retain_graph=True)
            model.requires_grad_(True)
            #model_registration.requires_grad_(False)
            mse_loss.backward()
            #model_registration.requires_grad_(True)
            
            #loss.backward()
            optimizer.step()
            optimizer_registration.step()
            # epoch loss
            loss_batch += loss.detach().item()
            loss_epoch += loss_batch
            if args.logging:
                wandb_batch_dict.update({'batch_loss': loss_batch})
                wandb.log(wandb_batch_dict)  # update logs per batch
            
        
        # collect epoch stats
        epoch_time = time.time() - start

        lr_SR = optimizer.param_groups[0]["lr"]
        lr_reg = optimizer_registration.param_groups[0]["lr"]
        if args.logging:
            wandb_epoch_dict.update({'epoch_no': epoch})
            wandb_epoch_dict.update({'epoch_time': epoch_time})
            wandb_epoch_dict.update({'epoch_loss': loss_epoch})
            wandb_epoch_dict.update({'lr_SR': lr_SR})
            wandb_epoch_dict.update({'lr_reg': lr_reg})

        if epoch == (config.TRAINING.EPOCHS -1):
            torch.save(model.state_dict(), model_path)
            torch.save(model_registration.state_dict(), model_path_reg)

        #if epoch < 25 or epoch > 100:
        scheduler.step()
        scheduler_registration.step()
        
        ################ INFERENCE #######################
        print('Inference...')
        model_inference = model
        model_inference_registration = model_registration
        model_inference.eval()
        model_inference_registration.eval()

        # start inference
        start = time.time()
        
        x_dim_c1, y_dim_c1, z_dim_c1 = dataset.get_contrast1_dim()
        x_dim_c2, y_dim_c2, z_dim_c2 = dataset.get_contrast2_dim()

        out = np.zeros((int(x_dim_c1*y_dim_c1*z_dim_c1 + x_dim_c2*y_dim_c2*z_dim_c2), 8))
        model_inference.to(device)
        model_inference_registration.to(device)
        for batch_idx, (data) in enumerate(infer_dataloader_contrast1):
            data.requires_grad_()
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
            batch_size = 10000
            out[batch_idx*batch_size:(batch_idx*batch_size + len(output)),:2] = output.cpu().detach().numpy() 
            
        contrast1_last_batch_idx = batch_idx*batch_size + len(output)
        
        for batch_idx, (data) in enumerate(infer_dataloader_contrast2):
            data.requires_grad_()
            array_idx = contrast1_last_batch_idx + batch_idx*batch_size
            out[array_idx:(array_idx + len(data))] = inference_iteration_contrast2(model, model_inference_registration, data, **training_args)

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
