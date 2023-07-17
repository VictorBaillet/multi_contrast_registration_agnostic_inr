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

from utils.utils import fast_trilinear_interpolation, center_of_mass
from utils.visualization.visualization import generate_NIFTIs, compute_and_log_metrics
from utils.config.config import create_losses, process_config, parse_args, create_input_mapper 

from models.serial_registration.training import forward_iteration, inference_iteration_contrast2
from models.serial_registration.config import training_config 



class serial_registration_irn:
    def __call__(self, inputs, is_moving_image):
        if torch.cuda.is_available():
            inputs = inputs.to(self.device)
        if is_moving_image:
            inputs = inputs*np.pi
            registration_output, _ = self.network_registration(inputs) 
            registration_output = torch.mul(registration_output, self.training_args['format_im']).float()
            output = self.network(self.input_mapper(torch.add(registration_output, inputs)))
        
        else:
            if self.config.MODEL.USE_FF:
                inputs = self.input_mapper(inputs)
            elif self.config.MODEL.USE_SIREN:
                inputs = inputs*np.pi

            output = self.network(inputs)
            
        return output
        
    def __init__(self, config_dict):
        self.config, model_config, model_registration_config, data_config, self.training_args = training_config(config_dict)
    
        self.network, self.model_name, self.optimizer, self.scheduler = model_config
        self.network_registration, self.optimizer_registration, self.scheduler_registration = model_registration_config
        self.dataset, self.train_dataloader, self.infer_dataloader, self.infer_dataloader_contrast1, self.infer_dataloader_contrast2, self.input_mapper = data_config 
        self.device = self.training_args['device']
        self.logging = self.config.SETTINGS.LOGGING
        self.score = {}
        
    def fit(self):
        # Seeding
        torch.manual_seed(self.config.TRAINING.SEED)
        np.random.seed(self.config.TRAINING.SEED)
        
        
        for epoch in range(self.config.TRAINING.EPOCHS):
            self.wandb_epoch_dict = {}
            
            self._training_iteration(epoch)
            model_intensities = self._inference()
            self._evaluation(model_intensities, epoch)
            
            if self.logging:
                wandb.log(self.wandb_epoch_dict)  # update logs per epoch
                
    def save_model(self, path):
        torch.save(self.network.state_dict(), path)
        
    def save_images(self, contrast, path):
        x_dim, y_dim, z_dim = self.dataset.get_dim(contrast=contrast, resolution='gt')
        
        model_intensities = self._inference()
        threshold = len(self.dataset.get_coordinates(contrast=1, resolution='gt'))
        if contrast == 0:
            model_intensities_contrast = model_intensities[:threshold,0] 
        else:
            model_intensities_contrast = model_intensities[threshold:,1] 
        
        label_arr = np.array(model_intensities_contrast, dtype=np.float32)
        model_intensities_contrast= np.clip(label_arr.reshape(-1, 1), 0, 1)
        
        img_contrast = model_intensities_contrast.reshape((x_dim, y_dim, z_dim))
        
        affine = np.array(self.dataset.get_affine(contrast=contrast, resolution='gt'))
        img = nib.Nifti1Image(img_contrast, affine)
        nib.save(img, path)

        return img
        
    def load_model(self, path):
        self.network.load_state_dict(torch.load(path))
        
    def score(self):
        return self.score
        
    def _training_iteration(self, epoch):
        print('-----------------')
        print("Epoch n°", epoch)
        print('Training the model...')
        # Set model to train
        self.network.train()
        self.network_registration.train()
        
        loss_epoch = 0.0
        cc_loss_registration_epoch = 0.0
        start = time.time()
        i = 0
        for batch_idx, (data, labels, mask) in enumerate(self.train_dataloader): #(data, labels, segm) in enumerate(train_dataloader):
            wandb_batch_dict = {}
            loss_batch = 0
            data = data.requires_grad_(True)
            
            # Forward pass
            registration_loss, mse_loss, wandb_batch_dict = forward_iteration(self.network, self.network_registration, data, labels, 
                                                                              wandb_batch_dict, epoch, **self.training_args)
                    
            # zero gradients
            self.optimizer.zero_grad()
            self.optimizer_registration.zero_grad()
            # backprop
            loss= registration_loss + mse_loss
    
            registration_loss.to(torch.float)
            registration_loss.to(device=self.device)
            mse_loss.to(torch.float)
            mse_loss.to(device=self.device)
            
            self.network.requires_grad_(False)
            registration_loss.backward(retain_graph=True)
            self.network.requires_grad_(True)
            #model_registration.requires_grad_(False)
            mse_loss.backward()
            #model_registration.requires_grad_(True)
            
            #loss.backward()
            self.optimizer.step()
            self.optimizer_registration.step()
            # epoch loss
            loss_batch += loss.detach().item()
            loss_epoch += loss_batch
            if self.logging:
                wandb_batch_dict.update({'batch_loss': loss_batch})
                wandb.log(wandb_batch_dict)  # update logs per batch
                
                
  
        

        epoch_time = time.time() - start

        lr_SR = self.optimizer.param_groups[0]["lr"]
        lr_reg = self.optimizer_registration.param_groups[0]["lr"]
        self.wandb_epoch_dict.update({'epoch_no': epoch})
        self.wandb_epoch_dict.update({'epoch_time': epoch_time})
        self.wandb_epoch_dict.update({'epoch_loss': loss_epoch})
        self.wandb_epoch_dict.update({'lr_SR': lr_SR})
        self.wandb_epoch_dict.update({'lr_reg': lr_reg})

        #if epoch < 25 or epoch > 100:
        self.scheduler.step()
        self.scheduler_registration.step()
        
            
    def _inference(self):
        print('Inference...')
        network_inference = self.network
        network_inference_registration = self.network_registration
        network_inference.eval()
        network_inference_registration.eval()

        # start inference
        start = time.time()
        
        x_dim_c1, y_dim_c1, z_dim_c1 = self.dataset.get_dim(contrast=1, resolution='gt')
        x_dim_c2, y_dim_c2, z_dim_c2 = self.dataset.get_dim(contrast=2, resolution='gt')

        out = np.zeros((int(x_dim_c1*y_dim_c1*z_dim_c1 + x_dim_c2*y_dim_c2*z_dim_c2), 8))
        network_inference.to(self.device)
        network_inference_registration.to(self.device)
        for batch_idx, (data) in enumerate(self.infer_dataloader_contrast1):
            data.requires_grad_()
            raw_data = data
            if torch.cuda.is_available():
                data = data.to(self.device)
                
            if self.config.MODEL.USE_FF:
                data = self.input_mapper(data)
            elif self.config.MODEL.USE_SIREN:
                data = data*np.pi
            else:
                data = data
                
            output = network_inference(data)
            batch_size = 10000
            out[batch_idx*batch_size:(batch_idx*batch_size + len(output)),:2] = output.cpu().detach().numpy() 
            
        contrast1_last_batch_idx = batch_idx*batch_size + len(output)
        
        for batch_idx, (data) in enumerate(self.infer_dataloader_contrast2):
            data.requires_grad_()
            array_idx = contrast1_last_batch_idx + batch_idx*batch_size
            out[array_idx:(array_idx + len(data))] = inference_iteration_contrast2(network_inference, network_inference_registration, 
                                                                                   data, **self.training_args)

        model_intensities=out
        
        inference_time = time.time() - start
        self.wandb_epoch_dict.update({'inference_time': inference_time})

            
        return model_intensities
                        
    def _evaluation(self, model_intensities, epoch):
        print("Generating NIFTIs.")
        pred_contrast1, pred_contrast2, gt_contrast1, gt_contrast2, wandb_epoch_dict = generate_NIFTIs(self.dataset, 
                                                                                                       model_intensities,  
                                                                                                       epoch, self.wandb_epoch_dict, 
                                                                                                       self.config)

        mask_c1 = self.dataset.get_mask(contrast=1, resolution='gt')
        mask_c2 = self.dataset.get_mask(contrast=2, resolution='gt')

        self.wandb_epoch_dict = compute_and_log_metrics(gt_contrast1, gt_contrast2, pred_contrast1, pred_contrast2, mask_c1, mask_c2, 
                                                   self.training_args['lpips_loss'], self.device, self.wandb_epoch_dict)


        
        
        
        
        
    