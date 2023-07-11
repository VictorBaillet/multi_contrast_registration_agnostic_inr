import os
import sys
import time

import nibabel as nib
import numpy as np

import torch
import wandb
import gc

sys.path.append(os.getcwd())

from utils.utils import dict2obj
from utils.config.config import parse_args 
from utils.visualization.visualization import generate_NIFTIs, compute_and_log_metrics
from utils.utils import fast_trilinear_interpolation, center_of_mass

from experiments.parallel_registration.training import forward_iteration, inference_iteration
from experiments.parallel_registration.config import training_config 



class parallel_registration_irn:
    def __call__(self):
        """"""
        
    def __init__(self, config_dict):
        self.config, model_config, data_config, directories, self.training_args = training_config(config_dict)
    
        self.network, self.model_name, self.optimizer, self.scheduler = model_config
        self.dataset, self.train_dataloader, self.infer_dataloader, self.input_mapper = data_config 
        self.weight_dir, self.image_dir = directories
        self.device = self.training_args['device']
        self.logging = self.config.SETTINGS.LOGGING
        
    def fit(self):
        # Seeding
        torch.manual_seed(self.config.TRAINING.SEED)
        np.random.seed(self.config.TRAINING.SEED)
        
        
        for epoch in range(self.config.TRAINING.EPOCHS):
            self.model_name_epoch = f'{self.model_name}_e{int(epoch)}_model.pt'  
            self.wandb_epoch_dict = {}
            
            self._training_iteration(epoch)
            model_intensities = self._inference()
            self._evaluation(model_intensities, epoch)
            
            if self.logging:
                wandb.log(self.wandb_epoch_dict)  # update logs per epoch
        
    def _training_iteration(self, epoch):
        print('-----------------')
        print("Epoch n°", epoch)
        print('Training the model...')
        # Set model to train
        self.network.train()

        model_path = os.path.join(self.weight_dir, self.model_name_epoch)

        loss_epoch = 0.0
        cc_loss_registration_epoch = 0.0
        start = time.time()
        i = 0
        for batch_idx, (data, labels, mask) in enumerate(self.train_dataloader): #(data, labels, segm) in enumerate(train_dataloader):
            loss_batch = 0
            wandb_batch_dict = {}
            data = data.requires_grad_(True)
            
            # Forward pass
            loss, cc_loss_registration, wandb_batch_dict = forward_iteration(self.network, data, labels, mask, wandb_batch_dict, 
                                                                             epoch, self.model_name, **self.training_args)
                    
            # zero gradients
            self.optimizer.zero_grad()
            # backprop
            loss = loss.to(torch.float)
            loss = loss.to(device=self.device)
            loss.backward()
            self.optimizer.step()
            # epoch loss
            loss_batch += loss.detach().cpu().item()
            loss_epoch += loss_batch
            cc_loss_registration_epoch += cc_loss_registration.detach().item()
            if self.logging:
                wandb_batch_dict.update({'batch_loss': loss_batch})
                wandb.log(wandb_batch_dict)  # update logs per batch
                
                
                
  
        
        # collect epoch stats
        epoch_time = time.time() - start

        lr = self.optimizer.param_groups[0]["lr"]
        self.wandb_epoch_dict.update({'epoch_no': epoch})
        self.wandb_epoch_dict.update({'epoch_time': epoch_time})
        self.wandb_epoch_dict.update({'epoch_loss': loss_epoch})
        self.wandb_epoch_dict.update({'cc_loss_registration_epoch': cc_loss_registration_epoch})
        self.wandb_epoch_dict.update({'lr': lr})

        if epoch == (self.config.TRAINING.EPOCHS -1):
            torch.save(self.network.state_dict(), model_path)

        if epoch < 25 or epoch > 125:
            self.scheduler.step()
        
            
    def _inference(self):
        print('Inference...')
        model_inference = self.network
        model_inference.eval()

        # start inference
        start = time.time()
        
        x_dim_c1, y_dim_c1, z_dim_c1 = self.dataset.get_dim(contrast=1, resolution='gt')
        x_dim_c2, y_dim_c2, z_dim_c2 = self.dataset.get_dim(contrast=2, resolution='gt')

        out = np.zeros((int(x_dim_c1*y_dim_c1*z_dim_c1 + x_dim_c2*y_dim_c2*z_dim_c2), 8))
        model_inference.to(self.device)
        batch_size = 10000
        for batch_idx, (data) in enumerate(self.infer_dataloader):
            
            data.requires_grad_()            
            out[batch_idx*batch_size:(batch_idx*batch_size + len(data))] = inference_iteration(model_inference, data, **self.training_args)

        model_intensities=out
        
        inference_time = time.time() - start
        if self.logging:
            self.wandb_epoch_dict.update({'inference_time': inference_time})
            
        return model_intensities
                        
    def _evaluation(self, model_intensities, epoch):
        print("Generating NIFTIs.")
        pred_contrast1, pred_contrast2, gt_contrast1, gt_contrast2, wandb_epoch_dict = generate_NIFTIs(self.dataset, 
                                                                                                       model_intensities, 
                                                                                                       self.image_dir, 
                                                                                                       self.model_name_epoch, 
                                                                                                       epoch, self.wandb_epoch_dict, 
                                                                                                       self.config)

        mask_c1 = self.dataset.get_mask(contrast=1, resolution='gt')
        mask_c2 = self.dataset.get_mask(contrast=2, resolution='gt')

        self.wandb_epoch_dict = compute_and_log_metrics(gt_contrast1, gt_contrast2, pred_contrast1, pred_contrast2, mask_c1, mask_c2, 
                                                   self.training_args['lpips_loss'], self.device, self.wandb_epoch_dict)


        
        
        
        
        
    