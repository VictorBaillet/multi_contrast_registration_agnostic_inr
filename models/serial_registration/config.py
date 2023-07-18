import os
import torch
import pathlib
import numpy as np
import wandb

from utils.utils import dict2obj
from utils.config.config import create_losses, process_config, create_input_mapper 
from models.serial_registration.experiment_utils.utils_config import create_network, create_datasets, compute_dataset_artifacts


def general_config(config_dict, verbose=True):
    # Init arguments 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))
    
    config = dict2obj(config_dict)

    # Seeding
    torch.manual_seed(config.TRAINING.SEED)
    np.random.seed(config.TRAINING.SEED)
    
    # Training device
    device = f'cuda:{config.TRAINING.GPU_DEVICE}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Network configuration
    network, network_registration = create_network(config.NETWORK, device)
    input_mapper = create_input_mapper(config.NETWORK, device)
    
    if verbose:
        print(f'Number of SR MLP parameters {sum(p.numel() for p in network.parameters())}')
        print(f'Number of Registration MLP parameters {sum(p.numel() for p in network_registration.parameters())}')

    # Losses configuration
    lpips_loss, criterion, mi_criterion, cc_criterion = create_losses(config, device)  
    
    # optimizer
    if config.TRAINING.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=config.TRAINING.LR)#, weight_decay=5e-5)
        optimizer_registration = torch.optim.Adam(network_registration.parameters(), lr=3e-4, weight_decay=5e-5)

    else:
        raise ValueError('Optim not defined!')
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max= config.TRAINING.EPOCHS_cos)
    scheduler_registration = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_registration, T_max= config.TRAINING.EPOCHS_cos)
     
    training_args = {'config':config,
                     'device':device,
                     'input_mapper':input_mapper,
                     'lpips_loss':lpips_loss,
                     'criterion':criterion,
                     'mi_criterion':mi_criterion,
                     'cc_criterion':cc_criterion,}
    
    model_config = (network, optimizer, scheduler, input_mapper)
    model_registration_config = (network_registration, optimizer_registration, scheduler_registration)
    
    return config, model_config, model_registration_config, training_args

def data_config(config, data_path, contrast_1, contrast_2, dataset_name, device, verbose=True):
    # Load Data
    dataset, dataloaders = create_datasets(config.TRAINING, data_path, contrast_1, contrast_2, dataset_name, verbose)

    
    moving_image, rev_affine, min_coords, max_coords, difference_center_of_mass, format_im = compute_dataset_artifacts(dataset, device)
    
    dataset_artifacts = {'moving_image':moving_image,
                         'min_coords':min_coords,
                         'max_coords':max_coords,
                         'rev_affine':rev_affine,
                         'difference_center_of_mass':difference_center_of_mass,
                         'format_im':format_im}
    
    return dataset, dataloaders, dataset_artifacts