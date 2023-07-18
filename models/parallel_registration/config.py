import os
import torch
import pathlib
import numpy as np
import wandb

from utils.utils import dict2obj
from utils.config.config import create_losses, process_config, create_input_mapper 
from models.parallel_registration.experiment_utils.utils_config import create_model, create_datasets, compute_dataset_artifacts


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

    # Model configuration
    model = create_model(config.NETWORK, device)
    input_mapper = create_input_mapper(config.NETWORK, device)
    
    if verbose:
        print(f'Number of MLP parameters {sum(p.numel() for p in model.parameters())}')

    # Losses configuration
    lpips_loss, criterion, mi_criterion, cc_criterion = create_losses(config, device)  

    # optimizer
    if config.TRAINING.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAINING.LR)#, weight_decay=5e-5)
    else:
        raise ValueError('Optim not defined!')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max= config.TRAINING.EPOCHS_cos)

    training_args = {'config':config,
                     'device':device,
                     'input_mapper':input_mapper,
                     'lpips_loss':lpips_loss,
                     'criterion':criterion,
                     'mi_criterion':mi_criterion,
                     'cc_criterion':cc_criterion,}
    
    model_config = (model, optimizer, scheduler, input_mapper)
    
    return config, model_config, training_args

def data_config(config, data_path, contrast_1, contrast_2, dataset_name, device, verbose=True):
    # Load Data
    dataset, train_dataloader, infer_dataloader = create_datasets(config.TRAINING, data_path, 
                                                                             contrast_1, contrast_2, dataset_name, verbose)
    
    fixed_image, rev_affine, min_coords, max_coords, difference_center_of_mass, format_im = compute_dataset_artifacts(dataset, device)
    
    dataset_artifacts = {'fixed_image':fixed_image,
                         'min_coords':min_coords,
                         'max_coords':max_coords,
                         'rev_affine':rev_affine,
                         'difference_center_of_mass':difference_center_of_mass,
                         'format_im':format_im}
    
    return dataset, train_dataloader, infer_dataloader, dataset_artifacts