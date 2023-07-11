import numpy as np
import argparse
import yaml
import lpips

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset.dataset import MultiModalDataset, InferDataset
from utils.loss_functions.loss_functions import MILossGaussian, NMI, NCC
from utils.config.utils_config import input_mapping, get_string


def process_config(args):
    """
    Load the configuration file and update it with the command line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.

    Returns
    -------
    config : object
        Configuration object.
    config_dict : dict
        Configuration dictionary.
    """
    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        
    if args.logging:
        config_dict["SETTINGS"]["LOGGING"] = True
        
    
    
    return config_dict

def create_input_mapper(config, device):
    """
    Create an input mapper for the Fourier features.

    Parameters
    ----------
    config : object
        Configuration object.
    device : str
        Training device.

    Returns
    -------
    input_mapper : object
        Input mapper for the Fourier features.
    """
    mapping_size = config.FOURIER.MAPPING_SIZE  # of FF
    B_gauss = torch.tensor(np.random.normal(scale=config.FOURIER.FF_SCALE, size=(config.FOURIER.MAPPING_SIZE, 3)), dtype=torch.float32).to(device)
    input_mapper = input_mapping(B=B_gauss, factor=config.FOURIER.FF_FACTOR).to(device)
    return input_mapper

def create_losses(config, config_dict, model_name, device):
    """
    Create the losses for the model.

    Parameters
    ----------
    config : object
        Configuration object.
    config_dict : dict
        Configuration dictionary.
    model_name : str
        Model name.
    device : str
        Training device.

    Returns
    -------
    lpips_loss : object
        LPIPS loss.
    criterion : object
        Similarity criterion.
    mi_criterion : object
        Mutual information criterion.
    cc_criterion : object
        Cross-correlation criterion.
    model_name : str
        Model name.
    """
    lpips_loss = lpips.LPIPS(net='alex', verbose=False).to(device)

    model_name = f'{model_name}_NUML_{config.MODEL.NUM_LAYERS}_N_{config.MODEL.HIDDEN_CHANNELS}_D_{config.MODEL.DROPOUT}_'     
    
    # Loss
    if config.TRAINING.SIMILARITY_LOSS == 'L1Loss':
        criterion = nn.L1Loss()
    elif config.TRAINING.SIMILARITY_LOSS == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError('Loss function not defined!')

    model_name = f'{model_name}_{config.TRAINING.SIMILARITY_LOSS}__{config.TRAINING.LOSS_WEIGHT.MSE_C1}__{config.TRAINING.LOSS_WEIGHT.MSE_C2}_'     

    # custom losses in addition to normal loss
    mi_criterion, cc_criterion = None, None
    if config.TRAINING.USE_MI:
        mi_criterion = MILossGaussian(num_bins=config.MI_CC.MI_NUM_BINS, sample_ratio=config.MI_CC.MI_SAMPLE_RATIO, gt_val=None)#config.MI_CC.GT_VAL)
        model_name = f'{model_name}_{get_string(config_dict["MI_CC"])}_'     
    
    if config.TRAINING.USE_CC:
        cc_criterion = NCC()
        model_name = f'{model_name}_{get_string(config_dict["MI_CC"])}_'    
        
    if config.TRAINING.USE_NMI:
        mi_criterion = NMI(intensity_range=(0,1), nbins=config.MI_CC.MI_NUM_BINS, sigma=config.MI_CC.NMI_SIGMA)
        model_name = f'{model_name}_{get_string(config_dict["MI_CC"])}_'  
        
    return lpips_loss, criterion, mi_criterion, cc_criterion, model_name

def parse_args():
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Arguments.
    """
    parser = argparse.ArgumentParser(description='Train Neural Implicit Function for a single scan.')
    parser.add_argument('--experiment_name', type=str, help='Experiment name', default=None)    
    parser.add_argument('--config', default='config.yaml', help='config file (.yaml) containing the hyper-parameters for training.')
    parser.add_argument('--logging', action='store_true')
    return parser.parse_args()
