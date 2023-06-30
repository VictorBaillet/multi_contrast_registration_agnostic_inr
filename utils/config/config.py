import numpy as np
import argparse
import yaml
import lpips

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.utils import dict2obj
from utils.dataset.dataset import MultiModalDataset, InferDataset
from utils.loss_functions.loss_functions import MILossGaussian, NMI, NCC
from utils.config.utils_config import input_mapping, get_string


def process_config(args):
    # Load the config 
    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config_dict)
    
    # we bypass lr, epoch and batch_size if we provide them via arparse
    if args.lr != None:
        config.TRAINING.LR = args.lr
        config_dict["TRAINING"]["LR"] = args.lr
    
    if args.batch_size != None:
        config.TRAINING.BATCH_SIZE = args.batch_size
        config_dict["TRAINING"]["BATCH_SIZE"] = args.batch_size
    
    if args.epochs != None:
        config.TRAINING.EPOCHS = args.epochs
        config_dict["TRAINING"]["EPOCHS"] = args.epochs

    # dataset specific
    if args.subject_id != None:
        config.DATASET.SUBJECT_ID = args.subject_id
        config_dict["DATASET"]["SUBJECT_ID"] = args.subject_id
    
    return config, config_dict

def create_input_mapper(config, device):
    mapping_size = config.FOURIER.MAPPING_SIZE  # of FF
    B_gauss = torch.tensor(np.random.normal(scale=config.FOURIER.FF_SCALE, size=(config.FOURIER.MAPPING_SIZE, 3)), dtype=torch.float32).to(device)
    input_mapper = input_mapping(B=B_gauss, factor=config.FOURIER.FF_FACTOR).to(device)
    return input_mapper

def create_losses(config, config_dict, model_name, device):
    lpips_loss = lpips.LPIPS(net='alex', verbose=False).to(device)

    model_name = f'{model_name}_NUML_{config.MODEL.NUM_LAYERS}_N_{config.MODEL.HIDDEN_CHANNELS}_D_{config.MODEL.DROPOUT}_'     
    
    # Loss
    if config.TRAINING.LOSS == 'L1Loss':
        criterion = nn.L1Loss()
    elif config.TRAINING.LOSS == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError('Loss function not defined!')

    model_name = f'{model_name}_{config.TRAINING.LOSS}__{config.TRAINING.LOSS_MSE_C1}__{config.TRAINING.LOSS_MSE_C2}_'     

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
    parser = argparse.ArgumentParser(description='Train Neural Implicit Function for a single scan.')
    parser.add_argument('--config', default='config.yaml', help='config file (.yaml) containing the hyper-parameters for training.')
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0], help="GPU ID following PCI order.")

    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)

    # patient
    parser.add_argument('--subject_id', type=str, default=None)
    parser.add_argument('--experiment_no', type=int, default=None)
    return parser.parse_args()
