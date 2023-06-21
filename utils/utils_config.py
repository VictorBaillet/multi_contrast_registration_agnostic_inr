from dataset.dataset import MultiModalDataset, InferDataset
import argparse
from model import MLPv1, MLPv2, Siren, WireReal, MLPregv1, MLPregv2
import lpips
from loss_functions import MILossGaussian, NMI, NCC
from utils.utils import input_mapping, get_string
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch


def process_config(config, config_dict, args):
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

def create_model(config, config_dict, device):
    input_mapper = None
    # Model Selection
    model_name = (
                f'{config.SETTINGS.PROJECT_NAME}_subid-{config.DATASET.SUBJECT_ID}_'
                f'ct1LR-{config.DATASET.LR_CONTRAST1}_ct2LR-{config.DATASET.LR_CONTRAST2}_'
                f's_{config.TRAINING.SEED}_shuf_{config.TRAINING.SHUFFELING}_'
    )


    # output_size
    if config.TRAINING.CONTRAST1_ONLY or config.TRAINING.CONTRAST2_ONLY:
        output_size = 1
        if config.TRAINING.CONTRAST1_ONLY:
            model_name = f'{model_name}_CT1_ONLY_'
        else:
            model_name = f'{model_name}_CT2_ONLY_'

    else:
        output_size = 2

    # Embeddings
    if config.MODEL.USE_FF:
        mapping_size = config.FOURIER.MAPPING_SIZE  # of FF
        input_size = 2* mapping_size
        B_gauss = torch.tensor(np.random.normal(scale=config.FOURIER.FF_SCALE, size=(config.FOURIER.MAPPING_SIZE, 3)), dtype=torch.float32).to(device)
        input_mapper = input_mapping(B=B_gauss, factor=config.FOURIER.FF_FACTOR).to(device)
        model_name = f'{model_name}_FF_{get_string(config_dict["FOURIER"])}_'

    else:
        input_size = 3

    # Model Selection
    if config.MODEL.USE_SIREN:
        model = Siren(in_features=input_size, out_features=output_size, hidden_features=config.MODEL.HIDDEN_CHANNELS,
                    hidden_layers=config.MODEL.NUM_LAYERS, first_omega_0=config.SIREN.FIRST_OMEGA_0, hidden_omega_0=config.SIREN.HIDDEN_OMEGA_0)   # no dropout implemented
        model_name = f'{model_name}_SIREN_{get_string(config_dict["SIREN"])}_'
    elif config.MODEL.USE_WIRE_REAL:
        model = WireReal(in_features=input_size, out_features=output_size, hidden_features=config.MODEL.HIDDEN_CHANNELS,
                    hidden_layers=config.MODEL.NUM_LAYERS, 
                    first_omega_0=config.WIRE.WIRE_REAL_FIRST_OMEGA_0, hidden_omega_0=config.WIRE.WIRE_REAL_HIDDEN_OMEGA_0,
                    first_s_0=config.WIRE.WIRE_REAL_FIRST_S_0, hidden_s_0=config.WIRE.WIRE_REAL_HIDDEN_S_0
                    )
        model_name = f'{model_name}_WIRE_{get_string(config_dict["WIRE"])}_'   
    
    else:
        if config.MODEL.USE_TWO_HEADS:
            if (config.TRAINING.CONTRAST1_ONLY or config.TRAINING.CONTRAST2_ONLY) == True:
                raise ValueError('Do not use MLPv2 for single contrast.')
            if config.MODEL.USE_REGISTRATION:
                model = MLPregv1(input_size=input_size, output_size=output_size, hidden_size=config.MODEL.HIDDEN_CHANNELS,
                        num_layers=config.MODEL.NUM_LAYERS, dropout=config.MODEL.DROPOUT)
                model_name = f'{model_name}_MLPregv1_'
            else:
                model = MLPv2(input_size=input_size, output_size=output_size, hidden_size=config.MODEL.HIDDEN_CHANNELS,
                            num_layers=config.MODEL.NUM_LAYERS, dropout=config.MODEL.DROPOUT)
                model_name = f'{model_name}_MLP2_'
        else:
            model = MLPv1(input_size=input_size, output_size=output_size, hidden_size=config.MODEL.HIDDEN_CHANNELS,
                        num_layers=config.MODEL.NUM_LAYERS, dropout=config.MODEL.DROPOUT)
            model_name = f'{model_name}_MLP2_'

    model.to(device)
    
    return model, model_name, input_mapper

def create_losses(config, config_dict, model_name, device):
    lpips_loss = lpips.LPIPS(net='alex').to(device)

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

def create_datasets(config):
    dataset = MultiModalDataset(
                    image_dir = config.SETTINGS.DIRECTORY,
                    name = config.SETTINGS.PROJECT_NAME,
                    subject_id=config.DATASET.SUBJECT_ID,
                    contrast1_LR_str=config.DATASET.LR_CONTRAST1,
                    contrast2_LR_str=config.DATASET.LR_CONTRAST2, 
                    )
    
    train_dataloader = DataLoader(dataset, batch_size=config.TRAINING.BATCH_SIZE, 
                                 shuffle=config.TRAINING.SHUFFELING, 
                                 num_workers=config.SETTINGS.NUM_WORKERS)
    
    mgrid_contrast1 = dataset.get_contrast1_coordinates()
    mgrid_contrast2 = dataset.get_contrast2_coordinates()
    mgrid_affine_contrast1 = dataset.get_contrast1_affine()
    mgrid_affine_contrast2 = dataset.get_contrast2_affine()

    infer_data_contrast = InferDataset(torch.cat((mgrid_contrast1, mgrid_contrast2), dim=0))
    threshold = len(mgrid_contrast1)
    infer_dataloader = torch.utils.data.DataLoader(infer_data_contrast,
                                               batch_size=5000,
                                               shuffle=False,
                                               num_workers=config.SETTINGS.NUM_WORKERS)
    
    return dataset, train_dataloader, infer_dataloader, threshold