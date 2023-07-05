import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader

from utils.utils import center_of_mass
from utils.dataset.dataset_utils import norm_grid
from utils.dataset.dataset import MultiModalDataset, InferDataset
from experiments.parallel_registration.model import MLPv1, MLPv2, Siren, WireReal, MLPregv1, MLPregv2, MLP_SIRENreg


def create_model(config, config_dict, device):
    # Model Selection
    model_name = (
                f'{config.SETTINGS.PROJECT_NAME}_subid-{config.DATASET.SUBJECT_ID}_'
                f'ct1LR-{config.DATASET.LR_CONTRAST1}_ct2LR-{config.DATASET.LR_CONTRAST2}_'
                f's_{config.TRAINING.SEED}_shuf_{config.TRAINING.SHUFFELING}_'
    )
    # Embeddings
    if config.MODEL.USE_FF:
        mapping_size = config.FOURIER.MAPPING_SIZE  # of FF
        input_size = 2* mapping_size
    else:
        input_size = 3
        
    output_size = 2

    # Model Selection
    if config.MODEL.USE_SIREN:
        model = Siren(in_features=input_size, out_features=output_size, hidden_features=config.MODEL.HIDDEN_CHANNELS,
                    hidden_layers=config.MODEL.NUM_LAYERS, first_omega_0=config.SIREN.FIRST_OMEGA_0, hidden_omega_0=config.SIREN.HIDDEN_OMEGA_0)
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
            model = MLPregv1(input_size=input_size, output_size=output_size, hidden_size=config.MODEL.HIDDEN_CHANNELS,
                    num_layers=config.MODEL.NUM_LAYERS, dropout=config.MODEL.DROPOUT)
            model_name = f'{model_name}_MLPregv1_'
        else:
            model = MLPv1(input_size=input_size, output_size=output_size, hidden_size=config.MODEL.HIDDEN_CHANNELS,
                        num_layers=config.MODEL.NUM_LAYERS, dropout=config.MODEL.DROPOUT)
            model_name = f'{model_name}_MLP2_'

    model.to(device)
    
    return model, model_name

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
                                               batch_size=10000,
                                               shuffle=False,
                                               num_workers=config.SETTINGS.NUM_WORKERS)
    
    
    return dataset, train_dataloader, infer_dataloader, threshold

def compute_dataset_artifacts(dataset, device):
    # Dimensions of the two images
    x_dim_c1, y_dim_c1, z_dim_c1 = dataset.get_contrast1_dim()
    x_dim_c2, y_dim_c2, z_dim_c2 = dataset.get_contrast2_dim()
    x_dim_c2_lr, y_dim_c2_lr, z_dim_c2_lr = dataset.get_contrast2_lr_dim()
    x_dim_c1_lr, y_dim_c1_lr, z_dim_c1_lr = dataset.get_contrast1_lr_dim()

    # Image to be registered
    fixed_image_unprocessed = dataset.get_contrast2_intensities().reshape((x_dim_c2_lr, y_dim_c2_lr, z_dim_c2_lr))
    # Créer un nouveau tableau y de taille (n+1), (m+1), (p+1) rempli de zéros
    n, m, p = fixed_image_unprocessed.shape
    fixed_image = torch.zeros((n+2, m+2, p+2))

    # Copier les valeurs de x dans y sauf pour les indices i=0 ou n, j=0 ou m et k=0 ou p
    fixed_image[1:-1, 1:-1, 1:-1] = fixed_image_unprocessed

    moving_image = dataset.get_contrast1_intensities().reshape((x_dim_c1_lr, y_dim_c1_lr, z_dim_c1_lr))

    
    # Maximum and minimum coordinates of the training points (used in fast_trilinear_interpolation)
    coord_c2 = dataset.get_contrast2_data().cpu().numpy()
    affine = dataset.get_contrast2_affine().cpu().numpy()
    rev_affine = np.linalg.inv(affine[:3,:3])
    res = rev_affine @ coord_c2.T
    res = res.T
    affine1 = dataset.get_contrast1_affine().cpu().numpy()
    rev_affine1 = np.linalg.inv(affine1)

    center_of_mass_c2 = nib.affines.apply_affine(affine, center_of_mass(fixed_image.cpu().numpy()))
    center_of_mass_c1 = nib.affines.apply_affine(affine1, center_of_mass(moving_image.cpu().numpy()))
    max_coords = [np.max(res[:,i]) for i in range(3)]
    min_coords = [-np.max(-res[:,i]) for i in range(3)]
    rev_affine = torch.tensor(rev_affine, device=device)

    dim_c1 = [x_dim_c1_lr, y_dim_c1_lr, z_dim_c1_lr]
    dim_c2 = [x_dim_c2_lr, y_dim_c2_lr, z_dim_c2_lr]
    for i in range(3):
        center_of_mass_c2[i] = norm_grid(center_of_mass_c2[i], 0, dim_c2[i], smin=min_coords[i], smax=max_coords[i])  
        center_of_mass_c1[i] = norm_grid(center_of_mass_c1[i], 0, dim_c1[i], smin=min_coords[i], smax=max_coords[i])  


    difference_center_of_mass = torch.tensor((center_of_mass_c2 - center_of_mass_c1), device=device, dtype=float)
    format_im = torch.sub(torch.tensor(max_coords, device=device, dtype=float), 
                          torch.tensor(min_coords, device=device, dtype=float))
    
    return fixed_image, rev_affine, min_coords, max_coords, difference_center_of_mass, format_im 