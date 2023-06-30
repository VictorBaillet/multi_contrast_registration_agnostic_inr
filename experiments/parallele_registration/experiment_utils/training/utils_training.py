import numpy as np
import torch

from utils.utils import fast_trilinear_interpolation
from utils.loss_functions.loss_functions import compute_jacobian_loss, compute_hyper_elastic_loss, compute_bending_energy


def config_data(data, labels, device, config, input_mapper):
    contrast1_mask = (labels[:,0] != -1.0)
    contrast1_labels = labels[contrast1_mask,0]
    contrast1_labels = contrast1_labels.reshape(-1,1).to(device=device)
    contrast1_segm = None #segm[contrast1_mask,:]
    contrast1_data = data[contrast1_mask,:]

    contrast2_mask = (labels[:,1] != -1.0)
    contrast2_labels = labels[contrast2_mask,1]
    contrast2_labels = contrast2_labels.reshape(-1,1).to(device=device)
    contrast2_segm = None #segm[contrast2_mask,:]
    contrast2_data = data[contrast2_mask,:]
    
    #data = torch.cat((contrast1_data,contrast2_data), dim=0)
    data = contrast1_data
    
    if torch.cuda.is_available():
        data, contrast1_labels, contrast2_labels  = data.to(device=device), contrast1_labels.to(device=device), contrast2_labels.to(device=device)
    
    raw_data = data
    if config.MODEL.USE_FF:
        data = input_mapper(data)
    elif config.MODEL.USE_SIREN:
        data = data*np.pi
        
    return raw_data, data, contrast1_labels, contrast2_labels

def process_output(target, raw_data, threshold, fixed_image, rev_affine, max_coords, min_coords, format_im, config, device):
    mse_target1 = target[:threshold,0]  # contrast1 output for contrast1 coordinate
    mse_target2 = target[:,1]  # contrast2 output for contrast2 coordinate
    registration_target = target[:,2:5].to(device=device)
    
    if config.MI_CC.MI_USE_PRED:
        mi_target1 = target[:,0:1]
        mi_target2 = target[:,1:2]

    elif config.TRAINING.USE_MI or config.TRAINING.USE_NMI or config.TRAINING.USE_CC:
        mi_target1 = target[:len(contrast1_labels),0][contrast1_segm.squeeze()]  # contrast2 output for contrast1 coordinate !! ETRANGE !!
        mi_target2 = target[len(contrast1_labels):,1][contrast2_segm.squeeze()]   # contrast1 output for contrast2 coordinate
    
    registration_target = torch.mul(registration_target, format_im)
    coord_temp = torch.add(registration_target, raw_data.to(device=device))
    
    contrast2_interpolated = fast_trilinear_interpolation(
        fixed_image,
        coord_temp[:, 0],
        coord_temp[:, 1],
        coord_temp[:, 2],
        min_coords,
        max_coords,
        device,
        rev_affine
    )
    
    return mse_target1, mse_target2, contrast2_interpolated, registration_target, mi_target1, mi_target2

def compute_similarity_loss(mse_target1, contrast1_labels, mse_target2, contrast2_interpolated, criterion, mi_criterion, cc_criterion):
    
    mse_loss_c1 = criterion(mse_target1, contrast1_labels.squeeze()) 

    mse_loss_c2 = criterion(mse_target2, contrast2_interpolated.squeeze().detach())  
    
    cc_loss_registration = cc_criterion(contrast1_labels.unsqueeze(0).unsqueeze(0).detach(),
                                        contrast2_interpolated.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
    
    mi_loss_registration = mi_criterion(contrast1_labels.unsqueeze(0).unsqueeze(0).detach(),
                                        contrast2_interpolated.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
    
    return mse_loss_c1, mse_loss_c2, cc_loss_registration, mi_loss_registration

def compute_regularization_loss(raw_data, registration_target, difference_center_of_mass, format_im, device):
    norm_loss = torch.tensor(0, device=device, dtype=float)
    difference_center_of_mass = difference_center_of_mass.float()
    for i in range(3):
        norm_loss += torch.mean(((registration_target[:, i] - 0*difference_center_of_mass[i])/(1/2*format_im[i]))**2)
        
    norm_registration = torch.sum(torch.linalg.vector_norm(registration_target, dim=1))
    jacobian_loss = compute_jacobian_loss(raw_data.to(device=device), registration_target, device, batch_size=len(raw_data))
    hyper_elastic_loss = compute_hyper_elastic_loss(raw_data.to(device=device), registration_target, device, batch_size=len(raw_data))
    bending_energy = compute_bending_energy(raw_data.to(device=device), registration_target, device, batch_size=len(raw_data))
    
    return norm_loss, jacobian_loss, hyper_elastic_loss, bending_energy, norm_registration

def update_wandb_batch_dict(name_to_metrics, wandb_batch_dict):
    for metric in name_to_metrics:
        wandb_batch_dict.update({metric[0]: metric[1].item()})
        
    return wandb_batch_dict
        
