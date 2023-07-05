import numpy as np
import torch

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
    
    data = torch.cat((contrast1_data,contrast2_data), dim=0)
    
    if torch.cuda.is_available():
        data, contrast1_labels, contrast2_labels  = data.to(device=device), contrast1_labels.to(device=device), contrast2_labels.to(device=device)
        contrast1_data, contrast2_data = contrast1_data.to(device=device), contrast2_data.to(device=device)
    
    raw_data = data
    if config.MODEL.USE_FF:
        contrast1_data = input_mapper(contrast1_data)
    elif config.MODEL.USE_SIREN:
        contrast2_data = contrast2_data*np.pi
    
    raw_contrast2_data = contrast2_data
    contrast2_data = contrast2_data*np.pi
        
    return raw_data,raw_contrast2_data, contrast1_data, contrast2_data, contrast1_labels, contrast2_labels

def process_output(target_contrast1, target_contrast2, raw_data, config):
    mi_target1 = torch.cat([target_contrast1[:, 0], target_contrast2[:, 0]], dim=0)
    mi_target2 = torch.cat([target_contrast1[:, 1], target_contrast2[:, 1]], dim=0)
        
    mse_target1 = target_contrast1[:,0]
    mse_target2 = target_contrast2[:,1]
    
    return mse_target1, mse_target2, mi_target1, mi_target2
    
def compute_similarity_loss(mse_target1, contrast1_labels, mse_target2, contrast2_labels, 
                            mi_target1, mi_target2, criterion, mi_criterion, cc_criterion):
    
    mse_loss_c1 = criterion(mse_target1, contrast1_labels.squeeze()) 
    mse_loss_c2 = criterion(mse_target2, contrast2_labels.squeeze())  
    cc_loss_registration = cc_criterion(mi_target1.unsqueeze(0).unsqueeze(0).detach(), mi_target2.unsqueeze(0).unsqueeze(0))
    mi_loss_registration = mi_criterion(mi_target1.unsqueeze(0).unsqueeze(0).detach(), mi_target2.unsqueeze(0).unsqueeze(0))
    
    return mse_loss_c1, mse_loss_c2, cc_loss_registration, mi_loss_registration

def compute_regularization_loss(data_contrast2, registration_target, difference_center_of_mass, format_im, device):
    norm_loss = torch.tensor(0, device=device, dtype=float)
    difference_center_of_mass = difference_center_of_mass.float()
    for i in range(3):
        norm_loss += torch.mean(((registration_target[:, i] - 0*difference_center_of_mass[i])/(1/2*format_im[i]))**2)

    norm_registration = torch.sum(torch.linalg.vector_norm(registration_target, dim=1))
    
    jacobian_loss = compute_jacobian_loss(data_contrast2.to(device=device), registration_target, device, batch_size=len(data_contrast2))
    hyper_elastic_loss = compute_hyper_elastic_loss(data_contrast2.to(device=device), registration_target, device, batch_size=len(data_contrast2))
    bending_energy = compute_bending_energy(data_contrast2.to(device=device), registration_target, device, batch_size=len(data_contrast2))
    
    return norm_loss, jacobian_loss, hyper_elastic_loss, bending_energy, norm_registration

def update_wandb_batch_dict(name_to_metrics, wandb_batch_dict):
    for metric in name_to_metrics:
        wandb_batch_dict.update({metric[0]: metric[1].detach().item()})
        
    return wandb_batch_dict
        
