import torch
import numpy as np
from utils.utils import fast_trilinear_interpolation
from models.serial_registration.experiment_utils.utils_training import config_data, process_output, compute_similarity_loss, compute_regularization_loss, update_wandb_batch_dict

from utils.loss_functions.utils_loss import compute_jacobian_matrix
    

def forward_iteration(network, network_registration, raw_data, labels, wandb_batch_dict, epoch, config, device, input_mapper, moving_image, 
                      criterion, mi_criterion, cc_criterion, min_coords, max_coords, rev_affine, difference_center_of_mass, format_im, **kwargs):
    
    raw_data, raw_contrast2_data, data_contrast1, data_contrast2, contrast1_labels, contrast2_labels = config_data(raw_data, labels, device, config, input_mapper)    
    
    target_contrast1 = network(data_contrast1)
    registration_target, _ = network_registration(data_contrast2) 
    registration_target = torch.mul(registration_target, format_im).float()
    target_contrast2 = network(input_mapper(torch.add(registration_target, raw_contrast2_data)))
    
    mse_target1, mse_target2, mi_target1, mi_target2 = process_output(target_contrast1, target_contrast2, raw_data, config)
    
    mse_loss_c1, mse_loss_c2, cc_loss_registration, mi_loss_registration = compute_similarity_loss(mse_target1, contrast1_labels,
                                                                                                   mse_target2, contrast2_labels,
                                                                                                   mi_target1, mi_target2,
                                                                                                   criterion, mi_criterion, cc_criterion)
    
    norm_loss, jacobian_loss, hyper_elastic_loss, bending_energy, norm_registration = compute_regularization_loss(data_contrast2, 
                                                                                                                registration_target,
                                                                                                                difference_center_of_mass, format_im,
                                                                                                                device)
    weights = config.TRAINING.LOSS_WEIGHT
    
    mse_loss = weights.MSE_C1*mse_loss_c1 + weights.MSE_C2*mse_loss_c2    
    
    registration_loss = weights.JACOBIAN*jacobian_loss + weights.BENDING_ENERGY*bending_energy 
    registration_loss =+ (weights.CC*cc_loss_registration - weights.MI*mi_loss_registration)  
    if norm_loss > 1:
        registration_loss += weights.NORM_LOSS*norm_loss**2
    if hyper_elastic_loss < 100:
        registration_loss += weights.HYPER_ELASTIC*hyper_elastic_loss**2
        
    metrics = [('mse_loss' ,mse_loss), ('registration_loss' ,registration_loss), ('mse_loss_c1' ,mse_loss_c1), 
               ('mse_loss_c2',mse_loss_c2), ('cc_loss_registration',cc_loss_registration), 
               ('mi_loss_registration' ,mi_loss_registration), ('norm_registration',norm_registration), ('norm_loss',norm_loss), 
               ('jacobian_loss',jacobian_loss), ('hyper_elastic_loss',hyper_elastic_loss), ('bending_energy',bending_energy)]
    
    wandb_batch_dict = update_wandb_batch_dict(metrics, wandb_batch_dict)
    
    wandb_batch_dict.update({'x_registration': torch.mean(registration_target[:, 0]).detach().item()})
    wandb_batch_dict.update({'y_registration': torch.mean(registration_target[:, 1]).detach().item()})
    wandb_batch_dict.update({'z_registration': torch.mean(registration_target[:, 2]).detach().item()})
    
    
    return registration_loss, mse_loss, wandb_batch_dict

def inference_iteration_contrast2(network, network_registration, raw_data, config, device, moving_image, input_mapper,
                                  min_coords, max_coords, rev_affine, format_im, **kwargs):
    
    if torch.cuda.is_available():
        raw_data = raw_data.to(device)
    data = raw_data*np.pi
            
    registration_output, _ = network_registration(data) 
    registration_output = torch.mul(registration_output, format_im).float()
    output = network(input_mapper(torch.add(registration_output, raw_data)))
    jac = compute_jacobian_matrix(data, registration_output, device)
    jac_norm = torch.norm(jac, dim=(1, 2)).unsqueeze(1)
    #det = torch.det(jac) - 1

    registration_norm = torch.norm(registration_output, dim=1).unsqueeze(1)
    
    coord_temp = torch.add(registration_output, raw_data).to(device=device)
    #coord_temp = raw_data
    contrast2_interpolated = fast_trilinear_interpolation(
        moving_image,
        coord_temp[:, 0],
        coord_temp[:, 1],
        coord_temp[:, 2],
        min_coords,
        max_coords,
        device,
        rev_affine
    )
    
    contrast2_interpolated = contrast2_interpolated.unsqueeze(1)
    res = torch.concat([output, registration_output, contrast2_interpolated, jac_norm, registration_norm], dim=1).cpu().detach().numpy()
    return res
