import torch
import numpy as np
from experiments.parallel_registration.experiment_utils.utils_training import config_data, process_output, compute_similarity_loss, compute_regularization_loss, update_wandb_batch_dict
from utils.loss_functions.utils_loss import compute_jacobian_matrix
    

def forward_iteration(model, raw_data, labels, mask, wandb_batch_dict, epoch, model_name, config, device, input_mapper, 
                      fixed_image, criterion, mi_criterion, cc_criterion, min_coords, max_coords, rev_affine,
                      difference_center_of_mass, format_im, **kwargs):
    
    raw_data, data, contrast1_labels, contrast2_labels, contrast1_mask, contrast2_mask = config_data(raw_data, labels, mask, 
                                                                                                     device, config, input_mapper)
    
    target = model(data)
    
    if config.MODEL.USE_SIREN or config.MODEL.USE_WIRE_REAL: 
        target, _ = target
    
    mse_target1, mse_target2, contrast2_interpolated, registration_target, mi_target1, mi_target2 = process_output(target, raw_data, 
                                                                                                                   len(contrast1_labels), 
                                                                                                                   fixed_image, 
                                                                                                                   rev_affine, max_coords,
                                                                                                                   min_coords, format_im, config, 
                                                                                                                   device)
    reg_lr_multiplier = 1
    
    mse_loss_c1, mse_loss_c2, cc_loss_registration, mi_loss_registration = compute_similarity_loss(mse_target1, contrast1_labels, contrast1_mask,
                                                                                                   mse_target2, contrast2_interpolated,
                                                                                                   criterion, mi_criterion, cc_criterion)
    
    norm_loss, jacobian_loss, hyper_elastic_loss, bending_energy, norm_registration = compute_regularization_loss(raw_data, registration_target, 
                                                                                                                difference_center_of_mass, format_im,
                                                                                                                device)
    weights = config.TRAINING.LOSS_WEIGHT
    similarity_loss = weights.MSE_C1*mse_loss_c1 + weights.MSE_C2*mse_loss_c2
    similarity_loss += reg_lr_multiplier*(weights.CC*cc_loss_registration - 0*weights.MI*mi_loss_registration)

    regularization_loss = weights.BENDING_ENERGY*bending_energy + weights.JACOBIAN*jacobian_loss
    if norm_loss > 1:
        regularization_loss += reg_lr_multiplier*weights.NORM_LOSS*norm_loss**2
    if hyper_elastic_loss < 100:
        regularization_loss += weights.HYPER_ELASTIC*hyper_elastic_loss
        
    loss = similarity_loss + regularization_loss
    
    metrics = [('similarity_loss',similarity_loss), ('regularization_loss',regularization_loss), ('mse_loss_c1' ,mse_loss_c1), 
                ('mse_loss_c2',mse_loss_c2), ('cc_loss_registration',cc_loss_registration), 
                ('mi_loss_registration' ,mi_loss_registration), ('norm_registration',norm_registration), ('norm_loss',norm_loss), 
                ('jacobian_loss',jacobian_loss), ('hyper_elastic_loss',hyper_elastic_loss), ('bending_energy',bending_energy)]
    
    wandb_batch_dict = update_wandb_batch_dict(metrics, wandb_batch_dict)
    
    wandb_batch_dict.update({'x_registration': torch.mean(registration_target[:, 0]).detach().item()})
    wandb_batch_dict.update({'y_registration': torch.mean(registration_target[:, 1]).detach().item()})
    wandb_batch_dict.update({'z_registration': torch.mean(registration_target[:, 2]).detach().item()})
                     
    return loss, cc_loss_registration, wandb_batch_dict

def inference_iteration(model, raw_data, config, device, input_mapper, fixed_image, min_coords, max_coords, rev_affine,
                        difference_center_of_mass, format_im, **kwargs):
    if torch.cuda.is_available():
        raw_data = raw_data.to(device)
                
    if config.MODEL.USE_FF:
        data = input_mapper(raw_data)
    elif config.MODEL.USE_SIREN:
        data = raw_data*np.pi
    else:
        data = raw_data

    output = model(data)
    
    if config.MODEL.USE_SIREN or config.MODEL.USE_WIRE_REAL:
        output, _ = output
    
    _, _, contrast2_interpolated, registration_target, _, _ = process_output(output, raw_data, 
                                                                             0, fixed_image, rev_affine, max_coords,
                                                                             min_coords, format_im, config, device)
    
    registration = output[:,2:4].to(device=device)
    registration = torch.cat((registration, torch.zeros_like(registration[:, 0:1])), dim=1)

    jac = compute_jacobian_matrix(raw_data, registration, device)
    jac_norm = torch.norm(jac, dim=(1, 2)).unsqueeze(1)
    #det = torch.det(jac) - 1
    registration_norm = torch.norm(registration, dim=1).unsqueeze(1)
    contrast2_interpolated = contrast2_interpolated.unsqueeze(1)
        
    res = torch.concat([output[:,:2], registration, contrast2_interpolated, jac_norm, registration_norm], dim=1).cpu().detach().numpy() 
    return res


