import torch
import numpy as np
import time
import torch.nn as nn
from utils.utils import input_mapping, get_string, fast_trilinear_interpolation
from loss_functions import compute_jacobian_loss, compute_hyper_elastic_loss, compute_bending_energy
from utils.utils import input_mapping, compute_metrics, dict2obj, get_string, compute_mi_hist, compute_mi
from utils.utils_visualization import show_slices_gt


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
    

def forward_iteration(model, raw_data, labels, wandb_batch_dict, epoch, model_name, config, args, device, input_mapper, 
                            fixed_image, lpips_loss, criterion, mi_criterion, cc_criterion, min_coords, max_coords, rev_affine):
    raw_data, data, contrast1_labels, contrast2_labels = config_data(raw_data, labels, device, config, input_mapper)    
    target = model(data)
    
    if config.MODEL.USE_SIREN or config.MODEL.USE_WIRE_REAL: # TODO check syntax compatibility
        target, _ = target
    
    # compute the loss on both modalities!
    # target = torch.where(intensity_index, target[:, 1], target[:, 0])
    mse_target1 = target[:len(contrast1_labels),0]  # contrast1 output for contrast1 coordinate
    #mse_target2 = target[len(contrast1_labels):,1]  # contrast2 output for contrast2 coordinate
    #registration_target = target[len(contrast1_labels):,2:5].to(device=device)
    registration_target = target[:,2:5].to(device=device)
    # target_mse = torch.cat((mi_target1, mi_target2), dim=0)
    
    if config.MI_CC.MI_USE_PRED:
        mi_target1 = target[:,0:1]
        mi_target2 = target[:,1:2]

    elif config.TRAINING.USE_MI or config.TRAINING.USE_NMI or config.TRAINING.USE_CC:
        mi_target1 = target[:len(contrast1_labels),0][contrast1_segm.squeeze()]  # contrast2 output for contrast1 coordinate !! ETRANGE !!
        mi_target2 = target[len(contrast1_labels):,1][contrast2_segm.squeeze()]   # contrast1 output for contrast2 coordinate
        
    #coord_temp = torch.add(registration_target, raw_data[len(contrast1_labels):].to(device=device))
    coord_temp = torch.add(registration_target, raw_data.to(device=device))
    
    #print(registration_target)
    #coord_temp = raw_data[len(contrast1_labels):].to(device=device)
    
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
    #mi_target2 = contrast2_interpolated.unsqueeze(1)

    
    loss = criterion(mse_target1, contrast1_labels.squeeze()) + criterion(target[:,1], contrast2_interpolated.squeeze())

    wandb_batch_dict.update({'mse_loss': loss.item()})
    
    bonus_loss = torch.linalg.matrix_norm(registration_target)
    
    wandb_batch_dict.update({'bonus_loss': bonus_loss.item()})
    """
    if epoch < 50:
        loss += 10*bonus_loss
    
    jacobian_loss = compute_jacobian_loss(raw_data.to(device=device), registration_target, batch_size=len(data))
    
    #bending_energy = compute_bending_energy(raw_data.to(device=device), registration_target, batch_size=len(data))
    wandb_batch_dict.update({'jacobian_loss': jacobian_loss.item()})
    
    wandb_batch_dict.update({'hyperlastic_loss': hyperlastic_loss.item()})
    #wandb_batch_dict.update({'bending_energy': bending_energy.item()})
    loss += 0.05*hyperlastic_loss #+ 0.1*jacobian_loss + 0.1*hyperlastic_loss #+ 0.01*bending_energy
    """
    jacobian_loss = compute_jacobian_loss(raw_data.to(device=device), registration_target, batch_size=len(data))
    wandb_batch_dict.update({'jacobian_loss': jacobian_loss.item()})
    loss += 0.05*jacobian_loss
    if epoch > -1:
        hyperlastic_loss = compute_hyper_elastic_loss(raw_data.to(device=device), registration_target, batch_size=len(data))
        wandb_batch_dict.update({'hyperlastic_loss': hyperlastic_loss.item()})
        loss += 0.05*hyperlastic_loss
    # mutual information loss
    if config.TRAINING.USE_MI or config.TRAINING.USE_NMI:
        if config.MI_CC.MI_USE_PRED:
            mi_loss = mi_criterion(mi_target1.unsqueeze(0).unsqueeze(0).detach(), mi_target2.unsqueeze(0).unsqueeze(0))
            if epoch > -1:
                loss -= 0.001*config.MI_CC.LOSS_MI*(mi_loss)
            if args.logging:
                wandb_batch_dict.update({'mi_loss': (mi_loss).item()})
        else:
            mi_loss1 = mi_criterion(mi_target1.unsqueeze(0).unsqueeze(0), contrast1_labels[contrast1_segm].unsqueeze(0).unsqueeze(0))
            mi_loss2 = mi_criterion(mi_target2.unsqueeze(0).unsqueeze(0), contrast2_labels[contrast2_segm].unsqueeze(0).unsqueeze(0))
            #loss += config.MI_CC.LOSS_MI*(mi_loss1+mi_loss2).to(torch.float)
            if args.logging:
                wandb_batch_dict.update({'mi_loss': (mi_loss1+mi_loss2).item()})
                
    if config.TRAINING.USE_CC:
        '''
        cc_loss1 = cc_criterion(mi_target1.unsqueeze(0).unsqueeze(0), contrast1_labels[contrast1_segm].unsqueeze(0).unsqueeze(0))
        cc_loss2 = cc_criterion(mi_target2.unsqueeze(0).unsqueeze(0), contrast2_labels[contrast2_segm].unsqueeze(0).unsqueeze(0))
        #loss += config.MI_CC.LOSS_CC*(cc_loss1+cc_loss2)
        if args.logging:
            wandb_batch_dict.update({'cc_loss': -(cc_loss1+cc_loss2).item()})
        '''
        cc_loss = cc_criterion(mi_target1.unsqueeze(0).unsqueeze(0).detach(), mi_target2.unsqueeze(0).unsqueeze(0))
        if epoch > -1:
            loss += 0.001*cc_loss
        if args.logging:
            wandb_batch_dict.update({'cc_loss': -(cc_loss).item()})
    
            
    return loss, wandb_batch_dict

def compute_and_log_metrics(gt_contrast1, gt_contrast2, pred_contrast1, pred_contrast2, mask_c1, mask_c2, lpips_loss, device, wandb_epoch_dict, args):
    metrics_contrast1 = compute_metrics(gt=gt_contrast1.copy(), pred=pred_contrast1.copy(), mask=mask_c1, lpips_loss=lpips_loss, device=device)
    metrics_contrast2 = compute_metrics(gt=gt_contrast2.copy(), pred=pred_contrast2.copy(), mask=mask_c2, lpips_loss=lpips_loss, device=device)
    '''
    metrics_mi_true = compute_mi_hist(gt_contrast1.copy(), gt_contrast2.copy(), mask_c2, bins=32)
    metrics_mi_1 = compute_mi_hist(pred_contrast1.copy(), gt_contrast2.copy(), mask_c2, bins=32)
    metrics_mi_2 = compute_mi_hist(pred_contrast2.copy(), gt_contrast1.copy(), mask_c2, bins=32)
    metrics_mi_pred = compute_mi_hist(pred_contrast1.copy(), pred_contrast2.copy(), mask_c2, bins=32)

    metrics_mi_approx = compute_mi(pred_contrast1.copy(), pred_contrast2.copy(), mask_c2,device)
    '''
    if args.logging:
        wandb_epoch_dict.update({f'contrast1_ssim': metrics_contrast1["ssim"]})
        wandb_epoch_dict.update({f'contrast1_psnr': metrics_contrast1["psnr"]})
        wandb_epoch_dict.update({f'contrast1_lpips': metrics_contrast1["lpips"]})
        wandb_epoch_dict.update({f'contrast2_ssim': metrics_contrast2["ssim"]})
        wandb_epoch_dict.update({f'contrast2_psnr': metrics_contrast2["psnr"]})
        wandb_epoch_dict.update({f'contrast2_lpips': metrics_contrast2["lpips"]})
        #wandb_epoch_dict.update({f'mutual_information': metrics_mi_pred["mi"]})
        #wandb_epoch_dict.update({f'mutual_information_appox': metrics_mi_approx["mi"]})
        #wandb_epoch_dict.update({f'MI_error_contrast1': np.abs(metrics_mi_1["mi"]-metrics_mi_true["mi"])})
        #wandb_epoch_dict.update({f'MI_error_contrast2': np.abs(metrics_mi_2["mi"]-metrics_mi_true["mi"])})
        #wandb_epoch_dict.update({f'MI_error_pred': np.abs(metrics_mi_pred["mi"]-metrics_mi_true["mi"])})

        
    return wandb_epoch_dict

