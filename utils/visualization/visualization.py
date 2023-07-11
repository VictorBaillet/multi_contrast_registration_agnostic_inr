import numpy as np
import nibabel as nib
import wandb
import os

from utils.visualization.utils_visualization import show_slices_gt, show_slices_registration, show_jacobian_det, compute_metrics


def generate_NIFTIs(dataset, model_intensities, image_dir, model_name_epoch, epoch, wandb_epoch_dict, config):
    x_dim_c1, y_dim_c1, z_dim_c1 = dataset.get_dim(contrast=1, resolution='gt')
    x_dim_c2, y_dim_c2, z_dim_c2 = dataset.get_dim(contrast=2, resolution='gt')
    threshold = len(dataset.get_coordinates(contrast=1, resolution='gt'))
    model_intensities_contrast1 = model_intensities[:threshold,0] # contrast1
    model_intensities_contrast2 = model_intensities[threshold:,1] # contrast2
    model_intensities_contrast2_interpolated = model_intensities[threshold:,5] # contrast2
    model_registration = model_intensities[threshold:,2:5]
    model_registration_jac_det = model_intensities[threshold:,6]
    model_registration_norm = model_intensities[threshold:,7]

    label_arr = np.array(model_intensities_contrast1, dtype=np.float32)
    model_intensities_contrast1= np.clip(label_arr.reshape(-1, 1), 0, 1)

    label_arr = np.array(model_intensities_contrast2, dtype=np.float32)
    model_intensities_contrast2= np.clip(label_arr.reshape(-1, 1), 0, 1)
    
    label_arr = np.array(model_intensities_contrast2_interpolated, dtype=np.float32)
    model_intensities_contrast2_interpolated= np.clip(label_arr.reshape(-1, 1), 0, 1)

    img_contrast1 = model_intensities_contrast1.reshape((x_dim_c1, y_dim_c1, z_dim_c1))#.cpu().numpy()
    img_contrast2 = model_intensities_contrast2.reshape((x_dim_c2, y_dim_c2, z_dim_c2))#.cpu().numpy()
    img_contrast2_interpolated = model_intensities_contrast2_interpolated.reshape((x_dim_c2, y_dim_c2, z_dim_c2))#.cpu().numpy()
    img_registration = model_registration.reshape((x_dim_c2, y_dim_c2, z_dim_c2, 3))#.cpu().numpy()
    model_registration_jac_det = model_registration_jac_det.reshape((x_dim_c2, y_dim_c2, z_dim_c2))#.cpu().numpy()
    model_registration_norm = model_registration_norm.reshape((x_dim_c2, y_dim_c2, z_dim_c2))

    gt_contrast1 = dataset.get_intensities(contrast=1, resolution='gt').reshape((x_dim_c1, y_dim_c1, z_dim_c1)).cpu().numpy()
    gt_contrast2 = dataset.get_intensities(contrast=2, resolution='gt').reshape((x_dim_c2, y_dim_c2, z_dim_c2)).cpu().numpy()

    label_arr = np.array(gt_contrast1, dtype=np.float32)
    gt_contrast1= np.clip(gt_contrast1.reshape(-1, 1), 0, 1).reshape((x_dim_c1, y_dim_c1, z_dim_c1))

    label_arr = np.array(gt_contrast2, dtype=np.float32)
    gt_contrast2= np.clip(gt_contrast2.reshape(-1, 1), 0, 1).reshape((x_dim_c2, y_dim_c2, z_dim_c2))

    pred_contrast1 = img_contrast1
    pred_contrast2 = img_contrast2
    
    mgrid_affine_contrast1 = dataset.get_affine(contrast=1, resolution='gt')
    mgrid_affine_contrast2 = dataset.get_affine(contrast=2, resolution='gt')
    affine_c1 = np.array(mgrid_affine_contrast1)
    affine_c2 = np.array(mgrid_affine_contrast2)
    
    img = nib.Nifti1Image(img_contrast1, affine_c1)

    if epoch == (config.TRAINING.EPOCHS -1):
        nib.save(img, os.path.join(image_dir, model_name_epoch.replace("model.pt", f"_ct1.nii.gz")))

    slice_0 = img_contrast1[int(x_dim_c1/2), :, :]
    slice_1 = img_contrast1[:, int(y_dim_c1/2), :]
    slice_2 = img_contrast1[:, :, int(z_dim_c1/2)]

    bslice_0 = gt_contrast1[int(x_dim_c1/2), :, :]
    bslice_1 = gt_contrast1[:, int(y_dim_c1/2), :]
    bslice_2 = gt_contrast1[:, :, int(z_dim_c1/2)]

    im = show_slices_gt([slice_0, slice_1, slice_2],[bslice_0, bslice_1, bslice_2], epoch)
    image = wandb.Image(im, caption=f"{config.DATASET.LR_CONTRAST1} prediction vs gt.")
    wandb_epoch_dict.update({f"{config.DATASET.LR_CONTRAST1}": image})

    slice_0 = img_contrast2[int(x_dim_c2/2), :, :]
    slice_1 = img_contrast2[:, int(y_dim_c2/2), :]
    slice_2 = img_contrast2[:, :, int(z_dim_c2/2)]

    im = show_slices_gt([slice_0, slice_1, slice_2],[bslice_0, bslice_1, bslice_2], epoch)
    image = wandb.Image(im, caption=f"{config.DATASET.LR_CONTRAST2} prediction vs {config.DATASET.LR_CONTRAST1} gt.")
    wandb_epoch_dict.update({f"{config.DATASET.LR_CONTRAST2} prediction vs {config.DATASET.LR_CONTRAST1} gt": image})
        
    img = nib.Nifti1Image(img_contrast2, affine_c2)
    if epoch == (config.TRAINING.EPOCHS -1):
        nib.save(img, os.path.join(image_dir, model_name_epoch.replace("model.pt", f"_ct2.nii.gz")))

    slice_0 = img_contrast2_interpolated[int(x_dim_c2/2), :, :]
    slice_1 = img_contrast2_interpolated[:, int(y_dim_c2/2), :]
    slice_2 = img_contrast2_interpolated[:, :, int(z_dim_c2/2)]

    bslice_0 = gt_contrast2[int(x_dim_c2/2), :, :]
    bslice_1 = gt_contrast2[:, int(y_dim_c2/2), :]
    bslice_2 = gt_contrast2[:, :, int(z_dim_c2/2)]
    
    registration_slice_0 = img_registration[int(x_dim_c2/2), :, :, :]
    registration_slice_1 = img_registration[:, int(y_dim_c2/2), :, :]
    registration_slice_2 = img_registration[:, :, int(z_dim_c2/2), :]
    
    reg_jac_det_slice_0 = model_registration_jac_det[int(x_dim_c2/2), :, :]
    reg_jac_det_slice_1 = model_registration_jac_det[:, int(y_dim_c2/2), :]
    reg_jac_det_slice_2 = model_registration_jac_det[:, :, int(z_dim_c2/2)]
    
    registration_norm_slice_0 = model_registration_norm[int(x_dim_c2/2), :, :]
    registration_norm_slice_1 = model_registration_norm[:, int(y_dim_c2/2), :]
    registration_norm_slice_2 = model_registration_norm[:, :, int(z_dim_c2/2)]

    im = show_slices_gt([slice_0, slice_1, slice_2],[bslice_0, bslice_1, bslice_2], epoch)
    image = wandb.Image(im, caption=f"{config.DATASET.LR_CONTRAST2} prediction (reversed registration) vs gt.")
    wandb_epoch_dict.update({f"{config.DATASET.LR_CONTRAST2}": image})
        
    im_registration = show_slices_registration([registration_slice_0, registration_slice_1, registration_slice_2], epoch)
    image = wandb.Image(im_registration, caption=f"{config.DATASET.LR_CONTRAST2} registration field")
    wandb_epoch_dict.update({f"{config.DATASET.LR_CONTRAST2} registration field": image})

    im_registration_jac_det = show_jacobian_det([reg_jac_det_slice_0, reg_jac_det_slice_1, reg_jac_det_slice_2], epoch, f"Jacobian determinant map after {epoch}.")
    
    image = wandb.Image(im_registration_jac_det, caption=f"{config.DATASET.LR_CONTRAST2} registration jacobian determinant map")
    wandb_epoch_dict.update({f"{config.DATASET.LR_CONTRAST2} registration jacobian determinant map": image})
        
    im_registration_norm = show_jacobian_det([registration_norm_slice_0, registration_norm_slice_1, registration_norm_slice_2], epoch, f"Norm of the registration function after {epoch}.")
    
    image = wandb.Image(im_registration_norm, caption=f"{config.DATASET.LR_CONTRAST2} registration norm map")
    wandb_epoch_dict.update({f"{config.DATASET.LR_CONTRAST2} registration norm map": image})
        
        
    return pred_contrast1, pred_contrast2, gt_contrast1, gt_contrast2, wandb_epoch_dict



def compute_and_log_metrics(gt_contrast1, gt_contrast2, pred_contrast1, pred_contrast2, mask_c1, mask_c2, lpips_loss, device, wandb_epoch_dict):
    metrics_contrast1 = compute_metrics(gt=gt_contrast1.copy(), pred=pred_contrast1.copy(), mask=mask_c1, lpips_loss=lpips_loss, device=device)
    metrics_contrast2 = compute_metrics(gt=gt_contrast2.copy(), pred=pred_contrast2.copy(), mask=mask_c2, lpips_loss=lpips_loss, device=device)
    '''
    metrics_mi_true = compute_mi_hist(gt_contrast1.copy(), gt_contrast2.copy(), mask_c2, bins=32)
    metrics_mi_1 = compute_mi_hist(pred_contrast1.copy(), gt_contrast2.copy(), mask_c2, bins=32)
    metrics_mi_2 = compute_mi_hist(pred_contrast2.copy(), gt_contrast1.copy(), mask_c2, bins=32)
    metrics_mi_pred = compute_mi_hist(pred_contrast1.copy(), pred_contrast2.copy(), mask_c2, bins=32)

    metrics_mi_approx = compute_mi(pred_contrast1.copy(), pred_contrast2.copy(), mask_c2,device)
    '''
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
