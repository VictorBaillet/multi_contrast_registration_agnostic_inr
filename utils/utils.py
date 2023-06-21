import torch
import numpy as np
import json
import os
from typing import List, Tuple, Optional
import torch.nn as nn
from math import log, sqrt
import wandb
from utils.utils_visualization import show_slices_gt, show_slices_registration, show_jacobian_det
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from loss_functions import MILossGaussian, NMI, NCC
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler

from dataset.dataset_utils import norm_grid
import nibabel as nib
import nibabel.processing as nip
import nibabel.orientations as nio

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

'''
def get_string(my_dict):
    result = '_'.join([f"{key}-{value}" for key, value in my_dict.items()])
    return result
'''
def get_string(my_dict):
    return '_'.join([str(value) for value in my_dict.values()])


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def min_max_scale(X, s_min, s_max):
    x_min, x_max = X.min(), X.max()
    return torch.tensor((X - x_min) / (x_max - x_min) * (s_max - s_min) + s_min)


# from official FF repository
class input_mapping(nn.Module):
    def __init__(self, B=None, factor=1.0):
        super(input_mapping, self).__init__()
        self.B = factor * B
    
    def forward(self, x):

        x_proj = (2. * np.pi * x) @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def compute_metrics(gt, pred, mask, lpips_loss, device):

    if type(mask) == torch.Tensor:
        mask = mask.float().cpu().numpy()

    assert mask.max() == 1.0, 'Mask Format incorrect.'
    #assert mask.min() == 0.0, 'Mask Format incorrect.'

    gt -= gt.min()#gt[mask == 1].min()
    gt /= gt.max()
    #gt *= mask

    pred -= pred.min()#[mask == 1].min()
    pred /= pred.max()
    #pred *= mask

    ssim = structural_similarity(gt, pred, data_range=1)
    psnr = peak_signal_noise_ratio(gt, pred, data_range=1)

    x, y, z = pred.shape

    lpips_val = 0

    for i in range(x):
        pred_t = torch.tensor(pred[i,:,:]).reshape(1, y, z).repeat(3,1,1).to(device)
        gt_t = torch.tensor(gt[i,:,:]).reshape(1, y, z).repeat(3,1,1).to(device)
        lpips_val += lpips_loss(gt_t, pred_t)

    for i in range(y):
        pred_t = torch.tensor(pred[:,i,:]).reshape(1, x, z).repeat(3,1,1).to(device)
        gt_t = torch.tensor(gt[:,i,:]).reshape(1, x, z).repeat(3,1,1).to(device)
        lpips_val += lpips_loss(gt_t, pred_t)

    for i in range(z):
        pred_t = torch.tensor(pred[:,:,i]).reshape(1, x, y).repeat(3,1,1).to(device)
        gt_t = torch.tensor(gt[:,:,i]).reshape(1, x, y).repeat(3,1,1).to(device)
        lpips_val += lpips_loss(gt_t, pred_t)

    lpips_val /= (x+y+z)

    vals = {}
    vals["ssim"]= ssim
    vals["psnr"]= psnr
    vals["lpips"] = lpips_val.item()

    return vals


def compute_mi(pred1, pred2, mask, device):
    if type(mask) == torch.Tensor:
        mask = mask.float()

    mi_metric = MILossGaussian(num_bins=32).to(device)
    pred1 = torch.tensor(pred1[mask==1]).to(device).unsqueeze(0).unsqueeze(0)
    pred2 = torch.tensor(pred2[mask==1]).to(device).unsqueeze(0).unsqueeze(0)
    
    vals = {}
    vals['mi'] = mi_metric(pred1, pred2)
    return vals


# from: https://matthew-brett.github.io/teaching/mutual_information.html
def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


# from: https://matthew-brett.github.io/teaching/mutual_information.html
def compute_mi_hist(img1, img2, mask, bins=32):

    if type(mask) == torch.Tensor:
        mask = mask.float().cpu().numpy()

    if type(img1) == torch.Tensor():
        img1 = img1.cpu().numpy()

    if type(img2)== torch.Tensor():
        img2 = img2.cpu().numpy()  

    # only inside of the brain
    img1 = img1[mask==1]
    img2 = img2[mask==1]

    hist_2d, _, _= np.histogram2d(
        img1.ravel(),
        img2.ravel(),
        bins=bins)

    vals = {}
    vals['mi'] = mutual_information(hist_2d)
    return vals


def resample_nib(img, voxel_spacing=(1, 1, 1), order=3):
    """Resamples the nifti from its original spacing to another specified spacing
    
    Parameters:
    ----------
    img: nibabel image
    voxel_spacing: a tuple of 3 integers specifying the desired new spacing
    order: the order of interpolation
    
    Returns:
    ----------
    new_img: The resampled nibabel image 
    
    """
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(np.rint([
        shp[0] * zms[0] / voxel_spacing[0],
        shp[1] * zms[1] / voxel_spacing[1],
        shp[2] * zms[2] / voxel_spacing[2]
        ]).astype(int))
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=-1024)
    print("[*] Image resampled to voxel size:", voxel_spacing)
    return new_img

def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices, min_coord, max_coord, device, rev_affine):
    """Interpolate the intensity of voxels based of the closest known neighboors
    
    Parameters:
    ----------
    input_array: Torch array containing the intensity of the known voxels
    x_indices, y_indices, z_indices: Torch array, x,y,z indices of the voxels to extrapolate
    min_coord, max_coord: minimum and maximum coordinates from all the training points (for normalization purposes)
    device: training device
    rev_affine: 4x4 Torch matrice corresponding to the affine transformation "coordinate in space" -> "(x, y, z) coordinates in input_array"
    
    Returns:
    ----------
    output: Torch array, extrapolated intensity of the desired voxels
    
    """
    indices = torch.cat((x_indices.unsqueeze(0), y_indices.unsqueeze(0), z_indices.unsqueeze(0)), dim=0).to(torch.float)
    input_array = input_array.to(device=device).to(torch.float)
    indices = torch.matmul(rev_affine.to(torch.float), indices)
    x_indices = indices[0]
    y_indices = indices[1]
    z_indices = indices[2]
    x_indices = norm_grid(x_indices, min_coord[0], max_coord[0], smin=0)
    y_indices = norm_grid(y_indices, min_coord[1], max_coord[1], smin=0)
    z_indices = norm_grid(z_indices, min_coord[2], max_coord[2], smin=0)
    x_indices = (x_indices) * (input_array.shape[0] - 1)
    y_indices = (y_indices) * (input_array.shape[1] - 1) 
    z_indices = (z_indices) * (input_array.shape[2] - 1)  

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    
    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = (x_indices - x0)
    y = (y_indices - y0)
    z = (z_indices - z0)
    
    
    output = (input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )

    return output

def generate_NIFTIs(dataset, model_intensities, image_dir, model_name_epoch, epoch, wandb_epoch_dict, config, args):
    x_dim_c1, y_dim_c1, z_dim_c1 = dataset.get_contrast1_dim()
    x_dim_c2, y_dim_c2, z_dim_c2 = dataset.get_contrast2_dim()
    threshold = len(dataset.get_contrast1_coordinates())
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

    gt_contrast1 = dataset.get_contrast1_gt().reshape((x_dim_c1, y_dim_c1, z_dim_c1)).cpu().numpy()
    gt_contrast2 = dataset.get_contrast2_gt().reshape((x_dim_c2, y_dim_c2, z_dim_c2)).cpu().numpy()

    label_arr = np.array(gt_contrast1, dtype=np.float32)
    gt_contrast1= np.clip(gt_contrast1.reshape(-1, 1), 0, 1).reshape((x_dim_c1, y_dim_c1, z_dim_c1))

    label_arr = np.array(gt_contrast2, dtype=np.float32)
    gt_contrast2= np.clip(gt_contrast2.reshape(-1, 1), 0, 1).reshape((x_dim_c2, y_dim_c2, z_dim_c2))

    pred_contrast1 = img_contrast1
    pred_contrast2 = img_contrast2
    
    mgrid_affine_contrast1 = dataset.get_contrast1_affine()
    mgrid_affine_contrast2 = dataset.get_contrast2_affine()
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
    if args.logging:
        image = wandb.Image(im, caption=f"{config.DATASET.LR_CONTRAST1} prediction vs gt.")
        wandb_epoch_dict.update({f"{config.DATASET.LR_CONTRAST1}": image})
        
    slice_0 = img_contrast2[int(x_dim_c1/2), :, :]
    slice_1 = img_contrast2[:, int(y_dim_c1/2), :]
    slice_2 = img_contrast2[:, :, int(z_dim_c1/2)]

    im = show_slices_gt([slice_0, slice_1, slice_2],[bslice_0, bslice_1, bslice_2], epoch)
    if args.logging:
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
    if args.logging:
        image = wandb.Image(im, caption=f"{config.DATASET.LR_CONTRAST2} prediction (reversed registration) vs gt.")
        wandb_epoch_dict.update({f"{config.DATASET.LR_CONTRAST2}": image})
        
    im_registration = show_slices_registration([registration_slice_0, registration_slice_1, registration_slice_2], epoch)
    if args.logging:
        image = wandb.Image(im_registration, caption=f"{config.DATASET.LR_CONTRAST2} registration field")
        wandb_epoch_dict.update({f"{config.DATASET.LR_CONTRAST2} registration field": image})
        
    im_registration_jac_det = show_jacobian_det([reg_jac_det_slice_0, reg_jac_det_slice_1, reg_jac_det_slice_2], epoch, f"Jacobian determinant map after {epoch}.")
    if args.logging:
        image = wandb.Image(im_registration_jac_det, caption=f"{config.DATASET.LR_CONTRAST2} registration jacobian determinant map")
        wandb_epoch_dict.update({f"{config.DATASET.LR_CONTRAST2} registration jacobian determinant map": image})
        
    im_registration_norm = show_jacobian_det([registration_norm_slice_0, registration_norm_slice_1, registration_norm_slice_2], epoch, f"Norm of the registration function after {epoch}.")
    
    if args.logging:
        image = wandb.Image(im_registration_norm, caption=f"{config.DATASET.LR_CONTRAST2} registration norm map")
        wandb_epoch_dict.update({f"{config.DATASET.LR_CONTRAST2} registration norm map": image})
        
    return pred_contrast1, pred_contrast2, gt_contrast1, gt_contrast2, wandb_epoch_dict

def compute_jacobian_loss(input_coords, output, batch_size=None):
    """Compute the jacobian regularization loss."""

    # Compute Jacobian matrices
    jac = compute_jacobian_matrix(input_coords, output)

    # Compute determinants and take norm
    loss = torch.det(jac) - 1
    loss = torch.linalg.norm(loss, 1)

    return loss / batch_size


def compute_jacobian_matrix(input_coords, output, add_identity=True):
    """Compute the Jacobian matrix of the output wrt the input."""

    jacobian_matrix = torch.zeros(input_coords.shape[0], 3, 3)
    for i in range(3):
        jacobian_matrix[:, i, :] = gradient(input_coords, output[:, i])
        if add_identity:
            jacobian_matrix[:, i, i] += torch.ones_like(jacobian_matrix[:, i, i])
    return jacobian_matrix



def gradient(input_coords, output, grad_outputs=None):
    """Compute the gradient of the output wrt the input."""

    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, [input_coords], grad_outputs=grad_outputs, create_graph=True,
    )[0]
    return grad

def compute_hyper_elastic_loss(
    input_coords, output, batch_size=None, alpha_l=1, alpha_a=1, alpha_v=1
):
    """Compute the hyper-elastic regularization loss."""

    grad_u = compute_jacobian_matrix(input_coords, output, add_identity=False)
    grad_y = grad_u
    """
    for i in range(3):
        grad_y[:, i, i] += torch.ones_like(grad_y[:, i, i])
    """
    grad_y = compute_jacobian_matrix(
        input_coords, output, add_identity=True
    )  # This is slow, faster to infer from grad_u
    
    # Compute length loss
    length_loss = torch.linalg.norm(grad_u, dim=(1, 2))
    length_loss = torch.pow(length_loss, 2)
    length_loss = torch.sum(length_loss)
    length_loss = 0.5 * alpha_l * length_loss

    # Compute cofactor matrices for the area loss
    cofactors = torch.zeros(batch_size, 3, 3)

    # Compute elements of cofactor matrices one by one (Ugliest solution ever?)
    cofactors[:, 0, 0] = torch.det(grad_y[:, 1:, 1:])
    cofactors[:, 0, 1] = torch.det(grad_y[:, 1:, 0::2])
    cofactors[:, 0, 2] = torch.det(grad_y[:, 1:, :2])
    cofactors[:, 1, 0] = torch.det(grad_y[:, 0::2, 1:])
    cofactors[:, 1, 1] = torch.det(grad_y[:, 0::2, 0::2])
    cofactors[:, 1, 2] = torch.det(grad_y[:, 0::2, :2])
    cofactors[:, 2, 0] = torch.det(grad_y[:, :2, 1:])
    cofactors[:, 2, 1] = torch.det(grad_y[:, :2, 0::2])
    cofactors[:, 2, 2] = torch.det(grad_y[:, :2, :2])

    # Compute area loss
    area_loss = torch.pow(cofactors, 2)
    area_loss = torch.sum(area_loss, dim=1)
    area_loss = area_loss - 1
    area_loss = torch.maximum(area_loss, torch.zeros_like(area_loss))
    area_loss = torch.pow(area_loss, 2)
    area_loss = torch.sum(area_loss)  # sum over dimension 1 and then 0
    area_loss = alpha_a * area_loss

    # Compute volume loss
    volume_loss = torch.det(grad_y)
    volume_loss = torch.mul(torch.pow(volume_loss - 1, 4), torch.pow(volume_loss, -2))
    volume_loss = torch.sum(volume_loss)
    volume_loss = alpha_v * volume_loss

    # Compute total loss
    loss = length_loss + area_loss + volume_loss

    return loss / batch_size

def compute_bending_energy(input_coords, output, batch_size=None):
    """Compute the bending energy."""

    jacobian_matrix = compute_jacobian_matrix(input_coords, output, add_identity=False)

    dx_xyz = torch.zeros(input_coords.shape[0], 3, 3)
    dy_xyz = torch.zeros(input_coords.shape[0], 3, 3)
    dz_xyz = torch.zeros(input_coords.shape[0], 3, 3)
    for i in range(3):
        dx_xyz[:, i, :] = gradient(input_coords, jacobian_matrix[:, i, 0])
        dy_xyz[:, i, :] = gradient(input_coords, jacobian_matrix[:, i, 1])
        dz_xyz[:, i, :] = gradient(input_coords, jacobian_matrix[:, i, 2])

    dx_xyz = torch.square(dx_xyz)
    dy_xyz = torch.square(dy_xyz)
    dz_xyz = torch.square(dz_xyz)

    loss = (
        torch.mean(dx_xyz[:, :, 0])
        + torch.mean(dy_xyz[:, :, 1])
        + torch.mean(dz_xyz[:, :, 2])
    )
    loss += (
        2 * torch.mean(dx_xyz[:, :, 1])
        + 2 * torch.mean(dx_xyz[:, :, 2])
        + torch.mean(dy_xyz[:, :, 2])
    )

    return loss / batch_size
