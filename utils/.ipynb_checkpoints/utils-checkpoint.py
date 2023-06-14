import torch
import numpy as np
import json
import os
from typing import List, Tuple, Optional
import torch.nn as nn
from math import log, sqrt
import wandb
from utils.visualization_utils import show_slices_gt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from utils.loss_functions import MILossGaussian, NMI, NCC
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

    scaler = MinMaxScaler()
    label_arr = np.array(model_intensities_contrast1, dtype=np.float32)
    model_intensities_contrast1= scaler.fit_transform(label_arr.reshape(-1, 1))

    label_arr = np.array(model_intensities_contrast2, dtype=np.float32)
    model_intensities_contrast2= scaler.fit_transform(label_arr.reshape(-1, 1))

    img_contrast1 = model_intensities_contrast1.reshape((x_dim_c1, y_dim_c1, z_dim_c1))#.cpu().numpy()
    img_contrast2 = model_intensities_contrast2.reshape((x_dim_c2, y_dim_c2, z_dim_c2))#.cpu().numpy()

    gt_contrast1 = dataset.get_contrast1_gt().reshape((x_dim_c1, y_dim_c1, z_dim_c1)).cpu().numpy()
    gt_contrast2 = dataset.get_contrast2_gt().reshape((x_dim_c2, y_dim_c2, z_dim_c2)).cpu().numpy()

    label_arr = np.array(gt_contrast1, dtype=np.float32)
    gt_contrast1= scaler.fit_transform(gt_contrast1.reshape(-1, 1)).reshape((x_dim_c1, y_dim_c1, z_dim_c1))

    label_arr = np.array(gt_contrast2, dtype=np.float32)
    gt_contrast2= scaler.fit_transform(gt_contrast2.reshape(-1, 1)).reshape((x_dim_c2, y_dim_c2, z_dim_c2))

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
        
    img = nib.Nifti1Image(img_contrast2, affine_c2)
    if epoch == (config.TRAINING.EPOCHS -1):
        nib.save(img, os.path.join(image_dir, model_name_epoch.replace("model.pt", f"_ct2.nii.gz")))

    slice_0 = img_contrast2[int(x_dim_c2/2), :, :]
    slice_1 = img_contrast2[:, int(y_dim_c2/2), :]
    slice_2 = img_contrast2[:, :, int(z_dim_c2/2)]

    bslice_0 = gt_contrast2[int(x_dim_c2/2), :, :]
    bslice_1 = gt_contrast2[:, int(y_dim_c2/2), :]
    bslice_2 = gt_contrast2[:, :, int(z_dim_c2/2)]

    im = show_slices_gt([slice_0, slice_1, slice_2],[bslice_0, bslice_1, bslice_2], epoch)
    if args.logging:
        image = wandb.Image(im, caption=f"{config.DATASET.LR_CONTRAST2} prediction vs gt.")
        wandb_epoch_dict.update({f"{config.DATASET.LR_CONTRAST2}": image})
        
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
"""
[tensor([1.0842e-01, 1.1990e-01, 1.8486e-01, 4.7692e-01, 6.1050e-03, 3.0208e-01,
        6.9353e-02, 1.6777e-01, 1.3919e-02, 0.0000e+00, 5.8608e-03, 3.1746e-03,
        1.0891e-01, 4.8840e-03, 9.0598e-02, 5.1282e-03, 4.5177e-02, 7.2625e-01,
        2.4444e-01, 2.9792e-02, 8.0586e-03, 4.0537e-02, 1.4652e-03, 4.0952e-01,
        1.3187e-02, 1.3651e-01, 5.3724e-03, 6.8864e-02, 2.6325e-01, 0.0000e+00,
        9.7680e-04, 1.7094e-03, 1.6606e-02, 0.0000e+00, 2.8376e-01, 7.5214e-02,
        0.0000e+00, 5.8608e-03, 7.3260e-04, 1.1233e-02, 9.7680e-04, 3.4261e-01,
        1.2601e-01, 9.4628e-01, 1.7729e-01, 1.9683e-01, 2.1807e-01, 0.0000e+00,
        1.7094e-03, 0.0000e+00, 7.3260e-04, 2.6129e-02, 1.8999e-01, 5.6410e-02,
        1.1600e-01, 0.0000e+00, 1.3138e-01, 0.0000e+00, 6.8010e-01, 0.0000e+00,
        6.1050e-03, 1.7094e-03, 7.8144e-03, 9.7436e-02, 6.9353e-02, 2.2564e-01,
        4.1514e-03, 0.0000e+00, 5.3016e-01, 1.0256e-02, 1.2210e-03, 0.0000e+00,
        3.8828e-02, 3.0379e-01, 0.0000e+00, 4.8840e-04, 1.1966e-02, 3.8095e-02,
        1.5092e-01, 6.7839e-01, 0.0000e+00, 9.2186e-01, 2.1148e-01, 2.6300e-01,
        5.4945e-02, 5.9829e-02, 1.7729e-01, 1.7827e-01, 1.4652e-03, 4.3956e-03,
        2.4420e-04, 1.4652e-03, 2.2320e-01, 3.8095e-02, 2.9109e-01, 3.4432e-02,
        0.0000e+00, 1.9927e-01, 7.1551e-02, 1.9536e-02, 0.0000e+00, 2.4420e-04,
        2.7179e-01, 2.1245e-02, 3.4188e-03, 3.3944e-02, 9.5238e-03, 7.0330e-02,
        2.1978e-03, 5.5702e-01, 2.5836e-01, 2.2686e-01, 1.9536e-03, 1.7143e-01,
        1.2576e-01, 2.1734e-01, 0.0000e+00, 3.3578e-01, 5.1770e-02, 4.6398e-03,
        2.1001e-02, 0.0000e+00, 1.7998e-01, 2.4420e-03, 0.0000e+00, 2.4420e-04,
        4.8840e-01, 2.1587e-01, 5.5189e-02, 6.2515e-02, 1.0501e-02, 0.0000e+00,
        8.7424e-02, 0.0000e+00, 2.7619e-01, 2.4420e-04, 0.0000e+00, 2.7839e-02,
        1.4652e-03, 2.2442e-01, 0.0000e+00, 0.0000e+00, 1.3309e-01, 8.9133e-02,
        7.3260e-04, 0.0000e+00, 1.0989e-02, 1.2210e-03, 1.4652e-03, 0.0000e+00,
        1.2723e-01, 1.2210e-03, 0.0000e+00, 0.0000e+00, 1.0000e+00, 2.3126e-01,
        8.4493e-02, 8.0830e-02, 1.1966e-02, 3.9805e-02, 0.0000e+00, 1.1722e-02,
        0.0000e+00, 2.4420e-04, 6.5934e-03, 0.0000e+00, 2.0513e-02, 0.0000e+00],
       device='cuda:0'), tensor([6.1783e-02, 1.9683e-01, 2.0562e-01, 2.8181e-01, 3.9072e-03, 3.1477e-01,
        6.8132e-02, 1.3724e-01, 8.0586e-03, 0.0000e+00, 6.6911e-02, 4.6398e-03,
        7.8388e-02, 2.4420e-03, 5.0794e-02, 3.4188e-03, 1.1868e-01, 4.9084e-01,
        3.3553e-01, 3.9072e-02, 9.2796e-03, 6.1050e-02, 7.3260e-04, 2.8278e-01,
        3.5653e-02, 1.2479e-01, 1.4652e-03, 1.9292e-02, 4.0855e-01, 0.0000e+00,
        7.3260e-04, 2.4420e-03, 1.8071e-02, 0.0000e+00, 2.3907e-01, 9.0598e-02,
        0.0000e+00, 7.5702e-03, 9.7680e-04, 6.3492e-03, 4.8840e-04, 3.0403e-01,
        1.2454e-01, 9.0647e-01, 1.9634e-01, 0.0000e+00, 2.3028e-01, 0.0000e+00,
        2.4420e-03, 0.0000e+00, 1.4652e-03, 0.0000e+00, 2.2076e-01, 1.8095e-01,
        7.2527e-02, 0.0000e+00, 1.1355e-01, 0.0000e+00, 5.6410e-01, 0.0000e+00,
        7.8144e-03, 3.4188e-03, 1.2943e-02, 3.0769e-02, 5.1526e-02, 2.0586e-01,
        8.3028e-03, 2.4420e-04, 3.7705e-01, 1.4164e-02, 2.1978e-03, 0.0000e+00,
        1.1502e-01, 2.7668e-01, 0.0000e+00, 4.8840e-04, 9.7680e-04, 1.7094e-03,
        1.5849e-01, 3.8632e-01, 0.0000e+00, 8.7595e-01, 2.1636e-01, 1.3578e-01,
        8.6203e-02, 2.7839e-02, 2.0904e-01, 1.9365e-01, 7.3260e-04, 3.4188e-03,
        1.4652e-03, 3.1746e-03, 2.8376e-01, 2.6618e-02, 4.5397e-01, 1.2454e-02,
        0.0000e+00, 2.2589e-01, 6.0317e-02, 1.4164e-02, 0.0000e+00, 3.4188e-03,
        2.5104e-01, 0.0000e+00, 6.1050e-03, 3.1746e-02, 5.8608e-03, 6.8376e-02,
        4.8840e-03, 6.5275e-01, 2.7473e-01, 1.4823e-01, 7.5702e-03, 1.9634e-01,
        0.0000e+00, 3.7949e-01, 0.0000e+00, 9.4725e-01, 5.3480e-02, 7.3260e-04,
        1.4652e-03, 0.0000e+00, 2.1538e-01, 4.8840e-04, 0.0000e+00, 4.8840e-04,
        6.6398e-01, 2.0513e-01, 6.3004e-02, 3.2234e-02, 4.5910e-02, 0.0000e+00,
        5.5922e-02, 0.0000e+00, 2.1563e-01, 4.8840e-04, 0.0000e+00, 4.1270e-02,
        1.7094e-03, 1.6239e-01, 0.0000e+00, 0.0000e+00, 1.0379e-01, 5.0794e-02,
        1.4652e-03, 0.0000e+00, 1.1722e-02, 9.7680e-04, 2.4420e-03, 0.0000e+00,
        9.4506e-02, 2.4420e-03, 0.0000e+00, 0.0000e+00, 1.0000e+00, 2.3126e-01,
        1.2283e-01, 5.9829e-02, 2.4664e-02, 9.7680e-02, 0.0000e+00, 1.4652e-02,
        0.0000e+00, 1.2210e-03, 6.8376e-03, 2.2955e-02, 1.3675e-02, 0.0000e+00],
       device='cuda:0'), tensor([1.9341e-01, 1.6313e-01, 1.9414e-01, 5.1941e-01, 5.8608e-03, 3.0037e-01,
        2.4176e-02, 1.5995e-01, 5.3724e-03, 0.0000e+00, 3.8828e-02, 4.3956e-03,
        8.3028e-02, 3.6630e-03, 1.1429e-01, 1.1233e-02, 4.6154e-02, 7.2918e-01,
        3.8046e-01, 5.6166e-02, 1.1722e-02, 3.0525e-02, 1.4652e-03, 4.5470e-01,
        0.0000e+00, 2.1026e-01, 4.3956e-03, 9.0354e-03, 2.4860e-01, 0.0000e+00,
        9.7680e-04, 1.7094e-03, 2.0269e-02, 0.0000e+00, 2.5324e-01, 8.1807e-02,
        0.0000e+00, 4.6398e-03, 4.8840e-04, 9.5238e-03, 1.7094e-03, 2.6862e-01,
        9.5971e-02, 8.5226e-01, 2.1270e-01, 1.9902e-01, 2.2589e-01, 0.0000e+00,
        4.8840e-04, 0.0000e+00, 1.9536e-03, 1.2943e-02, 1.4921e-01, 1.2454e-02,
        1.6703e-01, 0.0000e+00, 1.9194e-01, 0.0000e+00, 7.3651e-01, 0.0000e+00,
        9.5238e-03, 1.9536e-03, 1.2454e-02, 6.9841e-02, 5.3236e-02, 1.8632e-01,
        1.7094e-03, 4.8840e-04, 4.8938e-01, 1.4896e-02, 1.4652e-03, 0.0000e+00,
        5.1526e-02, 3.1453e-01, 0.0000e+00, 9.7680e-04, 0.0000e+00, 1.2015e-01,
        1.6996e-01, 7.6606e-01, 0.0000e+00, 5.8193e-01, 2.1197e-01, 2.3736e-01,
        1.2161e-01, 8.7179e-02, 1.6581e-01, 2.4176e-01, 7.3260e-04, 4.1514e-03,
        4.8840e-04, 3.6630e-03, 2.6740e-01, 2.6862e-02, 4.8840e-01, 2.9304e-02,
        0.0000e+00, 2.0904e-01, 8.8400e-02, 4.0781e-02, 0.0000e+00, 1.9536e-03,
        2.1221e-01, 5.6166e-03, 4.3956e-03, 5.4212e-02, 1.5385e-02, 8.0586e-02,
        2.6862e-03, 3.5385e-01, 2.4982e-01, 1.1575e-01, 3.9072e-03, 2.0098e-01,
        3.2576e-01, 8.4737e-02, 0.0000e+00, 6.6398e-01, 5.5922e-02, 2.6862e-03,
        2.0024e-02, 0.0000e+00, 8.1563e-02, 2.6862e-03, 0.0000e+00, 2.4420e-04,
        6.0049e-01, 1.9390e-01, 5.6899e-02, 9.3284e-02, 2.8571e-02, 0.0000e+00,
        5.8608e-02, 0.0000e+00, 2.7546e-01, 9.7680e-04, 0.0000e+00, 5.0794e-02,
        7.3260e-04, 2.0952e-01, 0.0000e+00, 0.0000e+00, 1.2576e-01, 8.2295e-02,
        2.4420e-04, 0.0000e+00, 5.8608e-03, 1.2210e-03, 4.8840e-04, 0.0000e+00,
        1.1282e-01, 4.3956e-03, 0.0000e+00, 0.0000e+00, 1.0000e+00, 2.4005e-01,
        5.1526e-02, 1.1893e-01, 8.0586e-03, 3.1990e-02, 0.0000e+00, 1.3675e-02,
        0.0000e+00, 0.0000e+00, 6.5934e-03, 9.7192e-02, 1.2698e-02, 0.0000e+00],
       device='cuda:0'), tensor([2.2149e-01, 1.6606e-01, 2.0391e-01, 4.8840e-01, 3.6630e-03, 2.5324e-01,
        7.9853e-02, 1.3236e-01, 1.2210e-02, 0.0000e+00, 3.0525e-02, 2.9304e-03,
        6.0317e-02, 5.6166e-03, 3.5165e-02, 1.0501e-02, 1.2454e-02, 8.3492e-01,
        2.3810e-01, 4.2979e-02, 1.8559e-02, 2.5885e-02, 1.9536e-03, 5.9219e-01,
        9.3773e-02, 1.0183e-01, 1.7094e-03, 2.4420e-03, 3.0525e-01, 0.0000e+00,
        9.7680e-04, 5.6166e-03, 5.6166e-03, 0.0000e+00, 2.3199e-01, 9.3529e-02,
        0.0000e+00, 6.8376e-03, 1.7094e-03, 5.1282e-03, 7.3260e-04, 2.8107e-01,
        5.7631e-02, 8.7643e-01, 1.8632e-01, 2.0830e-01, 2.2442e-01, 0.0000e+00,
        1.2210e-03, 0.0000e+00, 2.1978e-03, 1.2698e-02, 1.3431e-01, 2.1245e-02,
        1.4359e-01, 0.0000e+00, 1.5800e-01, 0.0000e+00, 6.3883e-01, 0.0000e+00,
        8.7912e-03, 6.3492e-03, 7.0818e-03, 7.4237e-02, 6.8864e-02, 2.0562e-01,
        1.2210e-03, 7.3260e-04, 4.3663e-01, 1.3919e-02, 7.3260e-04, 0.0000e+00,
        4.9328e-02, 3.1258e-01, 0.0000e+00, 4.8840e-04, 2.7839e-02, 6.8132e-02,
        1.2894e-01, 7.3162e-01, 0.0000e+00, 4.1685e-01, 1.6288e-01, 2.3150e-01,
        1.1600e-01, 6.5201e-02, 1.5775e-01, 2.0586e-01, 2.4420e-04, 4.3956e-03,
        1.4652e-03, 5.6166e-03, 2.2100e-01, 3.3700e-02, 8.5031e-01, 3.0769e-02,
        0.0000e+00, 1.9267e-01, 1.1477e-01, 8.3028e-03, 0.0000e+00, 1.4652e-03,
        2.8913e-01, 8.9621e-02, 3.6630e-03, 6.6911e-02, 1.5385e-02, 6.8864e-02,
        3.4188e-03, 2.2515e-01, 2.1294e-01, 5.7631e-02, 6.8376e-03, 2.1905e-01,
        3.7118e-01, 4.2491e-02, 0.0000e+00, 9.3431e-01, 5.4457e-02, 3.1746e-03,
        7.6679e-02, 0.0000e+00, 1.1209e-01, 2.6862e-03, 0.0000e+00, 7.3260e-04,
        4.2466e-01, 2.0098e-01, 8.4005e-02, 1.0891e-01, 1.6361e-02, 0.0000e+00,
        5.6166e-02, 0.0000e+00, 3.2405e-01, 2.4420e-04, 0.0000e+00, 1.0598e-01,
        4.8840e-04, 2.2686e-01, 0.0000e+00, 0.0000e+00, 1.5385e-01, 1.3700e-01,
        1.7094e-03, 0.0000e+00, 3.6630e-03, 3.4188e-03, 2.9304e-03, 0.0000e+00,
        1.0525e-01, 3.6630e-03, 0.0000e+00, 0.0000e+00, 1.0000e+00, 2.5470e-01,
        8.2784e-02, 7.8632e-02, 3.2723e-02, 6.7399e-02, 0.0000e+00, 1.3431e-02,
        0.0000e+00, 2.6862e-03, 8.3028e-03, 0.0000e+00, 2.7350e-02, 0.0000e+00],
       device='cuda:0'), tensor([1.7167e-01, 1.2576e-01, 1.9463e-01, 4.4567e-01, 2.6862e-03, 2.8962e-01,
        1.5263e-01, 1.1868e-01, 6.8376e-03, 0.0000e+00, 2.1490e-02, 5.6166e-03,
        8.1074e-02, 4.6398e-03, 9.7436e-02, 9.7680e-03, 7.5214e-02, 4.8010e-01,
        3.7924e-01, 2.7350e-02, 1.1233e-02, 4.2979e-02, 4.8840e-04, 2.1807e-01,
        2.5397e-02, 1.2894e-01, 1.4652e-03, 0.0000e+00, 3.1013e-01, 0.0000e+00,
        2.6862e-03, 1.2210e-03, 2.2711e-02, 0.0000e+00, 2.5788e-01, 7.3504e-02,
        0.0000e+00, 5.1282e-03, 4.8840e-04, 2.6862e-03, 1.2210e-03, 2.3590e-01,
        1.3455e-01, 8.5275e-01, 2.1954e-01, 2.5153e-02, 2.4347e-01, 0.0000e+00,
        2.1978e-03, 0.0000e+00, 1.2210e-03, 9.7680e-04, 2.3175e-01, 1.7949e-01,
        1.3944e-01, 0.0000e+00, 1.3993e-01, 0.0000e+00, 6.4322e-01, 0.0000e+00,
        5.1282e-03, 2.4420e-03, 1.0989e-02, 2.2222e-02, 4.8840e-02, 2.2759e-01,
        7.8144e-03, 2.4420e-04, 4.5958e-01, 1.3431e-02, 1.7094e-03, 0.0000e+00,
        7.8877e-02, 2.8303e-01, 0.0000e+00, 9.7680e-04, 0.0000e+00, 3.6142e-02,
        1.7070e-01, 6.8010e-01, 0.0000e+00, 4.8547e-01, 1.8559e-01, 1.7143e-01,
        1.7753e-01, 8.8645e-02, 2.0611e-01, 2.7790e-01, 0.0000e+00, 9.7680e-03,
        7.3260e-04, 3.1746e-03, 2.5592e-01, 1.4896e-02, 8.0904e-01, 2.6129e-02,
        0.0000e+00, 2.0611e-01, 1.4017e-01, 2.0024e-02, 0.0000e+00, 2.6862e-03,
        1.2479e-01, 0.0000e+00, 9.5238e-03, 5.4945e-02, 4.8840e-03, 4.9084e-02,
        3.4188e-03, 5.7582e-01, 2.3907e-01, 3.3773e-01, 6.8376e-03, 2.2344e-01,
        3.6386e-02, 1.5482e-01, 0.0000e+00, 1.0000e+00, 6.6422e-02, 9.7680e-04,
        1.6361e-02, 0.0000e+00, 2.2369e-01, 1.7094e-03, 0.0000e+00, 2.4420e-04,
        6.4298e-01, 2.2027e-01, 7.1795e-02, 2.6862e-02, 0.0000e+00, 0.0000e+00,
        4.4444e-02, 0.0000e+00, 1.8437e-01, 1.7094e-03, 0.0000e+00, 0.0000e+00,
        2.4420e-04, 1.0769e-01, 0.0000e+00, 0.0000e+00, 4.9328e-02, 4.9817e-02,
        1.7094e-03, 0.0000e+00, 1.3431e-02, 7.3260e-04, 1.7094e-03, 0.0000e+00,
        7.8388e-02, 2.9304e-03, 0.0000e+00, 0.0000e+00, 1.0000e+00, 2.1636e-01,
        1.4261e-01, 7.5214e-02, 7.5702e-03, 5.8120e-02, 0.0000e+00, 9.0354e-03,
        0.0000e+00, 4.8840e-04, 4.8840e-03, 0.0000e+00, 1.2698e-02, 0.0000e+00],
       device='cuda:0'), tensor([2.1270e-01, 1.6996e-01, 1.9853e-01, 2.1270e-01, 2.4420e-03, 2.9426e-01,
        8.2784e-02, 1.3553e-01, 8.3028e-03, 0.0000e+00, 7.5946e-02, 9.2796e-03,
        7.8877e-02, 3.4188e-03, 5.6166e-03, 1.0012e-02, 3.3700e-02, 6.9670e-01,
        4.0806e-01, 9.2796e-03, 4.3956e-03, 2.5641e-02, 9.7680e-04, 5.2454e-01,
        1.8242e-01, 1.2723e-01, 3.1746e-03, 2.4420e-02, 3.7021e-01, 0.0000e+00,
        1.7094e-03, 5.6166e-03, 9.0354e-03, 0.0000e+00, 2.7253e-01, 6.9109e-02,
        0.0000e+00, 4.8840e-03, 2.4420e-04, 1.0012e-02, 9.7680e-04, 2.0635e-01,
        1.7241e-01, 6.4444e-01, 1.8974e-01, 2.2466e-02, 2.1880e-01, 0.0000e+00,
        1.2210e-03, 0.0000e+00, 2.4420e-03, 5.2747e-02, 2.1123e-01, 5.8120e-02,
        1.1990e-01, 0.0000e+00, 1.3016e-01, 0.0000e+00, 5.4823e-01, 0.0000e+00,
        6.1050e-03, 9.7680e-04, 1.3675e-02, 4.5177e-02, 5.0305e-02, 2.1441e-01,
        6.1050e-03, 9.7680e-04, 2.6032e-01, 1.5140e-02, 1.2210e-03, 0.0000e+00,
        7.5946e-02, 2.7741e-01, 0.0000e+00, 7.3260e-04, 0.0000e+00, 5.1282e-02,
        1.5189e-01, 3.5897e-01, 0.0000e+00, 8.6886e-01, 2.4713e-01, 2.0440e-01,
        8.4493e-02, 4.7375e-02, 2.0952e-01, 2.0391e-01, 7.3260e-04, 1.0012e-02,
        1.9536e-03, 3.9072e-03, 2.5641e-01, 1.8559e-02, 8.0806e-01, 2.9060e-02,
        0.0000e+00, 2.1026e-01, 2.7350e-02, 1.0745e-02, 0.0000e+00, 4.1514e-03,
        1.1648e-01, 0.0000e+00, 5.1282e-03, 3.8095e-02, 4.3956e-03, 6.7888e-02,
        5.3724e-03, 5.2894e-01, 2.1807e-01, 3.6361e-01, 2.1978e-03, 1.7216e-01,
        5.4945e-02, 9.9878e-02, 0.0000e+00, 1.0000e+00, 7.1062e-02, 8.5470e-03,
        3.2234e-02, 0.0000e+00, 2.1465e-01, 2.9304e-03, 0.0000e+00, 9.7680e-04,
        6.2271e-01, 2.2882e-01, 3.7363e-02, 4.0049e-02, 6.8620e-02, 0.0000e+00,
        3.5897e-02, 0.0000e+00, 1.5751e-01, 2.4420e-04, 0.0000e+00, 0.0000e+00,
        1.4652e-03, 2.0757e-01, 0.0000e+00, 0.0000e+00, 1.7705e-01, 6.1294e-02,
        2.1978e-03, 0.0000e+00, 7.5702e-03, 2.9304e-03, 1.7094e-03, 0.0000e+00,
        8.0342e-02, 7.5702e-03, 0.0000e+00, 0.0000e+00, 1.0000e+00, 1.9829e-01,
        1.4017e-01, 9.5971e-02, 3.1258e-02, 6.8620e-02, 0.0000e+00, 1.5140e-02,
        0.0000e+00, 9.7680e-04, 3.1746e-03, 1.5067e-01, 2.2711e-02, 0.0000e+00],
       device='cuda:0'), tensor([2.8352e-01, 1.0232e-01, 2.1221e-01, 3.4945e-01, 7.0818e-03, 3.0085e-01,
        9.9634e-02, 1.1209e-01, 1.2210e-02, 0.0000e+00, 3.6874e-02, 8.7912e-03,
        7.6435e-02, 3.1746e-03, 6.1050e-02, 7.5702e-03, 5.6166e-03, 6.7741e-01,
        2.7619e-01, 4.6154e-02, 1.6117e-02, 2.9060e-02, 1.2210e-03, 3.4676e-01,
        1.5702e-01, 9.6459e-02, 7.3260e-04, 0.0000e+00, 4.8376e-01, 0.0000e+00,
        1.7094e-03, 2.1978e-03, 2.0024e-02, 0.0000e+00, 2.6398e-01, 7.9365e-02,
        0.0000e+00, 4.3956e-03, 7.3260e-04, 5.6166e-03, 4.8840e-04, 2.3394e-01,
        1.1990e-01, 8.4737e-01, 2.2271e-01, 2.2711e-02, 2.2320e-01, 0.0000e+00,
        1.4652e-03, 0.0000e+00, 1.7094e-03, 2.5885e-02, 2.1905e-01, 5.3724e-03,
        1.4286e-01, 0.0000e+00, 1.3773e-01, 0.0000e+00, 5.7656e-01, 0.0000e+00,
        1.7094e-03, 7.3260e-04, 1.1233e-02, 4.9084e-02, 8.0830e-02, 2.6203e-01,
        8.5470e-03, 7.3260e-04, 3.4139e-01, 6.3492e-03, 7.3260e-04, 0.0000e+00,
        7.6190e-02, 2.9035e-01, 0.0000e+00, 9.7680e-04, 0.0000e+00, 1.0208e-01,
        1.6288e-01, 6.3272e-01, 0.0000e+00, 2.8694e-01, 1.5995e-01, 2.1050e-01,
        1.1844e-01, 9.7680e-02, 2.1490e-01, 2.5568e-01, 7.3260e-04, 1.0501e-02,
        1.7094e-03, 7.0818e-03, 2.5568e-01, 2.1490e-02, 7.5995e-01, 2.5397e-02,
        0.0000e+00, 2.0024e-01, 1.0720e-01, 1.0256e-02, 0.0000e+00, 1.7094e-03,
        2.0122e-01, 0.0000e+00, 3.6630e-03, 5.1526e-02, 3.6630e-03, 4.6886e-02,
        9.0354e-03, 4.8352e-01, 1.9780e-01, 4.5910e-01, 1.9536e-03, 2.0317e-01,
        4.9084e-02, 1.1795e-01, 0.0000e+00, 1.0000e+00, 8.1074e-02, 8.7912e-03,
        6.7155e-02, 0.0000e+00, 1.9878e-01, 3.9072e-03, 0.0000e+00, 7.3260e-04,
        5.1624e-01, 2.2247e-01, 6.1294e-02, 3.7118e-02, 3.9316e-02, 0.0000e+00,
        3.5897e-02, 0.0000e+00, 1.3333e-01, 2.4420e-04, 0.0000e+00, 0.0000e+00,
        2.1978e-03, 1.0720e-01, 0.0000e+00, 0.0000e+00, 1.6313e-01, 3.0281e-02,
        2.9304e-03, 0.0000e+00, 1.4408e-02, 2.6862e-03, 1.2210e-03, 0.0000e+00,
        9.8901e-02, 5.8608e-03, 0.0000e+00, 0.0000e+00, 1.0000e+00, 2.2686e-01,
        1.8022e-01, 4.8596e-02, 9.0354e-03, 5.2259e-02, 0.0000e+00, 1.2943e-02,
        0.0000e+00, 7.3260e-04, 4.8840e-03, 0.0000e+00, 3.0769e-02, 0.0000e+00],
       device='cuda:0')]
tensor([[ 0.3330, -0.0260, -0.8061],
        [-0.2160,  0.3768, -0.1627],
        [-0.0213, -0.2338, -0.2575],
        ...,
        [ 0.3449,  0.5707,  0.3074],
        [ 0.1826,  0.4520, -0.7284],
        [ 0.1796,  0.7911, -0.3928]], device='cuda:0',
       grad_fn=<ToCopyBackward0>)
tensor([5.6193e-03, 1.1209e-01, 1.6191e-01, 1.9902e-01, 7.3264e-04, 2.3248e-01,
        9.5970e-02, 3.0989e-01, 9.5239e-03, 0.0000e+00, 2.5494e-01, 6.1050e-03,
        2.7888e-01, 3.4188e-03, 2.4421e-03, 1.3675e-02, 1.8559e-02, 8.3492e-01,
        1.9501e-06, 1.7753e-01, 1.7582e-02, 4.3223e-02, 7.3259e-04, 3.3114e-01,
        2.0928e-01, 3.9072e-02, 2.1978e-03, 8.1318e-02, 3.1526e-01, 0.0000e+00,
        1.4652e-03, 3.6630e-03, 1.3309e-01, 0.0000e+00, 3.0550e-01, 9.6214e-02,
        0.0000e+00, 8.5471e-03, 1.4652e-03, 9.7680e-03, 1.7094e-03, 2.1465e-01,
        2.0440e-01, 9.6679e-01, 2.3468e-01, 1.8193e-01, 2.5104e-01, 0.0000e+00,
        2.9304e-03, 0.0000e+00, 1.7094e-03, 7.5458e-02, 1.1233e-02, 4.3223e-02,
        4.9573e-02, 0.0000e+00, 8.1074e-02, 0.0000e+00, 4.4689e-01, 0.0000e+00,
        8.5470e-03, 5.8608e-03, 1.0012e-02, 5.7875e-02, 9.6459e-02, 1.6923e-01,
        7.3265e-04, 7.3260e-04, 2.7351e-02, 9.2796e-03, 7.3260e-04, 0.0000e+00,
        6.5201e-02, 3.7949e-01, 0.0000e+00, 3.4668e-08, 6.0684e-01, 2.8571e-02,
        1.1893e-01, 2.4786e-01, 0.0000e+00, 9.4041e-01, 3.2259e-01, 1.8169e-01,
        1.1600e-01, 4.7620e-02, 1.6679e-01, 2.3028e-01, 9.7680e-04, 1.9536e-03,
        4.8842e-04, 1.4652e-03, 2.5836e-01, 6.8376e-03, 1.0000e+00, 2.2955e-02,
        0.0000e+00, 2.1783e-01, 1.4628e-01, 1.4896e-02, 0.0000e+00, 1.7094e-03,
        2.5934e-01, 4.5666e-02, 3.6630e-03, 5.1771e-02, 1.6361e-02, 5.1038e-02,
        2.6863e-03, 2.4542e-01, 1.8828e-01, 2.3321e-01, 2.9305e-03, 3.0769e-02,
        2.2003e-01, 1.1160e-01, 0.0000e+00, 2.9060e-02, 1.8437e-01, 2.4420e-03,
        1.9682e-01, 0.0000e+00, 2.0759e-02, 3.6387e-08, 0.0000e+00, 9.7680e-04,
        2.3468e-01, 2.5324e-01, 6.7888e-02, 6.1783e-02, 5.6411e-02, 0.0000e+00,
        1.3431e-02, 0.0000e+00, 4.2735e-02, 2.4421e-04, 0.0000e+00, 1.2869e-01,
        3.6629e-03, 2.1465e-01, 0.0000e+00, 0.0000e+00, 1.3922e-02, 1.3578e-01,
        7.3259e-04, 0.0000e+00, 1.7582e-02, 2.9304e-03, 1.2210e-03, 0.0000e+00,
        1.3016e-01, 1.7094e-03, 3.9072e-03, 0.0000e+00, 3.3895e-01, 3.2991e-01,
        5.4212e-02, 8.8645e-02, 1.1233e-02, 9.7680e-02, 0.0000e+00, 6.8377e-03,
        0.0000e+00, 7.3260e-04, 6.5934e-03, 1.8413e-01, 2.8327e-02, 0.0000e+00],
       device='cuda:0')
tensor([34, 10, 32, 41, 53, 53, 17, 50, 36, 46, 22, 42, 41, 51, 24, 24, 53,  8,
        37, 41,  5, 13, 11, 27, 25, 32, 38, 21, 33, 24, 53, 21, 33, 19, 13, 16,
        37, 52, 29, 46, 61, 27, 26, 49, 36, 34, 35, 45, 14, 50, 25, 23, 57, 44,
        24, 26,  2, 41, 25, 43, 11, 52, 43, 33, 31, 44, 34, 38, 41, 11,  8, 16,
        49, 25, 41, 58, 11, 41, 42,  8, 49, 48, 54, 26, 46, 47,  5, 21,  1, 13,
        32, 59,  5, 32, 31, 17, 41, 22, 11, 21, 57, 26, 25, 14, 53, 22, 11, 54,
        21, 12, 13, 48, 39, 24, 33, 29,  6, 51, 43, 35, 50, 18, 48, 33, 56,  6,
        32, 47, 44, 25, 16, 14, 46, 53, 32, 32, 49, 32, 58, 24,  0, 37, 36, 40,
         5, 13, 13, 31,  6,  9, 48, 21, 22,  7, 26, 57, 11, 10, 37, 49, 26, 31,
        21, 36,  9, 50, 53,  9], device='cuda:0') tensor([176,  74, 196, 110, 276, 295, 184,  13, 100, 189, 155, 153,  20, 110,
        290, 245,   5, 127, 221, 157, 106, 253, 231, 186, 123,  31,  17,  70,
         23,  33, 170, 293, 305, 266, 247, 148,  12, 210, 275, 306, 273,  14,
        159, 310, 184,  73, 251, 264, 315, 293, 291,  87,  91,  40, 152, 225,
        260, 141, 136, 138,  88,  31,  87,  29,  84, 140, 163, 203, 144, 165,
        240, 278, 117, 215, 118, 279, 223, 194,  47, 207, 318, 236,  88,  48,
         28,  59, 208, 240, 248,  67, 227, 177, 101, 240,  84, 300,   0,  66,
         71, 275, 121, 292, 294, 274, 295, 236, 161, 104, 144, 214,  47, 147,
        145, 224, 203,  46,  38,  16, 317, 275, 223, 192, 190, 235, 128, 255,
        296, 277,  98, 259, 124,  14,  12, 189, 301, 228, 264,  80,   7,  90,
        202, 255, 176, 194,  60, 111, 234, 162, 236,  95, 221, 274, 130,   2,
        124,  49, 258, 166, 249, 142, 102,  40, 146,  65, 171, 211,   6, 124],
       device='cuda:0') tensor([ 51, 155,  87, 119, 294, 185, 294,  44,  97,  42, 145, 140, 107, 244,
         99, 186, 183, 107, 304, 103, 127, 129, 264, 151, 107, 279, 216, 312,
        303, 189, 202, 110, 215, 163, 122,  89, 212,  86, 114,  31, 275,  74,
        136,  13,   6, 315, 254, 198, 150, 126, 122, 166, 201, 175,  85, 312,
         56, 286,  14, 254, 174, 274, 237,  12, 256,  94, 224, 216,  22, 165,
        198,  92, 110, 205, 117, 304, 165, 203, 315, 197, 196, 256, 303,  41,
        306, 236,  35, 119,  30, 214, 280, 245, 135, 184,  78, 123,  16,  38,
        167, 203, 280, 314,  18, 238,  93,  55, 166, 299, 113,  10, 316, 260,
        249, 190,  14, 205,  56,  56, 235, 146, 301,  40, 259, 211, 271, 158,
        176, 151,  88, 291,  88, 137, 305, 186, 272,  99,   4, 172, 307, 185,
          6, 288,  95, 275, 129, 129, 203, 162,  55, 214, 264, 126, 149, 181,
        230, 199, 251, 151, 290,   7, 138, 241, 126,   3, 117, 214, 200,  37],
       device='cuda:0')
e
[tensor([7.3260e-03, 7.3260e-04, 5.8608e-03, 3.4872e-01, 0.0000e+00, 0.0000e+00,
        2.2418e-01, 5.3724e-03, 3.1697e-01, 3.7607e-02, 8.6789e-01, 0.0000e+00,
        4.8840e-04, 1.1966e-02, 0.0000e+00, 0.0000e+00, 1.2210e-03, 7.6679e-02,
        2.1929e-01, 2.8083e-02, 6.0806e-02, 1.1722e-02, 4.8596e-02, 1.9170e-01,
        7.5458e-02, 0.0000e+00, 2.4420e-04, 1.4017e-01, 9.7680e-04, 2.4420e-04,
        3.2967e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 7.5702e-03, 2.9939e-01,
        4.8840e-04, 1.9341e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.9304e-03,
        2.6471e-01, 0.0000e+00, 5.2015e-02, 4.0733e-01, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 1.9780e-02, 1.9292e-02, 7.3260e-04,
        2.3932e-02, 0.0000e+00, 0.0000e+00, 4.7814e-01, 1.2088e-01, 2.5031e-01,
        7.3260e-04, 2.4420e-04, 1.1477e-02, 1.1966e-02, 1.7924e-01, 2.2320e-01,
        2.3810e-01, 5.9341e-02, 2.6374e-02, 4.4933e-02, 1.0208e-01, 0.0000e+00,
        3.7607e-02, 2.2198e-01, 6.1050e-02, 3.7509e-01, 8.3028e-02, 9.7680e-04,
        9.7680e-04, 2.5275e-01, 0.0000e+00, 1.9463e-01, 2.8352e-01, 1.7094e-03,
        7.3260e-04, 3.4188e-03, 1.2527e-01, 1.9048e-02, 1.4408e-02, 2.4420e-04,
        3.7582e-01, 8.8645e-02, 1.3431e-02, 4.1319e-01, 4.5543e-01, 0.0000e+00,
        0.0000e+00, 8.7668e-02, 7.0818e-03, 0.0000e+00, 3.4066e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 5.6166e-03, 1.6630e-01, 2.9792e-01,
        2.4640e-01, 1.3187e-02, 1.9536e-03, 1.7607e-01, 1.9072e-01, 1.5263e-01,
        6.5690e-02, 1.4652e-03, 9.7680e-04, 1.0989e-02, 0.0000e+00, 0.0000e+00,
        1.7900e-01, 4.7375e-02, 1.3675e-01, 6.1050e-02, 2.9011e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 6.1148e-01, 6.4225e-02, 5.7143e-02, 9.7680e-04,
        2.4420e-04, 4.7131e-02, 0.0000e+00, 1.6606e-02, 0.0000e+00, 6.8376e-03,
        4.8840e-04, 2.9304e-03, 1.0305e-01, 0.0000e+00, 7.2772e-02, 6.6349e-01,
        2.1978e-03, 5.5653e-01, 1.4017e-01, 2.6203e-01, 1.5385e-02, 5.6166e-03,
        1.6996e-01, 0.0000e+00, 1.2454e-01, 7.3260e-04, 4.0781e-02, 7.3260e-04,
        0.0000e+00, 4.2002e-02, 4.3956e-03, 7.8144e-03, 1.4164e-02, 4.8840e-04,
        2.0098e-01, 1.1233e-02, 2.5201e-01, 1.3529e-01, 4.8840e-04, 3.1990e-02],
       device='cuda:0'), tensor([1.5873e-02, 5.6166e-03, 9.5238e-03, 6.3101e-01, 1.5629e-02, 0.0000e+00,
        2.7497e-01, 1.4164e-02, 4.5665e-02, 2.9548e-02, 9.1477e-01, 5.1600e-01,
        4.8840e-04, 1.0379e-01, 0.0000e+00, 0.0000e+00, 4.8840e-04, 4.9328e-02,
        2.2784e-01, 7.0085e-02, 0.0000e+00, 4.6398e-03, 4.2076e-01, 1.7387e-01,
        6.7155e-02, 7.3260e-04, 4.8840e-04, 2.4469e-01, 2.4420e-04, 1.4652e-03,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.8559e-02, 4.3272e-01,
        7.3260e-04, 1.5385e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.4420e-03,
        3.9487e-01, 0.0000e+00, 7.9121e-02, 7.3626e-01, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 6.5934e-03, 2.5885e-02, 9.7680e-04,
        1.6166e-01, 8.0342e-02, 0.0000e+00, 2.1758e-01, 1.1795e-01, 2.3321e-01,
        3.1746e-03, 0.0000e+00, 1.6850e-02, 2.6862e-02, 2.0098e-01, 2.2222e-01,
        1.4457e-01, 2.1490e-02, 2.1245e-02, 4.5177e-02, 9.8901e-02, 0.0000e+00,
        4.6154e-02, 1.3797e-01, 2.6129e-02, 2.8547e-01, 1.0232e-01, 1.0452e-01,
        4.8840e-04, 2.8107e-01, 0.0000e+00, 3.8730e-01, 2.5543e-01, 7.0818e-03,
        1.9536e-03, 3.6630e-03, 7.7167e-02, 2.2711e-02, 1.1722e-02, 5.3724e-03,
        2.5885e-01, 5.6899e-02, 5.3724e-03, 3.8706e-01, 4.4078e-01, 0.0000e+00,
        0.0000e+00, 3.3284e-01, 9.7680e-04, 0.0000e+00, 2.3175e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0256e-02, 1.4725e-01, 3.1551e-01,
        1.6386e-01, 2.9548e-02, 4.8840e-04, 1.7411e-01, 1.8584e-01, 1.4212e-01,
        8.3272e-02, 2.4420e-04, 2.4420e-03, 5.6166e-03, 0.0000e+00, 0.0000e+00,
        1.7534e-01, 2.8083e-02, 1.6874e-01, 6.6422e-02, 2.7253e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 4.8400e-01, 4.4689e-02, 1.7582e-02, 9.7680e-04,
        0.0000e+00, 5.5678e-02, 0.0000e+00, 1.7338e-02, 0.0000e+00, 8.0586e-03,
        4.8840e-04, 9.0354e-03, 7.9853e-02, 0.0000e+00, 7.7656e-02, 7.1087e-01,
        3.1746e-03, 6.2979e-01, 1.1331e-01, 2.6081e-01, 1.5629e-02, 3.1746e-03,
        1.6777e-01, 0.0000e+00, 3.5165e-02, 7.3260e-04, 8.3272e-02, 1.9536e-03,
        0.0000e+00, 2.4664e-02, 1.7094e-02, 1.9536e-02, 9.0354e-03, 9.7680e-04,
        9.9145e-02, 2.1490e-02, 1.9585e-01, 1.5238e-01, 4.8840e-04, 9.9145e-02],
       device='cuda:0'), tensor([1.1233e-02, 7.3260e-04, 1.9536e-02, 3.6654e-01, 1.0745e-02, 0.0000e+00,
        2.1807e-01, 3.6630e-03, 4.4005e-01, 1.6850e-02, 7.4432e-01, 0.0000e+00,
        2.6862e-03, 6.9597e-02, 0.0000e+00, 0.0000e+00, 7.3260e-04, 6.2271e-02,
        2.2955e-01, 1.8559e-02, 4.3223e-02, 1.5629e-02, 8.1563e-02, 1.1575e-01,
        1.4408e-01, 0.0000e+00, 4.8840e-04, 1.3822e-01, 0.0000e+00, 2.4420e-04,
        1.4310e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.9304e-03, 2.2662e-01,
        2.4420e-04, 2.1490e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.4420e-03,
        2.9744e-01, 0.0000e+00, 7.9853e-02, 3.6606e-01, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 5.1282e-03, 1.5629e-02, 7.3260e-04,
        8.2051e-02, 4.5397e-01, 0.0000e+00, 4.4103e-01, 1.5018e-01, 2.5079e-01,
        4.8840e-04, 2.4420e-04, 6.8376e-03, 9.2796e-03, 1.8388e-01, 2.1758e-01,
        2.5006e-01, 6.8132e-02, 3.2234e-02, 6.5201e-02, 1.1062e-01, 0.0000e+00,
        5.1770e-02, 2.2173e-01, 5.2991e-02, 3.4750e-01, 1.0842e-01, 1.6606e-02,
        2.4420e-04, 2.2589e-01, 0.0000e+00, 1.9096e-01, 2.6764e-01, 2.1978e-03,
        1.2210e-03, 9.7680e-04, 1.1697e-01, 5.8608e-03, 1.0745e-02, 1.7094e-03,
        1.1477e-01, 9.0598e-02, 1.4164e-02, 2.2711e-01, 5.2137e-01, 0.0000e+00,
        0.0000e+00, 3.4921e-02, 5.1282e-03, 0.0000e+00, 1.8999e-01, 9.7680e-04,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 2.9304e-03, 1.3773e-01, 3.1600e-01,
        2.7399e-01, 2.9304e-02, 0.0000e+00, 1.6654e-01, 2.0147e-01, 1.6703e-01,
        8.5958e-02, 1.4652e-03, 7.3260e-04, 7.0818e-03, 0.0000e+00, 0.0000e+00,
        1.9048e-01, 3.5653e-02, 1.3187e-01, 5.2015e-02, 2.5568e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 6.8498e-01, 3.2650e-01, 9.1087e-02, 9.7680e-04,
        9.7680e-04, 2.2711e-02, 0.0000e+00, 7.3260e-03, 0.0000e+00, 2.4420e-03,
        0.0000e+00, 4.1514e-03, 5.0061e-02, 3.5897e-02, 0.0000e+00, 6.8913e-01,
        1.7094e-03, 5.3455e-01, 1.4408e-01, 2.6252e-01, 2.4664e-02, 7.8144e-03,
        1.6752e-01, 0.0000e+00, 1.4408e-01, 2.4420e-04, 7.9609e-02, 7.3260e-04,
        0.0000e+00, 1.2821e-01, 0.0000e+00, 2.6862e-02, 1.6361e-02, 2.4420e-04,
        2.0513e-01, 6.3492e-03, 1.6386e-01, 1.6215e-01, 4.8840e-04, 1.0012e-01],
       device='cuda:0'), tensor([4.3956e-03, 1.2210e-03, 3.3944e-02, 3.6361e-01, 0.0000e+00, 0.0000e+00,
        1.9634e-01, 4.8840e-03, 4.1685e-01, 4.1758e-02, 9.7241e-01, 1.8315e-02,
        4.8840e-04, 2.2955e-02, 0.0000e+00, 0.0000e+00, 9.7680e-04, 4.9328e-02,
        2.2955e-01, 4.1026e-02, 5.9585e-02, 8.0586e-03, 4.4444e-02, 7.1551e-02,
        1.0281e-01, 4.8840e-04, 1.2210e-03, 1.6313e-01, 4.8840e-04, 1.2210e-03,
        2.9035e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.8376e-03, 4.2955e-01,
        4.8840e-04, 1.8901e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.7094e-03,
        2.7399e-01, 0.0000e+00, 8.9866e-02, 3.6972e-01, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 9.7680e-03, 2.6374e-02, 7.3260e-04,
        1.9829e-01, 3.8462e-01, 0.0000e+00, 4.2930e-01, 1.3114e-01, 2.4982e-01,
        1.9536e-03, 4.8840e-04, 5.1282e-03, 1.3431e-02, 1.8364e-01, 2.0537e-01,
        2.4054e-01, 9.3773e-02, 0.0000e+00, 4.9328e-02, 8.2784e-02, 0.0000e+00,
        5.4212e-02, 3.2601e-01, 4.2002e-02, 3.7070e-01, 7.6435e-02, 1.9536e-02,
        0.0000e+00, 2.3028e-01, 0.0000e+00, 8.2540e-02, 2.9035e-01, 3.1746e-03,
        1.2210e-03, 2.4420e-04, 7.1551e-02, 1.5385e-02, 7.0818e-03, 2.9304e-03,
        4.4689e-02, 7.5946e-02, 1.7094e-03, 1.0256e-01, 4.6252e-01, 0.0000e+00,
        0.0000e+00, 3.9316e-02, 1.9536e-03, 0.0000e+00, 1.8462e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 6.1050e-03, 1.7656e-01, 2.7985e-01,
        2.4420e-01, 3.9805e-02, 0.0000e+00, 1.6435e-01, 1.9634e-01, 9.5971e-02,
        9.6215e-02, 7.3260e-04, 1.9536e-03, 3.1746e-03, 0.0000e+00, 0.0000e+00,
        2.0049e-01, 2.2711e-02, 1.4652e-01, 4.2979e-02, 2.6618e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 6.6593e-01, 4.0391e-01, 1.1233e-01, 4.8840e-04,
        2.4420e-04, 1.0989e-02, 0.0000e+00, 2.9548e-02, 0.0000e+00, 1.9536e-03,
        7.3260e-04, 1.2210e-03, 5.1770e-02, 4.2735e-02, 7.6679e-02, 6.3297e-01,
        1.2210e-03, 5.5775e-01, 1.3553e-01, 2.5983e-01, 1.9292e-02, 3.1746e-03,
        2.4274e-01, 0.0000e+00, 9.9145e-02, 1.2210e-03, 9.6948e-02, 1.4652e-03,
        0.0000e+00, 8.8889e-02, 0.0000e+00, 2.7839e-02, 8.3028e-03, 0.0000e+00,
        2.0904e-01, 5.1282e-03, 7.8632e-02, 1.2234e-01, 0.0000e+00, 1.3797e-01],
       device='cuda:0'), tensor([2.8816e-02, 6.1050e-03, 3.6630e-02, 4.2613e-01, 5.1770e-02, 0.0000e+00,
        2.7521e-01, 1.1966e-02, 1.3919e-01, 2.1490e-02, 9.0452e-01, 4.6178e-01,
        1.2210e-03, 2.5495e-01, 0.0000e+00, 0.0000e+00, 4.8840e-04, 6.3248e-02,
        2.0073e-01, 5.1770e-02, 3.1258e-02, 2.6862e-03, 3.7387e-01, 1.2552e-01,
        1.1038e-01, 7.3260e-04, 4.8840e-04, 2.5104e-01, 2.4420e-04, 1.4652e-03,
        3.8339e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.8376e-03, 2.5885e-01,
        4.8840e-04, 1.8242e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 4.8840e-04,
        3.2576e-01, 0.0000e+00, 1.0354e-01, 6.9597e-01, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 3.4188e-03, 1.5140e-02, 1.7094e-03,
        1.5653e-01, 5.4212e-02, 0.0000e+00, 3.1477e-01, 1.4212e-01, 2.1685e-01,
        1.9536e-03, 4.8840e-04, 1.5629e-02, 1.4164e-02, 3.3748e-01, 1.8632e-01,
        1.7778e-01, 3.7851e-02, 2.7350e-02, 1.0501e-02, 9.8169e-02, 0.0000e+00,
        2.2222e-02, 3.1893e-01, 3.2234e-02, 2.7179e-01, 1.5311e-01, 9.3040e-02,
        4.8840e-04, 2.1929e-01, 0.0000e+00, 3.6630e-01, 3.0208e-01, 5.3724e-03,
        9.7680e-04, 3.4188e-03, 4.5421e-02, 1.1477e-02, 1.4408e-02, 5.3724e-03,
        3.3211e-02, 8.0098e-02, 1.7094e-03, 1.5556e-01, 4.6203e-01, 0.0000e+00,
        0.0000e+00, 2.8987e-01, 3.9072e-03, 0.0000e+00, 1.6557e-01, 7.3260e-04,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 2.9304e-03, 1.3797e-01, 3.3162e-01,
        2.4640e-01, 3.2234e-02, 0.0000e+00, 1.4969e-01, 2.1490e-01, 1.3773e-01,
        7.6190e-02, 7.3260e-04, 1.4652e-03, 3.9072e-03, 0.0000e+00, 0.0000e+00,
        1.8120e-01, 1.7094e-02, 1.4383e-01, 7.4237e-02, 2.4005e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 5.3162e-01, 1.5873e-01, 2.8327e-02, 1.4652e-03,
        0.0000e+00, 3.5165e-02, 0.0000e+00, 1.5873e-02, 0.0000e+00, 6.3492e-03,
        2.4420e-04, 1.3675e-02, 4.3223e-02, 3.7118e-02, 6.5201e-02, 6.9841e-01,
        3.1746e-03, 6.0269e-01, 1.1355e-01, 2.7033e-01, 2.1978e-02, 5.1282e-03,
        2.0073e-01, 0.0000e+00, 6.3736e-02, 4.8840e-04, 7.0330e-02, 2.6862e-03,
        0.0000e+00, 9.0354e-02, 0.0000e+00, 4.3956e-02, 1.1722e-02, 2.4420e-04,
        1.7558e-01, 1.4164e-02, 2.6227e-01, 1.6972e-01, 4.8840e-04, 2.7839e-02],
       device='cuda:0'), tensor([1.7827e-02, 1.9536e-03, 1.1233e-02, 4.6178e-01, 0.0000e+00, 0.0000e+00,
        2.3028e-01, 6.8376e-03, 1.2967e-01, 3.0769e-02, 6.0904e-01, 9.5482e-02,
        1.9536e-03, 5.2259e-02, 0.0000e+00, 0.0000e+00, 1.2210e-03, 5.0061e-02,
        2.5128e-01, 4.9817e-02, 0.0000e+00, 1.8559e-02, 3.3138e-01, 1.2479e-01,
        3.3944e-02, 9.7680e-04, 7.3260e-04, 2.4884e-01, 9.7680e-04, 7.3260e-04,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 5.8608e-03, 7.0452e-01,
        4.8840e-04, 1.5751e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.7094e-03,
        3.9902e-01, 0.0000e+00, 8.1563e-02, 6.2882e-01, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 2.1978e-03, 2.2711e-02, 1.9536e-03,
        1.1087e-01, 1.3431e-01, 0.0000e+00, 4.5714e-01, 1.3358e-01, 2.7375e-01,
        8.0586e-03, 4.8840e-04, 9.0354e-03, 1.0012e-02, 1.8950e-01, 2.2100e-01,
        1.9560e-01, 2.0757e-02, 2.0024e-02, 5.1526e-02, 9.4750e-02, 0.0000e+00,
        6.9353e-02, 2.3370e-01, 1.6361e-02, 2.6545e-01, 7.0574e-02, 6.5690e-02,
        1.4652e-03, 2.6422e-01, 0.0000e+00, 4.2833e-01, 2.0513e-01, 4.3956e-03,
        7.3260e-04, 4.6398e-03, 1.0061e-01, 6.5934e-03, 9.5238e-03, 1.4652e-03,
        6.7399e-02, 9.7192e-02, 2.4664e-02, 1.2454e-01, 5.3211e-01, 0.0000e+00,
        0.0000e+00, 3.2234e-01, 9.7680e-04, 0.0000e+00, 2.4151e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 1.8071e-02, 1.5556e-01, 3.0183e-01,
        3.9316e-02, 1.7827e-02, 1.7094e-03, 1.8168e-01, 1.9365e-01, 1.3871e-01,
        6.9597e-02, 1.7094e-03, 3.1746e-03, 3.6630e-03, 0.0000e+00, 0.0000e+00,
        1.8632e-01, 1.7094e-03, 2.3468e-01, 6.5446e-02, 2.4200e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 5.2650e-01, 0.0000e+00, 8.7668e-02, 9.7680e-04,
        2.4420e-04, 9.2063e-02, 0.0000e+00, 2.8083e-02, 0.0000e+00, 4.3956e-03,
        4.8840e-04, 1.2210e-02, 7.3993e-02, 9.1087e-02, 4.0781e-02, 7.6386e-01,
        4.8840e-03, 5.9292e-01, 1.6630e-01, 2.5421e-01, 4.3956e-03, 4.6398e-03,
        2.3272e-01, 0.0000e+00, 3.2479e-02, 7.3260e-04, 8.6447e-02, 1.7094e-03,
        0.0000e+00, 9.7680e-03, 0.0000e+00, 3.2723e-02, 5.3724e-03, 7.3260e-04,
        2.4176e-02, 2.6862e-02, 2.2613e-01, 1.5604e-01, 2.4420e-04, 6.8376e-02],
       device='cuda:0'), tensor([9.2796e-03, 2.4420e-03, 2.6129e-02, 3.5653e-01, 0.0000e+00, 0.0000e+00,
        2.3712e-01, 7.8144e-03, 8.8400e-02, 3.1013e-02, 6.0122e-01, 4.2735e-02,
        2.1978e-03, 2.1294e-01, 0.0000e+00, 0.0000e+00, 1.7094e-03, 5.0794e-02,
        2.2784e-01, 5.5922e-02, 1.0256e-02, 5.1282e-03, 2.9158e-01, 1.6752e-01,
        1.1477e-01, 2.4420e-04, 4.8840e-04, 2.5519e-01, 9.7680e-04, 7.3260e-04,
        1.1282e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.2943e-02, 6.3199e-01,
        4.8840e-04, 1.4139e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 9.7680e-04,
        3.1429e-01, 0.0000e+00, 9.9390e-02, 6.1734e-01, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 4.3956e-03, 2.1978e-02, 0.0000e+00,
        7.5214e-02, 5.4457e-02, 0.0000e+00, 4.6203e-01, 1.5531e-01, 2.5495e-01,
        5.8608e-03, 2.4420e-04, 9.2796e-03, 4.6398e-03, 2.9866e-01, 1.9560e-01,
        2.1001e-01, 4.5177e-02, 2.1978e-03, 2.9548e-02, 7.8632e-02, 0.0000e+00,
        2.4664e-02, 3.3480e-01, 3.8584e-02, 2.7033e-01, 1.6288e-01, 1.0989e-02,
        4.8840e-04, 2.2735e-01, 0.0000e+00, 3.6996e-01, 2.9988e-01, 5.8608e-03,
        4.8840e-04, 5.1282e-03, 9.3040e-02, 3.1746e-03, 5.6166e-03, 7.3260e-04,
        2.5641e-02, 9.6459e-02, 3.4188e-03, 9.2063e-02, 4.3126e-01, 0.0000e+00,
        0.0000e+00, 2.8327e-01, 4.1514e-03, 0.0000e+00, 1.4945e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 1.1233e-02, 1.5629e-01, 3.1770e-01,
        1.5311e-01, 2.1734e-02, 0.0000e+00, 1.7607e-01, 1.9365e-01, 1.2527e-01,
        7.8144e-02, 2.4420e-04, 5.3724e-03, 6.5934e-03, 0.0000e+00, 0.0000e+00,
        1.8486e-01, 4.1514e-03, 2.1074e-01, 5.3480e-02, 2.4029e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 4.8132e-01, 1.1526e-01, 6.1783e-02, 1.4652e-03,
        0.0000e+00, 7.5214e-02, 0.0000e+00, 4.8107e-02, 0.0000e+00, 1.7094e-03,
        2.4420e-04, 9.7680e-04, 2.7595e-02, 5.1526e-02, 9.7192e-02, 7.1770e-01,
        1.2210e-03, 5.9878e-01, 1.5336e-01, 2.3101e-01, 1.0745e-02, 2.1978e-03,
        2.3590e-01, 0.0000e+00, 4.6398e-03, 7.3260e-04, 7.2039e-02, 1.7094e-03,
        0.0000e+00, 8.1074e-02, 1.2698e-02, 1.8071e-02, 1.5140e-02, 0.0000e+00,
        8.9866e-02, 2.2222e-02, 2.0440e-01, 1.4774e-01, 4.8840e-04, 5.8608e-02],
       device='cuda:0')]
tensor([[ 0.7098, -0.3270,  0.7668],
        [ 0.2396,  0.1351,  0.4510],
        [ 0.5019,  0.4064,  0.0981],
        ...,
        [ 0.3449,  0.3968, -0.0833],
        [ 0.3629, -0.4723, -0.1437],
        [ 0.1341,  0.0280, -0.8404]], device='cuda:0',
       grad_fn=<ToCopyBackward0>)
tensor([3.9075e-03, 8.0586e-03, 1.9048e-02, 2.2442e-01, 2.3346e-01, 0.0000e+00,
        2.9524e-01, 1.2210e-02, 3.1477e-01, 1.3358e-01, 3.1380e-01, 1.7656e-01,
        4.8843e-04, 2.8571e-01, 0.0000e+00, 1.3550e-07, 7.3260e-04, 2.5153e-02,
        3.3358e-01, 1.8901e-01, 4.8595e-02, 3.6632e-03, 8.5472e-03, 2.2759e-01,
        2.6374e-02, 7.3260e-04, 7.3260e-04, 1.6606e-02, 7.6998e-09, 1.2210e-03,
        2.4958e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.9537e-03, 3.8510e-01,
        4.8840e-04, 2.9305e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 6.3492e-03,
        2.6471e-01, 0.0000e+00, 5.0061e-02, 4.9475e-01, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 9.2796e-03, 9.2797e-03, 1.2210e-03,
        1.4086e-08, 2.5519e-01, 0.0000e+00, 2.2076e-01, 1.2332e-01, 2.1563e-01,
        3.4188e-03, 1.2210e-03, 7.8144e-03, 2.3932e-02, 2.1294e-01, 4.5422e-02,
        2.2686e-01, 4.5665e-02, 1.5629e-02, 4.6886e-02, 2.4371e-01, 0.0000e+00,
        3.0525e-02, 1.5604e-01, 3.7363e-02, 2.7521e-01, 7.6679e-02, 1.6605e-02,
        1.2227e-09, 8.7911e-02, 0.0000e+00, 8.5470e-02, 2.5177e-01, 5.1282e-03,
        2.4420e-04, 4.1514e-03, 1.3626e-01, 1.9048e-02, 9.2795e-03, 1.4653e-03,
        2.7961e-08, 8.2296e-02, 1.2211e-03, 4.2002e-01, 4.7961e-01, 0.0000e+00,
        2.0140e-15, 1.3285e-01, 5.3724e-03, 0.0000e+00, 2.3077e-01, 4.1758e-02,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 1.9537e-03, 1.6239e-01, 3.0574e-01,
        2.0146e-01, 2.6618e-02, 4.8842e-04, 1.2552e-01, 2.0855e-01, 1.0916e-01,
        4.9084e-02, 1.7094e-03, 3.1746e-03, 4.6399e-03, 0.0000e+00, 0.0000e+00,
        2.5519e-01, 6.2515e-02, 2.2332e-07, 4.9817e-02, 2.3736e-01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 6.4664e-01, 3.7705e-01, 4.8107e-02, 4.8841e-04,
        4.8840e-04, 7.3992e-02, 0.0000e+00, 3.2967e-02, 0.0000e+00, 7.3260e-03,
        2.4420e-04, 1.9292e-02, 2.8816e-02, 4.8841e-02, 1.5873e-02, 6.9939e-01,
        4.8840e-04, 7.0183e-01, 1.8315e-01, 2.6252e-01, 7.5701e-03, 7.0817e-03,
        2.1490e-02, 0.0000e+00, 1.6874e-01, 4.8842e-04, 5.8852e-02, 2.1978e-03,
        0.0000e+00, 7.1061e-02, 2.5055e-01, 9.7681e-03, 1.4164e-02, 7.3260e-04,
        2.0098e-01, 2.0269e-02, 1.0818e-01, 8.8401e-02, 2.4421e-04, 1.2357e-01],
       device='cuda:0')
tensor([17,  7, 48, 44, 21,  4, 24, 16, 16, 56, 23, 43, 48, 40,  9, 46, 49, 25,
         7, 27, 44, 29, 38, 57,  9, 52, 32, 43, 53, 18, 39, 17, 60, 37, 57, 44,
        43, 27, 45, 49, 41, 22, 44, 39, 37, 33, 13,  1,  7, 10, 58, 17, 10, 52,
        38, 12,  5, 42, 57, 10, 57, 22, 21, 34, 17, 12, 29, 26, 11, 31, 10, 18,
        19, 37, 43, 52, 37, 24,  9, 38, 51, 30, 51, 19, 20, 45, 57, 53, 52, 31,
         8, 52, 50, 50, 61, 38, 21, 40, 22, 37, 39, 54,  8, 56, 62,  5, 49, 55,
        41, 20, 45, 45, 25, 59, 35, 23,  8, 14, 29, 15, 18,  5, 13, 37, 45, 20,
        49, 25, 29, 46, 37,  6,  4, 14, 37,  8, 52, 32,  0, 50,  5,  9, 16, 16,
        46,  6, 38, 16, 17, 28, 33, 15, 14, 16, 30, 52, 47, 23, 24,  9,  8, 26,
        41, 36, 21,  8, 51, 37, 42,  9, 43,  3,  9, 10, 38,  6, 47],
       device='cuda:0') tensor([286, 112, 128,  71,  88,  38,  18, 197, 319, 202, 292,  39,  48, 118,
         55, 297,  16, 233, 284,  96, 120,  71, 181, 230, 171,  79, 211, 217,
         53, 242,  58, 186, 242, 319, 191,  47,  10, 311,  59,  58,  51, 268,
        135, 127,  56, 312, 105, 154, 234, 245,  33, 169, 281, 195, 278,  26,
         41,  33, 204, 184, 102, 223,  25, 311, 307, 260, 319, 319, 258, 110,
        219, 278,  31, 182, 123, 214, 311, 273, 165,  84, 292,  77,  75, 118,
        107, 155, 106,  67, 294, 104, 216, 218, 112, 238, 230, 241,  55, 272,
        168, 298, 140, 198, 301, 210,  12, 266, 200, 165, 255, 168, 283, 190,
         93, 178, 116, 118,  47, 138, 182,  20, 293, 177, 282,  13, 256,  56,
        161,  28,  25, 181, 169,   2, 280,  12, 272, 213, 253, 190,  38,  46,
         83,  56, 229, 179, 125, 154, 299, 102, 105, 211,  58, 199, 159,  99,
        297, 124, 210, 208,  54, 283, 170, 197,  32, 131, 215, 302, 110, 138,
        302, 318,   4, 252, 289, 246,  56, 280, 243], device='cuda:0') tensor([207,  74, 171,  88, 297,  17,  98, 178, 283, 270, 243,  51, 183,   2,
         27,  31, 189, 201, 166, 249, 269,  23, 313, 274, 205,  69, 178,  61,
        109, 230, 156, 206, 265, 204, 202, 271, 248,  80, 111, 297, 109, 134,
         73,  75, 299, 119,  23,  40, 175, 154, 315,  35, 248,  76, 136,   5,
        133, 130, 201,  59, 184,  38, 221, 218, 118,  19, 110,  53, 164, 247,
         52, 243, 256, 297, 250,  79,  17, 186,  32,  47, 158,  54, 152, 246,
        192, 200, 200, 288, 175,  58,  20,  81, 301, 122, 281, 306,  30,   5,
        235, 283,  57, 302, 200, 250, 313,  28, 283, 149,  16, 177, 203, 195,
        199, 248, 241, 265,   4,  63,  13, 166, 143,  29, 127,  18,  11, 191,
        103,  19, 103,  27, 192, 140,   9,  43, 120, 191, 268,  98,   4, 124,
        122,  27, 284, 106,  36, 155,  43, 109, 223, 270, 305, 158,  58,   3,
        313, 271, 147,  70,   5, 133,   9, 225, 106, 282,  34,  96, 141,  19,
        141, 221, 160,  72, 220, 249, 222,  58,  48], device='cuda:0')
"""