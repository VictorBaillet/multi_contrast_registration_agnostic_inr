import torch
import numpy as np
import json

from utils.dataset.dataset_utils import norm_grid

def center_of_mass(x):
    """
    Compute the center of mass of an array.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    center_of_mass : ndarray
        The coordinates of the center of mass of the input array.
    """
    m = np.sum(x)
    if m == 0:
        return np.array([0.0, 0.0, 0.0])
    else:
        i = np.arange(x.shape[0]).reshape(x.shape[0], 1, 1)
        j = np.arange(x.shape[1]).reshape(1, x.shape[1], 1)
        k = np.arange(x.shape[2]).reshape(1, 1, x.shape[2])
        return np.array([np.sum(i * x) / m, np.sum(j * x) / m, np.sum(k * x) / m])

    
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def min_max_scale(X, s_min, s_max):
    """
    Scale the input array to the given range.

    Parameters
    ----------
    X : ndarray
        Input array.
    s_min : float
        Minimum value of the output range.
    s_max : float
        Maximum value of the output range.

    Returns
    -------
    scaled_X : tensor
        The scaled input array.
    """
    x_min, x_max = X.min(), X.max()
    return torch.tensor((X - x_min) / (x_max - x_min) * (s_max - s_min) + s_min)




def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices, min_coord, max_coord, device, rev_affine):
    """
    Interpolate the intensity of voxels based on the closest known neighbors.

    Parameters
    ----------
    input_array : tensor
        Torch array containing the intensity of the known voxels.
    x_indices : tensor
        Torch array containing the x indices of the voxels to extrapolate.
    y_indices : tensor
        Torch array containing the y indices of the voxels to extrapolate.
    z_indices : tensor
        Torch array containing the z indices of the voxels to extrapolate.
    min_coord : tuple
        Minimum coordinates from all the training points (for normalization purposes).
    max_coord : tuple
        Maximum coordinates from all the training points (for normalization purposes).
    device : str
        Training device.
    rev_affine : tensor
        4x4 Torch matrix corresponding to the affine transformation "coordinate in space" -> "(x, y, z) coordinates in input_array".

    Returns
    -------
    output : tensor
        Extrapolated intensity of the desired voxels.
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