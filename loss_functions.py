import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor
from typing import Optional, Tuple
from torch.nn.modules.loss import _Loss
import pdb


from utils.utils_loss import nmi_gauss, nmi_gauss_mask, StableStd, finite_diff, param_ndim_setup, ncc_mask, ncc, gradient, compute_jacobian_matrix

class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.weight: Optional[Tensor]


class NCC(_Loss):
    def __init__(self, use_mask: bool = False):
        super().__init__()
        if use_mask:
            self.forward = self.masked_metric
        else:
            self.forward = self.metric

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        return -ncc(fixed, warped)

    def masked_metric(self, fixed: Tensor, warped: Tensor, mask: Tensor) -> Tensor:
        return -ncc_mask(fixed, warped, mask)



class NMI(_Loss):
    """Normalized mutual information metric.

    As presented in the work by `De Vos 2020: <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11313/113130R/Mutual-information-for-unsupervised-deep-learning-image-registration/10.1117/12.2549729.full?SSO=1>`_

    """

    def __init__(
        self,
        intensity_range: Optional[Tuple[float, float]] = None,
        nbins: int = 32,
        sigma: float = 0.1,
        use_mask: bool = False,
    ):
        super().__init__()
        self.intensity_range = intensity_range
        self.nbins = nbins
        self.sigma = sigma
        if use_mask:
            self.forward = self.masked_metric
        else:
            self.forward = self.metric

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed.min(), fixed.max()
                warped_range = warped.min(), warped.max()

        bins_fixed = torch.linspace(
            fixed_range[0],
            fixed_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )
        bins_warped = torch.linspace(
            warped_range[0],
            warped_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )

        return -nmi_gauss(
            fixed, warped, bins_fixed, bins_warped, sigma=self.sigma
        ).mean()

    def masked_metric(self, fixed: Tensor, warped: Tensor, mask: Tensor) -> Tensor:
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed.min(), fixed.max()
                warped_range = warped.min(), warped.max()

        bins_fixed = torch.linspace(
            fixed_range[0],
            fixed_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )
        bins_warped = torch.linspace(
            warped_range[0],
            warped_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )

        return -nmi_gauss_mask(
            fixed, warped, bins_fixed, bins_warped, mask, sigma=self.sigma
        )


class MILossGaussian(nn.Module):
    """
    Mutual information loss using Gaussian kernel in KDE
    """
    def __init__(self,
                 vmin=0.0,
                 vmax=1.0,
                 num_bins=16, # 32 or 64 
                 sample_ratio=1.0, #0.7 for t1 and t2
                 normalised=True,
                 gt_val=None,
                 ):
        super(MILossGaussian, self).__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.sample_ratio = sample_ratio
        self.normalised = normalised
        self.gt_val = gt_val

        # set the std of Gaussian kernel so that FWHM is one bin width
        bin_width = (vmax - vmin) / num_bins
        self.sigma = bin_width * (1/(2 * math.sqrt(2 * math.log(2))))

        # set bin edges
        self.num_bins = num_bins
        self.bins = torch.linspace(self.vmin, self.vmax, self.num_bins, requires_grad=False).unsqueeze(1)

    def _compute_joint_prob(self, x, y):
        """
        Compute joint distribution and entropy
        Input shapes (N, 1, prod(sizes))
        """
        # cast bins
        self.bins = self.bins.type_as(x)

        # calculate Parzen window function response (N, #bins, H*W*D)
        win_x = torch.exp(-(x - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_x = win_x / (math.sqrt(2 * math.pi) * self.sigma)
        win_y = torch.exp(-(y - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_y = win_y / (math.sqrt(2 * math.pi) * self.sigma)

        # calculate joint histogram batch
        hist_joint = win_x.bmm(win_y.transpose(1, 2))  # (N, #bins, #bins)

        # normalise joint histogram to get joint distribution
        hist_norm = hist_joint.flatten(start_dim=1, end_dim=-1).sum(dim=1) + 1e-5
        p_joint = hist_joint / hist_norm.view(-1, 1, 1)  # (N, #bins, #bins) / (N, 1, 1)

        return p_joint

    def forward(self, x, y):
        """
        Calculate (Normalised) Mutual Information Loss.
        Args:
            x: (torch.Tensor, size (N, 1, *sizes))
            y: (torch.Tensor, size (N, 1, *sizes))
        Returns:
            (Normalise)MI: (scalar)
        """
        if self.sample_ratio < 1.:
            # random spatial sampling with the same number of pixels/voxels
            # chosen for every sample in the batch
            numel_ = np.prod(x.size()[2:])
            idx_th = int(self.sample_ratio * numel_)
            idx_choice = torch.randperm(int(numel_))[:idx_th]

            x = x.view(x.size()[0], 1, -1)[:, :, idx_choice]
            y = y.view(y.size()[0], 1, -1)[:, :, idx_choice]

        # make sure the sizes are (N, 1, prod(sizes))
        x = x.flatten(start_dim=2, end_dim=-1)
        y = y.flatten(start_dim=2, end_dim=-1)


        # compute joint distribution
        p_joint = self._compute_joint_prob(x, y)
        p_joint = p_joint/ p_joint.sum()

        # marginalise the joint distribution to get marginal distributions
        # batch size in dim0, x bins in dim1, y bins in dim2
        p_x = torch.sum(p_joint, dim=2)
        p_y = torch.sum(p_joint, dim=1)

        #px_py = p_x[:, None] * p_y[None, :]
        # px_py = (p_y.T @ p_x).unsqueeze(0)

        px_py = torch.outer(p_x.squeeze(), p_y.squeeze()).unsqueeze(0)
        nzs = p_joint >0 #>0

        # calculate entropy
        #ent_x = - torch.sum(p_x * torch.log(p_x + 1e-5), dim=1)  # (N,1)
        #ent_y = - torch.sum(p_y * torch.log(p_y + 1e-5), dim=1)  # (N,1)
        #ent_joint = - torch.sum(p_joint * torch.log(p_joint + 1e-5), dim=(1, 2))  # (N,1)

        '''

        if self.normalised:
            if self.gt_val:
                return (torch.mean((ent_x + ent_y) / ent_joint)-self.gt_val)**2
            else:
                return torch.mean((ent_x + ent_y) / ent_joint)
        else:
            if self.gt_val:
                return (torch.mean(ent_x + ent_y - ent_joint)-self.gt_val)**2
            else:
                return torch.mean(ent_x + ent_y - ent_joint)
        '''
        # pdb.set_trace()

        if self.gt_val:
            return ((p_joint[nzs] * torch.log(p_joint[nzs]/ (px_py[nzs]+ 1e-12))).sum()-self.gt_val)**2
        else:
            return (p_joint[nzs] * torch.log(p_joint[nzs]/ (px_py[nzs]+ 1e-12))).sum()




class LNCCLoss(nn.Module):
    """
    Local Normalized Cross Correlation loss
    Adapted from VoxelMorph implementation:
    https://github.com/voxelmorph/voxelmorph/blob/5273132227c4a41f793903f1ae7e27c5829485c8/voxelmorph/torch/losses.py#L7
    """
    def __init__(self, window_size=7):
        super(LNCCLoss, self).__init__()
        self.window_size = window_size

    def forward(self, x, y):
        # products and squares
        xsq = x * x
        ysq = y * y
        xy = x * y

        # set window size
        ndim = x.dim() - 2
        window_size = param_ndim_setup(self.window_size, ndim)

        # summation filter for convolution
        sum_filt = torch.ones(1, 1, *window_size).type_as(x)

        # set stride and padding
        stride = (1,) * ndim
        padding = tuple([math.floor(window_size[i]/2) for i in range(ndim)])

        # get convolution function of the correct dimension
        conv_fn = getattr(F, f'conv{ndim}d')

        # summing over window by convolution
        x_sum = conv_fn(x, sum_filt, stride=stride, padding=padding)
        y_sum = conv_fn(y, sum_filt, stride=stride, padding=padding)
        xsq_sum = conv_fn(xsq, sum_filt, stride=stride, padding=padding)
        ysq_sum = conv_fn(ysq, sum_filt, stride=stride, padding=padding)
        xy_sum = conv_fn(xy, sum_filt, stride=stride, padding=padding)

        window_num_points = np.prod(window_size)
        x_mu = x_sum / window_num_points
        y_mu = y_sum / window_num_points

        cov = xy_sum - y_mu * x_sum - x_mu * y_sum + x_mu * y_mu * window_num_points
        x_var = xsq_sum - 2 * x_mu * x_sum + x_mu * x_mu * window_num_points
        y_var = ysq_sum - 2 * y_mu * y_sum + y_mu * y_mu * window_num_points

        lncc = cov * cov / (x_var * y_var + 1e-5)

        return -torch.mean(lncc)
    
def compute_jacobian_loss(input_coords, output, batch_size=None):
    """Compute the jacobian regularization loss."""

    # Compute Jacobian matrices
    jac = compute_jacobian_matrix(input_coords, output)

    # Compute determinants and take norm
    loss = torch.det(jac) - 1
    loss = torch.linalg.norm(loss, 1)

    return loss / batch_size


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


def l2reg_loss(u):
    """L2 regularisation loss"""
    derives = []
    ndim = u.size()[1]
    for i in range(ndim):
        derives += [finite_diff(u, dim=i)]
    loss = torch.cat(derives, dim=1).pow(2).sum(dim=1).mean()
    return loss


def bending_energy_loss(u):
    """Bending energy regularisation loss"""
    derives = []
    ndim = u.size()[1]
    # 1st order
    for i in range(ndim):
        derives += [finite_diff(u, dim=i)]
    # 2nd order
    derives2 = []
    for i in range(ndim):
        derives2 += [finite_diff(derives[i], dim=i)]  # du2xx, du2yy, (du2zz)
    derives2 += [math.sqrt(2) * finite_diff(derives[0], dim=1)]  # du2dxy
    if ndim == 3:
        derives2 += [math.sqrt(2) * finite_diff(derives[0], dim=2)]  # du2dxz
        derives2 += [math.sqrt(2) * finite_diff(derives[1], dim=2)]  # du2dyz

    assert len(derives2) == 2 * ndim
    loss = torch.cat(derives2, dim=1).pow(2).sum(dim=1).mean()
    return loss


def total_variation_loss(img, edge_map=None):
     if edge_map != None:
          edge_map = 1.0 -edge_map.permute(0,2,3,1)
     else:
          edge_map = torch.ones(img.shape).cuda()
     tv_h1 = (torch.pow(img[:,:,:,1:,:]-img[:,:,:,:-1,:], 2)*edge_map[:,:,:,1:,:]).sum()
     tv_w1 = (torch.pow(img[:,:,1:,:,:]-img[:,:,:-1,:,:], 2)*edge_map[:,:,1:,:,:]).sum()
     tv_d1 = (torch.pow(img[:,1:,:,:,:]-img[:,-1:,:,:,:], 2)*edge_map[:,1:,:,:,:]).sum()
     
     tv_h2 = (torch.pow(img[:,:,:,1:,:]-img[:,:,:,:-1,:], 2)*edge_map[:,:,:,:-1,:]).sum()
     tv_w2 = (torch.pow(img[:,:,1:,:,:]-img[:,:,:-1,:,:], 2)*edge_map[:,:,:-1,:,:]).sum()
     tv_d2 = (torch.pow(img[:,1:,:,:,:]-img[:,:-1,:,:,:], 2)*edge_map[:,:-1,:,:,:]).sum()

     return 0.163*(tv_h1+tv_w1+tv_d1+tv_h2+tv_w2+tv_d2)/edge_map.sum()