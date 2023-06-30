#import matplotlib
##matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def show_slices(slices, epoch):
    """ Function to display row of image slices """
    plt.close()
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    plt.suptitle(f"Selected Centre Slices of NERF-based brain after {epoch}.")
    return fig

def show_slices_gt(slices, gt_slices, epoch):
    """ Function to display row of image slices """
    plt.close()
    fig, axes = plt.subplots(2, len(slices), dpi=150)

    for i, slice in enumerate(slices):
        axes[0][i].imshow(slice.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    for i, slice in enumerate(gt_slices):
        axes[1][i].imshow(slice.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    plt.suptitle(f"Selected Centre Slices of the MRI after {epoch}.")
    plt.tight_layout()
    return fig

def show_slices_registration(slices, epoch):
    plt.close()
    fig, axes = plt.subplots(1, len(slices), dpi=150)
    x_step = [5, 5, 5]

    for i, slice in enumerate(slices):
        step = x_step[i]
        coordinates = all_coordinates(slice.shape[:-1])
        coord = filter_coordinates(coordinates, coordinates, step, step).reshape(-1, 2)
        new_slice = filter_coordinates(slice, coordinates, step, step).reshape(-1, 3)
        
        axes[i].quiver(coord[:,0], coord[:,1], 10*new_slice[:,0], 10*new_slice[:,1])
        axes[i].set_aspect(aspect=1.0)
    """
    pos1 = axes[1].get_position(original=False)
    pos0 = axes[0].get_position(original=False)
    new_width = pos1.height * (axes[0].get_xlim()[1] / axes[0].get_ylim()[1])
    axes[0].set_position([pos0.x0 - (new_width - pos0.width), pos1.y0, new_width, pos1.height])
    """
        
    plt.suptitle(f"Registration fiels after {epoch}.")
    plt.tight_layout()
    return fig

def filter_coordinates(array, coordinates, n, m):
    mask = np.mod(coordinates[:, :, 0], n) == 0
    mask &= np.mod(coordinates[:, :, 1], m) == 0
    filtered_coordinates = array[mask]
    return filtered_coordinates

def all_coordinates(shape):
    indices = np.meshgrid(*[np.arange(n) for n in shape], indexing='ij')
    coordinates = np.transpose(indices, (1, 2, 0))
    return coordinates

def show_jacobian_det(slices, epoch, suptitle):
    plt.close()
    fig, axes = plt.subplots(1, len(slices), dpi=150)
    im = []
    for i, slice in enumerate(slices):
        im.append(axes[i].imshow(slice.T, cmap='Reds', interpolation='nearest'))
        axes[i].invert_yaxis()
        
    fig.colorbar(im[0], ax=axes.ravel().tolist(), orientation='horizontal')
    plt.subplots_adjust(bottom=0.5)
    plt.suptitle(suptitle)
    #plt.tight_layout()
    return fig
    

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
