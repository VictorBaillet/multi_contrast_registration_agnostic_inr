#import matplotlib
##matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def show_slices(slices, epoch):
    """ Function to display row of image slices """
    plt.close()
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.suptitle(f"Selected Centre Slices of NERF-based brain after {epoch}.")
    return fig

def show_slices_gt(slices, gt_slices, epoch):
    """ Function to display row of image slices """
    plt.close()
    fig, axes = plt.subplots(2, len(slices), dpi=150)

    for i, slice in enumerate(slices):
        axes[0][i].imshow(slice.T, cmap="gray", origin="lower")
    for i, slice in enumerate(gt_slices):
        axes[1][i].imshow(slice.T, cmap="gray", origin="lower")
    plt.suptitle(f"Selected Centre Slices of NERF-based brain after {epoch}.")
    plt.tight_layout()
    return fig

def show_slices_registration(slices, epoch):
    plt.close()
    fig, axes = plt.subplots(1, len(slices), dpi=150)
    coords = []
    for i, slice in enumerate(slices):
        coords.append(list(np.zeros((slice.shape[0], slice.shape[1], 2))))
        for x in range(len(slice)):
            for y in range(len(slice[0])):
                coords[-1][x][y] = [x, y]
                
    for i, slice in enumerate(slices):
        slice = slice.reshape(-1, slice.shape[-1])
        coord = np.array(coords[i])
        coord = coord.reshape(-1, coord.shape[-1])
        
        axes[i].quiver(coord[::10,0], coord[::10,1], 10*slice[::10, 0], 10*slice[::10, 1])
        
    plt.suptitle(f"Registration fiels after {epoch}.")
    plt.tight_layout()
    return fig
    
