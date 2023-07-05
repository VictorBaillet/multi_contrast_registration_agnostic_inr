import nibabel
import numpy as np
import torch
from tqdm import tqdm
# import SimpleITK as sitk
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler

from scipy import ndimage
from skimage import filters


def norm_grid(grid, xmin, xmax, smin=-1, smax=1):
    """
    Normalize grid values.

    Parameters
    ----------
    grid : numpy.ndarray
        Grid.
    xmin : float
        Minimum x value.
    xmax : float
        Maximum x value.
    smin : float, optional
        Minimum s value. (default: -1)
    smax : float, optional
        Maximum s value. (default: 1)

    Returns
    -------
    numpy.ndarray
        Normalized grid values.
    """

    def min_max_scale(X, x_min, x_max, s_min, s_max):
        return (X - x_min)/(x_max - x_min)*(s_max - s_min) + s_min

    return min_max_scale(X=grid, x_min=xmin, x_max=xmax, s_min=smin, s_max=smax)


def get_crop_indexes(image1: nibabel.Nifti1Image, image2: nibabel.Nifti1Image,
                   image1_lr: nibabel.Nifti1Image, image2_lr: nibabel.Nifti1Image):
    """
    Get crop indexes of the largest shared field of view among the images.

    Parameters
    ----------
    image1 : nibabel.Nifti1Image
        Contrast 1.
    image2 : nibabel.Nifti1Image
        Contrast 2.
    image1_lr : nibabel.Nifti1Image
        Contrast 1 low resolution.
    image2_lr : nibabel.Nifti1Image
        Contrast 2 low resolution.

    Returns
    -------
    tuple
        Crop indexes.
    """
    img1_affine = image1.affine
    (x1, y1, z1) = image1.shape
    
    img2_affine = image2.affine
    (x2, y2, z2) = image2.shape
    
    img1_lr_affine = image1_lr.affine
    img2_lr_affine = image2_lr.affine
    
    img1_corners = np.array([[0,       0,    0],
                            [x1-1,    0,    0],
                            [0,    y1-1,    0],
                            [x1-1, y1-1,    0],
                            [0,       0, z1-1],
                            [x1-1,    0, z1-1],
                            [x1-1, y1-1, z1-1]])
    
    img2_corners = np.array([[0,       0,    0],
                            [x2-1,    0,    0],
                            [0,    y2-1,    0],
                            [x2-1, y2-1,    0],
                            [0,       0, z2-1],
                            [x2-1,    0, z2-1],
                            [x2-1, y2-1, z2-1]])
    
    img1_corners_coordinates = []
    img2_corners_coordinates = []
    
    for coord in img1_corners:
        img1_corners_coordinates.append(nib.affines.apply_affine(img1_affine, coord))
    
    for coord in img2_corners:
        img2_corners_coordinates.append(nib.affines.apply_affine(img2_affine, coord))
    
    img1_corners_coordinates = np.array(img1_corners_coordinates)
    img2_corners_coordinates = np.array(img2_corners_coordinates)
    
    
    coord_min = []
    coord_max = []
    for i in range(3):
        coord_min.append(max(np.min(img1_corners_coordinates[:,i]), np.min(img2_corners_coordinates[:,i])))
        coord_max.append(min(np.max(img1_corners_coordinates[:,i]), np.max(img2_corners_coordinates[:,i])))
        
    
    img1_crop_index = coord_to_index(img1_affine, coord_min, coord_max)
    img2_crop_index = coord_to_index(img2_affine, coord_min, coord_max)
    img1_lr_crop_index = coord_to_index(img1_lr_affine, coord_min, coord_max)
    img2_lr_crop_index = coord_to_index(img2_lr_affine, coord_min, coord_max)
    
    
    return img1_crop_index, img2_crop_index, img1_lr_crop_index, img2_lr_crop_index

def coord_to_index(affine, coord_min, coord_max):
    """
    Convert coordinates to index.

    Parameters
    ----------
    affine : array_like
        Affine.
    coord_min : list
        Minimum coordinates.
    coord_max : list
        Maximum coordinates.

    Returns
    -------
    list
        Crop indexes.
    """
    img_crop = []
    rev_affine = np.eye(4)
    rev_affine[:3,:3] = np.linalg.inv(affine[:3, :3])
    rev_affine[:3, 3] = rev_affine[:3,:3] @ (-affine[:3, 3])
    
    min_index = nib.affines.apply_affine(rev_affine, np.array(coord_min))
    max_index = nib.affines.apply_affine(rev_affine, np.array(coord_max))
    img_crop = [min_index, max_index]
    
    return np.maximum(np.array(img_crop, dtype=int), 0)

def crop_images(image1: nibabel.Nifti1Image, image2: nibabel.Nifti1Image,
                image1_lr: nibabel.Nifti1Image, image2_lr: nibabel.Nifti1Image):
    """
    Crop images to the largest shared field of view.

    Parameters
    ----------
    image1 : nibabel.Nifti1Image
        Contrast 1.
    image2 : nibabel.Nifti1Image
        Contrast 2.
    image1_lr : nibabel.Nifti1Image
        Contrast 1 low resolution.
    image2_lr : nibabel.Nifti1Image
        Contrast 2 low resolution.

    Returns
    -------
    tuple
        Cropped images.
    """
    img1_crop, img2_crop, img1_lr_crop, img2_lr_crop = get_crop_indexes(image1, image2, image1_lr, image2_lr)
    
    def slice_image(image, crop):
        cropped_image = image.slicer[crop[0,0]:crop[1,0],
                                     crop[0,1]:crop[1,1],
                                     crop[0,2]:crop[1,2]]
        return cropped_image
    
    return slice_image(image1, img1_crop), slice_image(image2, img2_crop), slice_image(image1_lr, img1_lr_crop), slice_image(image2_lr, img2_lr_crop)
    
def get_image_coordinate_grid_nib(image: nibabel.Nifti1Image, slice=False):
    """
    Get information anout the image coordinate grid.

    Parameters
    ----------
    image : nibabel.Nifti1Image
        Image.

    Returns
    -------
    dict
        Image coordinate grid information.
    """
    img_header = image.header
    img_data = image.get_fdata()
    img_affine = image.affine
    (x, y, z) = image.shape

    label = []
    coordinates = []

    for i in tqdm(range(x)):
        for j in range(y):
            for k in range(z):
                coordinates.append(nib.affines.apply_affine(img_affine, np.array(([i, j, k]))))
                label.append(img_data[i, j, k])

    # convert to numpy array
    coordinates_arr = np.array(coordinates, dtype=np.float32)
    label_arr = np.array(label, dtype=np.float32)

    def min_max_scale(X, s_min, s_max):
        x_min, x_max = X.min(), X.max()
        return (X - x_min)/(x_max - x_min)*(s_max - s_min) + s_min

    coordinates_arr_norm = min_max_scale(X=coordinates_arr, s_min=-1, s_max=1)

    scaler = MinMaxScaler()

    label_arr_norm = scaler.fit_transform(label_arr.reshape(-1, 1))

    if slice:

        coordinates_arr = coordinates_arr.reshape(x,y,z,3)
        label_arr = label_arr.reshape(x,y,z,1)

        coordinates_arr = coordinates_arr[:,:,::3,:].reshape(-1,3)
        label_arr = label_arr[:,:,::3,:].reshape(-1,1)
            
        coordinates_arr_norm = coordinates_arr_norm.reshape(x,y,z,3)
        label_arr_norm = label_arr_norm.reshape(x,y,z,1)

        coordinates_arr_norm = coordinates_arr_norm[:,:,::3,:].reshape(-1,3)
        label_arr_norm = label_arr_norm[:,:,::3,:].reshape(-1,1)
            
    # normalize intensities and coordinates
    # image_grid_norm = torch.tensor(image_grid, dtype=torch.float32)
    # image_data_norm = torch.tensor(label_arr_norm, dtype=torch.float32).view(-1,1)

    x_min, y_min, z_min = nib.affines.apply_affine(img_affine, np.array(([0, 0, 0])))
    x_max, y_max, z_max = nib.affines.apply_affine(img_affine, np.array(([x, y, z])))

    boundaries = dict()
    boundaries['xmin'] = x_min
    boundaries['ymin'] = y_min
    boundaries['zmin'] = z_min
    boundaries['xmax'] = x_max
    boundaries['ymax'] = y_max
    boundaries['zmax'] = z_max

    image_dict = {
        'boundaries': boundaries,
        'affine': torch.tensor(img_affine),
        'origin': torch.tensor(np.array([0])),
        'spacing': torch.tensor(np.array(img_header["pixdim"][1:4])),
        'dim': torch.tensor(np.array([x, y, z])),
        'intensity': torch.tensor(label_arr, dtype=torch.float32).view(-1, 1),
        'intensity_norm': torch.tensor(label_arr_norm, dtype=torch.float32).view(-1, 1),
        'coordinates': torch.tensor(coordinates_arr, dtype=torch.float32),
        'coordinates_norm': torch.tensor(coordinates_arr_norm, dtype=torch.float32),
        'image_data' : img_data,
    }
    return image_dict

def calculate_sobel_filter(image: nibabel.Nifti1Image):
    img_data = image.get_fdata()

    sobelx = ndimage.sobel(img_data, axis=0)
    sobely = ndimage.sobel(img_data, axis=1)
    sobelz = ndimage.sobel(img_data, axis=2)

    sobel = np.sqrt(sobelx**2 + sobely**2, sobelz**2)

    scaler = MinMaxScaler()
    sobel = scaler.fit_transform(sobel.reshape(-1, 1)).reshape(img_data.shape)

    return sobel


def calculate_sobel_median_filter(image: nibabel.Nifti1Image, median_filter_size=(1,1,1)):
    img_data = image.get_fdata()

    sobelx = ndimage.sobel(img_data, axis=0)
    sobely = ndimage.sobel(img_data, axis=1)
    sobelz = ndimage.sobel(img_data, axis=2)

    sobel = np.sqrt(sobelx**2 + sobely**2, sobelz**2)

    sobel = ndimage.median_filter(sobel, size=median_filter_size)

    scaler = MinMaxScaler()
    sobel = scaler.fit_transform(sobel.reshape(-1, 1)).reshape(img_data.shape)

    return sobel


def calculate_laplacian(image: nibabel.Nifti1Image):

    img_data = image.get_fdata()
    # Apply a Gaussian filter to smooth the image
    img_smooth = filters.gaussian(img_data, sigma=1)
    # Calculate the Laplacian of the smoothed image using the filters.laplace function

    laplacian = np.abs(filters.laplace(img_smooth))
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(laplacian.reshape(-1, 1)).reshape(img_data.shape)

    return scaled



