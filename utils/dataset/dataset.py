import os
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset

from utils.dataset.dataset_utils import get_image_coordinate_grid_nib, norm_grid, crop_images

class _BaseDataset(Dataset):
    """Base dataset class"""

    def __init__(self, image_dir):
        super(_BaseDataset, self).__init__()
        self.image_dir = image_dir
        
        assert os.path.exists(image_dir), f"Image Directory does not exist: {image_dir}!"

    def __getitem__(self, index):
        """ Load data and pre-process """
        raise NotImplementedError

    def __len__(self) -> int:
        r"""Returns the number of coordinates stored in the dataset."""
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

class MultiModalDataset(_BaseDataset):
    """
    Dataset of contrast 1 and contrast 2 image sequence of the same patient.
    These could be e.g. T1w and T2w brain images,
    an T1w and T2w spine image, etc.
    Parameters
    ----------
    image_dir : str
        Image directory.
    name : str
        Dataset name.
    contrast1_LR_str : str
        Contrast 1 LR string.
    contrast2_LR_str : str
        Contrast 2 LR string.
    """

    def __init__(self, image_dir, name, contrast1_LR_str, contrast2_LR_str, verbose=True):
        super(MultiModalDataset, self).__init__(image_dir)
        self.dataset_name = name
        self.contrast1_LR_str = contrast1_LR_str
        self.contrast2_LR_str = contrast2_LR_str
        self.contrast1_LR_mask_str = contrast1_LR_str.replace("LR", "mask_LR")
        self.contrast2_LR_mask_str = contrast2_LR_str.replace("LR", "mask_LR")
        self.contrast1_GT_str = contrast1_LR_str.replace("_LR", "")
        self.contrast2_GT_str = contrast2_LR_str.replace("_LR", "")
        self.contrast1_GT_mask_str = "bodymask"
        self.contrast2_GT_mask_str = "bodymask"
        self.verbose = verbose
        

        self.dataset_name = (
            f'{self.dataset_name}_'
            f'{self.contrast1_LR_str}_{self.contrast1_GT_str}_'
            f'{self.contrast2_LR_str}_{self.contrast2_GT_str}_'
            f'{self.contrast1_LR_mask_str}_{self.contrast2_LR_mask_str}_'
            f'{self.contrast1_GT_mask_str}_{self.contrast2_GT_mask_str}_'
            f'.pt'
        )

        files = sorted(list(Path(self.image_dir).rglob('*.nii.gz'))) 
        files = [str(x) for x in files]


        # flair3 and flair3d_LR or t1 and t1_LR
        gt_contrast1 = [x for x in files if self.contrast1_GT_str in x and self.contrast1_LR_str not in x and 'mask' not in x][0]
        gt_contrast2 = [x for x in files if self.contrast2_GT_str in x and self.contrast2_LR_str not in x and 'mask' not in x][0]

        lr_contrast1 = [x for x in files if self.contrast1_LR_str in x and 'mask' not in x][0]
        lr_contrast2 = [x for x in files if self.contrast2_LR_str in x and 'mask' not in x][0]

        lr_contrast1_mask = [x for x in files if self.contrast1_LR_mask_str in x and 'mask' in x][0]
        lr_contrast2_mask = [x for x in files if self.contrast2_LR_mask_str in x and 'mask' in x][0]

        gt_contrast1_mask = [x for x in files if self.contrast1_GT_mask_str in x and 'mask' in x][0]
        gt_contrast2_mask = [x for x in files if self.contrast2_GT_mask_str in x and 'mask' in x][0]

        self.lr_contrast1 = lr_contrast1
        self.lr_contrast2 = lr_contrast2
        self.lr_contrast1_mask = lr_contrast1_mask
        self.lr_contrast2_mask = lr_contrast2_mask
        self.gt_contrast1 = gt_contrast1
        self.gt_contrast2 = gt_contrast2
        self.gt_contrast1_mask = gt_contrast1_mask
        self.gt_contrast2_mask = gt_contrast2_mask
        
        self.dataset_path = os.path.join(os.path.join(os.getcwd(), "data/preprocessed_data"), self.dataset_name)
        
        if os.path.isfile(self.dataset_path):
            if verbose:
                print("Dataset available : ", self.dataset_name)
                print("skipping preprocessing.")
            dataset = torch.load(self.dataset_path)
            self.data = dataset["data"]
            self.labels = dataset["labels"]
            self.mask = dataset["mask"]
            self.len = dataset["len"]
            self.contrasts_data = dataset["contrasts_data"]

        else:
            self.len = 0
            self.data = []
            self.labels = []
            self._process()

    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> Tuple[dict, dict]:
        data = self.data[idx]
        label = self.labels[idx]
        mask = self.mask[idx]
        return data, label, mask

    def _process(self):
        if self.verbose:
            print(f"Using {self.lr_contrast1} as contrast1.")
            print(f"Using {self.lr_contrast2} as contrast2.")

            print(f"Using {self.lr_contrast1_mask} as contrast1 mask.")
            print(f"Using {self.lr_contrast2_mask} as contrast2 mask.")

            print(f"Using {self.gt_contrast1} as gt contrast1.")
            print(f"Using {self.gt_contrast2} as gt contrast2.")

            print(f"Using {self.gt_contrast1_mask} as gt contrast1 mask.")
            print(f"Using {self.gt_contrast2_mask} as gt contrast2 mask.")
        
        lr_contrast1_image = nib.load(str(self.lr_contrast1))
        lr_contrast2_image = nib.load(str(self.lr_contrast2))
        gt_contrast1_image = nib.load(str(self.gt_contrast1))
        gt_contrast2_image = nib.load(str(self.gt_contrast2))
        
        lr_contrast1_mask = nib.load(str(self.lr_contrast1_mask))
        lr_contrast2_mask = nib.load(str(self.lr_contrast2_mask))
        gt_contrast1_mask = nib.load(str(self.gt_contrast1_mask))
        gt_contrast2_mask = nib.load(str(self.gt_contrast2_mask))
        
        gt_contrast1_cropped, gt_contrast2_cropped, lr_contrast1_cropped, lr_contrast2_cropped = crop_images(gt_contrast1_image, gt_contrast2_image,
                                                                                                             lr_contrast1_image, lr_contrast2_image)
        
        gt_contrast1_mask, gt_contrast2_mask, lr_contrast1_mask, lr_contrast2_mask = crop_images(gt_contrast1_mask, gt_contrast2_mask,
                                                                                                 lr_contrast1_mask, lr_contrast2_mask)
        
        self.lr_contrast1_dict = get_image_coordinate_grid_nib(lr_contrast1_cropped)
        self.lr_contrast2_dict = get_image_coordinate_grid_nib(lr_contrast2_cropped)
        
        self.lr_contrast1_dict["mask"] = get_image_coordinate_grid_nib(lr_contrast1_mask)["intensity"].bool()
        self.lr_contrast2_dict["mask"] = get_image_coordinate_grid_nib(lr_contrast2_mask)["intensity"].bool()

        data_contrast1, labels_contrast1 = self.lr_contrast1_dict["coordinates"], self.lr_contrast1_dict["intensity_norm"]
        data_contrast2, labels_contrast2 = self.lr_contrast2_dict["coordinates"], self.lr_contrast2_dict["intensity_norm"]
                
        min1, max1 = data_contrast1.min(), data_contrast1.max()
        min2, max2 = data_contrast2.min(), data_contrast2.max()

        min_c, max_c = np.min(np.array([min1, min2])), np.max(np.array([max1, max2]))

        data_contrast1 = norm_grid(data_contrast1, xmin=min_c, xmax=max_c)
        data_contrast2 = norm_grid(data_contrast2, xmin=min_c, xmax=max_c)
        
        self.lr_contrast1_dict["coordinates"] = data_contrast1
        self.lr_contrast2_dict["coordinates"] = data_contrast2
                                
        labels_contrast1_stack = torch.cat((labels_contrast1, torch.ones(labels_contrast1.shape)*-1), dim=1)
        labels_contrast2_stack = torch.cat((torch.ones(labels_contrast2.shape)*-1, labels_contrast2), dim=1)
        
        # assemble the data and labels
        self.data = torch.cat((data_contrast1, data_contrast2), dim=0)
        self.labels = torch.cat((labels_contrast1_stack, labels_contrast2_stack), dim=0)
        self.mask = torch.cat((self.lr_contrast1_dict["mask"], self.lr_contrast2_dict["mask"]), dim=0)
        self.len = len(self.labels)

        # store the GT images to compute SSIM and other metrics!
        self.gt_contrast1_dict = get_image_coordinate_grid_nib(gt_contrast1_cropped)
        self.gt_contrast2_dict = get_image_coordinate_grid_nib(gt_contrast2_cropped)

        self.coordinates_contrast1 = self.gt_contrast1_dict["coordinates"]
        self.coordinates_contrast2 = self.gt_contrast2_dict["coordinates"]
                
        self.gt_contrast1_dict['mask'] = torch.tensor(gt_contrast1_mask.get_fdata()).bool()
        self.gt_contrast2_dict['mask'] = torch.tensor(gt_contrast2_mask.get_fdata()).bool()
        
        self.gt_contrast1_dict["coordinates"] = norm_grid(self.coordinates_contrast1, xmin=min_c, xmax=max_c)
        self.gt_contrast2_dict["coordinates"] = norm_grid(self.coordinates_contrast2, xmin=min_c, xmax=max_c)
        
        self.contrasts_data = {'lr_contrast1': self.lr_contrast1_dict,
                               'lr_contrast2': self.lr_contrast2_dict,
                               'gt_contrast1': self.gt_contrast1_dict,
                               'gt_contrast2': self.gt_contrast2_dict}

        # store to avoid preprocessing
        dataset = {
            'len': self.len,
            'data': self.data,
            'mask': self.mask,
            'labels': self.labels,
            'contrasts_data': self.contrasts_data}
        
        if not os.path.exists(os.path.join(os.path.join(os.getcwd(), "data/preprocessed_data"))):
            os.makedirs(os.path.join(os.path.join(os.getcwd(), "data/preprocessed_data")))
        torch.save(dataset, self.dataset_path)
        
    def get_labels(self):
        return self.labels
    
    def get_mask(self):
        return self.mask
    
    def get_data(self):
        return self.data
    
    def get_intensities(self, contrast, resolution):
        image_name = f'{resolution}_contrast{contrast}'
        return self.contrasts_data[image_name]["intensity_norm"]

    def get_coordinates(self, contrast, resolution):
        image_name = f'{resolution}_contrast{contrast}'
        return self.contrasts_data[image_name]["coordinates"]

    def get_affine(self, contrast, resolution):
        image_name = f'{resolution}_contrast{contrast}'
        return self.contrasts_data[image_name]["affine"]
    
    def get_dim(self, contrast, resolution):
        image_name = f'{resolution}_contrast{contrast}'
        return self.contrasts_data[image_name]["dim"]

    def get_mask(self, contrast, resolution):
        image_name = f'{resolution}_contrast{contrast}'
        return self.contrasts_data[image_name]['mask']

    def get_image_data(self, contrast, resolution):
        image_name = f'{resolution}_contrast{contrast}'
        return self.contrasts_data[image_name]['img_data']
    
    def get_gt_similarity(self):
        return self.gt_similarity
    
    def get_lr_similarity(self):
        return self.lr_similarity

class InferDataset(Dataset):
    def __init__(self, grid):
        super(InferDataset, self,).__init__()
        self.grid = grid

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, idx):
        data = self.grid[idx]
        return data

    
