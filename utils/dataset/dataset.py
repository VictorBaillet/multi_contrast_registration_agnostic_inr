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
    subject_id : str
        Subject ID.
    contrast1_LR_str : str
        Contrast 1 LR string.
    contrast2_LR_str : str
        Contrast 2 LR string.
    """

    def __init__(self, image_dir, name, subject_id, contrast1_LR_str, contrast2_LR_str):
        super(MultiModalDataset, self).__init__(image_dir)
        self.dataset_name = name
        self.subject_id = subject_id
        self.contrast1_LR_str = contrast1_LR_str
        self.contrast2_LR_str = contrast2_LR_str
        self.contrast1_LR_mask_str = contrast1_LR_str.replace("LR", "mask_LR")
        self.contrast2_LR_mask_str = contrast2_LR_str.replace("LR", "mask_LR")
        self.contrast1_GT_str = contrast1_LR_str.replace("_LR", "")
        self.contrast2_GT_str = contrast2_LR_str.replace("_LR", "")
        self.contrast1_GT_mask_str = "bodymask"#"brainmask"
        self.contrast2_GT_mask_str = "bodymask"#"brainmask"
        
        #Cross correlation/Mutual information between gt images and lr images -- Computed in the config of each project.
        self.gt_similarity = 0
        self.lr_similarity = 0
        
        

        self.dataset_name = (
            f'{self.dataset_name}_'
            f'{self.subject_id}_'
            f'{self.contrast1_LR_str}_{self.contrast1_GT_str}_'
            f'{self.contrast2_LR_str}_{self.contrast2_GT_str}_'
            f'{self.contrast1_LR_mask_str}_{self.contrast2_LR_mask_str}_'
            f'{self.contrast1_GT_mask_str}_{self.contrast2_GT_mask_str}_'
            f'.pt'
        )

        files = sorted(list(Path(self.image_dir).rglob('*.nii.gz'))) 
        files = [str(x) for x in files]


        # only keep NIFTIs that follow specific subject 
        files = [k for k in files if self.subject_id in k]

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
        
        #gt_contrast2_dict = get_image_coordinate_grid_nib(nib.load(str(self.gt_contrast2)))
        #self.gt_constrast2_img_data = gt_contrast2_dict["image_data"]
        
        self.dataset_path = os.path.join(os.path.join(os.getcwd(), "projects/preprocessed_data"), self.dataset_name)
        
        if os.path.isfile(self.dataset_path):
            print("Dataset available : ", self.dataset_name)
            dataset = torch.load(self.dataset_path)
            self.data = dataset["data"]
            self.data_contrast1 = dataset["data_contrast1"]
            self.data_contrast2 = dataset["data_contrast2"]
            self.labels_contrast1 = dataset["labels_contrast1"]
            self.labels_contrast2 = dataset["labels_contrast2"]
            self.dim_lr_contrast1 = dataset["dim_lr_contrast1"]
            self.dim_lr_contrast2 = dataset["dim_lr_contrast2"]
            self.label = dataset["label"]
            self.mask = dataset["mask"]
            self.affine_contrast1 = dataset["affine_contrast1"]
            self.affine_contrast2 = dataset["affine_contrast2"]
            self.dim_contrast1 = dataset["dim_contrast1"]
            self.dim_contrast2 = dataset["dim_contrast2"]
            self.len = dataset["len"]
            self.gt_contrast1 = dataset["gt_contrast1"]
            self.gt_contrast2 = dataset["gt_contrast2"]
            self.gt_contrast1_mask = dataset["gt_contrast1_mask"]
            self.gt_contrast2_mask = dataset["gt_contrast2_mask"]
            self.coordinates_contrast1 = dataset["coordinates_contrast1"]
            self.coordinates_contrast2 = dataset["coordinates_contrast2"]
            print("skipping preprocessing.")

        else:
            self.len = 0
            self.data = []
            self.label = []
            self._process()

    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> Tuple[dict, dict]:
        data = self.data[idx]
        label = self.label[idx]
        #mask = self.mask[idx]
        return data, label #, mask

    def get_intensities(self):
        return self.label
    
    def get_contrast1_intensities(self):
        return self.labels_contrast1
    
    def get_contrast2_intensities(self):
        return self.labels_contrast2
    
    def get_mask(self):
        return self.mask

    def get_contrast1_coordinates(self):
        return self.coordinates_contrast1
    
    def get_contrast2_coordinates(self):
        return self.coordinates_contrast2

    def get_contrast1_affine(self):
        return self.affine_contrast1
    
    def get_contrast2_affine(self):
        return self.affine_contrast2
    
    def get_contrast1_dim(self):
        return self.dim_contrast1
    
    def get_contrast2_dim(self):
        return self.dim_contrast2
    
    def get_contrast1_lr_dim(self):
        return self.dim_lr_contrast1
    
    def get_contrast2_lr_dim(self):
        return self.dim_lr_contrast2
    
    def get_contrast2_gt(self):
        return self.gt_contrast2
        
    def get_contrast1_gt(self):
        return self.gt_contrast1
        
    def get_contrast2_gt_mask(self):
        return self.gt_contrast2_mask
    
    def get_contrast1_gt_mask(self):
        return self.gt_contrast1_mask
    
    def get_contrast2_gt_image_data(self):
        return self.gt_constrast2_img_data
    
    def get_data(self):
        return self.data
    
    def get_contrast1_data(self):
        return self.data_contrast1
    
    def get_contrast2_data(self):
        return self.data_contrast2
    
    def get_gt_similarity(self):
        return self.gt_similarity
    
    def get_lr_similarity(self):
        return self.lr_similarity

    def _process(self):

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
        
        gt_contrast1_cropped, gt_contrast2_cropped, lr_contrast1_cropped, lr_contrast2_cropped = crop_images(gt_contrast1_image, gt_contrast2_image,
                                                                                                             lr_contrast1_image, lr_contrast2_image)
        
        contrast1_dict = get_image_coordinate_grid_nib(lr_contrast1_cropped)
        contrast2_dict = get_image_coordinate_grid_nib(lr_contrast2_cropped)
        
        contrast1_mask_dict = get_image_coordinate_grid_nib(nib.load(str(self.lr_contrast1_mask)))
        contrast2_mask_dict = get_image_coordinate_grid_nib(nib.load(str(self.lr_contrast2_mask)))

        data_contrast2 = contrast2_dict["coordinates"]
        data_contrast1 = contrast1_dict["coordinates"]
                
        min1, max1 = data_contrast1.min(), data_contrast1.max()
        min2, max2 = data_contrast2.min(), data_contrast2.max()

        min_c, max_c = np.min(np.array([min1, min2])), np.max(np.array([max1, max2]))

        data_contrast1 = norm_grid(data_contrast1, xmin=min_c, xmax=max_c)
        data_contrast2 = norm_grid(data_contrast2, xmin=min_c, xmax=max_c)
                

        self.data_contrast1 = data_contrast1
        self.data_contrast2 = data_contrast2

        labels_contrast2 = contrast2_dict["intensity_norm"]
        labels_contrast1 = contrast1_dict["intensity_norm"]
        
        self.labels_contrast1 = labels_contrast1
        self.labels_contrast2 = labels_contrast2
        
        self.dim_lr_contrast1 = contrast1_dict["dim"]
        self.dim_lr_contrast2 = contrast2_dict["dim"]
        
        
        mask_contrast2 = contrast2_mask_dict["intensity_norm"].bool()
        mask_contrast1 = contrast1_mask_dict["intensity_norm"].bool()
        
        labels_contrast1_stack = torch.cat((labels_contrast1, torch.ones(labels_contrast1.shape)*-1), dim=1)
        labels_contrast2_stack = torch.cat((torch.ones(labels_contrast2.shape)*-1, labels_contrast2), dim=1)
        
        # assemble the data and labels
        self.data = torch.cat((data_contrast1, data_contrast2), dim=0)
        self.label = torch.cat((labels_contrast1_stack, labels_contrast2_stack), dim=0)
        self.mask = torch.cat((mask_contrast1, mask_contrast2), dim=0)
        self.len = len(self.label)

        # store the GT images to compute SSIM and other metrics!
        gt_contrast1_dict = get_image_coordinate_grid_nib(gt_contrast1_cropped)
        gt_contrast2_dict = get_image_coordinate_grid_nib(gt_contrast2_cropped)

        self.gt_contrast2 = gt_contrast2_dict["intensity_norm"]
        self.gt_contrast1 = gt_contrast1_dict["intensity_norm"]
        
        self.gt_constrast2_img_data = gt_contrast2_dict["image_data"]

        self.gt_contrast2_mask = torch.tensor(nib.load(self.gt_contrast2_mask).get_fdata()).bool()
        self.gt_contrast1_mask = torch.tensor(nib.load(self.gt_contrast1_mask).get_fdata()).bool()

        self.coordinates_contrast1 = gt_contrast1_dict["coordinates"]
        self.coordinates_contrast2 = gt_contrast2_dict["coordinates"]
        
        self.coordinates_contrast1 = norm_grid(self.coordinates_contrast1, xmin=min_c, xmax=max_c)
        self.coordinates_contrast2 = norm_grid(self.coordinates_contrast2, xmin=min_c, xmax=max_c)
                
        self.gt_contrast2 = self.gt_contrast2
        self.gt_contrast1 = self.gt_contrast1
        
        self.gt_contrast1_mask = self.gt_contrast1_mask
        self.gt_contrast2_mask = self.gt_contrast2_mask
        
        self.coordinates_contrast1 = self.coordinates_contrast1
        self.coordinates_contrast2 = self.coordinates_contrast2

        self.affine_contrast1 = gt_contrast1_dict["affine"]
        self.affine_contrast2 = gt_contrast2_dict["affine"]
        self.dim_contrast1 = gt_contrast1_dict["dim"]
        self.dim_contrast2 = gt_contrast2_dict["dim"]

        # store to avoid preprocessing
        dataset = {
            'len': self.len,
            'data': self.data,
            'data_contrast1': data_contrast1,
            'data_contrast2': data_contrast2,
            'labels_contrast2': labels_contrast2,
            'labels_contrast1': labels_contrast1,
            'dim_lr_contrast1': self.dim_lr_contrast1,
            'dim_lr_contrast2': self.dim_lr_contrast2,
            'mask': self.mask,
            'label': self.label,
            'affine_contrast1': self.affine_contrast1,
            'affine_contrast2': self.affine_contrast2,
            'gt_contrast1': self.gt_contrast1,
            'gt_contrast2': self.gt_contrast2,
            'gt_contrast1_mask': self.gt_contrast1_mask,
            'gt_contrast2_mask': self.gt_contrast2_mask,
            'dim_contrast1': self.dim_contrast1,
            'dim_contrast2': self.dim_contrast2,
            'coordinates_contrast1': self.coordinates_contrast1,
            'coordinates_contrast2': self.coordinates_contrast2,
        }
        if not os.path.exists(os.path.join(os.path.join(os.getcwd(), "projects/preprocessed_data"))):
            os.makedirs(os.path.join(os.path.join(os.getcwd(), "projects/preprocessed_data")))
        torch.save(dataset, self.dataset_path)

class InferDataset(Dataset):
    def __init__(self, grid):
        super(InferDataset, self,).__init__()
        self.grid = grid

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, idx):
        data = self.grid[idx]
        return data

if __name__ == '__main__':

    dataset = MultiModalDataset(
                image_dir='miccai',
                name='miccai_dataset',          
                )

    print("Passed.")
    
