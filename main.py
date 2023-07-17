import os 
import sys
import argparse
import pathlib
from utils.config.config import parse_args, process_config 
from models.parallel_registration.parallel_registration import parallel_registration_irn
from models.serial_registration.serial_registration import serial_registration_irn

"""
Rajouter methodes : .save et .load
"""


def main():
    args = parse_args()
    experiment_name = args.experiment_name
    project_folder = os.path.join('models', experiment_name)
    config_folder = os.path.join(project_folder, 'configs')
    args.config = os.path.join(config_folder, args.config)
    
    config_dict = process_config(args)
    
    if experiment_name == "parallel_registration":
        net = parallel_registration_irn(config_dict)
    if experiment_name == "serial_registration":
        net = serial_registration_irn(config_dict)
        
    net.fit()
    
    save_folder = f'runs/{config_dict["SETTINGS"]["PROJECT_NAME"]}/'
    im_dir = save_folder + 'images'
    weights_dir = save_folder + 'weights'
    pathlib.Path(weights_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(im_dir).mkdir(parents=True, exist_ok=True)
    path_contrast_1 = os.path.join(im_dir, 'contrast_1.nii.gz')
    path_contrast_2 = os.path.join(im_dir, 'contrast_2.nii.gz')
    
    net.save_images(contrast=1, path=path_contrast_1)
    net.save_images(contrast=2, path=path_contrast_2)
    net.save_model(os.path.join(weights_dir, 'model.pt'))
    
if __name__ == '__main__':
    main()