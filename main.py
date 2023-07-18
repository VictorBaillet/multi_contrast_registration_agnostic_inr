import os 
import sys
import argparse
import pathlib
from utils.config.config import parse_args, process_config 
from models.parallel_registration.parallel_registration import parallel_registration_irn
from models.serial_registration.serial_registration import serial_registration_irn


def main():
    args = parse_args()
    config_dict = process_config(args)
        
    model_config = config_dict['MODEL']
    dataset_config = config_dict['DATASET']
    wandb_config = config_dict['WANDB']
    
    if args.experiment_name == "parallel_registration":
        net = parallel_registration_irn(model_config)
    if args.experiment_name == "serial_registration":
        net = serial_registration_irn(model_config)
    
    
    net.config_wandb(logging=wandb_config['LOGGING'], project_name=wandb_config['PROJECT_NAME'])
    net.fit(data_path=dataset_config['PATH'], 
            contrast_1=dataset_config['LR_CONTRAST1'], 
            contrast_2=dataset_config['LR_CONTRAST2'], 
            dataset_name=dataset_config['DATASET_NAME'])
    
    
    contrast_1_path, contrast_2_path, model_path = compute_paths(config_dict)
    
    net.save_images(contrast=1, path=contrast_1_path)
    net.save_images(contrast=2, path=contrast_2_path)
    net.save_model(model_path)
    
def compute_paths(config_dict):
    save_folder = f'runs/{config_dict["DATASET"]["DATASET_NAME"]}/'
    im_dir = save_folder + 'images'
    weights_dir = save_folder + 'weights'
    pathlib.Path(weights_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(im_dir).mkdir(parents=True, exist_ok=True)
    contrast_1_path = os.path.join(im_dir, 'contrast_1.nii.gz')
    contrast_2_path = os.path.join(im_dir, 'contrast_2.nii.gz')
    model_path = os.path.join(weights_dir, 'model.pt')
    
    return contrast_1_path, contrast_2_path, model_path
    
if __name__ == '__main__':
    main()