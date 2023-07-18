import wandb
import yaml
import os
import sys

sys.path.append(os.getcwd())
from utils.config.config import parse_args
from sweep.sweep_utils import compute_train_function


def main():
    args = parse_args()
    config_path = os.path.join('sweep/configs', args.config)
    model_config_path = os.path.join(config_path, 'model.yaml')
    sweep_config_path = os.path.join(config_path, 'sweep.yaml')
    
    with open(model_config_path) as f:
        model_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        
    with open(sweep_config_path) as f:
        sweep_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        
    project_name = 'sweep_' + model_config_dict['WANDB']['PROJECT_NAME']
        
    sweep_id = wandb.sweep(sweep_config_dict, project=project_name)
    
    train = compute_train_function(model_config_dict)
    
    wandb.agent(sweep_id, train, count=30)
    
if __name__=='__main__':
    main()
