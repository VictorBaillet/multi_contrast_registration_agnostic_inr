import wandb

from models.parallel_registration.parallel_registration import parallel_registration_irn
from models.serial_registration.serial_registration import serial_registration_irn

def update_config(config_dict, config):
    for key in config_dict:
        if key in config:
            config_dict[key] = config[key]
        elif isinstance(config_dict[key], dict):
            config_dict[key] = update_config(config_dict[key], config)
    
    return config_dict

def compute_train_function(model_config_dict):
    def train():
        config_dict = model_config_dict
        with wandb.init(config=None):
            # If called by wandb.agent, as below,
            # this config will be set by Sweep Controller
            config = wandb.config
            config_dict = update_config(config_dict, config)

            model_config = config_dict['MODEL']
            dataset_config = config_dict['DATASET']

            net = parallel_registration_irn(model_config, verbose=False)
            net.logging = True

            net.fit(data_path=dataset_config['PATH'], 
                    contrast_1=dataset_config['LR_CONTRAST1'], 
                    contrast_2=dataset_config['LR_CONTRAST2'], 
                    dataset_name=dataset_config['DATASET_NAME'])

            wandb.log({'loss' : net.get_score()['loss']})
            
    return train