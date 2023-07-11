import os 
import sys
import argparse
from utils.config.config import parse_args, process_config 
from experiments.parallel_registration.parallel_registration import parallel_registration_irn
from experiments.serial_registration.serial_registration import serial_registration_irn

"""
Rajouter methodes : .save et .load
"""


def main():
    args = parse_args()
    experiment_name = args.experiment_name
    project_folder = os.path.join('experiments', experiment_name)
    config_folder = os.path.join(project_folder, 'configs')
    args.config = os.path.join(config_folder, args.config)
    
    config_dict = process_config(args)
    
    if experiment_name == "parallel_registration":
        net = parallel_registration_irn(config_dict)
    if experiment_name == "serial_registration":
        net = serial_registration_irn(config_dict)
        
    net.fit()
    
if __name__ == '__main__':
    main()