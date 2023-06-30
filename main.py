import os 
import sys
import argparse


def main():
    import os

    parser = argparse.ArgumentParser(description='script description')
    parser.add_argument('--experiment_name', type=str, help='Experiment name')
    parser.add_argument('--config', type=str, help='Config file name')
    parser.add_argument('--logging', action='store_true')
    
    args = parser.parse_args()
    
    project_folder = os.path.join('experiments', args.experiment_name)
    config_folder = os.path.join(project_folder, 'configs')
    config_file = os.path.join(config_folder, args.config)

    for filename in os.listdir(project_folder):
        if filename.endswith('.py') and filename.startswith('main'):
            if args.logging:
                os.system(f'python {os.path.join(project_folder, filename)} --config {config_file} --logging')
            else:
                os.system(f'python {os.path.join(project_folder, filename)} --config {config_file}')



if __name__ == '__main__':
    main()