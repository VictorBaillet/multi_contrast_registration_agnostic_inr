import os
import torch
import pathlib
import numpy as np
import wandb

from utils.config.config import create_losses, process_config, create_input_mapper 
from experiments.parallel_registration.experiment_utils.config.utils_config import create_model, create_datasets, compute_dataset_artifacts


def training_config(args):
    # Init arguments 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

    config, config_dict = process_config(args)

    # Logging run
    if args.logging:
        wandb.login()
        wandb.init(config=config_dict, project=config.SETTINGS.PROJECT_NAME)

    # Make directory for models
    weight_dir = f'runs/{config.SETTINGS.PROJECT_NAME}_weights'
    image_dir = f'runs/{config.SETTINGS.PROJECT_NAME}_images'

    pathlib.Path(weight_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(image_dir).mkdir(parents=True, exist_ok=True)

    # Seeding
    torch.manual_seed(config.TRAINING.SEED)
    np.random.seed(config.TRAINING.SEED)

    # Training device
    device = f'cuda:{config.SETTINGS.GPU_DEVICE}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Model configuration
    model, model_name = create_model(config, config_dict, device)
    input_mapper = create_input_mapper(config, device)

    print(f'Number of MLP parameters {sum(p.numel() for p in model.parameters())}')

    # Losses configuration
    lpips_loss, criterion, mi_criterion, cc_criterion, model_name = create_losses(config, config_dict, model_name, device)  

    # optimizer
    if config.TRAINING.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAINING.LR)#, weight_decay=5e-5)
        """
        registration_layers = [param for name, param in model.named_parameters() if 'registration' in name]
        other_layers = [param for name, param in model.named_parameters() if not('registration' in name)]
        optimizer = torch.optim.Adam([{'params': other_layers},
                                     {'params': registration_layers, 'lr': 1e-8}],
                                    lr =config.TRAINING.LR)
        """
        model_name = f'{model_name}_{config.TRAINING.OPTIM}_{config.TRAINING.LR}_'    
    else:
        raise ValueError('Optim not defined!')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max= config.TRAINING.EPOCHS)

    # Load Data
    dataset, train_dataloader, infer_dataloader, threshold = create_datasets(config)
    
    fixed_image, rev_affine, min_coords, max_coords, difference_center_of_mass, format_im = compute_dataset_artifacts(dataset, device)

    training_args = {'config':config,
                     'args':args,
                     'device':device,
                     'input_mapper':input_mapper,
                     'fixed_image':fixed_image,
                     'lpips_loss':lpips_loss,
                     'criterion':criterion,
                     'mi_criterion':mi_criterion,
                     'cc_criterion':cc_criterion,
                     'min_coords':min_coords,
                     'max_coords':max_coords,
                     'rev_affine':rev_affine,
                     'difference_center_of_mass':difference_center_of_mass,
                     'format_im':format_im}
    
    model_config = (model, model_name, optimizer, scheduler)
    data_config = (dataset, train_dataloader, infer_dataloader, input_mapper)
    directories = (weight_dir, image_dir)
    
    return config, model_config, data_config, directories, training_args

