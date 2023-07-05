# Parallel registration model

Registration and image reconstruction share the first layers of the model. These weights are optimized according to the reconstruction and registration losses simultaneously.
Given coordinates (x, y, z) the model outputs the intensity of the corresponding voxel for the two registered contrasts and a registration vector that maps the point to its unregistered coordinates. 

## Model structure

The model is a MLP, the first layers are shared and there is a specialized head for each task :

![The three heads model](https://github.com/VictorBaillet/multi_contrast_registration_agnostic_inr/assets/105466709/8cad3c22-50bb-4c7c-a2de-54ca0fd2fbde)

The MLP can be either a ReLu-MLP with Fourier features or a SIREN.

## Losses 

In this example we register a T2w image to a T1w image :  
If we note φ the registration function (in this case the inverse of the registration moving to fixed image) and INR T1w (respectively INR T2w) the model's T1w intensity output (respectively T2w), we have :  

![Losses relationship](https://github.com/VictorBaillet/multi_contrast_registration_agnostic_inr/assets/105466709/69df6196-b7e0-4229-9816-858ffb63035e)

The final loss is equal to $\alpha L2(INR \ \ \ T1w, T1w) + \beta MI(INR \ \ \ T1w, INR \ \ \ T2w) + \gamma L2(φ(INR \ \ \ T2w), T2w)$.

## Configs

In the config.yaml files you can change, among others, the parameters of the model and the weight of each loss.

## Related google slides

[2023-05-04 progress report](https://docs.google.com/presentation/d/1Mf8d2fuZPoIiBFSIMw2ye3mM0Mg1gzOGNzUxOgzWNEE/edit?usp=drive_link)
