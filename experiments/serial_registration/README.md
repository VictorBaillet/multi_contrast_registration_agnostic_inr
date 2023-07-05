# Serial registration model

Registration and image reconstruction don't share layers in the model. Some layers are optimized according to both reconstruction and registration losses simultaneously 
but the tasks are done by distinct layers.
Given coordinates (x, y, z) the model outputs the intensity of the corresponding voxel for the two registered contrasts. The registration is done upstream 
of the recontruction model.

## Model structure

The model consists of two sub-models. One for registration that takes coordinates of a point from the moving image and output the coordinates of 
the corresponding point in the fixed image. The reconstruction model takes coordinates of a point in the fixed image space and output the intensity of the voxel
for the two registered contrasts.

![Serial registration model](https://github.com/VictorBaillet/multi_contrast_registration_agnostic_inr/assets/105466709/705dea42-90d7-4db4-b555-2723f7307431)

The first model is only used on points from the moving image space, coordinates of points from the fixed image space can be directly given to the reconstruction model.

## Losses

The losses are similar to the parallel registration strategy. However the training process is different, 
the registration model is only used with data points from the moving image :

![Fixed image data forward pass](https://github.com/VictorBaillet/multi_contrast_registration_agnostic_inr/assets/105466709/e6434517-5e42-4e31-8530-f69c0a787248)

![Moving image data forward pass](https://github.com/VictorBaillet/multi_contrast_registration_agnostic_inr/assets/105466709/70f3163b-8ceb-487a-913f-5c04278da513)

## Config

In the config.yaml files you can change, among others, the parameters of the model and the weight of each loss.

## Related google slide

[2023-07-05 Progress report] (https://docs.google.com/presentation/d/14BUUlKPjN2aiMHQe6H01RsE6wDLntVs9Boq5mgTMVAI/edit#slide=id.p)
