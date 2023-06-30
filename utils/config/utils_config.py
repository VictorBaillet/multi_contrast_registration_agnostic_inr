import torch.nn as nn
import torch
import numpy as np

# from official FF repository
class input_mapping(nn.Module):
    def __init__(self, B=None, factor=1.0):
        super(input_mapping, self).__init__()
        self.B = factor * B
    
    def forward(self, x):

        x_proj = (2. * np.pi * x) @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
def get_string(my_dict):
    return '_'.join([str(value) for value in my_dict.values()])