import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import numpy as np

def fc_block(in_size, out_size, dropout,*args, **kwargs):
    return nn.Sequential(
        nn.Linear(in_size, out_size, *args, **kwargs),
        nn.ReLU(),
        nn.Dropout(dropout),
    )

class MLPv1(nn.Module):
    def __init__(self, input_size=3, hidden_size=512, output_size=1, dropout=0, num_layers=5):
        super(MLPv1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.num_layers = num_layers
        fc_blocks = [fc_block(self.hidden_size,self.hidden_size, self.dropout) for i in range(self.num_layers)]
        self.fc_in = nn.Linear(self.input_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.output_size)
        self.fc = nn.Sequential(*fc_blocks)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        # flatten image
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc_in(x))
        x = self.dropout_layer(x)
        x = self.fc(x)
        # add output layer
        x = self.fc_out(x)
        return x
    

class MLPv2(nn.Module):
    def __init__(self, input_size=3, hidden_size=512, output_size=1, dropout=0, num_layers=5):
        super(MLPv2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.num_layers = num_layers
        fc_blocks = [fc_block(self.hidden_size,self.hidden_size, self.dropout) for i in range(self.num_layers)]
        self.fc_in = nn.Linear(self.input_size, self.hidden_size)
        self.fc_out1 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.fc_out2 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.out1 = nn.Linear(self.hidden_size//2, self.output_size//2)
        self.out2 = nn.Linear(self.hidden_size//2, self.output_size//2)
        self.fc = nn.Sequential(*fc_blocks)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        # flatten image
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc_in(x))
        x = self.dropout_layer(x)
        x = self.fc(x)
        # add output layer
        x1 = x
        x2 = x
        x1 = self.out1(F.relu(self.fc_out1(x1)))
        x2 = self.out2(F.relu(self.fc_out2(x2)))
        return torch.cat((x1,x2),dim=1)


class MLPv3(nn.Module):
    def __init__(self, input_size=3, hidden_size=512, output_size=1, dropout=0, num_layers=5, num_layers_head=3):
        super(MLPv3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_layers_head = num_layers_head

        fc_blocks = [fc_block(self.hidden_size,self.hidden_size, self.dropout) for i in range(self.num_layers)]
        self.fc_in = nn.Linear(self.input_size, self.hidden_size)
        self.fc_out1 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.fc_out2 = nn.Linear(self.hidden_size, self.hidden_size//2)
        fc_head1 = [fc_block(self.hidden_size//2,self.hidden_size//2, self.dropout) for i in range(self.num_layers_head)]
        fc_head2 = [fc_block(self.hidden_size//2,self.hidden_size//2, self.dropout) for i in range(self.num_layers_head)]    
        self.fc_head1 = nn.Sequential(*fc_head1)
        self.fc_head2 = nn.Sequential(*fc_head2)
        self.out1 = nn.Linear(self.hidden_size//2, self.output_size//2)
        self.out2 = nn.Linear(self.hidden_size//2, self.output_size//2)
        self.fc = nn.Sequential(*fc_blocks)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        # flatten image
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc_in(x))
        x = self.dropout_layer(x)
        x = self.fc(x)
        # split heads layer
        x1 = x
        x2 = x
        
        x1 = F.relu(self.fc_out1(x1))
        x2 = F.relu(self.fc_out2(x2))

        x1 = self.out1(self.fc_head1(x1))
        x2 = self.out2(self.fc_head1(x2))
        return torch.cat((x1,x2),dim=1)
    
class MLPregv1(nn.Module):
    def __init__(self, input_size=3, hidden_size=512, output_size=1, dropout=0, num_layers=5):
        super(MLPregv1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.num_layers = num_layers
        fc_blocks = [fc_block(self.hidden_size,self.hidden_size, self.dropout) for i in range(self.num_layers)]
        self.fc_in = nn.Linear(self.input_size, self.hidden_size)
        self.fc_contrast1 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.fc_contrast2 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.fc_registration = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.out_contrast1 = nn.Linear(self.hidden_size//2, self.output_size//2)
        self.out_contrast2 = nn.Linear(self.hidden_size//2, self.output_size//2)
        self.out_registration = nn.Linear(self.hidden_size//2, 3)
        self.out_x_registration = nn.Linear(1, 1)
        nn.init.zeros_(self.out_x_registration.weight)
        nn.init.zeros_(self.out_x_registration.bias)
        self.out_y_registration = nn.Linear(1, 1)
        nn.init.zeros_(self.out_y_registration.weight)
        nn.init.zeros_(self.out_y_registration.bias)
        self.out_z_registration = nn.Linear(1, 1)
        nn.init.zeros_(self.out_z_registration.weight)
        nn.init.zeros_(self.out_z_registration.bias)
        self.fc = nn.Sequential(*fc_blocks)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        # flatten image
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc_in(x))
        x = self.dropout_layer(x)
        x = self.fc(x)
        # add output layer
        x1 = x
        x2 = x
        x3 = x
        x1 = (self.out_contrast1(F.relu(self.fc_contrast1(x1))))
        x2 = (self.out_contrast2(F.relu(self.fc_contrast2(x2))))
        x3 = self.out_registration(F.relu(self.fc_registration(x3)))
        x_reg = self.out_x_registration(x3[:,0:1])
        y_reg = self.out_y_registration(x3[:,1:2])
        z_reg = self.out_z_registration(x3[:,2:3])
        return torch.cat((x1,x2,x_reg,y_reg,z_reg),dim=1)
        #return torch.cat((x1,x2,x_reg, y_reg),dim=1)

class MLPregv2(nn.Module):
    def __init__(self, input_size=3, hidden_size=512, output_size=1, dropout=0, num_layers=5, num_layers_head=3):
        super(MLPregv2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_layers_head = num_layers_head

        fc_blocks = [fc_block(self.hidden_size,self.hidden_size, self.dropout) for i in range(self.num_layers)]
        self.fc_in = nn.Linear(self.input_size, self.hidden_size)
        self.fc_out1 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.fc_out2 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.fc_out3 = nn.Linear(self.hidden_size, self.hidden_size//2)
        fc_head1 = [fc_block(self.hidden_size//2,self.hidden_size//2, self.dropout) for i in range(self.num_layers_head)]
        fc_head2 = [fc_block(self.hidden_size//2,self.hidden_size//2, self.dropout) for i in range(self.num_layers_head)]    
        fc_head3 = [fc_block(self.hidden_size//2,self.hidden_size//2, self.dropout) for i in range(self.num_layers_head)]    
        self.fc_head1 = nn.Sequential(*fc_head1)
        self.fc_head2 = nn.Sequential(*fc_head2)
        self.fc_head3 = nn.Sequential(*fc_head3)
        self.out1 = nn.Linear(self.hidden_size//2, self.output_size//2)
        self.out2 = nn.Linear(self.hidden_size//2, self.output_size//2)
        self.out3 = nn.Linear(self.hidden_size//2, 3)
        self.fc = nn.Sequential(*fc_blocks)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        # flatten image
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc_in(x))
        x = self.dropout_layer(x)
        x = self.fc(x)
        # split heads layer
        x1 = x
        x2 = x
        x3 = x
        
        x1 = F.relu(self.fc_out1(x1))
        x2 = F.relu(self.fc_out2(x2))
        x3 = F.relu(self.fc_out3(x3))

        x1 = self.out1(self.fc_head1(x1))
        x2 = self.out2(self.fc_head2(x2))
        x3 = self.out3(self.fc_head3(x3))
        return torch.cat((x1,x2,x3),dim=1)

class MLP_SIRENreg(nn.Module):
    def __init__(self, input_size=3, hidden_size=512, output_size=1, dropout=0, num_layers=5, first_omega_0=30):
        super(MLP_SIRENreg, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.num_layers = num_layers
        fc_blocks = [fc_block(self.hidden_size,self.hidden_size, self.dropout) for i in range(self.num_layers)]
        self.fc_in = nn.Linear(self.input_size, self.hidden_size)
        self.fc_contrast1 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.fc_contrast2 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.out_contrast1 = nn.Linear(self.hidden_size//2, self.output_size//2)
        self.out_contrast2 = nn.Linear(self.hidden_size//2, self.output_size//2)
        self.fc = nn.Sequential(*fc_blocks)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        self.net_registration = []
        self.net_registration.append((SineLayer(self.hidden_size, self.hidden_size//2,
                                  is_first=True, omega_0=first_omega_0)))
        self.net_registration.append((SineLayer(self.hidden_size//2, 3,
                                  is_first=False, omega_0=first_omega_0)))

        self.net_registration = nn.Sequential(*self.net_registration)


    def forward(self, x):
        # flatten image
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc_in(x))
        x = self.dropout_layer(x)
        x = self.fc(x)
        # add output layer
        x1 = x
        x2 = x
        x3 = x
        x1 = (self.out_contrast1(F.relu(self.fc_contrast1(x1))))
        x2 = (self.out_contrast2(F.relu(self.fc_contrast2(x2))))
        x3 = self.net_registration(x3)
        return torch.cat((x1,x2,x3),dim=1)
    

# source: https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=Eo1TYp2ePynt
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input_data):
        return torch.sin(self.omega_0 * self.linear(input_data))

    def forward_with_intermediate(self, input_data):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input_data)
        return torch.sin(intermediate), intermediate


# paper: https://arxiv.org/pdf/2301.05187.pdf

class WireLayerNonComplex(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, s_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.s_0 = s_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input_data):
        return torch.sin(self.omega_0 * self.linear(input_data))*torch.exp(-((self.s_0 * self.linear(input_data))*(self.s_0 * self.linear(input_data))))

    def forward_with_intermediate(self, input_data):
        # For visualization of activation distributions
        intermediate_1 = self.omega_0 * self.linear(input_data)
        intermediate_2 = self.s_0 * self.linear(input_data)
        return torch.sin(intermediate_1)*torch.exp(-torch.square(intermediate_2)), intermediate_1, intermediate_2



class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


class WireReal(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30., first_s_0=10, hidden_s_0=10):
        super().__init__()

        self.net = []
        self.net.append(WireLayerNonComplex(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0, s_0=first_s_0))

        for i in range(hidden_layers):
            self.net.append(WireLayerNonComplex(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0, s_0=hidden_s_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(WireLayerNonComplex(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0, s_0=hidden_s_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, WireLayerNonComplex):
                x, intermed1, intermed2 = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed1.retain_grad()
                    intermed2.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed1 + intermed2 #unsure if this makes sense (JM)
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
