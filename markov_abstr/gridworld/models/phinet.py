import numpy as np
import torch
import torch.nn

from .nnutils import Network, Reshape, conv2d_size_out
from markov_abstr.gridworld.models.simplenet import SimpleNet

class PhiNet(Network):
    def __init__(self,
                 input_shape=2,
                 n_latent_dims=4,
                 n_hidden_layers=1,
                 n_units_per_layer=32,
                 final_activation=torch.nn.Tanh,
                 network_arch='mlp'):
        super().__init__()
        self.input_shape = input_shape

        if network_arch == 'mlp':
            shape_flat = np.prod(self.input_shape)
            self.layers = []
            self.layers.append(Reshape(-1, shape_flat))
            self.layers.append(
                SimpleNet(
                    n_inputs=shape_flat,
                    n_outputs=n_latent_dims,
                    n_hidden_layers=n_hidden_layers,
                    n_units_per_layer=n_units_per_layer,
                    activation=torch.nn.Tanh,
                    final_activation=final_activation,
                ))

        elif network_arch == 'curl':
            final_size = conv2d_size_out(self.input_shape, (3, 3), 2)
            final_size = conv2d_size_out(final_size, (3, 3), 1)
            final_size = conv2d_size_out(final_size, (3, 3), 1)
            final_size = conv2d_size_out(final_size, (3, 3), 1)
            output_size = final_size[0] * final_size[1] * 32
            self.layers = [
                torch.nn.Conv2d(self.input_shape[0], 32, kernel_size=(4, 4), stride=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1),
                torch.nn.ReLU(),
                Reshape(-1, output_size),
                torch.nn.Linear(output_size, n_latent_dims),
                torch.nn.LayerNorm(n_latent_dims),
                torch.nn.Tanh(),
            ]
        else:
            raise NotImplementedError

        self.phi = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        z = self.phi(x)
        return z
