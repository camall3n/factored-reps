import numpy as np
import torch
import torch.nn

from ..nnutils import Module, one_hot
from ..mlp import MLP

class ContrastiveNet(Module):
    def __init__(self, n_latent_dims=4, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()
        self.frozen = False

        self.layers = []
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(2 * n_latent_dims, 1)])
        else:
            self.layers.extend(
                [torch.nn.Linear(2 * n_latent_dims, n_units_per_layer),
                 torch.nn.Tanh()])
            self.layers.extend(
                [torch.nn.Linear(n_units_per_layer, n_units_per_layer),
                 torch.nn.Tanh()] * (n_hidden_layers - 1))
            self.layers.extend([torch.nn.Linear(n_units_per_layer, 1)])
        self.layers.extend([torch.nn.Sigmoid()])
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        fakes = self.model(context).squeeze()
        return fakes

class ActionContrastiveNet(Module):
    def __init__(self, n_actions, n_latent_dims, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()
        self.frozen = False
        self.n_actions = n_actions

        self.layers = []
        input_size = 2 * n_latent_dims + n_actions
        self.model = MLP(
            n_inputs=input_size,
            n_outputs=1,
            n_hidden_layers=n_hidden_layers,
            n_units_per_layer=n_units_per_layer,
            activation=torch.nn.Tanh,
            final_activation=torch.nn.Sigmoid,
        )

    def forward(self, z0, a, z1):
        a_onehot = one_hot(a, self.n_actions)
        context = torch.cat((z0, a_onehot, z1), -1)
        fakes = self.model(context).squeeze()
        return fakes
