import numpy as np
import torch
import torch.nn

from ..nnutils import Network

class InvNet(Network):
    def __init__(self,
                 n_actions,
                 n_latent_dims=4,
                 n_hidden_layers=1,
                 n_units_per_layer=32,
                 dropout_prob=0.0,
                 temperature=1.0):
        super().__init__()
        self.n_actions = n_actions
        self.temperature = temperature

        self.layers = []
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(2 * n_latent_dims, n_actions)])
        else:
            self.layers.extend(
                [torch.nn.Linear(2 * n_latent_dims, n_units_per_layer),
                 torch.nn.Tanh()])
            if dropout_prob > 0:
                self.layers.append(torch.nn.Dropout(dropout_prob))
            remaining_layer_group = [
                torch.nn.Linear(n_units_per_layer, n_units_per_layer),
                torch.nn.Tanh()
            ] + ([torch.nn.Dropout(dropout_prob)] if dropout_prob > 0 else [])
            self.layers.extend(remaining_layer_group * (n_hidden_layers - 1))
            self.layers.extend([torch.nn.Linear(n_units_per_layer, n_actions)])

        self.inv_model = torch.nn.Sequential(*self.layers)

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        a_logits = self.inv_model(context) / self.temperature
        return a_logits
