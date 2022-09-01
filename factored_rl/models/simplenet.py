import numpy as np
import torch
import torch.nn

from .nnutils import Network

class SimpleNet(Network):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_hidden_layers=1,
                 n_units_per_layer=32,
                 activation=torch.nn.Tanh,
                 final_activation=None):
        super().__init__()
        self.n_outputs = n_outputs
        self.frozen = False

        self.layers = []
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(n_inputs, n_outputs)])
        else:
            self.layers.extend([torch.nn.Linear(n_inputs, n_units_per_layer), activation()])
            self.layers.extend(
                [torch.nn.Linear(n_units_per_layer, n_units_per_layer),
                 activation()] * (n_hidden_layers - 1))
            self.layers.extend([torch.nn.Linear(n_units_per_layer, n_outputs)])

        if final_activation is not None:
            try:
                self.layers.extend([final_activation()])
            except TypeError:
                self.layers.extend([final_activation])

        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
