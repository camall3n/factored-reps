from typing import List, Optional

import numpy as np
import torch
import torch.nn

from factored_rl import configs
from .nnutils import Module, ActivationType, build_activation

def coerce_to_int(x):
    try:
        return int(x)
    except TypeError:
        pass

    if hasattr(x, '__len__'):
        assert len(x) == 1
        x = x[0]

    return x

class MLP(Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_hidden_layers=1,
        n_units_per_layer=32,
        activation: Optional[ActivationType] = torch.nn.Tanh, # activation or list thereof for internal layers
        final_activation: Optional[ActivationType] = None, # activation or list thereof for final layer
    ):# yapf: disable
        super().__init__()
        self.n_outputs = coerce_to_int(n_outputs)
        self.frozen = False

        self.n_layers = n_hidden_layers + 1
        layer_sizes = [n_inputs] + [n_units_per_layer] * n_hidden_layers + [n_outputs]
        assert len(layer_sizes) == self.n_layers + 1

        # build list of lists of activations
        if not isinstance(activation, List):
            activation = [activation]
        if not isinstance(final_activation, List):
            final_activation = [final_activation]
        activations = [activation] * n_hidden_layers + [final_activation]
        assert len(activations) == self.n_layers

        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            for ac in activations[i]:
                if ac is not None:
                    self.layers.append(build_activation(ac, layer_sizes[i + 1]))

        self.model = torch.nn.Sequential(*self.layers)

    @classmethod
    def from_config(cls, n_inputs, n_outputs, cfg: configs.MLPConfig):
        return cls(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_hidden_layers=cfg.n_hidden_layers,
            n_units_per_layer=cfg.n_units_per_layer,
            activation=configs.instantiate(cfg.activation),
            final_activation=configs.instantiate(cfg.final_activation),
        )

    def forward(self, x):
        return self.model(x)
