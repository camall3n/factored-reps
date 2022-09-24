from typing import List, Tuple, Union

import numpy as np
import torch

from factored_rl.experiments import configs
from .nnutils import Module, Sequential, Reshape
from .cnn import CNN
from .mlp import MLP

class Network(Module):
    def __init__(self, input_shape: Tuple[int], output_shape: Union[int, Tuple[int]],
                 cfg: configs.ModelConfig):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        layers = []
        if cfg.architecture == 'cnn':
            if input_shape[-2:] != cfg.cnn.supported_2d_input_shape:
                raise ValueError(f'Input shape does not match supported 2D input shape: '
                                 f'{cfg.cnn.supported_2d_input_shape}')
            cnn = CNN(
                input_shape=(3, ) + cfg.cnn.supported_2d_input_shape,
                n_output_channels=cfg.cnn.n_output_channels,
                kernel_sizes=cfg.cnn.kernel_sizes,
                strides=cfg.cnn.strides,
                activation=configs.instantiate(cfg.cnn.activation),
                final_activation=configs.instantiate(cfg.cnn.final_activation),
            )
            layers.append(cnn)
            n_features = np.prod(cnn.output_shape)
        elif cfg.architecture == 'mlp':
            n_features = np.prod(input_shape) if len(input_shape) > 1 else input_shape[0]
        else:
            n_features = None
            raise NotImplementedError(f'Unknown architecture: {cfg.architecture}')

        mlp = MLP(
            n_inputs=n_features,
            n_outputs=output_shape,
            n_hidden_layers=cfg.mlp.n_hidden_layers,
            n_units_per_layer=cfg.mlp.n_units_per_layer,
            activation=configs.instantiate(cfg.mlp.activation),
            final_activation=configs.instantiate(cfg.mlp.final_activation),
        )
        if len(input_shape) > 1:
            layers.append(Reshape(-1, n_features))
        layers.append(mlp)

        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x)
