from typing import List, Tuple, Union

import numpy as np
import torch

from factored_rl import configs
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
            cnn = CNN.from_config(input_shape, cfg)
            layers.append(cnn)
            n_features = np.prod(cnn.output_shape)
        elif cfg.architecture == 'mlp':
            n_features = np.prod(input_shape) if len(input_shape) > 1 else input_shape[0]
        else:
            n_features = None
            raise NotImplementedError(f'Unknown architecture: {cfg.architecture}')

        if cfg.architecture == 'mlp' or len(input_shape) > 1:
            layers.append(Reshape(-1, n_features))

        if cfg.mlp.n_units_per_layer > 0:
            mlp = MLP.from_config(n_inputs=n_features, n_outputs=output_shape, cfg=cfg.mlp)
            layers.append(mlp)

        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x)
