import numpy as np
import torch

from .nnutils import Network, Sequential, Reshape
from . import CNN, MLP

class NatureDQN(Network):
    def __init__(self, input_shape=(4, 84, 84), n_actions=18):
        super().__init__()
        assert input_shape[-2:] == (84, 84)
        cnn = CNN(input_shape=input_shape,
                  n_output_channels=[32, 64, 64],
                  kernel_sizes=[8, 4, 3],
                  strides=[4, 2, 1])
        n_flattened = np.prod(cnn.output_shape)
        mlp = MLP(n_inputs=n_flattened,
                  n_outputs=n_actions,
                  n_hidden_layers=1,
                  n_units_per_layer=512,
                  activation=torch.nn.ReLU)
        self.model = Sequential(*[cnn, Reshape(-1, n_flattened), mlp])

    def forward(self, x):
        return self.model(x)
