import torch

from . import nnutils

class NatureDQN(nnutils.Network):
    def __init__(self, input_shape=(4, 84, 84), n_actions=18):
        super().__init__()
        shape = nnutils.get_conv2d_output_size(input_shape[1:], (8, 8), 4)
        shape = nnutils.get_conv2d_output_size(shape, (4, 4), 2)
        shape = nnutils.get_conv2d_output_size(shape, (3, 3), 1)
        flattened_size = shape[0] * shape[1] * 64
        self.encoder = nnutils.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=(8, 8), stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
            torch.nn.ReLU(),
            nnutils.Reshape(-1, flattened_size),
        )
        self.model = nnutils.Sequential(torch.nn.Linear(flattened_size, 512), torch.nn.ReLU(),
                                        torch.nn.Linear(512, n_actions))

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0) # add a batch dimension
        return self.model(self.encoder(x))
