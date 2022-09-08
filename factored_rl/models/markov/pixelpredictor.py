import numpy as np
import torch

from .autoencoder import AutoEncoder
from ..mlp import MLP
from ..nnutils import one_hot

class PixelPredictor(AutoEncoder):
    def __init__(self, args, n_actions, input_shape=2):
        super().__init__(args, n_actions, input_shape)
        self.transition_model = MLP(
            n_inputs=(args.latent_dims + n_actions),
            n_outputs=args.latent_dims,
            n_units_per_layer=args.n_units_per_layer,
            n_hidden_layers=args.n_hidden_layers,
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)

    def compute_loss(self, x0, a, x1):
        z0 = self.encode(x0)
        transition_input = torch.cat((z0, one_hot(a, self.n_actions)), dim=-1)
        z1 = self.transition_model(transition_input)
        loss = self.mse(x1, self.decode(z1))
        return loss

    def train_batch(self, x0, a, x1, *args, **kwargs):
        self.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(x0, a, x1)
        loss.backward()
        self.optimizer.step()
        return loss
