from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

from ..nnutils import Network, Reshape
from .phinet import PhiNet

class AutoEncoder(Network):
    def __init__(self, args, n_actions, input_shape=2):
        super().__init__()
        self.n_actions = n_actions

        self.coefs = args.coefs
        self.phi = PhiNet(input_shape=input_shape,
                          n_latent_dims=args.latent_dims,
                          n_units_per_layer=args.n_units_per_layer,
                          n_hidden_layers=args.n_hidden_layers)
        self.reverse_phi = nn.Sequential(
            PhiNet(input_shape=args.latent_dims,
                   n_latent_dims=self.phi.layers[0].shape[-1],
                   n_units_per_layer=args.n_units_per_layer,
                   n_hidden_layers=args.n_hidden_layers),
            Reshape(-1, *input_shape),
        )
        self.mse = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def encode(self, x0):
        return self.phi(x0)

    def decode(self, z0):
        return self.reverse_phi(z0)

    def compute_loss(self, x0):
        loss = self.mse(x0, self.decode(self.encode(x0)))
        return loss

    def train_batch(self, x0, *args, **kwargs):
        self.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(x0)
        loss.backward()
        self.optimizer.step()
        return loss
