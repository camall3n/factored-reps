from collections import defaultdict
from factored_reps.models.parents_net import ParentsNet
import numpy as np
import torch
import torch.nn

from markov_abstr.gridworld.models.nnutils import Network
from markov_abstr.gridworld.models.phinet import PhiNet
from markov_abstr.gridworld.models.invnet import InvNet
from markov_abstr.gridworld.models.fwdnet import FwdNet
from markov_abstr.gridworld.models.contrastivenet import ContrastiveNet
from markov_abstr.gridworld.models.invdiscriminator import InvDiscriminator

class FactorNet(Network):
    def __init__(self,
                 n_actions,
                 input_shape=2,
                 n_latent_dims=4,
                 n_hidden_layers=1,
                 n_units_per_layer=32,
                 lr=0.001,
                 coefs=None):
        super().__init__()
        self.n_actions = n_actions
        self.n_latent_dims = n_latent_dims
        self.lr = lr
        self.coefs = defaultdict(lambda: 1.0)
        if coefs is not None:
            for k, v in coefs.items():
                self.coefs[k] = v

        self.phi = PhiNet(input_shape=input_shape,
                          n_latent_dims=n_latent_dims,
                          n_units_per_layer=n_units_per_layer,
                          n_hidden_layers=n_hidden_layers)
        self.inv_model = InvNet(n_actions=n_actions,
                                n_latent_dims=n_latent_dims,
                                n_units_per_layer=n_units_per_layer,
                                n_hidden_layers=n_hidden_layers)
        self.discriminator = ContrastiveNet(n_latent_dims=n_latent_dims,
                                            n_hidden_layers=1,
                                            n_units_per_layer=n_units_per_layer)
        self.parents_model = ParentsNet(n_actions=n_actions,
                                        n_latent_dims=n_latent_dims,
                                        n_units_per_layer=n_units_per_layer,
                                        n_hidden_layers=n_hidden_layers)
        self.fwd_model = FwdNet(n_actions=n_actions,
                                n_latent_dims=n_latent_dims,
                                n_hidden_layers=n_hidden_layers,
                                n_units_per_layer=n_units_per_layer)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def inverse_loss(self, z0, z1, a):
        if self.coefs['L_inv'] == 0.0:
            return torch.tensor(0.0)
        a_hat = self.inv_model(z0, z1)
        return self.cross_entropy(input=a_hat, target=a)

    def ratio_loss(self, z0, z1):
        if self.coefs['L_rat'] == 0.0:
            return torch.tensor(0.0)
        N = len(z0)
        # shuffle next states
        idx = torch.randperm(N)
        z1_neg = z1.view(N, -1)[idx].view(z1.size())

        # concatenate positive and negative examples
        z0_extended = torch.cat([z0, z0], dim=0)
        z1_pos_neg = torch.cat([z1, z1_neg], dim=0)
        is_fake = torch.cat([torch.zeros(N), torch.ones(N)], dim=0)

        # Compute which ones are fakes
        fakes = self.discriminator(z0_extended, z1_pos_neg)
        return self.bce_loss(input=fakes, target=is_fake.float())

    def compute_fwd_loss(self, z1, z1_hat):
        if self.coefs['L_fwd'] == 0.0:
            return torch.tensor(0.0)
        return self.mse(z1, z1_hat)

    def distance_loss(self, z0, z1):
        if self.coefs['L_dis'] == 0.0:
            return torch.tensor(0.0)
        dz = torch.norm(z1 - z0, dim=-1, p=2)
        with torch.no_grad():
            max_dz = 0.1
        excess = torch.nn.functional.relu(dz - max_dz)
        return self.mse(excess, torch.zeros_like(excess))

    def compute_factored_loss(self, parents):
        if self.coefs['L_fac'] == 0.0:
            return torch.tensor(0.0)

        # TODO: how to compute factored loss?

        # 1. mean?
        loss = torch.mean(parents, dim=-1)

        # 2. sum?
        # loss = torch.sum(parents, dim=-1)

        # 3. ???

        loss = torch.mean(loss, dim=0)

        return loss

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def predict_a(self, z0, z1):
        raise NotImplementedError
        # a_logits = self.inv_model(z0, z1)
        # return torch.argmax(a_logits, dim=-1)

    def compute_loss(self, z0, a, parents, z1, z1_hat):
        loss = 0
        loss += self.coefs['L_inv'] * self.inverse_loss(z0, z1, a)
        loss += self.coefs['L_rat'] * self.ratio_loss(z0, z1)
        loss += self.coefs['L_dis'] * self.distance_loss(z0, z1)
        loss += self.coefs['L_fwd'] * self.compute_fwd_loss(z1, z1_hat)
        loss += self.coefs['L_fac'] * self.compute_factored_loss(parents)
        return loss

    def train_batch(self, x0, a, x1):
        self.train()
        self.optimizer.zero_grad()
        z0 = self.phi(x0)
        parents = self.parents_model(z0, a)
        z1_hat = self.fwd_model(z0 * parents, a)

        z1 = self.phi(x1)
        loss = self.compute_loss(z0, a, parents, z1, z1_hat)
        loss.backward()
        self.optimizer.step()
        return loss
