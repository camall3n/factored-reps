from collections import defaultdict
from markov_abstr.gridworld.models.featurenet import FeatureNet
from factored_reps.models.parents_net import ParentsNet
import numpy as np
import torch
import torch.nn

from markov_abstr.gridworld.models.nnutils import Network
from markov_abstr.gridworld.models.phinet import PhiNet
from markov_abstr.gridworld.models.invnet import InvNet
from markov_abstr.gridworld.models.fwdnet import FwdNet
from markov_abstr.gridworld.models.simplenet import SimpleNet
from markov_abstr.gridworld.models.contrastivenet import ContrastiveNet
from markov_abstr.gridworld.models.invdiscriminator import InvDiscriminator

class FactoredAutoencoder(Network):
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

        self.featurenet = FeatureNet(n_actions=n_actions,
                                     input_shape=input_shape,
                                     n_latent_dims=n_latent_dims,
                                     n_hidden_layers=n_hidden_layers,
                                     n_units_per_layer=n_units_per_layer,
                                     lr=lr,
                                     coefs=coefs)
        self.phi = self.featurenet.phi

        self.encoder = SimpleNet(n_inputs=n_latent_dims,
                                 n_outputs=n_latent_dims,
                                 n_hidden_layers=1,
                                 n_units_per_layer=32,
                                 final_activation=torch.nn.Tanh)
        self.decoder = SimpleNet(n_inputs=n_latent_dims,
                                 n_outputs=n_latent_dims,
                                 n_hidden_layers=1,
                                 n_units_per_layer=32,
                                 final_activation=torch.nn.Tanh)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()

        # don't include featurenet/phi parameters in optimizer
        parameters = (list(self.encoder.parameters()) + list(self.decoder.parameters()))
        self.optimizer = torch.optim.Adam(parameters, lr=self.lr)

    def compute_reconstruction_loss(self, z0, z0_hat, z1, z1_hat):
        if self.coefs['L_rec'] == 0.0:
            return torch.tensor(0.0)
        return (self.mse(z0, z0_hat) + self.mse(z1, z1_hat)) / 2.0

    def compute_factored_loss(self, z0, z1):
        eps = 1e-6
        dz = z1 - z0
        l1 = torch.sum(torch.abs(dz), dim=-1)
        lmax = torch.max(torch.abs(dz), dim=-1)[0]
        return torch.mean(l1 / lmax)

    def encode(self, x):
        z = self.phi(x)
        z_fac = self.encoder(z)
        return z_fac

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_loss(self, z0, z0_factored, z0_hat, z1, z1_factored, z1_hat):
        loss_info = {
            'L_rec': self.compute_reconstruction_loss(z0, z0_hat, z1, z1_hat),
            'L_fac': self.compute_factored_loss(z0_factored, z1_factored),
        }
        loss = 0
        for loss_type in ['L_rec', 'L_fac']:
            loss += self.coefs[loss_type] * loss_info[loss_type]
        loss_info['L'] = loss

        return loss_info

    def train_batch(self, x0, a, x1, test=False):
        _, _, fnet_loss_info = self.featurenet.train_batch(x0, a, x1, d=None, test=test)

        with torch.no_grad():
            z0 = self.phi(x0)
            z1 = self.phi(x1)

        if not test:
            self.train()
            self.optimizer.zero_grad()
        z0_factored = self.encoder(z0)
        z1_factored = self.encoder(z1)
        z0_hat = self.decoder(z0_factored)
        z1_hat = self.decoder(z1_factored)

        loss_info = self.compute_loss(z0, z0_factored, z0_hat, z1, z1_factored, z1_hat)
        if not test:
            loss_info['L'].backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                for loss_type, loss_value in fnet_loss_info.items():
                    if loss_type == 'L':
                        loss_info[loss_type] += loss_value
                    elif loss_type in ['L_inv', 'L_rat', 'L_dis']:
                        loss_info[loss_type] = loss_value

        return z0_factored, z1_factored, loss_info
