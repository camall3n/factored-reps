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

class FactoredFwdModel(Network):
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
        self.parents_model = ParentsNet(n_actions=n_actions,
                                        n_latent_dims=n_latent_dims,
                                        n_units_per_layer=n_units_per_layer,
                                        n_hidden_layers=n_hidden_layers,
                                        factored=True)
        self.fwd_model = FwdNet(n_actions=n_actions,
                                n_latent_dims=n_latent_dims,
                                n_hidden_layers=n_hidden_layers,
                                n_units_per_layer=n_units_per_layer,
                                factored=True)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()

        # don't include featurenet/phi parameters in optimizer
        parameters = (list(self.encoder.parameters()) + list(self.decoder.parameters()) +
                      list(self.parents_model.parameters()) + list(self.fwd_model.parameters()))
        self.optimizer = torch.optim.Adam(parameters, lr=self.lr)

    def compute_fwd_loss(self, z1, z1_hat):
        if self.coefs['L_fwd'] == 0.0:
            return torch.tensor(0.0)
        return self.mse(z1, z1_hat)

    def compute_factored_loss(self, parent_likelihood):
        if self.coefs['L_fac'] == 0.0:
            return torch.tensor(0.0)

        # TODO: how to compute factored loss?

        # 1. mean?
        loss = torch.mean(parent_likelihood, dim=-1)
        if parent_likelihood.ndim > 2:
            loss = torch.mean(loss, dim=0)

        # 2. sum?
        # loss = torch.sum(parent_likelihood, dim=-1)

        # 3. ???

        loss = torch.mean(loss, dim=0)

        return loss

    def encode(self, x):
        z = self.phi(x)
        z_fac = self.encoder(z)
        return z_fac

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_loss(self, parent_likelihood, z1, z1_hat):
        loss_info = {
            'L_fwd': self.compute_fwd_loss(z1, z1_hat),
            'L_fac': self.compute_factored_loss(parent_likelihood),
        }
        loss = 0
        for loss_type in ['L_fwd', 'L_fac']:
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
        with torch.no_grad():
            z1_factored = self.encoder(z1)
        parent_dependencies, parent_likelihood = self.parents_model(z0_factored, a)
        dz_hat = self.fwd_model(z0_factored, a, parent_dependencies)
        z1_factored_hat = z0_factored + dz_hat
        z1_hat = self.decoder(z1_factored_hat)

        loss_info = self.compute_loss(parent_likelihood, z1, z1_hat)
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