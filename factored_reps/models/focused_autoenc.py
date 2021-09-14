from markov_abstr.gridworld.models.featurenet import FeatureNet
from factored_reps.models.parents_net import ParentsNet
import numpy as np
import torch
import torch.nn

from markov_abstr.gridworld.models.nnutils import Network
from markov_abstr.gridworld.models.simplenet import SimpleNet

class FocusedAutoencoder(Network):
    def __init__(self, args, n_actions, input_shape=2, device='cpu'):
        super().__init__()
        self.n_actions = n_actions
        self.coefs = args.coefs
        self.device = device

        self.featurenet = FeatureNet(args, n_actions=n_actions, input_shape=input_shape)
        self.phi = self.featurenet.phi

        self.encoder = SimpleNet(n_inputs=args.markov_dims,
                                 n_outputs=args.latent_dims,
                                 n_hidden_layers=1,
                                 n_units_per_layer=32,
                                 final_activation=torch.nn.Tanh)
        self.decoder = SimpleNet(n_inputs=args.latent_dims,
                                 n_outputs=args.markov_dims,
                                 n_hidden_layers=1,
                                 n_units_per_layer=32,
                                 final_activation=torch.nn.Tanh)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()

        # don't include featurenet/phi parameters in optimizer
        parameters = (list(self.encoder.parameters()) + list(self.decoder.parameters()))
        self.optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    def compute_reconstruction_loss(self, z0, z0_hat, z1, z1_hat):
        if self.coefs.L_rec == 0.0:
            return torch.tensor(0.0).to(self.device)
        return (self.mse(z0, z0_hat) + self.mse(z1, z1_hat)) / 2.0

    def compute_focused_loss(self, z0, z1):
        if self.coefs.L_foc == 0.0:
            return torch.tensor(0.0).to(self.device)
        eps = 1e-6
        dz = z1 - z0
        l1 = torch.sum(torch.abs(dz), dim=-1)
        lmax = torch.max(torch.abs(dz), dim=-1)[0]
        return torch.mean(l1 / (lmax + eps))

    def encode(self, x):
        z = self.phi(x)
        z_fac = self.encoder(z)
        return z_fac

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_loss(self, z0, z0_factored, z0_hat, z1, z1_factored, z1_hat):
        loss_info = {
            'L_rec': self.compute_reconstruction_loss(z0, z0_hat, z1, z1_hat),
            'L_foc': self.compute_focused_loss(z0_factored, z1_factored),
        }
        loss = 0
        for loss_type in sorted(loss_info.keys()):
            loss += vars(self.coefs)[loss_type] * loss_info[loss_type]
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
