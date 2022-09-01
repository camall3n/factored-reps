import numpy as np
import torch
import torch.nn

from factored_rl.models.markov.featurenet import FeatureNet
from factored_rl.models.factored.parents_net import ParentsNet
from factored_rl.models.nnutils import Network
from factored_rl.models.simplenet import SimpleNet

class FocusedAutoencoder(Network):
    def __init__(self,
                 args,
                 n_actions,
                 n_input_dims,
                 n_latent_dims,
                 device='cpu',
                 backprop_next_state=True):
        super().__init__()
        self.n_actions = n_actions
        self.coefs = args.coefs
        self.device = device
        self.backprop_next_state = backprop_next_state

        self.encoder = SimpleNet(n_inputs=n_input_dims,
                                 n_outputs=n_latent_dims,
                                 n_hidden_layers=args.n_hidden_layers,
                                 n_units_per_layer=args.n_units_per_layer,
                                 final_activation=torch.nn.Tanh)
        self.decoder = SimpleNet(n_inputs=n_latent_dims,
                                 n_outputs=n_input_dims,
                                 n_hidden_layers=args.n_hidden_layers,
                                 n_units_per_layer=args.n_units_per_layer,
                                 final_activation=torch.nn.Tanh)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()

        # don't include featurenet/phi parameters in optimizer
        parameters = (list(self.encoder.parameters()) + list(self.decoder.parameters()))
        self.optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    def compute_reconstruction_loss(self, x0, x0_hat, x1, x1_hat):
        if self.coefs.L_rec == 0.0:
            return torch.tensor(0.0).to(self.device)
        return (self.mse(x0, x0_hat) + self.mse(x1, x1_hat)) / 2.0

    def compute_focused_loss(self, z0, z1):
        if self.coefs.L_foc == 0.0:
            return torch.tensor(0.0).to(self.device)
        eps = 1e-6
        dz = z1 - z0
        l1 = torch.sum(torch.abs(dz), dim=-1)
        lmax = torch.max(torch.abs(dz), dim=-1)[0]
        return torch.mean(l1 / (lmax + eps))

    def encode(self, x):
        z_fac = self.encoder(x)
        return z_fac

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        return self.decode(self.encode(x))

    def compute_loss(self, x0, z0_factored, x0_hat, x1, z1_factored, x1_hat):
        loss_info = {
            'L_rec': self.compute_reconstruction_loss(x0, x0_hat, x1, x1_hat),
            'L_foc': self.compute_focused_loss(z0_factored, z1_factored),
        }
        loss = 0
        for loss_type in sorted(loss_info.keys()):
            loss += vars(self.coefs)[loss_type] * loss_info[loss_type]
        loss_info['L'] = loss

        return loss_info

    def train_batch(self, x0, a, x1, test=False):
        if not test:
            self.train()
            self.optimizer.zero_grad()
        z0_factored = self.encoder(x0)
        x0_hat = self.decoder(z0_factored)
        with torch.set_grad_enabled(self.backprop_next_state):
            z1_factored = self.encoder(x1)
            x1_hat = self.decoder(z1_factored)

        loss_info = self.compute_loss(x0, z0_factored, x0_hat, x1, z1_factored, x1_hat)
        if not test:
            loss_info['L'].backward()
            self.optimizer.step()

        return z0_factored, z1_factored, loss_info
