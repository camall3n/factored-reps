import itertools

import numpy as np
import torch
import torch.nn

from markov_abstr.gridworld.models.nnutils import Network
from markov_abstr.gridworld.models.simplenet import SimpleNet

class CAENet(Network):
    """Counterfactual Autoencoder"""
    def __init__(self, args, n_actions, n_input_dims, n_latent_dims, device='cpu'):
        super().__init__()
        self.n_actions = n_actions
        self.n_input_dims = n_input_dims
        self.n_latent_dims = n_latent_dims
        self.coefs = args.coefs
        self.device = device

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

        if args.dist_mode in ['mse', 'l2', 'L2']:
            self.distanceLoss = torch.nn.MSELoss()
        elif args.dist_mode in ['mae', 'l1', 'L1']:
            self.distanceLoss = torch.nn.L1Loss()
        elif args.dist_mode in ['huber', 'Huber']:
            self.distanceLoss = torch.nn.HuberLoss()
        else:
            raise ValueError("Invalid dist_mode. Choices are: 'mse', 'l1', 'huber'")

        # don't include featurenet/phi parameters in optimizer
        params = itertools.chain(self.encoder.parameters(), self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=args.lr)  # TODO: tune beta1?

    def compute_focused_loss(self, z0, z1):
        if self.coefs.L_foc == 0.0:
            return torch.tensor(0.0).to(self.device)
        eps = 1e-6
        dz = z1 - z0
        l1 = torch.sum(torch.abs(dz), dim=-1)
        lmax = torch.max(torch.abs(dz), dim=-1)[0]
        return torch.mean(l1 / (lmax + eps))

    def generate_counterfactual_states(self, z):
        za_0 = z
        N = len(za_0)
        with torch.no_grad():
            # shuffle za to get alternate states zb
            shuffled_idx = np.random.permutation(np.arange(N))
            zb_0 = za_0[shuffled_idx, ...]

            # select one variable index per sample in zb
            selected_idx = torch.multinomial(torch.ones(self.n_latent_dims),
                                             num_samples=N,
                                             replacement=True).to(self.device)

            # extract selected values from zb transition
            zbi_0 = zb_0[torch.arange(N), ..., selected_idx]

            # Duplicate za again, and insert selected values from zb to form zc
            zc_0 = za_0.clone()
            zc_0[torch.arange(N), ..., selected_idx] = zbi_0

        z_aug = zc_0
        return z_aug

    def encode(self, x):
        z_fac = self.encoder(x)
        return z_fac

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        return self.decode(self.encode(x))

    def train_batch(self, x0, a, x1, test=False):
        # forward
        if not test:
            self.train()
        z0 = self.encoder(x0)
        z1 = self.encoder(x1)

        x0_hat = self.decoder(z0)
        z0_hat = self.encoder(x0_hat)

        z0_aug = self.generate_counterfactual_states(z0)
        x0_aug = self.decoder(z0_aug)
        z0_aug_hat = self.encoder(x0_aug)
        x0_aug_hat = self.decoder(z0_aug_hat)

        # backward
        if not test:
            self.optimizer.zero_grad()
        loss_info = {}
        loss_info['L_rec_x'] = self.distanceLoss(x0, x0_hat)
        loss_info['L_rec_x_aug'] = self.distanceLoss(x0_aug, x0_aug_hat)
        loss_info['L_rec_z'] = (self.distanceLoss(z0, z0_hat) +
                                self.distanceLoss(z0_aug, z0_aug_hat))
        loss_info['L_foc'] = self.compute_focused_loss(z0, z1)
        loss = 0
        for loss_type in sorted(loss_info.keys()):
            loss += vars(self.coefs)[loss_type] * loss_info[loss_type]
        if not test:
            loss.backward()
            self.optimizer.step()
        loss_info['L'] = loss

        return z0, z1, loss_info
