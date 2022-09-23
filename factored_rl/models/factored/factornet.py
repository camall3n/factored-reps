import numpy as np
import torch
import torch.nn

from factored_rl.models.factored.parents_net import ParentsNet
from factored_rl.models.nnutils import Module
from factored_rl.models.markov.phinet import PhiNet
from factored_rl.models.markov.invnet import InvNet
from factored_rl.models.markov.fwdnet import FwdNet
from factored_rl.models.markov.contrastivenet import ContrastiveNet

class FactorNet(Module):
    def __init__(self, args, n_actions, input_shape=2, device='cpu'):
        super().__init__()
        self.n_actions = n_actions
        self.max_dz = args.max_dz
        self.coefs = args.coefs
        self.device = device

        self.phi = PhiNet(input_shape=input_shape,
                          n_latent_dims=args.latent_dims,
                          n_units_per_layer=args.n_units_per_layer,
                          n_hidden_layers=args.n_hidden_layers,
                          network_arch=args.encoder_arch)
        self.inv_model = InvNet(n_actions=n_actions,
                                n_latent_dims=args.latent_dims,
                                n_units_per_layer=args.n_units_per_layer,
                                n_hidden_layers=args.n_hidden_layers)
        self.discriminator = ContrastiveNet(n_latent_dims=args.latent_dims,
                                            n_hidden_layers=args.n_hidden_layers_contrastive,
                                            n_units_per_layer=args.n_units_per_layer)
        self.parents_model = ParentsNet(n_actions=n_actions,
                                        n_latent_dims=args.latent_dims,
                                        n_units_per_layer=args.n_units_per_layer,
                                        n_hidden_layers=args.n_hidden_layers,
                                        factored=True)
        self.fwd_model = FwdNet(n_actions=n_actions,
                                n_latent_dims=args.latent_dims,
                                n_hidden_layers=args.n_hidden_layers,
                                n_units_per_layer=args.n_units_per_layer,
                                factored=True)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)

    def inverse_loss(self, z0, z1, a):
        if self.coefs.L_inv == 0.0:
            return torch.tensor(0.0).to(self.device)
        a_hat = self.inv_model(z0, z1)
        return self.cross_entropy(input=a_hat, target=a)

    def ratio_loss(self, z0, z1):
        if self.coefs.L_rat == 0.0:
            return torch.tensor(0.0).to(self.device)
        N = len(z0)
        # shuffle next states
        idx = torch.randperm(N)
        z1_neg = z1.view(N, -1)[idx].view(z1.size())

        # concatenate positive and negative examples
        z0_extended = torch.cat([z0, z0], dim=0)
        z1_pos_neg = torch.cat([z1, z1_neg], dim=0)
        with torch.no_grad():
            is_fake = torch.cat([torch.zeros(N), torch.ones(N)], dim=0).float().to(self.device)

        # Compute which ones are fakes
        fakes = self.discriminator(z0_extended, z1_pos_neg)
        return self.bce_loss(input=fakes, target=is_fake)

    def compute_fwd_loss(self, z1, z1_hat):
        if self.coefs.L_fwd == 0.0:
            return torch.tensor(0.0).to(self.device)
        return self.mse(z1, z1_hat)

    def distance_loss(self, z0, z1):
        if self.coefs.L_dis == 0.0:
            return torch.tensor(0.0).to(self.device)
        dz = torch.norm(z1 - z0, dim=-1, p=2)
        excess = torch.nn.functional.relu(dz - self.max_dz)
        return self.mse(excess, torch.zeros_like(excess))

    def compute_factored_loss(self, parent_likelihood):
        if self.coefs.L_fac == 0.0:
            return torch.tensor(0.0).to(self.device)

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
        return z

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def predict_a(self, z0, z1):
        raise NotImplementedError
        # a_logits = self.inv_model(z0, z1)
        # return torch.argmax(a_logits, dim=-1)

    def compute_loss(self, z0, a, parent_likelihood, z1, z1_hat):
        loss_info = {
            'L_inv': self.inverse_loss(z0, z1, a),
            'L_rat': self.ratio_loss(z0, z1),
            'L_dis': self.distance_loss(z0, z1),
            'L_fwd': self.compute_fwd_loss(z1, z1_hat),
            'L_fac': self.compute_factored_loss(parent_likelihood),
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
        z0 = self.phi(x0)
        parent_dependencies, parent_likelihood = self.parents_model(z0, a)
        dz_hat = self.fwd_model(z0, a, parent_dependencies)
        z1_hat = z0 + dz_hat

        z1 = self.phi(x1)
        loss_info = self.compute_loss(z0, a, parent_likelihood, z1, z1_hat)
        if not test:
            loss_info['L'].backward()
            self.optimizer.step()
        return z0, z1, loss_info
