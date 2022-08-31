import numpy as np
import torch
import torch.nn

from factored_reps.models.markov.featurenet import FeatureNet
from factored_reps.models.factored.parents_net import ParentsNet
from factored_reps.models.nnutils import Network
from factored_reps.models.markov.fwdnet import FwdNet
from factored_reps.models.simplenet import SimpleNet

class FactoredFwdModel(Network):
    def __init__(self, args, n_actions, input_shape=2, device='cpu'):
        super().__init__()

        self.coefs = args.coefs
        self.device = device

        self.featurenet = FeatureNet(args,
                                     n_actions=n_actions,
                                     input_shape=input_shape,
                                     latent_dims=args.markov_dims,
                                     device=self.device)
        if args.load_markov is not None:
            self.featurenet.load(args.load_markov, to=self.device)
        self.phi = self.featurenet.phi

        self.freeze_markov = args.freeze_markov
        if self.freeze_markov:
            self.featurenet.freeze()
            self.phi.freeze()

        self.encoder = SimpleNet(n_inputs=args.markov_dims,
                                 n_outputs=args.latent_dims,
                                 n_hidden_layers=args.n_hidden_layers_factored_autoenc,
                                 n_units_per_layer=args.n_units_per_layer,
                                 final_activation=torch.nn.Tanh)
        self.decoder = SimpleNet(n_inputs=args.latent_dims,
                                 n_outputs=args.markov_dims,
                                 n_hidden_layers=args.n_hidden_layers_factored_autoenc,
                                 n_units_per_layer=args.n_units_per_layer,
                                 final_activation=torch.nn.Tanh)
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

        # don't include featurenet/phi parameters in optimizer
        parameters = (list(self.encoder.parameters()) + list(self.decoder.parameters()) +
                      list(self.parents_model.parameters()) + list(self.fwd_model.parameters()))
        self.optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    def compute_fwd_loss(self, z1, z1_hat):
        if self.coefs.L_fwd == 0.0:
            return torch.tensor(0.0).to(self.device)
        return self.mse(z1, z1_hat)

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

    def compute_loss(self, z0, z0_rec, z0_factored, z1, z1_rec, z1_factored, z1_hat,
                     parent_likelihood):
        loss_info = {
            'L_fwd': self.compute_fwd_loss(z1, z1_hat),
            'L_fac': self.compute_factored_loss(parent_likelihood),
            'L_foc': self.compute_focused_loss(z0_factored, z1_factored),
            'L_rec': self.compute_reconstruction_loss(z0, z0_rec, z1, z1_rec)
        }
        loss = 0
        for loss_type in sorted(loss_info.keys()):
            loss += vars(self.coefs)[loss_type] * loss_info[loss_type]
        loss_info['L'] = loss

        return loss_info

    def train_batch(self, x0, a, x1, test=False):
        if self.freeze_markov:
            fnet_loss_info = dict()
        else:
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

        z0_rec = self.decoder(z0_factored)
        z1_rec = self.decoder(z1_factored)

        loss_info = self.compute_loss(z0, z0_rec, z0_factored, z1, z1_rec, z1_factored, z1_hat,
                                      parent_likelihood)
        if not test:
            loss_info['L'].backward()
            self.optimizer.step()

        with torch.no_grad():
            for loss_type, loss_value in fnet_loss_info.items():
                if loss_type == 'L':
                    loss_info[loss_type] += loss_value
                elif loss_type in ['L_inv', 'L_rat', 'L_dis']:
                    loss_info[loss_type] = loss_value

        return z0_factored, z1_factored, loss_info
