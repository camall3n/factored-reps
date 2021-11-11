import numpy as np
import torch
import torch.nn

from .nnutils import Network
from .phinet import PhiNet
from .invnet import InvNet
from .fwdnet import FwdNet
from .contrastivenet import ContrastiveNet, ActionContrastiveNet
from .invdiscriminator import InvDiscriminator

class FeatureNet(Network):
    def __init__(self, args, n_actions, input_shape=2, latent_dims=2, device='cpu'):
        super().__init__()
        self.n_actions = n_actions
        self.lr = args.learning_rate
        self.max_dz = args.max_dz
        self.coefs = args.coefs
        self.max_gradient_norm = args.markov_max_gradient_norm
        self.device = device

        self.phi = PhiNet(input_shape=input_shape,
                          n_latent_dims=latent_dims,
                          n_units_per_layer=args.n_units_per_layer,
                          n_hidden_layers=args.n_hidden_layers,
                          network_arch=args.encoder_arch)
        # self.fwd_model = FwdNet(n_actions=n_actions, n_latent_dims=latent_dims, n_hidden_layers=args.n_hidden_layers, n_units_per_layer=args.n_units_per_layer)
        self.inv_model = InvNet(n_actions=n_actions,
                                n_latent_dims=latent_dims,
                                n_units_per_layer=args.n_units_per_layer,
                                n_hidden_layers=args.n_hidden_layers_inverse_model,
                                dropout_prob=args.inverse_model_dropout_prob,
                                temperature=args.inverse_model_temperature)
        self.inv_discriminator = InvDiscriminator(n_actions=n_actions,
                                                  n_latent_dims=latent_dims,
                                                  n_units_per_layer=args.n_units_per_layer,
                                                  n_hidden_layers=args.n_hidden_layers)
        self.discriminator = ContrastiveNet(n_latent_dims=latent_dims,
                                            n_hidden_layers=1,
                                            n_units_per_layer=args.n_units_per_layer)
        self.action_discrim = ActionContrastiveNet(n_actions=n_actions,
                                                   n_latent_dims=latent_dims,
                                                   n_hidden_layers=1,
                                                   n_units_per_layer=args.n_units_per_layer)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.lr,
                                          weight_decay=args.markov_weight_decay)

    def inverse_loss(self, z0, z1, a):
        if self.coefs.L_inv == 0.0:
            return torch.tensor(0.0).to(self.device)
        a_hat = self.inv_model(z0, z1)
        return self.cross_entropy(input=a_hat, target=a)

    def contrastive_inverse_loss(self, z0, z1, a):
        if self.coefs.L_coinv == 0.0:
            return torch.tensor(0.0).to(self.device)
        N = len(z0)
        # shuffle next states
        idx = torch.randperm(N)  #BUG: why is this never being used?

        a_neg = torch.randint_like(a, low=0, high=self.n_actions)

        # concatenate positive and negative examples
        z0_extended = torch.cat([z0, z0], dim=0)
        z1_extended = torch.cat([z1, z1], dim=0)
        a_pos_neg = torch.cat([a, a_neg], dim=0)
        is_fake = torch.cat([torch.zeros(N), torch.ones(N)], dim=0)

        # Compute which ones are fakes
        fakes = self.inv_discriminator(z0_extended, z1_extended, a_pos_neg)
        return self.bce_loss(input=fakes, target=is_fake.float())

    def transition_ratio_loss(self, z0, a, z1, z1_neg):
        if self.coefs.L_rat == 0.0:
            return torch.tensor(0.0).to(self.device)
        N = len(z0)

        # concatenate positive and negative examples
        z0_extended = torch.cat([z0, z0], dim=0)
        a_extended = torch.cat([a, a], dim=0)
        z1_pos_neg = torch.cat([z1, z1_neg], dim=0)
        with torch.no_grad():
            is_fake = torch.cat([torch.zeros(N), torch.ones(N)], dim=0).float().to(self.device)

        # Compute which ones are fakes
        fakes = self.action_discrim(z0_extended, a_extended, z1_pos_neg)
        return self.bce_loss(input=fakes, target=is_fake)

    def ratio_loss(self, z0, z1, z1_neg):
        if self.coefs.L_rat == 0.0:
            return torch.tensor(0.0).to(self.device)
        N = len(z0)
        # shuffle next states
        # idx = torch.randperm(N)
        # z1_neg = z1.view(N, -1)[idx].view(z1.size())

        # concatenate positive and negative examples
        z0_extended = torch.cat([z0, z0], dim=0)
        z1_pos_neg = torch.cat([z1, z1_neg], dim=0)
        with torch.no_grad():
            is_fake = torch.cat([torch.zeros(N), torch.ones(N)], dim=0).float().to(self.device)

        # Compute which ones are fakes
        fakes = self.discriminator(z0_extended, z1_pos_neg)
        return self.bce_loss(input=fakes, target=is_fake)

    @staticmethod
    def get_negatives(buffer, idx, mode='all'):
        """Provide alternate x' (negative examples) for contrastive model"""

        shuffled_idx = np.random.permutation(idx)
        # replace all samples with one specific type of negative example
        if mode == 'random':
            negatives = buffer.retrieve(shuffled_idx, 'next_ob')  # x' -> \tilde x'
        # elif mode == 'same':
        #     negatives = buffer.retrieve(idx, 'ob')  # x' -> x
        elif mode == 'following':
            negatives = buffer.retrieve((idx + 1) % len(buffer), 'next_ob')  # x' -> x''
        elif mode == 'all':
            # replace samples with equal amounts of all types of negative example

            # draw from multiple sources of negative examples, depending on RNG
            r = np.random.rand(*idx.shape)

            # decide where to use the same obs again
            # to_keep = np.argwhere(r < 1 / 3).squeeze()

            # decide where to use the *subsequent* obs from the trajectory (i.e. x''), unless out of bounds
            to_offset = np.argwhere((r < 0.5) & ((idx + 1) < len(buffer))).squeeze()
            # to_offset = np.argwhere((1 / 3 <= r) & (r < 2 / 3)
            # & ((idx + 1) < len(buffer))).squeeze()
            # otherwise we'll draw a random obs from the buffer

            # replace x' samples with random samples
            negatives = buffer.retrieve(shuffled_idx, 'next_ob')
            # replace x' samples with x
            # negatives[to_keep] = buffer.retrieve(idx[to_keep], 'ob')
            # replace x' samples with x''
            negatives[to_offset] = buffer.retrieve(idx[to_offset] + 1, 'next_ob')
        else:
            raise NotImplementedError()

        return negatives

    def triplet_loss(self, z0, z1, z1_alt):
        if self.coefs.L_dis == 0.0:
            return torch.tensor(0.0).to(self.device)
        anchor = z0
        positive = z1
        negative = z1_alt
        d_pos = torch.norm(anchor - positive, dim=-1, p=2)
        d_neg = torch.norm(anchor - negative, dim=-1, p=2)
        margin = self.max_dz
        excess = torch.nn.functional.relu(d_pos - d_neg + margin)
        return torch.mean(excess, dim=0)

    def distance_loss(self, z0, z1):
        if self.coefs.L_dis == 0.0:
            return torch.tensor(0.0).to(self.device)
        dz = torch.norm(z1 - z0, dim=-1, p=2)
        excess = torch.nn.functional.relu(dz - self.max_dz)
        return self.mse(excess, torch.zeros_like(excess))

    def oracle_loss(self, z0, z1, d):
        if self.coefs.L_ora == 0.0:
            return torch.tensor(0.0).to(self.device)

        dz = torch.cat(
            [torch.norm(z1 - z0, dim=-1, p=2),
             torch.norm(z1.flip(0) - z0, dim=-1, p=2)], dim=0)

        with torch.no_grad():
            counts = 1 + torch.histc(d, bins=36, min=0, max=35)
            inverse_counts = counts.sum() / counts
            weights = inverse_counts[d.long()]
            weights = weights / weights.sum()

        loss = self.mse(dz, d / 10.0)
        # loss += torch.sum(weights * (dz - d / 20.0)**2) # weighted MSE
        # loss = -torch.nn.functional.cosine_similarity(dz, d, 0)
        return loss

    def encode(self, x):
        z = self.phi(x)
        return z

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def predict_a(self, x0, x1):
        with torch.no_grad():
            z0 = self.phi(x0)
            z1 = self.phi(x1)
            a_logits = self.inv_model(z0, z1)
        return torch.argmax(a_logits, dim=-1)

    def predict_is_fake(self, x0, x1):
        with torch.no_grad():
            z0 = self.phi(x0)
            z1 = self.phi(x1)
            fake_prob = self.discriminator(z0, z1)
        return fake_prob > 0.5

    def predict_is_fake_transition(self, x0, a, x1):
        with torch.no_grad():
            z0 = self.phi(x0)
            z1 = self.phi(x1)
            fake_prob = self.action_discrim(z0, a, z1)
        return fake_prob > 0.5

    def compute_loss(self, z0, a, z1, z1_alt):
        loss_info = {
            # 'L_coinv': self.contrastive_inverse_loss(z0, z1, a),
            # 'L_inv': self.inverse_loss(z0, z1, a),
            # 'L_rat': self.ratio_loss(z0, z1, z1_alt),
            'L_txr': self.transition_ratio_loss(z0, a, z1, z1_alt),
            # 'L_dis': self.distance_loss(z0, z1),
            'L_trp': self.triplet_loss(z0, z1, z1_alt),
            # 'L_ora': self.oracle_loss(z0, z1, d),
        }
        loss = 0
        for loss_type in sorted(loss_info.keys()):
            loss += vars(self.coefs)[loss_type] * loss_info[loss_type]
        loss_info['L'] = loss

        return loss_info

    def train_batch(self, x0, a, x1, x1_alt, d=None, test=False):
        if not test:
            self.train()
            self.optimizer.zero_grad()
        else:
            self.eval()
        z0 = self.phi(x0)
        z1 = self.phi(x1)
        z1_alt = self.phi(x1_alt)
        loss_info = self.compute_loss(z0, a, z1, z1_alt)
        if not test:
            loss_info['L'].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_gradient_norm)
            loss_info['grad_norm'] = grad_norm
            self.optimizer.step()
        return z0, z1, loss_info
