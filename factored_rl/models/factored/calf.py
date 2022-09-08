import itertools

import numpy as np
import torch
import torch.nn

from factored_rl.models.nnutils import Network, Identity
from factored_rl.models.mlp import MLP

class GANLoss(torch.nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.

    Copied from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = torch.nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = torch.nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class CALFNet(Network):
    """Counterfactual Adversarial Latent Factorization"""
    def __init__(self,
                 args,
                 n_actions,
                 n_input_dims,
                 n_latent_dims,
                 device='cpu',
                 backprop_next_state=True,
                 identity=False):
        super().__init__()
        self.n_actions = n_actions
        self.n_input_dims = n_input_dims
        self.n_latent_dims = n_latent_dims
        self.coefs = args.coefs
        self.device = device
        self.backprop_next_state = backprop_next_state
        self.identity = identity

        if self.identity:
            self.encoder = Identity()
            self.decoder = Identity()
            if self.n_input_dims != self.n_latent_dims:
                raise ValueError(
                    "'n_input_dims' must match 'n_latent_dims' when using identity autoencoder")
        else:
            self.encoder = MLP(n_inputs=n_input_dims,
                               n_outputs=n_latent_dims,
                               n_hidden_layers=args.n_hidden_layers,
                               n_units_per_layer=args.n_units_per_layer,
                               final_activation=torch.nn.Tanh)
            self.decoder = MLP(n_inputs=n_latent_dims,
                               n_outputs=n_input_dims,
                               n_hidden_layers=args.n_hidden_layers,
                               n_units_per_layer=args.n_units_per_layer,
                               final_activation=torch.nn.Tanh)

        self.discriminator = MLP(n_inputs=(2 * n_latent_dims),
                                 n_outputs=1,
                                 n_hidden_layers=args.n_hidden_layers,
                                 n_units_per_layer=args.n_units_per_layer,
                                 final_activation=None)

        self.criterionGAN = GANLoss(args.gan_mode).to(self.device)
        self.criterionCycle = torch.nn.MSELoss()

        # don't include featurenet/phi parameters in optimizer
        params_G = itertools.chain(self.encoder.parameters(), self.decoder.parameters())
        params_D = self.discriminator.parameters()
        self.optimizer_G = torch.optim.Adam(params_G, lr=args.lr_G) # TODO: tune beta1?
        self.optimizer_D = torch.optim.Adam(params_D, lr=args.lr_D) # TODO: tune beta1?

    def compute_reconstruction_loss(self, x0, x0_hat, x1, x1_hat):
        if self.coefs.L_rec == 0.0:
            return torch.tensor(0.0).to(self.device)
        x0_rec_loss = self.criterionCycle(x0, x0_hat)
        x1_rec_loss = self.criterionCycle(x1, x1_hat)
        return (x0_rec_loss + x1_rec_loss) * 0.5

    def compute_focused_loss(self, z0, z1):
        if self.coefs.L_foc == 0.0:
            return torch.tensor(0.0).to(self.device)
        eps = 1e-6
        dz = z1 - z0
        l1 = torch.sum(torch.abs(dz), dim=-1)
        lmax = torch.max(torch.abs(dz), dim=-1)[0]
        return torch.mean(l1 / (lmax + eps))

    def generate_calf_transitions(self, z0, z1):
        za_0, za_1 = z0, z1
        N = len(za_0)
        with torch.no_grad():
            # shuffle za to get alternate transitions zb
            shuffled_idx = np.random.permutation(np.arange(N))
            zb_0 = za_0[shuffled_idx, ...]
            zb_1 = za_1[shuffled_idx, ...]

            # select one variable index per sample in zb
            selected_idx = torch.multinomial(torch.ones(self.n_latent_dims),
                                             num_samples=N,
                                             replacement=True).to(self.device)

            # extract selected values from zb transition
            zbi_0 = zb_0[torch.arange(N), ..., selected_idx]
            zbi_1 = zb_1[torch.arange(N), ..., selected_idx]

            # Duplicate za again, and insert selected values from zb to form zc
            zc_0, zc_1 = za_0.clone(), za_1.clone()
            zc_0[torch.arange(N), ..., selected_idx] = zbi_0
            zc_1[torch.arange(N), ..., selected_idx] = zbi_1

        zz_real = torch.cat([za_0, za_1], dim=-1)
        zz_fake = torch.cat([zc_0, zc_1], dim=-1)
        return zz_real, zz_fake

    def compute_discriminator_loss(self, zz_real, zz_fake):
        # TODO: should actions be inputs as well?

        pred_real = self.discriminator(zz_real.detach())
        loss_real = self.criterionGAN(pred_real, target_is_real=True)

        pred_fake = self.discriminator(zz_fake)
        loss_fake = self.criterionGAN(pred_fake, target_is_real=False)
        loss_D = (loss_real + loss_fake) * 0.5

        return loss_D

    def compute_calf_loss(self, zz_real):
        # TODO: actions?

        return self.criterionGAN(self.discriminator(zz_real), target_is_real=False)

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
        z0_factored = self.encoder(x0)
        x0_hat = self.decoder(z0_factored)
        with torch.set_grad_enabled(self.backprop_next_state):
            z1_factored = self.encoder(x1)
            x1_hat = self.decoder(z1_factored)
        zz_real, zz_fake = self.generate_calf_transitions(z0_factored, z1_factored)

        # "generator"
        self.discriminator.freeze()
        if not test:
            self.optimizer_G.zero_grad()
        loss_info = {}
        loss_info['L_rec'] = self.compute_reconstruction_loss(x0, x0_hat, x1, x1_hat)
        loss_info['L_calf'] = self.compute_calf_loss(zz_real)
        loss_info['L_foc'] = self.compute_focused_loss(z0_factored, z1_factored)
        loss_G = 0
        for loss_type in sorted(loss_info.keys()):
            loss_G += vars(self.coefs)[loss_type] * loss_info[loss_type]
        if not test and not self.identity:
            loss_G.backward()
            self.optimizer_G.step()
        loss_info['L_G'] = loss_G

        # discriminator
        self.discriminator.unfreeze()
        if not test:
            self.optimizer_D.zero_grad()
        loss_D = self.compute_discriminator_loss(zz_real, zz_fake)
        if not test:
            loss_D.backward()
            self.optimizer_D.step()
        loss_info['L_D'] = loss_D
        loss_info['L'] = loss_D + loss_G

        return z0_factored, z1_factored, loss_info
