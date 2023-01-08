from typing import Tuple, Dict

import matplotlib.pyplot as plt
import optuna
import pytorch_lightning as pl
import torch
from torch import Tensor

from factored_rl import configs
from factored_rl.models.nnutils import Sequential, Reshape, one_hot
from factored_rl.models import Network, MLP, CNN, losses

class BaseModel(pl.LightningModule):
    def __init__(self, input_shape: Tuple, n_actions: int, cfg: configs.Config):
        super().__init__()
        self.cfg = cfg
        self.input_shape = tuple(input_shape)
        self.n_actions = n_actions
        self.process_configs()
        self.initialize_qnet()

    def process_configs(self):
        self.n_latent_dims = self.input_shape[0]
        self.output_shape = self.n_latent_dims

    def initialize_qnet(self):
        if self.cfg.model.qnet is not None:
            self.qnet = MLP.from_config(self.n_latent_dims, self.n_actions, self.cfg.model.qnet)

class EncoderModel(BaseModel):
    def __init__(self, input_shape: Tuple, n_actions: int, cfg: configs.Config):
        super().__init__(input_shape, n_actions, cfg)
        self.encoder = EncoderNet(self.input_shape, self.n_latent_dims, cfg.model)

    def process_configs(self):
        self.n_latent_dims = self.cfg.model.n_latent_dims
        self.output_shape = self.n_latent_dims

class AutoencoderModel(EncoderModel):
    def __init__(self, input_shape: Tuple, n_actions: int, cfg: configs.Config):
        super().__init__(input_shape, n_actions, cfg)
        self.decoder = DecoderNet(self.encoder, input_shape, cfg.model)
        self.subtract_mean_input = cfg.model.subtract_mean_input
        if self.subtract_mean_input:
            self.mean_input = torch.zeros(input_shape, device=cfg.model.device).detach()
            self.n_inputs_observed = 0

    def process_configs(self):
        self.n_latent_dims = self.cfg.model.n_latent_dims
        self.output_shape = self.n_latent_dims

    def update_mean_input(self, x: Tensor):
        if self.subtract_mean_input:
            with torch.no_grad():
                n_inputs = len(x)

                # https://math.stackexchange.com/questions/106700/incremental-averaging
                #   m_n = m_{n-1} + (k/n) * (a_n - m_{n-1})
                self.n_inputs_observed += n_inputs
                update_fraction = (n_inputs / self.n_inputs_observed)
                self.mean_input += update_fraction * (x.mean(dim=0) - self.mean_input)

    def encode(self, x):
        if self.subtract_mean_input:
            x = x - self.mean_input
        return self.encoder(x)

    def decode(self, z):
        final_activation = torch.sigmoid if self.cfg.model.arch.decoder == 'cnn' else lambda x: x
        x_hat = final_activation(self.decoder(z))
        if self.subtract_mean_input:
            x_hat = x_hat + self.mean_input
        return x_hat

    def training_step(self, batch, batch_idx):
        x = batch
        self.update_mean_input(x)
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = torch.nn.functional.mse_loss(input=x_hat, target=x)
        self.log('loss/reconst', loss)
        if self.global_step % self.cfg.trainer.log_every_n_steps == 0:
            self.log_images(x, x_hat, 'img/obs_reconst_diff')

        self.prune_if_necessary({'loss/reconst': loss})
        return loss

    def log_images(self, obs: Tensor, obs_hat: Tensor, log_str='img/obs_reconst_diff'):
        if self.cfg.transform.name == 'images':
            # stack images along H dimension
            N = min(24, len(obs))
            obs_stack = torch.cat((obs[:N], obs_hat[:N], (obs[:N] - obs_hat[:N])), dim=2)
            tensorboard = self.logger.experiment
            # The following add_images() calls crash intermittently, due to a matplotlib issue:
            # https://github.com/Lightning-AI/lightning/issues/11925#issuecomment-1111727962
            #
            # Workaround: add plt.pause(0.1) after each call
            tensorboard.add_images(log_str, obs_stack, self.global_step)
            plt.pause(0.1)

    def prune_if_necessary(self, losses: Dict):
        if self.cfg.tuner.tune_rep and self.cfg.tuner.tune_metric == 'reconst':
            self.trial.report(losses['loss/reconst'], step=self.global_step)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.trainer.rep_learning_rate)

class PairedAutoencoderModel(AutoencoderModel):
    def __init__(self, input_shape: Tuple, n_actions: int, cfg: configs.Config):
        super().__init__(input_shape, n_actions, cfg)
        self.n_actions = n_actions
        assert cfg.model.action_sampling is not None
        distance_modes = {
            'mse': torch.nn.functional.mse_loss,
        }
        if self.cfg.loss.distance not in distance_modes:
            raise RuntimeError(
                f"Distance mode {self.cfg.loss.distance} not in {list(distance_modes.keys())}")
        self.distance = distance_modes[self.cfg.loss.distance]

    def training_step(self, batch, batch_idx):
        ob = batch['ob']
        self.update_mean_input(ob)
        actions = one_hot(batch['action'], self.n_actions)
        next_ob = batch['next_ob']
        self.update_mean_input(next_ob)
        z = self.encode(ob)
        next_z = self.encode(next_ob)
        effects = next_z - z
        ob_hat = self.decode(z)
        next_ob_hat = self.decode(next_z)
        losses = {
            'actions': self.action_semantics_loss(actions, effects),
            'effects': self.effects_loss(effects),
            'reconst': self.reconstruction_loss(ob, next_ob, ob_hat, next_ob_hat),
        }
        loss = sum([losses[key] * self.cfg.loss[key] for key in losses.keys()])
        losses = {('loss/' + key): value for key, value in losses.items()}
        losses['loss/train_loss'] = loss
        self.log_dict(losses)
        if self.global_step % self.cfg.trainer.log_every_n_steps == 0:
            self.log_images(ob, ob_hat, 'img/obs_reconst_diff')
            self.log_images(next_ob, next_ob_hat, 'img/next_obs_reconst_diff')

        self.prune_if_necessary(losses)
        return loss

    def effects_loss(self, effects: Tensor):
        if self.cfg.loss.effects == 0:
            return 0.0
        effects_loss = losses.compute_sparsity(effects, self.cfg.loss.sparsity)
        return effects_loss

    def _get_action_residuals(self, actions: Tensor, effects: Tensor):
        if actions.dim() != 2:
            raise ValueError(f'actions must be a 2-D one-hot tensor; got dim = {actions.dim()}')
        action_mask = actions.unsqueeze(-1) #(N,A,1)
        action_effects = action_mask * effects.unsqueeze(dim=1) #(N,1,d)
        mean_action_effects = (action_effects.sum(dim=0, keepdim=True) /
                               (action_mask.sum(dim=0, keepdim=True) + 1e-9)) #(A,d)
        action_residuals = ((action_effects - mean_action_effects) * action_mask).sum(dim=1) #(N,d)
        return action_residuals

    def action_semantics_loss(self, actions: Tensor, effects: Tensor):
        if self.cfg.loss.actions == 0:
            return 0.0
        action_residuals = self._get_action_residuals(actions, effects)
        actions_loss = losses.compute_sparsity(action_residuals, self.cfg.loss.sparsity)
        return actions_loss

    def reconstruction_loss(self, ob: Tensor, next_ob: Tensor, ob_hat: Tensor,
                            next_ob_hat: Tensor):
        if self.cfg.loss.reconst == 0:
            return 0.0
        reconst_loss = (self.distance(input=ob_hat, target=ob) +
                        self.distance(input=next_ob_hat, target=next_ob)) / 2
        return reconst_loss

class EncoderNet(Network):
    def forward(self, x):
        is_batched = (tuple(x.shape) != tuple(self.input_shape))
        x = super().forward(x)
        if not is_batched:
            x = torch.squeeze(x, 0)
        return x

class DecoderNet(Network):
    def __init__(self, encoder: EncoderNet, output_shape, cfg: configs.ModelConfig):
        super(Network, self).__init__() # skip over the Network init function
        self.input_shape = encoder.output_shape
        self.output_shape = output_shape
        if cfg.arch.decoder == 'mlp':
            _, mlp = encoder.model
            cnn = None
            flattened_activation = None
        elif cfg.arch.decoder == 'cnn':
            cnn, _, mlp = encoder.model
            flattened_activation = configs.instantiate(cfg.cnn.final_activation)
        else:
            raise NotImplementedError(f'Unsupported architecture: {cfg.arch.decoder}')
        for layer in reversed(mlp.model):
            if hasattr(layer, 'out_features'):
                in_shape = layer.out_features
                break
        flattened_shape = mlp.model[0].in_features
        transposed_mlp = MLP(
            n_inputs=in_shape,
            n_outputs=flattened_shape,
            n_hidden_layers=cfg.mlp.n_hidden_layers,
            n_units_per_layer=cfg.mlp.n_units_per_layer,
            activation=configs.instantiate(cfg.mlp.activation),
            final_activation=flattened_activation,
        )
        if cfg.arch.decoder == 'mlp':
            unflattened_shape = output_shape
        else:
            unflattened_shape = cnn.layer_shapes[-1]
        transposed_reshape = Reshape(-1, *unflattened_shape)
        layers = [transposed_mlp, transposed_reshape]
        if cfg.arch.decoder != 'mlp':
            transposed_conv = CNN(
                input_shape=unflattened_shape,
                n_output_channels=self.make_reversed(cfg.cnn.n_output_channels)[1:] +
                [cnn.n_input_channels],
                kernel_sizes=self.make_reversed(cfg.cnn.kernel_sizes),
                strides=self.make_reversed(cfg.cnn.strides),
                activation=configs.instantiate(cfg.cnn.activation),
                final_activation=None,
                transpose=True,
                layer_shapes=self.make_reversed(cnn.layer_shapes),
            )
            layers.append(transposed_conv)
        self.model = Sequential(*layers)

    @staticmethod
    def make_reversed(x):
        try:
            return list(reversed(x))
        except TypeError:
            pass
        return x

    def forward(self, x):
        is_batched = (x.ndim > 1)
        x = super().forward(x)
        if not is_batched:
            x = torch.squeeze(x, 0)
        return x
