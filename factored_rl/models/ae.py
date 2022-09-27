from typing import Tuple

import pytorch_lightning as pl
import torch

from factored_rl import configs
from factored_rl.models.nnutils import Sequential, Reshape
from factored_rl.models import Network, MLP, CNN, losses

class Autoencoder(pl.LightningModule):
    def __init__(self, input_shape: Tuple, cfg: configs.Config):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.n_latent_dims = cfg.model.ae.n_latent_dims
        self.output_shape = self.input_shape
        self.cfg = cfg
        self.encoder = Encoder(input_shape, self.n_latent_dims, cfg.model)
        self.decoder = Decoder(self.encoder, input_shape, cfg.model)

    def training_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = torch.nn.functional.mse_loss(input=x_hat, target=x)
        self.log('loss/reconst', loss)
        return loss

    def configure_optimizers(self):
        partial_optimizer = configs.instantiate(self.cfg.trainer.optimizer)
        optimizer = partial_optimizer(self.parameters(), lr=self.cfg.trainer.learning_rate)
        return optimizer

class PairedAutoencoder(Autoencoder):
    def __init__(self, input_shape: Tuple, cfg: configs.Config):
        super().__init__(input_shape, cfg)
        assert cfg.model.action_sampling is not None
        distance_modes = {
            'mse': torch.nn.functional.mse_loss,
        }
        if self.cfg.losses.distance not in distance_modes:
            raise RuntimeError(
                f"Distance mode {self.cfg.losses.distance} not in {list(distance_modes.keys())}")
        self.distance = distance_modes[self.cfg.losses.distance]

    def training_step(self, batch, batch_idx):
        ob = batch['ob']
        next_ob = batch['next_ob']
        z = self.encoder(ob)
        next_z = self.encoder(next_ob)
        effects = next_z - z
        losses = {
            'effects': self.effects_loss(effects),
            'reconst': self.reconstruction_loss(ob, next_ob, z, next_z),
        }
        loss = sum([losses[key] * self.cfg.losses[key] for key in losses.keys()])
        losses = {('loss/' + key): value for key, value in losses.items()}
        losses['loss/train_loss'] = loss
        self.log_dict(losses)
        return loss

    def effects_loss(self, effects):
        if self.cfg.losses.effects == 0:
            return 0.0
        effects_loss = losses.compute_sparsity(effects, self.cfg.losses.sparsity)
        return effects_loss

    def reconstruction_loss(self, ob, next_ob, z, next_z):
        if self.cfg.losses.reconst == 0:
            return 0.0
        ob_hat = self.decoder(z)
        next_ob_hat = self.decoder(next_z)
        reconst_loss = (self.distance(input=ob_hat, target=ob) +
                        self.distance(input=next_ob_hat, target=next_ob)) / 2
        return reconst_loss

class Encoder(Network):
    def forward(self, x):
        is_batched = (tuple(x.shape) != tuple(self.input_shape))
        x = super().forward(x)
        if not is_batched:
            x = torch.squeeze(x, 0)
        return x

class Decoder(Network):
    def __init__(self, encoder: Encoder, output_shape, cfg: configs.ModelConfig):
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
