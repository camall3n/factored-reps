from typing import Tuple

from disent.frameworks.vae import BetaVae
from disent.model import AutoEncoder as DisentAutoencoder
from disent.model.ae import DecoderConv64
from disent.model.ae import EncoderConv64

from factored_rl.experiments import configs

def build_disent_model(input_shape: Tuple, cfg: configs.Config):
    if cfg.model.name == 'ae_cnn_64':
        # create the BetaVAE model
        # - adjusting the beta, learning rate, and representation size.
        return BetaVae(
            model=DisentAutoencoder(
                # z_multiplier is needed to output mu & logvar when parameterising normal distribution
                encoder=EncoderConv64(x_shape=input_shape,
                                      z_size=cfg.model.ae.n_latent_dims,
                                      z_multiplier=2),
                decoder=DecoderConv64(x_shape=input_shape, z_size=cfg.model.ae.n_latent_dims),
            ),
            cfg=BetaVae.cfg(
                optimizer=cfg.trainer.optimizer._target_.split('.')[-1].lower(),
                optimizer_kwargs=dict(lr=cfg.trainer.learning_rate),
                loss_reduction=cfg.model.vae.loss_reduction,
                beta=cfg.model.vae.beta,
            ))
    else:
        raise NotImplementedError(f"Don't know how to build disent model: {cfg.model.name}")
