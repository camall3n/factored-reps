from multiprocessing import freeze_support

# Args & hyperparams
import hydra
from factored_rl.experiments import configs

# Data
from factored_rl.experiments.common import initialize_env
from disent.dataset.data import GymEnvData
from disent.dataset import DisentIterDataset
from disent.dataset.transform import ToImgTensorF32
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Model
from disent.frameworks.vae import BetaVae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64
from disent.model.ae import EncoderConv64

# Training
from factored_rl.experiments.common import cpu_count

# ----------------------------------------
# Args & hyperparams
# ----------------------------------------

@hydra.main(config_path="../conf", config_name='config', version_base=None)
def main(cfg: configs.Config):
    configs.initialize_experiment(cfg, 'factorize')

    train_dl, input_shape = initialize_dataloader(cfg, cfg.seed)
    val_dl, _ = initialize_dataloader(cfg, cfg.seed + 1000000)

    model = initialize_model(input_shape, cfg)
    trainer = pl.Trainer(max_steps=cfg.trainer.max_steps,
                         overfit_batches=cfg.trainer.overfit_batches,
                         gpus=(1 if cfg.model.device == 'cuda' else 0))
    trainer.fit(model, train_dl, val_dl)

def initialize_dataloader(cfg: configs.TrainerConfig, seed: int = None):
    env = initialize_env(cfg, seed)
    data = GymEnvData(env, seed, transform=ToImgTensorF32())
    dataset = DisentIterDataset(data)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=cfg.trainer.batch_size,
                            num_workers=0 if cfg.trainer.quick else cpu_count(),
                            persistent_workers=False if cfg.trainer.quick else True)
    return dataloader, data.x_shape

def initialize_model(input_shape, cfg):
    if cfg.model.name == 'vae_64':
        # create the BetaVAE model
        # - adjusting the beta, learning rate, and representation size.
        module = BetaVae(
            model=AutoEncoder(
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
    return module

if __name__ == '__main__':
    freeze_support() # do this to make sure multiprocessing works correctly
    main()
