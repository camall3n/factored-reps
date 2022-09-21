from multiprocessing import freeze_support

# Args & hyperparams
import hydra
from factored_rl.experiments import configs

# Data
from factored_rl.experiments.common import initialize_env
from disent.dataset.data import GymEnvData
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Model
from disent.frameworks.vae import BetaVae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64
from disent.model.ae import EncoderConv64

# ----------------------------------------
# Args & hyperparams
# ----------------------------------------

@hydra.main(config_path="../conf", config_name='factorize', version_base=None)
def main(cfg: configs.FactorizeConfig):
    configs.initialize_experiment(cfg, 'factorize')

    env, train_dl = initialize_dataloader(cfg, cfg.seed)
    _, val_dl = initialize_dataloader(cfg, cfg.seed + 1000000)

    model = initialize_model(env, cfg)
    trainer = pl.Trainer(max_steps=cfg.trainer.max_steps,
                         overfit_batches=cfg.trainer.overfit_batches,
                         gpus=(1 if cfg.model.device.type == 'cuda' else 0))
    trainer.fit(model, train_dl, val_dl)

def initialize_dataloader(env, cfg: configs.TrainingConfig, seed: int = None):
    env = initialize_env(cfg, seed)
    dataset = GymEnvData(env, seed)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=cfg.training.batch_size,
                            num_workers=2,
                            persistent_workers=True)
    return env, dataloader

def initialize_model(env, cfg):
    if cfg.model == 'betavae':
        # create the BetaVAE model
        # - adjusting the beta, learning rate, and representation size.
        shape = env.observation_space.shape
        module = BetaVae(
            model=AutoEncoder(
                # z_multiplier is needed to output mu & logvar when parameterising normal distribution
                encoder=EncoderConv64(x_shape=shape, z_size=10, z_multiplier=2),
                decoder=DecoderConv64(x_shape=shape, z_size=10),
            ),
            cfg=BetaVae.cfg(
                optimizer=cfg.trainer.optimizer._target_.split('.')[-1].lower(),
                optimizer_kwargs=dict(lr=cfg.trainer.learning_rate),
                loss_reduction='mean_sum',
                beta=4,
            ))
    return module

if __name__ == '__main__':
    freeze_support() # do this to make sure multiprocessing works correctly
    main()
