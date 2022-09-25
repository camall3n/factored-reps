from multiprocessing import freeze_support

# Args & hyperparams
import hydra
from factored_rl.experiments import configs

# Data
from factored_rl.experiments.common import initialize_env, initialize_model
from disent.dataset.data import GymEnvData
from disent.dataset import DisentIterDataset
from disent.dataset.transform import ToImgTensorF32
from torch.utils.data import DataLoader

# Training
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
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
    # ckpt_callback = ModelCheckpoint(save_last=True)
    trainer = pl.Trainer(
        max_steps=cfg.trainer.max_steps,
        overfit_batches=cfg.trainer.overfit_batches,
        gpus=(1 if cfg.model.device == 'cuda' else 0),
        default_root_dir=cfg.dir,
        # checkpoint_callback=ckpt_callback,
    )
    trainer.fit(model, train_dl, val_dl)

def initialize_dataloader(cfg: configs.Config, seed: int = None):
    env = initialize_env(cfg, seed)
    dataset = GymEnvData(env, seed, transform=ToImgTensorF32())
    input_shape = dataset.x_shape
    if cfg.model.lib == 'disent':
        dataset = DisentIterDataset(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.trainer.batch_size,
        num_workers=0 if cfg.trainer.quick else cpu_count(),
        persistent_workers=False if cfg.trainer.quick else True,
        worker_init_fn=dataset.worker_init_fn,
    )
    return dataloader, input_shape

if __name__ == '__main__':
    freeze_support() # do this to make sure multiprocessing works correctly
    main()