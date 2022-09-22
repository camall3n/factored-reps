import pytest
from multiprocessing import freeze_support

# Args & hyperparams
import hydra
from omegaconf import OmegaConf

# Data
from factored_rl.experiments.common import initialize_env
from disent.dataset.data import GymEnvData
from disent.dataset import DisentIterDataset
from disent.dataset.sampling import SingleSampler
from disent.dataset.transform import ToImgTensorF32
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Training
from factored_rl.experiments.common import cpu_count

from multiprocessing import freeze_support

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from disent.dataset import DisentDataset
from disent.dataset.data import TaxiData64x64
from disent.dataset.sampling import SingleSampler
from disent.dataset.transform import ToImgTensorF32
from disent.frameworks.vae import BetaVae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64
from disent.model.ae import EncoderConv64

#%%

@pytest.fixture
def map_data():
    return TaxiData64x64()

@pytest.fixture
def map_dataset(map_data):
    return DisentDataset(dataset=map_data, transform=ToImgTensorF32())

@pytest.fixture
def map_dataloader(map_dataset):
    return DataLoader(dataset=map_dataset, batch_size=128, shuffle=True, num_workers=0)

@pytest.fixture
def cfg():
    return OmegaConf.create(
    """
    seed: 0
    model:
      architecture: 'cnn'
    env:
      name: taxi
      n_steps_per_episode: 50
      exploring_starts: true
      fixed_goal: false
      grayscale: false
      depot_dropoff_only: true
    trainer:
      batch_size: 128
      quick: false
      num_dataloader_workers: 0
    transform:
      name: images
      noise: true
      noise_std: 0.01
    """)

@pytest.fixture
def iter_data(cfg):
    env = initialize_env(cfg, cfg.seed)
    return GymEnvData(env, cfg.seed)

@pytest.fixture
def iter_dataset(iter_data):
    return DisentIterDataset(iter_data, transform=ToImgTensorF32())

@pytest.fixture
def iter_dataloader(iter_dataset, cfg):
    return DataLoader(dataset=iter_dataset,
                      batch_size=cfg.trainer.batch_size,
                      num_workers=0 if cfg.trainer.quick else cpu_count(),
                      persistent_workers=False if cfg.trainer.quick else True)

def test_env_observation_shapes(map_data, iter_data):
    map_ob = map_data.env.reset()[0]
    iter_ob = iter_data.env.reset()[0]
    assert map_ob.shape == iter_ob.shape

def test_data_shapes(map_data, iter_data):
    map_item = next(iter(map_data))
    iter_item = next(iter(iter_data))
    assert type(map_item) == type(iter_item)
    assert map_item.shape == iter_item.shape

def test_dataset_shapes(map_dataset, iter_dataset):
    map_dict = next(iter(map_dataset))
    iter_dict = next(iter(iter_dataset))
    assert type(map_dict) == type(iter_dict)
    assert type(map_dict['x_targ']) == type(iter_dict['x_targ'])
    assert len(map_dict['x_targ']) == len(iter_dict['x_targ'])
    assert type(map_dict['x_targ'][0]) == type(iter_dict['x_targ'][0])
    assert map_dict['x_targ'][0].shape == iter_dict['x_targ'][0].shape

def test_dataloader_batch_shapes(map_dataloader, iter_dataloader):
    map_batch = next(iter(map_dataloader))
    iter_batch = next(iter(iter_dataloader))
    assert type(map_batch) == type(iter_batch)
    assert type(map_batch['x_targ']) == type(iter_batch['x_targ'])
    assert len(map_batch['x_targ']) == len(iter_batch['x_targ'])
    assert type(map_batch['x_targ'][0]) == type(iter_batch['x_targ'][0])
    assert map_batch['x_targ'][0].shape == iter_batch['x_targ'][0].shape
