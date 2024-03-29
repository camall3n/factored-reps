import pytest

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils import cleanup, get_config
from disent.dataset import DisentDataset, DisentIterDataset
from disent.dataset.data import GymEnvData, TaxiData64x64
from disent.dataset.transform import ToImgTensorF32
from factored_rl.experiments.common import cpu_count, initialize_env

cleanup()

@pytest.fixture
def map_data():
    return TaxiData64x64()

@pytest.fixture
def map_dataset(map_data):
    return DisentDataset(dataset=map_data, transform=ToImgTensorF32())

@pytest.fixture
def map_dataloader(map_dataset):
    return DataLoader(dataset=map_dataset, batch_size=4, shuffle=True, num_workers=0)

@pytest.fixture
def cfg():
    return get_config([
        "script=disent",
        "experiment=pytest",
        "seed=0",
        "model=cnn_64",
        "env=taxi",
        "env.depot_dropoff_only=false",
        "trainer=rep.quick",
        "trainer.batch_size=4",
        "transform=images",
    ])

@pytest.fixture
def iter_data(cfg):
    env = initialize_env(cfg, cfg.seed)
    return GymEnvData(env, cfg.seed)

@pytest.fixture
def iter_dataset(iter_data):
    return DisentIterDataset(dataset=iter_data, transform=None)

@pytest.fixture
def iter_dataloader(iter_dataset, cfg):
    return DataLoader(dataset=iter_dataset,
                      batch_size=cfg.trainer.batch_size,
                      num_workers=0 if cfg.trainer.quick else cpu_count(),
                      persistent_workers=False if cfg.trainer.quick else True)

def test_env_observation_shapes(map_data, iter_data):
    map_ob = map_data.env.reset()[0]
    iter_ob = iter_data.env.reset()[0]
    assert np.moveaxis(map_ob, -1, -3).shape == iter_ob.shape

def test_data_shapes(map_data, iter_data):
    map_item = next(iter(map_data))
    iter_item = next(iter(iter_data))
    assert type(map_item) == type(iter_item)
    assert np.moveaxis(map_item, -1, -3).shape == iter_item.shape

def test_dataset_shapes(map_dataset, iter_dataset):
    map_dict = next(iter(map_dataset))
    iter_dict = next(iter(iter_dataset))
    assert type(map_dict) == type(iter_dict)
    assert type(map_dict['x_targ']) == type(iter_dict['x_targ'])
    assert len(map_dict['x_targ']) == len(iter_dict['x_targ'])
    assert isinstance(map_dict['x_targ'][0], torch.Tensor)
    assert isinstance(iter_dict['x_targ'][0], np.ndarray)
    assert map_dict['x_targ'][0].shape == iter_dict['x_targ'][0].shape

def test_dataloader_batch_shapes(map_dataloader, iter_dataloader):
    map_batch = next(iter(map_dataloader))
    iter_batch = next(iter(iter_dataloader))
    assert type(map_batch) == type(iter_batch)
    assert type(map_batch['x_targ']) == type(iter_batch['x_targ'])
    assert len(map_batch['x_targ']) == len(iter_batch['x_targ'])
    assert type(map_batch['x_targ'][0]) == type(iter_batch['x_targ'][0])
    assert map_batch['x_targ'][0].shape == iter_batch['x_targ'][0].shape

@pytest.fixture
def obs_data(cfg):
    env = initialize_env(cfg, cfg.seed)
    return GymEnvData(env, cfg.seed, action_sampling=None)

@pytest.fixture
def transition_data(cfg):
    env = initialize_env(cfg, cfg.seed)
    return GymEnvData(env, cfg.seed, action_sampling='all')

def test_obs_vs_transitions(obs_data, transition_data):
    obs_item = next(obs_data)
    transition_item = next(transition_data)
    assert type(obs_item) == type(transition_item['ob'])
    assert obs_item.shape == transition_item['ob'].shape
    assert (obs_item == transition_item['ob']).all()
