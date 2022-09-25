import pytest

import numpy as np
import gym
import torch
from torch.utils.data.dataloader import DataLoader

from disent.dataset.data import GymEnvData
from disent.dataset import DisentDataset

from disent.dataset.sampling import SingleSampler
from disent.metrics import metric_dci, metric_mig

from gym.wrappers.flatten_observation import FlattenObservation
from visgrid.envs.components import Grid
from visgrid.envs import GridworldEnv
from visgrid.wrappers import ToFloatWrapper, NormalizeWrapper, NoiseWrapper, wrap_gridworld

#%%
@pytest.fixture
def ring_env_data():
    grid = Grid.generate_ring(10, 10, 8)
    env = GridworldEnv.from_grid(grid, hidden_goal=True, should_render=False)
    data = GymEnvData(env)
    return data

def test_sample_single_valid_position(ring_env_data):
    pos = ring_env_data.sample_state()
    assert ring_env_data.env.unwrapped.is_valid_pos(pos)

def test_sample_multiple_valid_positions(ring_env_data):
    positions = ring_env_data.get_batch(batch_size=100)[-1]
    for pos in positions:
        assert ring_env_data.env.unwrapped.is_valid_pos(pos)

@pytest.fixture
def unwrapped_env_data():
    env = GridworldEnv(10, 10, hidden_goal=True, should_render=False)
    data = GymEnvData(env)
    return data

def test_sample_with_iter(ring_env_data):
    ob = next(ring_env_data)
    assert ob.shape == (2, )

@pytest.fixture
def wrapped_env_data():
    env = GridworldEnv(6,
                       6,
                       hidden_goal=True,
                       should_render=True,
                       dimensions=GridworldEnv.dimensions_6x6_to_18x18)
    env = wrap_gridworld(env)
    data = GymEnvData(env)
    return data

def test_wrapped_shapes(wrapped_env_data):
    ob = next(wrapped_env_data)
    assert ob.shape == (18, 18)

@pytest.fixture
def flattened_env_data():
    env = GridworldEnv(6,
                       6,
                       hidden_goal=True,
                       should_render=True,
                       dimensions=GridworldEnv.dimensions_6x6_to_18x18)
    env = wrap_gridworld(env)
    env = FlattenObservation(env)
    data = GymEnvData(env)
    return data

def test_compute_dci(unwrapped_env_data, flattened_env_data):
    def get_dci(data):
        dataset = DisentDataset(dataset=data, sampler=SingleSampler())
        return metric_dci(dataset, lambda x: x, num_train=100, num_test=50)['dci.disentanglement']

    disentangled_dci_score = get_dci(unwrapped_env_data)
    entangled_dci_score = get_dci(flattened_env_data)
    assert entangled_dci_score < disentangled_dci_score
    print(f'disentangled: {disentangled_dci_score}')
    print(f'entangled: {entangled_dci_score}')

def test_compute_mig(unwrapped_env_data, flattened_env_data):
    def get_mig(data):
        dataset = DisentDataset(dataset=data, sampler=SingleSampler())
        return metric_mig(dataset, lambda x: x, num_train=100)['mig.discrete_score']

    disentangled_mig_score = get_mig(unwrapped_env_data)
    entangled_mig_score = get_mig(flattened_env_data)
    assert entangled_mig_score < disentangled_mig_score
    print(f'disentangled: {disentangled_mig_score}')
    print(f'entangled: {entangled_mig_score}')

def test_gymenvdata_protocol_checking():
    class FakeGymEnv():
        pass

    with pytest.raises(AssertionError):
        GymEnvData(FakeGymEnv())

    with pytest.raises(AttributeError):
        GymEnvData(gym.make('CartPole-v1'))

def test_multiple_dataloader_workers(ring_env_data):
    dl_iter = iter(
        DataLoader(ring_env_data,
                   batch_size=20,
                   num_workers=2,
                   worker_init_fn=ring_env_data.get_worker_init_fn()))
    worker1_samples = next(dl_iter)
    worker2_samples = next(dl_iter)
    assert not (worker1_samples == worker2_samples).all()
