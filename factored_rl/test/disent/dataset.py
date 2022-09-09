import pytest

import numpy

from disent.dataset.data import GymEnvData
from disent.dataset import DisentDataset
from disent.frameworks.ae import Ae
from disent.model.ae import DecoderIdentity, EncoderIdentity

from disent.model import AutoEncoder
from disent.dataset.sampling import SingleSampler
from disent.dataset.transform import ToImgTensorF32
from disent.metrics import metric_mig

from visgrid.envs.components import Grid
from visgrid.envs import GridworldEnv
from visgrid.wrappers import NormalizedFloatWrapper, NoiseWrapper, ImageFrom1DWrapper

from factored_rl.wrappers import FactorPermutationWrapper

@pytest.fixture
def unwrapped_env_data():
    grid = Grid.generate_ring(10, 10, 8)
    env = GridworldEnv.from_grid(grid, hidden_goal=True, should_render=False)
    env = ImageFrom1DWrapper(env)
    data = GymEnvData(env)
    return data

def test_unwrapped_shapes(unwrapped_env_data):
    state = unwrapped_env_data.sample_factors()
    idx = unwrapped_env_data.pos_to_idx(state)
    obs = unwrapped_env_data._get_observation(idx)
    assert obs.shape == (2, 1, 1)

def test_sample_single_valid_position(unwrapped_env_data):
    pos = unwrapped_env_data.sample_factors()
    assert unwrapped_env_data.env.is_valid_pos(pos)

def test_sample_multiple_valid_positions(unwrapped_env_data):
    positions = unwrapped_env_data.sample_factors(size=100)
    for pos in positions:
        assert unwrapped_env_data.env.is_valid_pos(pos)

@pytest.fixture
def wrapped_env_data():
    grid = Grid.generate_ring(10, 10, 8)
    env = GridworldEnv.from_grid(grid, hidden_goal=True, should_render=False)
    env = FactorPermutationWrapper(env)
    env = NormalizedFloatWrapper(env)
    env = NoiseWrapper(env)
    env = ImageFrom1DWrapper(env)
    data = GymEnvData(env)
    return data

def test_wrapped_shapes(wrapped_env_data):
    state = wrapped_env_data.sample_factors()
    idx = wrapped_env_data.pos_to_idx(state)
    obs = wrapped_env_data._get_observation(idx)
    assert obs.shape == (2, 1, 1)

def test_compute_mig(unwrapped_env_data, wrapped_env_data):
    def get_mig(data):
        module = Ae(model=AutoEncoder(
            encoder=EncoderIdentity(x_shape=data.x_shape),
            decoder=DecoderIdentity(x_shape=data.x_shape),
        ))
        dataset = DisentDataset(dataset=data, sampler=SingleSampler(), transform=ToImgTensorF32())
        return metric_mig(dataset, lambda x: module.encode(x), num_train=200)['mig.discrete_score']

    disentangled_mig_score = get_mig(unwrapped_env_data)
    entangled_mig_score = get_mig(wrapped_env_data)
    assert entangled_mig_score < disentangled_mig_score
