import pytest

import numpy
import gym

from disent.dataset.data import GymEnvData
from disent.dataset import DisentDataset

from disent.dataset.sampling import SingleSampler
from disent.metrics import metric_dci

from visgrid.envs.components import Grid
from visgrid.envs import GridworldEnv
from visgrid.wrappers import NormalizedFloatWrapper, NoiseWrapper

from factored_rl.wrappers import ObservationPermutationWrapper, FactorPermutationWrapper

@pytest.fixture
def ring_env_data():
    grid = Grid.generate_ring(10, 10, 8)
    env = GridworldEnv.from_grid(grid, hidden_goal=True, should_render=False)
    data = GymEnvData(env)
    return data

def test_sample_single_valid_position(ring_env_data):
    pos = ring_env_data.sample_factors()
    assert ring_env_data.env.is_valid_pos(pos)

def test_sample_multiple_valid_positions(ring_env_data):
    positions = ring_env_data.sample_factors(size=100)
    for pos in positions:
        assert ring_env_data.env.is_valid_pos(pos)

@pytest.fixture
def unwrapped_env_data():
    env = GridworldEnv(10, 10, hidden_goal=True, should_render=False)
    data = GymEnvData(env)
    return data

def test_unwrapped_shapes(unwrapped_env_data):
    state = unwrapped_env_data.sample_factors()
    idx = unwrapped_env_data.pos_to_idx(state)
    obs = unwrapped_env_data._get_observation(idx)
    assert obs.shape == (2, )

@pytest.fixture
def wrapped_env_data():
    env = GridworldEnv(10, 10, hidden_goal=True, should_render=False)
    env = ObservationPermutationWrapper(env)
    env = NormalizedFloatWrapper(env)
    env = NoiseWrapper(env)
    data = GymEnvData(env)
    return data

def test_wrapped_shapes(wrapped_env_data):
    state = wrapped_env_data.sample_factors()
    idx = wrapped_env_data.pos_to_idx(state)
    obs = wrapped_env_data._get_observation(idx)
    assert obs.shape == (2, )

def test_compute_dci(unwrapped_env_data, wrapped_env_data):
    def get_dci(data):
        dataset = DisentDataset(dataset=data, sampler=SingleSampler())
        return metric_dci(dataset, lambda x: x, num_train=100, num_test=50)['dci.disentanglement']

    disentangled_dci_score = get_dci(unwrapped_env_data)
    entangled_dci_score = get_dci(wrapped_env_data)
    assert entangled_dci_score < disentangled_dci_score
    print(f'disentangled: {disentangled_dci_score}')
    print(f'entangled: {entangled_dci_score}')

def test_gymenvdata_protocol_checking():
    class FakeGymEnv():
        pass

    with pytest.raises(AssertionError):
        GymEnvData(FakeGymEnv())

    with pytest.raises(AssertionError):
        GymEnvData(gym.make('CartPole-v1'))
