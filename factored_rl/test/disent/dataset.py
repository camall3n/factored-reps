import pytest

from disent.dataset.data import GymEnvData
from visgrid.envs.components import Grid
from visgrid.envs import GridworldEnv
from factored_rl.wrappers import FactorPermutationWrapper
from visgrid.wrappers import NormalizedFloatWrapper, NoiseWrapper

@pytest.fixture
def unwrapped_env_data():
    grid = Grid.generate_ring(10, 10, 8)
    env = GridworldEnv.from_grid(grid, hidden_goal=True, should_render=False)
    data = GymEnvData(env)
    return data

def test_unwrapped_shapes(unwrapped_env_data):
    state = unwrapped_env_data.sample_factors()
    idx = unwrapped_env_data.pos_to_idx(state)
    obs = unwrapped_env_data._get_observation(idx)
    assert len(obs) == 2

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
    data = GymEnvData(env)
    return data

def test_wrapped_shapes(wrapped_env_data):
    state = wrapped_env_data.sample_factors()
    idx = wrapped_env_data.pos_to_idx(state)
    obs = wrapped_env_data._get_observation(idx)
    assert len(obs) == 2
