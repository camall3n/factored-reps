from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pickle
import seaborn as sns
import torch

from factored_rl import configs
from factored_rl.test.utils import get_config
from visgrid.envs.gridworld import GridworldEnv
from factored_rl.experiments.common import initialize_env, initialize_model
from factored_rl.experiments.factorize.run import initialize_dataloader

# Env
import gym
from gym.wrappers import FlattenObservation
from visgrid.envs import GridworldEnv, TaxiEnv
from factored_rl.wrappers import RotationWrapper, FactorPermutationWrapper, ObservationPermutationWrapper
from factored_rl.wrappers import MoveAxisToCHW, PolynomialBasisWrapper, FourierBasisWrapper, LegendreBasisWrapper
from visgrid.wrappers import GrayscaleWrapper, InvertWrapper, ToFloatWrapper, NormalizeWrapper, NoiseWrapper, ClipWrapper, GaussianBlurWrapper, wrap_gridworld

#%%
cfg = get_config([
    "experiment=pytest",
    "timestamp=false",
    "env=gridworld",
    # "env.grayscale=true",
    "transform=images",
    "model=ae/ae_mlp",
    "trainer=rep.quick",
    "model.n_latent_dims=2",
    "model.mlp.n_units_per_layer=32",
])
cfg.script = 'disent_vs_rep'
cfg.model.device = 'cpu'

env = GridworldEnv(6, 6, exploring_starts=cfg.env.exploring_starts, terminate_on_goal=True, fixed_goal=True, hidden_goal=True, should_render=True, dimensions=GridworldEnv.dimensions_6x6_to_18x18)
env = wrap_gridworld(env)
env = ToFloatWrapper(env)
env.reset(seed=cfg.seed)
env.action_space.seed(cfg.seed)

#%%
load_dir = '/Users/cam/dev/test-markov-abstr/markov_abstr/gridworld/results/models/markov-3k/'
model = initialize_model((324,), 4, cfg)

model_weights = torch.load(load_dir + 'phi-1_latest.pytorch')
new_state_dict = OrderedDict()
for key, value in model_weights.items():
    new_key = {
        "phi.1.weight": "model.1.model.0.weight",
        "phi.1.bias": "model.1.model.0.bias",
        "phi.3.weight": "model.1.model.2.weight",
        "phi.3.bias": "model.1.model.2.bias",
    }[key]
    new_state_dict[new_key] = value
model.encoder.load_state_dict(new_state_dict)

#%%
def get_obs(s):
    obs = env.get_observation(s)
    wrapper_list = []
    env_iter = env
    while hasattr(env_iter, 'observation'):
        wrapper_list.append(env_iter)
        env_iter = env_iter.env
    for wrapper in reversed(wrapper_list):
        obs = wrapper.observation(obs)
    return obs

def encode(obs: np.ndarray):
    with torch.no_grad():
        return model.encode(torch.asarray(obs))

def decode(z: np.ndarray):
    with torch.no_grad():
        return model.decode(torch.asarray(z))

def plot_obs(obs):
    if obs.ndim == 4:
        assert obs.shape[0] == 1
        return plot_obs(obs[0])
    plt.imshow(np.moveaxis(obs, 0, -1))
#%%
env.reset()
s = env.get_state()
obs, _ = env.reset()
obs = get_obs(s)
plot_obs(obs)
obs.shape
#%%
data = []
for agent_row in range(5):
    for agent_col in range(5):
        for obs_realization in range(10):
            data_item = {
                'agent_row': agent_row,
                'agent_col': agent_col,
                'obs_realization': obs_realization
            }
            s = np.asarray(list(data_item.values())[:-1])
            obs = np.expand_dims(get_obs(s), 0)
            z = encode(obs)[0]
            for i, _ in enumerate(z):
                data_item[f'z_{i}'] = z[i].item()
            data.append(data_item)
df = pd.DataFrame(data)

#%%
#%%
g = sns.PairGrid(df)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3, legend=False)

#%%
x_vars = [col for col in df.columns if 'z' in col and 'obs' not in col]
y_vars = [col for col in df.columns if 'z' not in col]
g = sns.PairGrid(df, x_vars=x_vars, y_vars=y_vars, hue='in_taxi')
g.map(sns.scatterplot)

#%%
x_vars = [col for col in df.columns if 'z' in col and 'obs' not in col]
y_vars = [col for col in df.columns if 'z' not in col]
g = sns.PairGrid(df, x_vars=x_vars, y_vars=y_vars)
g.map(sns.kdeplot)


#%%
x_vars = [col for col in df.columns if 'z' in col and 'obs' not in col]
y_vars = [col for col in df.columns if 'z' not in col]
g = sns.PairGrid(df, x_vars=x_vars, y_vars=y_vars)
g.map(sns.violinplot, orient='h', split=True, color='C0')
