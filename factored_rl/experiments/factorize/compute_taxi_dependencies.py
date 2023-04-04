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

#%%
cfg = get_config([
    "experiment=pytest",
    "timestamp=false",
    "env=taxi",
    "env.depot_dropoff_only=false",
    "transform=identity",
    "transform.noise=false",
])
env = initialize_env(cfg, cfg.seed).unwrapped

for _ in range(100):
    state, info = env.reset()
    action = env.action_space.sample()
    next_state, _, _, _, info = env.step(action)
    experience = {
        'state': state,
        'action': action,
        'next_state': next_state,
        'effect': next_state - state,
    }
    experience.update({f"s_{i}": s for i, s in enumerate(state)})
    experience.update({f"s'_{i}": sp for i, sp in enumerate(next_state)})
    experience


data['state'][0]
