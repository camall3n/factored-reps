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
data = next(iter(dl))

for _ in range(100):
    s, info = env.reset()
    a = env.action_space.sample()
    sp, _, _, _, info = env.step(a)
    for
data.keys()

data['state'][0]
