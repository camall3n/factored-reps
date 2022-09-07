import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import seeding
import torch
from tqdm import tqdm

from ....models.markov.featurenet import FeatureNet
from ....models.markov.autoencoder import AutoEncoder
from visgrid.envs import GridworldEnv
from visgrid.envs.components import Grid
from visgrid.wrappers.sensors import *

class Args:
    pass

args = Args()

args.rows = 6
args.cols = 6
args.walls = 'maze'

data = []
for seed in tqdm(range(1, 301)):

    seeding.seed(seed, np, torch)
    if args.walls == 'maze':
        env = GridworldEnv.from_saved_maze(rows=args.rows, cols=args.cols, seed=args.seed)
    else:
        env = GridworldEnv(rows=args.rows, cols=args.cols)
        if args.walls == 'spiral':
            env.grid = Grid.generate_spiral(rows=args.rows, cols=args.cols)
        elif args.walls == 'loop':
            env.grid = Grid.generate_spiral_with_shortcut(rows=args.rows, cols=args.cols)

    #% ------------------ Generate experiences ------------------
    n_samples = 20000
    states = [env.get_state()]
    actions = []
    for t in range(n_samples):
        a = np.random.choice(env.action_space)
        s, _, _ = env.step(a)
        states.append(s)
        actions.append(a)
    states = np.stack(states)
    s0 = np.asarray(states[:-1, :])
    c0 = s0[:, 0] * env.cols + s0[:, 1]
    s1 = np.asarray(states[1:, :])
    a = np.asarray(actions)

    _, state_counts = np.unique(states, axis=0, return_counts=True)
    seed_data = pd.DataFrame(state_counts, columns=['counts'])
    seed_data['seed'] = seed
    data.append(seed_data)

data = pd.concat(data)
sns.boxplot(data=data, x='seed', y='counts')
plt.ylim([0, 1200])
plt.show()
plt.figure()
plt.hist(data['counts'], bins=100)
plt.show()
