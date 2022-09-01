from argparse import Namespace
# import glob
# import gym
# import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm

from visgrid.taxi.taxi import VisTaxi5x5

args = Namespace
args.n_samples = 1000
args.n_steps_per_episode = 1

env = VisTaxi5x5(grayscale=False)

ds_list = []

done = True
ep_steps = 0
for i in tqdm(range(args.n_samples)):
    if done or ep_steps >= args.n_steps_per_episode:
        env.reset(goal=True, explore=True)
        state = env.get_state()
        ep_steps = 0

    a = random.choice(env.actions)
    next_ob, reward, done = env.step(a)
    next_state = env.get_state()
    ep_steps += 1

    ds = next_state - state
    ds_list.append(ds)

    state = next_state.copy()

#%% ------------------ Plot ds vs. ds correlation ------------------

s_deltas = np.stack(ds_list, axis=1)
z_deltas = s_deltas

n_factors = len(state)
n_vars = n_factors

all_deltas = np.concatenate((z_deltas, s_deltas))
correlation = np.corrcoef(all_deltas)[:n_vars, -n_factors:]
plt.imshow(correlation, vmin=-1, vmax=1)
for i in range(n_vars):
    for j in range(n_factors):
        text = plt.gca().text(j, i, np.round(correlation[i, j], 2),
                       ha="center", va="center", color="w")
plt.yticks(np.arange(n_vars))
plt.xticks(np.arange(n_factors))
plt.ylabel(r'Ground truth factor ($\Delta s$)')
plt.xlabel(r'Ground truth factor ($\Delta s$)')
plt.title('Correlation coefficients')
plt.colorbar()
images_dir = 'results/focused-taxi/images/'
os.makedirs(images_dir, exist_ok=True)
plt.savefig(images_dir + 'ground-truth-correlation-plot.png', facecolor='white')
plt.show()
