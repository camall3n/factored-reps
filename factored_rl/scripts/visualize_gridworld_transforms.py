import random
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seeding
import seaborn as sns
from tqdm import tqdm

from visgrid.envs import GridworldEnv
from visgrid.wrappers import *
from factored_rl.wrappers import *

def get_rgb_for_gridworld_states(states):
    states = states.astype(float)
    states -= states.min()
    states /= states.max()
    states = 2 * states - 1
    hue = np.arctan2(states[:, 1], states[:, 0]) / (2 * np.pi) + 0.5
    sat = np.linalg.norm(states, axis=-1)
    sat /= np.max(sat)
    sat = 0.15 + 0.85 * sat
    val = 0.8 * np.ones_like(hue)
    hsv = np.stack((hue, sat, val), axis=1)
    rgb = mpl.colors.hsv_to_rgb(hsv)
    return rgb

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for transform, ax in zip(['identity', 'rotate', 'permute_factors', 'permute_states'], axes):
    env = GridworldEnv(10,
                       10,
                       hidden_goal=True,
                       goal_position=(0, 0),
                       should_render=False,
                       dimensions=GridworldEnv.dimensions_onehot)

    if transform == 'permute_states':
        env = ObservationPermutationWrapper(env)
    elif transform == 'permute_factors':
        env = FactorPermutationWrapper(env)
    env = NormalizeWrapper(FloatWrapper(env), -1, 1)
    if transform == 'rotate':
        env = TransformWrapper(RotationWrapper(env), lambda x: x / np.sqrt(2))
    env = ClipWrapper(NoiseWrapper(env, sigma=0.01), -1, 1)
    obs = []
    states = []
    for _ in range(10000):
        ob, _ = env.reset()
        state = env.get_state()
        obs.append(ob)
        states.append(state)
    # env.plot(ob)
    # print(ob.shape)
    obs = np.stack(obs)
    states = np.stack(states)

    ax.scatter(x=obs[:, 0], y=obs[:, 1], c=get_rgb_for_gridworld_states(states))
    ax.set_aspect(1)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{transform}')

plt.tight_layout()
plt.savefig('images/disent_vs_rep/transforms.png', facecolor='white')
plt.show()
