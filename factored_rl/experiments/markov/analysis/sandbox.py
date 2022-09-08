import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import seaborn as sns
import seeding
import torch
from tqdm import tqdm

from visgrid.envs import GridworldEnv
from visgrid.wrappers.transforms import wrap_gridworld, NoiseWrapper

p = sns.color_palette(n_colors=6)
p[-2]

seeding.seed(1, np, random, torch)
env = GridworldEnv.from_saved_maze(rows=6, cols=6, seed=1)
env = GridworldEnv(rows=10,
                   cols=10,
                   agent_position=(0, 0),
                   goal_position=(3, 3),
                   should_render=False,
                   hidden_goal=True)
env = NoiseWrapper(env, sigma=0.2, truncation=0.4)
env.plot()
env.reset()

n_samples = 20000
states = [env.get_state()]
for t in range(n_samples):
    a = np.random.choice(env.action_space)
    s, _, _ = env.step(a)
    states.append(s)
states = np.stack(states)
s0 = np.asarray(states[:-1, :])
c0 = s0[:, 0] * env.cols + s0[:, 1]
s1 = np.asarray(states[1:, :])
x0 = sensor(s0)
x1 = sensor(s1)

ax = env.plot()
xx = x0[:, 1] + 0.5
yy = x0[:, 0] + 0.5
ax.scatter(xx, yy, c=c0, alpha=0.05)
plt.savefig('foo.png', facecolor='white')

#%%
env = GridworldEnv(rows=10,
                   cols=10,
                   agent_position=(0, 0),
                   goal_position=(3, 3),
                   should_render=True,
                   hidden_goal=True,
                   dimensions=GridworldEnv.dimensions_6x6_to_18x18)
env = wrap_gridworld(env)
seeding.seed(0, np, random, torch)

x = env.reset()
s = env.get_state()
imgs = []
for i in range(20):
    x = sensor(s)
    fig, ax = plt.subplots()
    fig.set_figwidth(fig.get_figheight())
    ax.imshow(x)
    ax.axis('off')
    fig.tight_layout()
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    imgs.append(img)
import imageio

imageio.mimwrite('x1-above-corner-distribution.mp4', imgs)
plt.imshow(x)
plt.axis('off')
plt.savefig('x1-above-corner.png')
#%%

x = env.step(3)[0]
s = env.get_state()
plt.imshow(x)
plt.axis('off')
plt.savefig('x0-at-corner.png')

#%%
env = GridworldEnv(6,
                   6,
                   should_render=True,
                   hidden_goal=True,
                   dimensions=GridworldEnv.dimensions_6x6_to_18x18)
env = wrap_gridworld(env)
x = env.reset()[0]
# print(x)
plt.imshow(x)

n_pixels = len(x.flatten())

#%%

n_steps = 1000

states = []
actions = []
observations = []

for t in tqdm(range(n_steps)):
    s = env.get_state()
    states.append(s)
    observations.append(x)

    a = np.random.randint(0, 4)
    x = env.step(a)[0]
    actions.append(a)

states.append(env.get_state())
observations.append(x)

next_states = states[1:]
states = states[:-1]

next_observations = observations[1:]
observations = observations[:-1]

experiences = list(zip(states, observations, actions, next_states, next_observations))

#%%
n_dims = 3
random_matrix = np.random.randn(n_pixels * 2, n_dims)

state_pairs = np.concatenate(list(zip(*[(s, sp) for (s, _, _, sp, _) in experiences])), axis=1)
obs_pairs = np.concatenate(list(zip(*[(x, xp) for (_, x, _, _, xp) in experiences])), axis=1)

pair_colors = np.unique(state_pairs, return_inverse=True, axis=0)[1]
n_colors = len(np.unique(state_pairs, axis=0))

pair_embeddings = np.matmul(obs_pairs, random_matrix)
obs_pair_distance = np.linalg.norm(np.stack(observations) - np.stack(next_observations), 2, axis=1)
state_pair_distance = np.linalg.norm(np.stack(states) - np.stack(next_states), 1, axis=1)

#%%
sns.boxenplot(x=state_pair_distance, y=obs_pair_distance)
sns.despine()
ax = plt.gca()
ax.set_xlabel('Graph distance')
ax.set_ylabel('L2 distance')
# ax.set_xticks([])
plt.show()

#%%

# x, y, z = zip(*pair_embeddings)
#
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(x,y,z, c=pair_colors)
# plt.show()
#
# assert False, 'Force quit'

#%%
n_observations = n_steps
D_states = np.zeros((n_observations, n_observations))
D_embeddings = np.zeros((n_observations, n_observations))
D_observations = np.zeros((n_observations, n_observations))
for row in range(n_observations):
    for col in range(n_observations):
        D_states[row, col] = np.linalg.norm(states[row] - states[col], 1)
        D_observations[row, col] = np.linalg.norm(observations[row] - observations[col], 2)
D_states[np.diag_indices(n_observations)] = np.nan

#%%
sns.boxenplot(x=D_states.flatten(), y=D_observations.flatten())
sns.despine()
ax = plt.gca()
ax.set_xlabel('Graph distance')
ax.set_ylabel('L2 distance')
# ax.set_xticks([])
plt.show()

#%%
x = sensor(env.step(0)[0])
# print(x)
plt.imshow(x)

#%%

def count_edges(n):
    n = 5
    inner_edges = 2 * n * (n - 1)
    outer_edges = 4 * (n - 2)
    corner_edges = 4
    level_edges = 2 * inner_edges + outer_edges + corner_edges
    inter_level_edges = n * n * (n - 1)
    return n * level_edges + 2 * inter_level_edges

vertices, edges = list(zip(*[(n**3, count_edges(n)) for n in range(2, 20)]))
plt.loglog(vertices, edges, '.-')

#%%
T_x = np.array([[.1, .2, .7, 0, 0, 0, 0, 0, 0], [0, 0, 0, .5, .4, .1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, .1, .6, .3], [0, 0, 0, .25, .2, .05, .05, .3, .15],
                [0, 0, 0, .25, .2, .05, .05, .3, .15], [0, 0, 0, .25, .2, .05, .05, .3, .15],
                [.1, .2, .7, 0, 0, 0, 0, 0, 0], [.1, .2, .7, 0, 0, 0, 0, 0, 0],
                [.1, .2, .7, 0, 0, 0, 0, 0, 0]])

for i in range(20):
    T_x = np.matmul(T_x, T_x)
T_x[0]

#%%
T_z = np.array([[.1, .2, .7], [0, .5, .5], [1, 0, 0]])

for i in range(20):
    T_z = np.matmul(T_z, T_z)
T_z[0][0] * np.array([.1, .2, .7])
T_z[0][1] * np.array([.5, .4, .1])
T_z[0][2] * np.array([.1, .6, .3])

#%%
T_efg = np.array([[0, 2 / 9, 7 / 9], [0, .5, .5], [.9, 0, 0.1]])
for i in range(20):
    T_efg = np.matmul(T_efg, T_efg)

T_efg[0]

#%%
T_23 = np.array([[.1, .2, .7, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, .5, .5],
                 [.1, .2, .7, 0, 0]])

for i in range(20):
    T_23 = np.matmul(T_23, T_23)

T_23[0]

#%%
#%%
Ta = np.array([[0, 1, 0, 0, 0], [0, 0, 0.5, 0.5, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1],
               [1, 0, 0, 0, 0]])

for i in range(10):
    Ta = np.matmul(Ta, Ta)

Ta[0]

#%%
T = np.array([
    [0.0, 0.75, 0.25],
    [0.6, 0.3, 0.1],
    [0.2, 0.6, 0.2],
])
for i in range(10):
    T = np.matmul(T, T)
T[0].round(4)
assert np.allclose(np.matmul(T[0], T), T)

#%%
T = np.array([
    [0.0, 0.75, 0.25],
    [1 / 3, .5, 1 / 6],
    [1, 0, 0],
])
for i in range(10):
    T = np.matmul(T, T)
T[0].round(4)
assert np.allclose(np.matmul(T[0], T), T)

#%%
T = np.array([
    [0.0, 0.75, 0.25],
    [2 / 3, 1 / 4, 1 / 12],
    [1, 0, 0],
])
for i in range(10):
    T = np.matmul(T, T)
T[0].round(4)
assert np.allclose(np.matmul(T[0], T), T)
#%%
T = np.array([
    [0.0, 0.75, 0.25],
    [1 / 3, 1 / 2, 1 / 6],
    [0.0, 0.75, 0.25],
])
for i in range(10):
    T = np.matmul(T, T)
T[0].round(4)
assert np.allclose(np.matmul(T[0], T), T)

#%%
T = np.array([
    [0.0, 0.75, 0.25],
    [2 / 3, 1 / 4, 1 / 12],
    [0.0, 0.75, 0.25],
])
for i in range(10):
    T = np.matmul(T, T)
T[0].round(4)
assert np.allclose(np.matmul(T[0], T), T)

#%%
T = np.array([
    [0.3 + .75 * .6, 0.1 + .25 * .6],
    [0.6 + .75 * .2, 0.2 + .25 * .2],
])
for i in range(10):
    T = np.matmul(T, T)
T[0].round(4)
assert np.allclose(np.matmul(T[0], T), T)

#%%
T = np.array([[0, 1 / 2, 1 / 2, 0], [1 / 2, 0, 0, 1 / 2], [0, 1, 0, 0], [1, 0, 0, 0]])
for i in range(10):
    T = np.matmul(T, T)
T[0].round(4)
assert np.allclose(np.matmul(T[0], T), T)

#%%
Ta = np.array([
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [.75, .25, 0, 0],
    [.5, .5, 0, 0],
])
Tb = np.array([
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [.75, .25, 0, 0],
    [.5, .5, 0, 0],
])
P0 = np.ones(len(Ta)) / len(Ta)
P = P0
for i in range(20):
    if i % 2 == 0:
        P = np.matmul(P, Ta)
    else:
        P = np.matmul(P, Tb)
    print(P)
assert np.allclose(np.matmul(P, T), P)
