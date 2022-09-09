import matplotlib.pyplot as plt
import numpy as np
import seeding
from sklearn.neighbors import KernelDensity

from visgrid.envs import GridworldEnv
from visgrid.wrappers.transforms import TransformWrapper, NoiseWrapper

seeding.seed(0, np)

env = GridworldEnv(rows=7, cols=4, should_render=False)
env = TransformWrapper(env, lambda obs: obs + (0.5, 0.5))
env = NoiseWrapper(env, 0.5)

n_samples = 20000
obs = [env.reset()[0]]
states = [env.get_state()]
actions = []
for t in range(n_samples):
    while True:
        a = np.random.choice(env.action_space)
        if env.can_run(a):
            break
    x = env.step(a)[0]
    s = env.get_state()
    obs.append(x)
    states.append(s)
    actions.append(a)
obs = np.stack(obs)
states = np.stack(states)
x0 = np.asarray(obs[:-1])
s0 = np.asarray(states[:-1, :])
c0 = s0[:, 0] * env.cols + s0[:, 1]
x1 = np.asarray(obs[1:])
s1 = np.asarray(states[1:, :])
a = np.asarray(actions)

def phi(x):
    z = x
    z[:, 0] -= 0.3 * x[:, 1]
    z[:, 1] += 0.4 * x[:, 0]
    z -= np.min(z)
    z /= np.max(z)
    return z

z0 = phi(x0)
z1 = phi(x1)

fig, ax = plt.subplots(1, 2)
ax[0].scatter(s0[:, 1], s0[:, 0], c=c0)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlabel('$s_0$')
ax[0].set_ylabel('$s_1$')
ax[1].scatter(z0[:, 0], z0[:, 1], c=c0)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xlabel('$z_0$')
ax[1].set_ylabel('$z_1$')
plt.show()

def fit_kde(x, bw=0.03):
    p = KernelDensity(bandwidth=bw, kernel='tophat')
    p.fit(x)
    return p

def MI(x, y):
    xy = np.concatenate([x, y], axis=-1)
    log_pxy = fit_kde(xy).score_samples(xy)
    log_px = fit_kde(x).score_samples(x)
    log_py = fit_kde(y).score_samples(y)
    log_ratio = log_pxy - log_px - log_py
    return np.mean(log_ratio)

print(MI(s0, z0))
print(MI(s0, s0))
