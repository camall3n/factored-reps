import imageio
import numpy as np
import matplotlib.pyplot as plt
import seeding
from tqdm import tqdm
import torch

from factored_rl.models.markov.nullabstraction import NullAbstraction
from factored_rl.models.factored.focused_autoenc import FocusedAutoencoder
from visgrid.envs import GridworldEnv
from factored_rl.entropy.mi import MI
from visgrid.wrappers.transforms import TransformWrapper

#%% ------------------ Define MDP ------------------
seeding.seed(0, np)

env = GridworldEnv(rows=6, cols=6, image_observations=False)
env = TransformWrapper(env, lambda obs: torch.as_tensor(obs).float())
env.reset_agent()

phi = NullAbstraction(-1, 2)

#%% ------------------ Generate experiences ------------------
n_samples = 1000
states = [env.get_state()]
actions = []
for t in range(n_samples):
    while True:
        a = np.random.choice(env.action_space)
        if env.can_run(a):
            break
    s, _, _ = env.step(a)
    states.append(s)
    actions.append(a)
states = torch.stack(states)
s0 = states[:-1, :]
s1 = states[1:, :]
c0 = s0[:, 0] * env.cols + s0[:, 1]
a = np.asarray(actions)

MI_max = MI(s0, s0)

z0 = phi(s0)
z1 = phi(s1)

entangler = FocusedAutoencoder(lr=0.03, coefs={'L_fac': -0.1})
disentangler = FocusedAutoencoder(lr=0.03, coefs={'L_fac': 0.1})

#%% ------------------ Train entangler ------------------
for update in tqdm(range(1000)):
    entangler.train_batch(z0, z1)
e0 = entangler(z0)
e1 = entangler(z1)

noise_machine = SensorChain([
    NoiseSensor(sigma=0.01),
])

e0n = noise_machine(e0.detach().numpy())
e1n = noise_machine(e1.detach().numpy())

#%%
def get_frame(ax, rep, title, save=''):
    rep = rep.detach().numpy()
    ax.clear()
    ax.scatter(rep[:, 0], rep[:, 1], c=c0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(r'$z_F^{(1)}$')
    ax.set_xlabel(r'$z_F^{(0)}$')
    plt.rcParams.update({'font.size': 22})
    fig = plt.gcf()
    fig.canvas.draw()
    fig.canvas.flush_events()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return frame

#%% ------------------ Train disentangler ------------------
seeding.seed(1, np)
disentangler = FocusedAutoencoder(lr=0.03, coefs={'L_fac': 0.1})
fig, ax = plt.subplots(figsize=(8, 8))
frames = []
e0no = sensor(e0n)
e1no = sensor(e1n)
for update in tqdm(range(100)):
    disentangler.train_batch(e0no, e1no)
    d0 = disentangler(e0no)
    d1 = disentangler(e1no)
    # if update % 10 == 0:
    frames.append(get_frame(ax, d0, r'$z_F$'))

imageio.mimwrite('results/factornet/disentangling.mp4', frames, fps=15)

#%%
def plot2d(rep, title, save=''):
    rep = rep #.detach().numpy()
    plt.scatter(rep[:, 0], rep[:, 1], c=c0)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(r'$z_F^{(0)}$')
    plt.ylabel(r'$z_F^{(1)}$')
    plt.title(title)
    if save != '':
        plt.savefig(save)
    plt.show()

plot2d(z0, title='True state (MI=1.0)', save='results/factornet/img1-true_state.png')
e_title = r'$z$' #.format(MI(s0, e0.detach().numpy())/MI_max)
plt.figure(figsize=(8, 8))
plot2d(e0n, title=e_title, save='results/factornet/img2-entangled.png')
d_title = r'$z_F$' #.format(MI(s0, d0.detach().numpy())/MI_max)
plt.figure(figsize=(8, 8))
plot2d(d0.detach().numpy(), title=d_title, save='results/factornet/img3-disentangled.png')
