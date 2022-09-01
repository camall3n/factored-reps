from argparse import Namespace
import glob
import gym
import json
#!! do not import matplotlib until you check input arguments
import numpy as np
import os
import pandas as pd
import platform
import random
import seaborn as sns
import seeding
from sklearn.neighbors import KernelDensity
import sys
import torch
from tqdm import tqdm
from factored_reps.agents.replaymemory import ReplayMemory

from factored_reps import utils
from visgrid.taxi.taxi import VisTaxi5x5
from visgrid.sensors import *
from factored_reps.models.markov.featurenet import FeatureNet
from factored_reps.models.factored.focused_autoenc import FocusedAutoencoder
from factored_reps.models.factored.calf import CALFNet
from factored_reps.models.factored.cae import CAENet
from factored_reps.models.debug.categorical_predictor import CategoricalPredictor
from factored_reps.experiments.factorize.analysis.heatmaps import add_heatmap_labels, diagonalize

#%% ------------------ Parse args/hyperparameters ------------------
if 'ipykernel' in sys.argv[0]:
    sys.argv += [
        "-t", 'exp00-test', '--load-experiences', 'exp02-factorize-multi-seeds', '-s', '1',
        '--save', '--quick'
    ]

parser = utils.get_parser()
# yapf: disable
parser.add_argument('-s','--seed', type=int, default=None,
                    help='Random seed')
parser.add_argument('-t','--tag', type=str, required=True,
                    help='Tag for identifying experiment')
parser.add_argument('--hyperparams', type=str, default='hyperparams/factorize-taxi.csv',
                    help='Path to hyperparameters csv file')
parser.add_argument('--save', action='store_true',
                    help='Save final network weights')
parser.add_argument('--quick', action='store_true',
                    help='Flag to reduce number of updates for quick testing')
parser.add_argument('--headless', action='store_true',
                    help='Enable headless (no graphics) mode for running on cluster')
parser.add_argument('--load-experiences', type=str, default=None,
                    help='Flag to force regeneration of experience data')
parser.add_argument('--remove-self-loops', action='store_true',
                    help='Prevent self-loop transitions from being added to replay buffer')
parser.add_argument("-f", "--fool_ipython", help="Dummy arg to fool ipython", default="1")
# yapf: enable

args = utils.parse_args_and_load_hyperparams(parser)
del args.fool_ipython

if args.headless:
    import matplotlib
    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Move all loss coefficients to a sub-namespace
coefs = Namespace(**{name: value for (name, value) in vars(args).items() if name[:2] == 'L_'})
for coef_name in vars(coefs):
    delattr(args, coef_name)
args.coefs = coefs

# markov_abstraction_tag = 'exp78-blast-markov_100__learningrate_0.001__latentdims_20'
markov_abstraction_tag = 'exp78-blast-markov_122__learningrate_0.001__latentdims_20'

prefix = os.path.expanduser('~/data-gdk/csal/factored/') if platform.system() == 'Linux' else ''
results_dir = prefix + 'results/'

args_filename = glob.glob(results_dir +
                          'taxi/logs/{}/args-*.txt'.format(markov_abstraction_tag))[0]
with open(args_filename, 'r') as args_file:
    markov_args = eval(args_file.read())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

log_dir = results_dir + 'focused-taxi/logs/' + str(args.tag)
models_dir = results_dir + 'focused-taxi/models/' + str(args.tag)
os.makedirs(log_dir, exist_ok=True)

train_log = open(log_dir + '/train-{}.txt'.format(args.seed), 'w')
test_log = open(log_dir + '/test-{}.txt'.format(args.seed), 'w')
with open(log_dir + '/args-{}.txt'.format(args.seed), 'w') as arg_file:
    arg_file.write(repr(args))

if args.seed is None:
    args.seed = markov_args.seed

seeding.seed(args.seed, np, random)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False

#%% ------------------ Define MDP ------------------
class StateAbstractionWrapper(gym.Wrapper):
    def __init__(self, env, state_abstraction_model):
        super().__init__(env)
        self.sensor = MoveAxisSensor(-1, 0)
        self.state_abstraction_model = state_abstraction_model
        self.device = next(self.state_abstraction_model.parameters()).device

    def reset(self):
        obs = self.env.reset(goal=False, explore=True)
        return self.encode(obs)

    def step(self, action):
        obs, reward, done = self.env.step(action)
        return self.encode(obs), reward, done

    def encode(self, obs):
        obs = self.sensor.observe(obs)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).float().to(self.device)
            abstract_state = self.state_abstraction_model(obs_tensor).cpu().numpy()
        return abstract_state

env = VisTaxi5x5(grayscale=markov_args.grayscale)
example_obs = MoveAxisSensor(-1, 0).observe(env.reset(goal=False, explore=True))

featurenet = FeatureNet(markov_args,
                        n_actions=len(env.actions),
                        input_shape=example_obs.shape,
                        latent_dims=markov_args.latent_dims,
                        device=device).to(device)
model_file = results_dir + 'taxi/models/{}/fnet-{}_best.pytorch'.format(
    markov_abstraction_tag, markov_args.seed)
featurenet.load(model_file, to=device)
featurenet.freeze()
phi = featurenet.phi
phi.freeze()
env = StateAbstractionWrapper(env, phi)

env.reset()

pass

#%% ------------------ Generate & store experiences ------------------
def generate_experiences(env, n_episodes, n_steps_per_episode, seed, quiet=False):
    experiences = []
    episodes = range(n_episodes)
    if not quiet:
        episodes = tqdm(episodes)
    for episode in episodes:
        seeding.seed(n_episodes * (seed - 1) + 1 + episode, np, random)
        ob = env.reset()
        state = env.get_state()
        for step in range(n_steps_per_episode):
            action = random.choice(env.actions)
            next_ob, reward, done = env.step(action)
            next_state = env.get_state()

            experience = {
                'episode': episode,
                'step': step,
                'ob': ob,
                'state': state,
                'action': action,
                'reward': reward,
                'next_ob': next_ob,
                'next_state': next_state,
                'done': done,
            }
            experiences.append(experience)

            ob = next_ob
            state = next_state
            if done:
                break

    return experiences

on_retrieve = {
    '_index_': lambda items: np.asarray(items),
    '*': lambda items: torch.as_tensor(np.asarray(items)).to(device),
    'ob': lambda items: items.float(),
    'next_ob': lambda items: items.float(),
    'action': lambda items: items.long()
}
replay_test = ReplayMemory(args.batch_size, on_retrieve)
replay_train = ReplayMemory(args.replay_buffer_size, on_retrieve)

if args.load_experiences is not None:
    memory_dir = os.path.join(results_dir, 'focused-taxi/memory', str(args.load_experiences))
    replay_train.load(memory_dir + '/seed_{}__replay_train.json'.format(args.seed))
    replay_test.load(memory_dir + '/seed_{}__replay_test.json'.format(args.seed))
else:
    memory_dir = os.path.join(results_dir, 'focused-taxi/memory', str(args.tag))
    n_test_episodes = 500
    n_train_episodes = int(np.ceil(args.replay_buffer_size / args.n_steps_per_episode))
    if args.quick:
        n_train_episodes = n_test_episodes
    test_seed = 1
    train_seed = 2 + args.seed
    print('Initializing replay buffer...')
    for buffer, n_episodes, seed in zip([replay_train, replay_test],
                                        [n_train_episodes, n_test_episodes],
                                        [train_seed, test_seed]):
        for exp in generate_experiences(env,
                                        n_episodes,
                                        n_steps_per_episode=args.n_steps_per_episode,
                                        seed=seed):
            if buffer is replay_test:
                s = exp['state']
            buffer.push(exp)

    replay_train.save(memory_dir, filename='seed_{}__replay_train'.format(args.seed))
    replay_test.save(memory_dir, filename='seed_{}__replay_test'.format(args.seed))

#%% ------------------ Define models ------------------
# facnet = FocusedAutoencoder(args,
#                             n_actions=len(env.actions),
#                             n_input_dims=markov_args.latent_dims,
#                             n_latent_dims=args.latent_dims,
#                             device=device,
#                             backprop_next_state=args.autoenc_backprop_next_state).to(device)
# facnet = CALFNet(args,
#                  n_actions=len(env.actions),
#                  n_input_dims=markov_args.latent_dims,
#                  n_latent_dims=args.latent_dims,
#                  device=device,
#                  backprop_next_state=args.autoenc_backprop_next_state,
#                  identity=args.identity_autoenc).to(device)
facnet = CAENet(args,
                n_actions=len(env.actions),
                n_input_dims=markov_args.latent_dims,
                n_latent_dims=args.latent_dims,
                device=device).to(device)
facnet.print_summary()

#%% ------------------ Define training/testing callbacks ------------------
def train_batch(test=False):
    buffer = replay_train if not test else replay_test
    batch_size = args.batch_size if not test else len(replay_test)
    fields = ['_index_', 'ob', 'state', 'action', 'next_ob', 'next_state']
    batch = buffer.sample(batch_size, fields) if not test else buffer.retrieve(fields=fields)
    idx, obs, states, actions, next_obs, next_states = batch

    # If requested, remove self-loop transitions
    if args.remove_self_loops:
        mask = (states != next_states).any(axis=-1)
    else:
        mask = np.ones(len(states), dtype=bool)
    if mask.any(axis=0):
        obs = obs[mask]
        actions = actions[mask]
        next_obs = next_obs[mask]

        z, _, loss_info = facnet.train_batch(obs, actions, next_obs, test=test)
        z = z.detach()
    else:
        z = torch.zeros_like(ob)
        loss_info = {}
    return z, loss_info

def convert_and_log_loss_info(log_file, loss_info, step):
    for loss_type, loss_value in loss_info.items():
        loss_info[loss_type] = loss_value.detach().cpu().numpy().tolist()
    loss_info['step'] = step

    json_str = json.dumps(loss_info)
    log_file.write(json_str + '\n')
    log_file.flush()

best_model_test_loss = np.inf

def test_rep(step):
    with torch.no_grad():
        facnet.eval()
        with torch.no_grad():
            z0, loss_info = train_batch(test=True)

    convert_and_log_loss_info(test_log, loss_info, step)
    is_best = False
    if (args.save and args.save_model_every_n_steps > 0 and step > 0
            and step % args.save_model_every_n_steps == 0):
        global best_model_test_loss
        current_loss = loss_info['L']
        if current_loss < best_model_test_loss:
            is_best = True
            best_model_test_loss = current_loss
        facnet.save('focused-autoenc-{}'.format(args.seed), models_dir, is_best=is_best)

    text = '\n'.join([key + ' = ' + str(val) for key, val in loss_info.items()])

    return (z0.cpu().numpy(), text), is_best

#%% ------------------ Run Experiment ------------------
args.n_expected_times_to_sample_experience = 5

probability_of_sampling = args.batch_size / replay_train.capacity
n_updates_per_buffer_fill = args.n_expected_times_to_sample_experience / probability_of_sampling
n_steps_per_update = replay_train.capacity / n_updates_per_buffer_fill
n_episodes_per_update = int(min(1, n_steps_per_update // args.n_steps_per_episode))

data = []
best_frame = 0
print('Training model...')
if args.quick:
    args.n_updates = 10
for step in tqdm(range(args.n_updates)):
    test_results, is_best = test_rep(step)

    train_loss_info = train_batch()[-1]
    convert_and_log_loss_info(train_log, train_loss_info, step)

    if args.load_experiences is None:
        for exp in generate_experiences(env,
                                        n_episodes=n_episodes_per_update,
                                        n_steps_per_episode=args.n_steps_per_episode,
                                        seed=train_seed + step,
                                        quiet=True):
            replay_train.push(exp)

#%% ------------------ Save results ------------------
if args.save:
    facnet.save('focused-autoenc-{}'.format(args.seed), models_dir)

train_log.close()
test_log.close()

#%% ------------------ Analyze results ------------------
torchify = lambda x: on_retrieve['ob'](on_retrieve['*'](x)).squeeze()

obs = []
next_obs = []
states = []
z_list = []
a_list = []
dz_list = []
ds_list = []

if args.quick:
    args.n_samples = 50
done = True
ep_steps = 0
for i in tqdm(range(args.n_samples)):
    if done or ep_steps >= args.n_steps_per_episode:
        ob = env.reset()
        state = env.get_state()
        ep_steps = 0

    with torch.no_grad():
        z = facnet.encode(torchify(ob)).cpu().numpy()

    a = random.choice(env.actions)
    next_ob, reward, done = env.step(a)
    next_state = env.get_state()
    ep_steps += 1

    with torch.no_grad():
        next_z = facnet.encode(torchify(next_ob)).cpu().numpy()

    dz = next_z - z
    ds = next_state - state
    dz_list.append(dz)
    ds_list.append(ds)
    obs.append(ob)
    next_obs.append(next_ob)
    states.append(state)
    z_list.append(z)
    a_list.append(a)

    ob = next_ob.copy()
    state = next_state.copy()
    z = next_z.copy()

#%% ------------------ Plot dz vs. ds correlation ------------------
z_deltas = np.stack(dz_list, axis=1)
s_deltas = np.stack(ds_list, axis=1)

def compute_focused_loss(dz):
    eps = 1e-14
    dz = dz.transpose()
    l1 = np.sum(np.abs(dz), axis=-1)
    lmax = np.max(np.abs(dz), axis=-1)
    return np.mean(l1 / (lmax + eps))

compute_focused_loss(s_deltas)
self_loops = (s_deltas.sum(axis=0) == 0)
compute_focused_loss(s_deltas[:, ~self_loops])
compute_focused_loss(z_deltas[:, ~self_loops])

taxi_pos_only = s_deltas.copy().astype(float)
taxi_pos_only[2:, :] = 0
s_shaped_noise = np.random.normal(scale=0.01, size=s_deltas.shape)
taxi_pos_only += s_shaped_noise
compute_focused_loss(taxi_pos_only[:, ~self_loops])
dz = taxi_pos_only

s_concat_with_noise = np.concatenate((s_deltas, s_deltas + s_shaped_noise))
compute_focused_loss(s_concat_with_noise[:, ~self_loops])

n_factors = len(state)
n_vars = len(z)

all_deltas = np.concatenate((z_deltas, s_deltas))
correlation = np.nan_to_num(np.corrcoef(all_deltas)[:n_vars, -n_factors:])
diag_correlation, y_ticks = diagonalize(np.abs(correlation))
plt.imshow(diag_correlation, vmin=0, vmax=1)
#add_heatmap_labels(diag_correlation)

def savefig(*args):
    plt.savefig(*args, facecolor='white', edgecolor='white')

ax = plt.gca()

ax.set_yticks(np.arange(n_vars))
ax.set_yticklabels(y_ticks)

plt.ylabel(r'Learned representation ($\Delta z$)')
plt.xlabel(r'Ground truth factor ($\Delta s$)')
plt.title('Correlation Magnitude')
plt.colorbar()
plt.tight_layout()
images_dir = results_dir + 'focused-taxi/images/{}/'.format(args.tag)
plot_dir = images_dir + '/correlation_dz_vs_ds/'
os.makedirs(plot_dir, exist_ok=True)
savefig(plot_dir + 'seed-{}-correlation_dz_vs_ds.png'.format(args.seed))
# plt.show()
plt.close()

#% ------------------ Plot z vs s correlation ------------------
n_factors = len(state)
n_vars = len(z)

s = np.stack(states, axis=1).astype(float)
z = np.stack(z_list, axis=1).astype(float)
state_vars = np.concatenate((z, s))
correlation = np.corrcoef(state_vars)[:n_vars, -n_factors:]
diag_correlation, y_ticks = diagonalize(np.abs(correlation))
plt.imshow(diag_correlation, vmin=0, vmax=1)
#add_heatmap_labels(diag_correlation)

ax = plt.gca()

ax.set_yticks(np.arange(n_vars))
ax.set_yticklabels(y_ticks)

plt.ylabel(r'Learned representation ($z$)')
plt.xlabel(r'Ground truth factor ($s$)')
plt.title('Correlation Magnitude')
plt.colorbar()
plt.tight_layout()
plot_dir = images_dir + '/correlation_z_vs_s/'
os.makedirs(plot_dir, exist_ok=True)
savefig(plot_dir + 'seed-{}-correlation_z_vs_s.png'.format(args.seed))
# plt.show()
plt.close()

#%% ------------------ Plot MI(z, s) ------------------
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

def compute_mi_matrix(s, z):
    n_factors = s.shape[0]
    n_vars = z.shape[0]
    mi_matrix = np.zeros((n_vars, n_factors))
    for i in range(n_factors):
        s_i = s[i, :][:, np.newaxis]
        for j in range(n_vars):
            z_j = z[j, :][:, np.newaxis]
            mi_matrix[j][i] = MI(s_i, z_j)
    return mi_matrix

# def compute_mi_matrix(s, z):
#     n_factors = s.shape[0]
#     n_vars = 1
#     mi_matrix = np.zeros((1, n_factors))
#     for i in range(n_factors):
#         s_i = s[i,:][:, np.newaxis]
#         mi_matrix[0][i] = MI(s_i, z.transpose())
#     return mi_matrix

s_entropy = compute_mi_matrix(s, s).diagonal()[np.newaxis, :]
mi_matrix = compute_mi_matrix(s, s)
mi_matrix = compute_mi_matrix(s, z)

n_vars, n_factors = mi_matrix.shape

diag_mi_matrix, y_ticks = diagonalize(np.abs(mi_matrix))
plt.imshow(diag_mi_matrix)

ax = plt.gca()

ax.set_yticks(np.arange(n_vars))
ax.set_yticklabels(y_ticks)

plt.ylabel(r'Learned representation ($z$)')
plt.xlabel(r'Ground truth factor ($s$)')
plt.title('Mutual Information')
plt.colorbar()
plt.tight_layout()
plot_dir = images_dir + '/mi_z_vs_s/'
os.makedirs(plot_dir, exist_ok=True)
savefig(plot_dir + 'seed-{}-mi_z_vs_s.png'.format(args.seed))
# plt.show()
plt.close()

# MI(s0, s1)
# MI(s0, s2)
# MI(s0, s3)
# MI(s0, s4)
# MI(s[:,0][-1,:][np.newaxis,:], s[:,1][np.newaxis,:])
# MI(s.transpose(), z.transpose())
# MI(z.transpose(), s.transpose())
# MI(z.transpose(), z.transpose())

#%% ------------------ Examine action effects ------------------
z_deltas = np.stack(dz_list, axis=1)
s_deltas = np.stack(ds_list, axis=1)
all_a = np.stack(a_list, axis=0)
all_obs = np.stack(obs, axis=1).squeeze().transpose()
all_z = np.stack(z_list, axis=1)

self_loops = (s_deltas.sum(axis=0) == 0)
s_shaped_noise = np.random.normal(scale=0.03, size=s_deltas.shape)

obs_deltas = np.stack([next_ob - ob for ob, next_ob in zip(obs, next_obs)],
                      axis=1).squeeze().transpose()

def plot_action_deltas(deltas, filename):
    n_vars = len(deltas)
    fig, axes = plt.subplots(5, 1, figsize=(8, 12))
    for action, action_name in enumerate(['left', 'right', 'up', 'down', 'interact']):
        experiences = (all_a == action)
        noise = np.random.normal(scale=0, size=deltas[:, experiences & ~self_loops].shape)
        action_specific_deltas = deltas[:, experiences & ~self_loops] + noise
        df = pd.DataFrame(
            {r'$z_{' + '{}'.format(i) + '}$': action_specific_deltas[i, :]
             for i in range(n_vars)})
        sns.violinplot(data=df, ax=axes[action])
        axes[action].set_title('a = {}'.format(action_name))
        axes[action].set_ylabel('Effect on var')
    axes[-1].set_xlabel('Var')
    plt.tight_layout()
    plot_dir = images_dir + '/' + filename + '/'
    os.makedirs(plot_dir, exist_ok=True)
    savefig(plot_dir + filename + '_seed-{}.png'.format(args.seed))
    # plt.show()
    plt.close()

plot_action_deltas(all_z, 'action_deltas_z')
plot_action_deltas(all_obs, 'action_deltas_x')
plot_action_deltas(z_deltas, 'action_deltas_dz')
plot_action_deltas(obs_deltas, 'action_deltas_dx')

#%%

compute_focused_loss(s_deltas)
self_loops = (s_deltas.sum(axis=0) == 0)
compute_focused_loss(s_deltas[:, ~self_loops])
compute_focused_loss(z_deltas[:, ~self_loops])

taxi_pos_only = s_deltas.copy().astype(float)
taxi_pos_only[2:, :] = 0
s_shaped_noise = np.random.normal(scale=0.01, size=s_deltas.shape)
taxi_pos_only += s_shaped_noise
compute_focused_loss(taxi_pos_only[:, ~self_loops])
dz = taxi_pos_only

s_concat_with_noise = np.concatenate((s_deltas, s_deltas + s_shaped_noise))
compute_focused_loss(s_concat_with_noise[:, ~self_loops])

n_factors = len(state)
n_vars = len(z)

all_deltas = np.concatenate((z_deltas, s_deltas))
correlation = np.corrcoef(all_deltas)[:n_vars, -n_factors:]
diag_correlation, y_ticks = diagonalize(np.abs(correlation))

# plt.imshow(diag_correlation, vmin=0, vmax=1)
# #add_heatmap_labels(diag_correlation)

# ax = plt.gca()

# ax.set_yticks(np.arange(n_vars))
# ax.set_yticklabels(y_ticks)

# plt.ylabel(r'Learned representation ($\Delta z$)')
# plt.xlabel(r'Ground truth factor ($\Delta s$)')
# plt.title('Correlation Magnitude')
# plt.colorbar()
# plt.tight_layout()
# plot_dir = images_dir + ''
# os.makedirs(images_dir, exist_ok=True)
# # savefig(images_dir + 'seed-{}-correlation-plot.png'.format(args.seed))
# plt.show()
# plt.close()

#%% ------------------ Define models ------------------

n_values_per_variable = [5, 5, 5, 5, 2]
predictor = CategoricalPredictor(
    n_inputs=markov_args.latent_dims,
    n_values=n_values_per_variable,
    learning_rate=markov_args.learning_rate,
).to(device)
models_dir = results_dir + 'taxi/models/{}'.format(markov_abstraction_tag)
predictor.load(models_dir + '/predictor-{}_best.pytorch'.format(markov_args.seed), to=device)
predictor.print_summary()

def generate_confusion_plots(s_actual, s_predicted):
    state_vars = ['taxi_row', 'taxi_col', 'passenger_row', 'passenger_col',
                  'in_taxi'][:len(s_actual[0])]
    n_passengers = 1
    n_values_per_variable = [5, 5] + ([5, 5, 2] * n_passengers)

    fig, axes = plt.subplots(len(state_vars), 1, figsize=(3, 2 * len(state_vars)))

    for state_var_idx, (state_var, n_values,
                        ax) in enumerate(zip(state_vars, n_values_per_variable, axes)):
        bins = n_values
        value_range = ((-0.5, n_values - 0.5), (0, n_values - 0.5))
        h = ax.hist2d(x=s_predicted[:, state_var_idx],
                      y=s_actual[:, state_var_idx],
                      bins=bins,
                      range=value_range)
        counts, xedges, yedges, im = h
        fig.colorbar(im, ax=ax)

        for i in range(len(yedges) - 1):
            for j in range(len(xedges) - 1):
                ax.text(xedges[j] + 0.5,
                        yedges[i] + 0.4,
                        int(counts.T[i, j]),
                        color="w",
                        ha="center",
                        va="center",
                        fontweight="bold")
        ax.set_title(state_var)
        ax.set_xlabel('predicted')
        ax.set_ylabel('actual')
    plt.tight_layout()
    # plt.show()
    return fig, axes

def analyze_results(s, z_markov, autoenc, predictor):
    z_hat = autoenc(z_markov)
    s_hat_from_z = predictor.predict(z_markov).detach().cpu().numpy()
    generate_confusion_plots(s, s_hat_from_z)
    plot_dir = images_dir + '/predictor_confusion_from_z/'
    os.makedirs(plot_dir, exist_ok=True)
    savefig(os.path.join(plot_dir, 'seed-{}-predictor_confusion_from_z.png'.format(args.seed)))
    plt.close()

    s_hat_from_z_hat = predictor.predict(z_hat).detach().cpu().numpy()
    generate_confusion_plots(s, s_hat_from_z_hat)
    plot_dir = images_dir + '/predictor_confusion_from_z_hat/'
    os.makedirs(plot_dir, exist_ok=True)
    savefig(os.path.join(plot_dir, 'seed-{}-predictor_confusion_from_z_hat.png'.format(args.seed)))
    plt.close()

all_s = np.asarray(states)
t_obs = torchify(np.asarray(obs))
analyze_results(all_s, t_obs, facnet, predictor)

#%% ----- Where is the passenger info? -----

z_markov = t_obs
z_f = facnet.encode(z_markov)

state_vars = ['taxi_row', 'taxi_col', 'passenger_row', 'passenger_col', 'in_taxi'][:len(all_s[0])]

def compute_accuracy(z_hat):
    s_hat = predictor.predict(z_hat).detach().cpu().numpy()
    accuracy = dict()
    for state_var_idx, state_var in enumerate(state_vars):
        s_predicted = s_hat[:, state_var_idx]
        s_actual = all_s[:, state_var_idx]
        n_correct = np.sum(s_predicted == s_actual)
        n_total = len(s_actual)
        accuracy[state_var] = n_correct / n_total
    return accuracy

acc_baseline = np.asarray(list(compute_accuracy(z_markov).values()))
z_hat = facnet(z_markov)
acc_reconstruction = np.asarray(list(compute_accuracy(z_hat).values()))

def compute_accuracy_delta(z_f):
    z_hat = facnet.decode(z_f)
    acc_factored = np.asarray(list(compute_accuracy(z_hat).values()))
    d_acc_factored = acc_factored - acc_baseline
    return d_acc_factored

print(compute_accuracy_delta(z_f))

#%%
# Separately set each variable to its mean value and measure accuracy:
n_vars = z_f.shape[-1]
df = pd.DataFrame()
for i in range(n_vars):
    z_f_alt = z_f.clone()
    z_f_alt[:, i] = z_f[:, i].mean()
    accuracy_deltas = compute_accuracy_delta(z_f_alt)
    df_entries = [{
        'state_var': state_var,
        'accuracy_delta': accuracy_delta,
        'var_idx': i,
    } for state_var, accuracy_delta in zip(state_vars, accuracy_deltas)]
    for df_entry in df_entries:
        df = df.append(df_entry, ignore_index=True)

sns.barplot(data=df, x='var_idx', y='accuracy_delta', hue='state_var')
plt.title('Change in accuracy, holding each individual variable fixed')
plot_dir = images_dir + '/intervention_single/'
os.makedirs(plot_dir, exist_ok=True)
savefig(plot_dir + 'seed-{}-intervention_single.png'.format(args.seed))
plt.close()

#%%
# Separately set all but one variable to the mean value and measure accuracy:
n_vars = z_f.shape[-1]
df = pd.DataFrame()
for i in range(n_vars):
    z_f_alt = z_f.mean(axis=0).repeat((len(z_f), 1))
    z_f_alt[:, i] = z_f[:, i].clone()
    accuracy_deltas = compute_accuracy_delta(z_f_alt)
    df_entries = [{
        'state_var': state_var,
        'accuracy_delta': accuracy_delta,
        'var_idx': i,
    } for state_var, accuracy_delta in zip(state_vars, accuracy_deltas)]
    for df_entry in df_entries:
        df = df.append(df_entry, ignore_index=True)
sns.barplot(data=df, x='var_idx', y='accuracy_delta', hue='state_var')
plt.ylim([-1.1, 0.01])
sns.move_legend(plt.gca(), loc='lower center', ncol=3)
plt.title('Change in accuracy, holding all but one variable fixed')
plot_dir = images_dir + '/intervention_remainder/'
os.makedirs(plot_dir, exist_ok=True)
savefig(plot_dir + 'seed-{}-intervention_remainder.png'.format(args.seed))
plt.close()
