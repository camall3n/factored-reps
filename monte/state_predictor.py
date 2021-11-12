from argparse import Namespace
import glob
import gzip
import itertools
import json
import pickle
import platform
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seeding
import torch
import torch.nn.functional as F
from tqdm import tqdm

from visgrid.gridworld import GridWorld
from visgrid.sensors import *
from factored_reps.models.factornet import FactorNet
from markov_abstr.gridworld.models.featurenet import FeatureNet
from factored_reps.utils import get_parser, load_hyperparams_and_inject_args
from markov_abstr.gridworld.models.simplenet import SimpleNet
from monte.fakemonteenv import FakeMonteEnv
from factored_reps.agents.replaymemory import ReplayMemory
from factored_reps import utils

parser = get_parser()
parser.add_argument('-s', '--seed', type=int, default=1)
parser.add_argument('-e', '--experiment', type=int, default=2)
parser.add_argument('-n', '--n_train_steps', type=int, default=4000)
parser.add_argument("-f", "--fool_ipython", help="Dummy arg to fool ipython", default="1")
args = parser.parse_args()
del args.fool_ipython

seeding.seed(args.seed, np, random)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False

n_train_steps = args.n_train_steps

filepaths = glob.glob('monte-results/logs/monte{:02d}*/args-{}.txt'.format(args.experiment, args.seed))
for filepath in filepaths:
    with open(filepath, 'r') as argsfile:
        line = argsfile.readline()
        args = eval(line)
    break

args.n_updates = n_train_steps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

output_dir = 'monte-results/analyze_markov_accuracy/{}/seed-{}'.format(
    'quick' if device.type == 'cpu' else args.tag,
    args.seed
)
os.makedirs(output_dir, exist_ok=True)

model_file = 'monte-results/models/{}/fnet-{}_latest.pytorch'.format(args.tag, args.seed)

#%% ------------------ Load environment ------------------
def load_trajectories(path):
    '''
    Returns a generator for getting states.
    Args:
        path (str): filepath of pkl file containing trajectories
        skip (int): number of trajectories to skip
    Returns:
        (generator): generator to be called for trajectories
    '''
    # print(f"[+] Loading trajectories from file '{path}'")
    with gzip.open(path, 'rb') as f:
        try:
            while True:
                traj = pickle.load(f)
                yield traj
        except EOFError:
            pass

prefix = os.path.expanduser('~/scratch/') if platform.system() == 'Linux' else ''
path = prefix+'monte/monte_rnd_with_reward_actions_full_trajectories.pkl.gz'
traj_iter = load_trajectories(path)

def generate_trajectory():
    experiences = []
    while not experiences:
        traj = next(traj_iter)
        if len(traj) <= 2:
            continue
        env = FakeMonteEnv()
        for experience in traj:
            experience = {
                'ob': experience[1],
                'action': experience[2],
                'reward': experience[3],
                'ram': experience[0],
            }
            state = env.parseRAM(experience['ram'])
            state = np.asarray([
                state['player_x'],
                state['player_y'],
            ])
            experience['next_state'] = state
            del experience['ram']
            experiences.append(experience)

        for prev_exp, curr_exp in zip(experiences[0:-1], experiences[1:]):
            prev_exp['next_ob'] = curr_exp['ob']
            curr_exp['state'] = prev_exp['next_state']

        # delete first/last items in list because they don't have state/next_ob
        experiences = experiences[1:-1]

    return experiences

def generate_experiences(n_episodes):
    trajectories = [generate_trajectory() for _ in range(n_episodes)]
    experiences = list(itertools.chain.from_iterable(trajectories))
    return experiences

#% ------------------ Generate & store experiences ------------------
on_retrieve = {
    '_index_': lambda items: np.asarray(items),
    '*': lambda items: torch.as_tensor(np.asarray(items)).to(device),
    'ob': lambda items: items.float(),
    'next_ob': lambda items: items.float(),
    'action': lambda items: items.long(),
    'state': lambda items: items.float(),
    'next_state': lambda items: items.float(),
}
replay_test = ReplayMemory(args.batch_size, on_retrieve)
if args.quick:
    args.replay_buffer_size = args.batch_size
# replay_train = ReplayMemory(args.replay_buffer_size, on_retrieve)
print('Initializing replay buffer...')
for buffer in [replay_test]:#, replay_train]:
    while len(buffer) < buffer.capacity:
        for exp in generate_experiences(n_episodes=1):
            buffer.push(exp)

fields = ['ob', 'state', 'action', 'reward', 'next_ob', 'next_state']
batch = replay_test.retrieve(fields=fields)
obs, states, actions, rewards, next_obs, next_states = batch

#%% ------------------ Load model ------------------
x = obs[0]
xp = next_obs[0]

fnet = FeatureNet(args,
                  n_actions=18,
                  input_shape=x.shape,
                  latent_dims=args.latent_dims,
                  device=device)
fnet.to(device)
fnet.load(model_file, to=device)
fnet.freeze()

predictor = SimpleNet(
    n_inputs=args.latent_dims,
    n_outputs=len(states[0]),
    n_hidden_layers=2,
    n_units_per_layer=32,
    activation=torch.nn.ReLU,
    final_activation=None,
).to(device)
predictor.optimizer = torch.optim.Adam(predictor.parameters(), lr=args.learning_rate)

#%% ------------------ Encode observations to latent states ------------------
# n_training = len(replay_train)
n_test = len(replay_test)

if device.type == 'cpu':
    # n_training = n_training // 10
    n_test = n_test // 10
    args.batch_size = 10
    args.n_updates = 2

with torch.no_grad():
    latent_states_test = fnet.encode(replay_test.retrieve(fields='ob'))

#%% ------------------ Define helper functions for training predictor ------------------
def train_batch(test=False):
    buffer = replay_test#replay_train if not test else replay_test
    batch_size = args.batch_size if not test else len(replay_test)
    fields = ['ob', 'state']
    batch = buffer.sample(batch_size, fields) if not test else buffer.retrieve(fields=fields)
    obs, states = batch

    with torch.no_grad():
        z = fnet.encode(obs)
    loss = process_batch(z, states, test=test)
    return loss

def compute_loss(z, s):
    s_hat = predictor(z)
    loss = F.mse_loss(input=s_hat, target=s)
    return loss

def process_batch(z, s, test=False):
    if not test:
        predictor.train()
        predictor.optimizer.zero_grad()
    else:
        predictor.eval()
    loss = compute_loss(z, s)
    if not test:
        loss.backward()
        predictor.optimizer.step()
    return loss

def convert_and_log_loss_info(log_file, loss_info, step):
    for loss_type, loss_value in loss_info.items():
        loss_info[loss_type] = loss_value.detach().cpu().numpy().tolist()
    loss_info['step'] = step

    json_str = json.dumps(loss_info)
    log_file.write(json_str + '\n')
    log_file.flush()

#%% ------------------ Train ground-truth state predictor ------------------
log_dir = os.path.join(output_dir, 'logs')
model_dir = os.path.join(output_dir, 'models')

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

loss_infos = []
with open(log_dir + '/train-{}.txt'.format(args.seed), 'w') as logfile:
    for step in tqdm(range(args.n_updates)):
        loss_info = dict()
        loss_info['test'] = train_batch(test=True)
        # loss_info['train'] = train_batch(test=False)
        convert_and_log_loss_info(logfile, loss_info, step)
        loss_infos.append(loss_info)

predictor.save('predictor-{}'.format(args.seed), model_dir)

#%% ------------------ Visualize training loss ------------------
data = pd.DataFrame(loss_infos).melt(id_vars=['step'],
                                     value_vars=['test'],#'train',
                                     var_name='mode',
                                     value_name='loss')
sns.lineplot(data=data, x='step', y='loss', hue='mode')
plt.savefig(os.path.join(output_dir, 'detached_mse_predictor_loss.png'),
            facecolor='white',
            edgecolor='white')

#%% ------------------ Predict ground-truth states ------------------
with torch.no_grad():
    state_reconstructions_test = predictor(latent_states_test).detach().cpu().numpy()

#%% ------------------ Analyze state predictions ------------------
s_actual = states.detach().cpu().numpy()
s_predicted = state_reconstructions_test

state_vars = ['player_x', 'player_y']

fig, axes = plt.subplots(len(state_vars), 1, figsize=(6, 2 * len(state_vars)))

for (state_var_idx, state_var), ax in zip(enumerate(state_vars), axes):
    actual_bins = len(np.unique(s_actual[:, state_var_idx]))
    predicted_bins = int(4 * actual_bins)
    h = ax.hist2d(x=s_predicted[:, state_var_idx],
                  y=s_actual[:, state_var_idx],
                  bins=(predicted_bins, actual_bins))
    fig.colorbar(h[3], ax=ax)
    ax.set_title(state_var)
    ax.set_xlabel('predicted')
    ax.set_ylabel('actual')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'detached_mse_confusion_hist.png'),
            facecolor='white',
            edgecolor='white')
plt.show()
