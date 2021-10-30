from argparse import Namespace
import glob
import json
import pickle
import platform
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seeding
import torch
from tqdm import tqdm

from visgrid.sensors import *
from markov_abstr.gridworld.models.featurenet import FeatureNet
from factored_reps.models.categorical_predictor import CategoricalPredictor
from factored_reps import utils

parser = utils.get_parser()
parser.add_argument('-s', '--seed', type=int, default=1)
parser.add_argument('--model_type',
                    type=str,
                    default='perfect',
                    choices=['perfect'],
                    help='Which type of representation learning method')
parser.add_argument('-t', '--tag', type=str, required=True, help='Tag for identifying experiment')
parser.add_argument('--save', action='store_true', help='Save final network weights')
parser.add_argument('--hyperparams',
                    type=str,
                    default='hyperparams/taxi.csv',
                    help='Path to hyperparameters csv file')
parser.add_argument("-f", "--fool_ipython", help="Dummy arg to fool ipython", default="1")
# yapf: enable

args = utils.parse_args_and_load_hyperparams(parser)

# Move all loss coefficients to a sub-namespace
coefs = Namespace(**{name: value for (name, value) in vars(args).items() if name[:2] == 'L_'})
for coef_name in vars(coefs):
    delattr(args, coef_name)
args.coefs = coefs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

log_dir = 'results/logs/' + str(args.tag)
models_dir = 'results/models/{}'.format(args.tag)
output_dir = 'results/analyze_markov_accuracy/{}/seed-{}'.format(args.tag, args.seed)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

train_log = open(log_dir + '/train-{}.txt'.format(args.seed), 'w')
test_log = open(log_dir + '/test-{}.txt'.format(args.seed), 'w')
with open(log_dir + '/args-{}.txt'.format(args.seed), 'w') as arg_file:
    arg_file.write(repr(args))

seeding.seed(args.seed, np, random)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False

#%% ------------------ Load experiences ------------------
prefix = os.path.expanduser('~/scratch/') if platform.system() == 'Linux' else ''
experiences_dir = os.path.join(prefix + 'results', 'taxi-experiences', args.taxi_experiences)
filename_pattern = os.path.join(experiences_dir, 'seed-*.pkl')

results_files = glob.glob(filename_pattern)

experiences_limit = 20 if device.type == 'cpu' else 5000

experiences = []
n_episodes = 0
for results_file in sorted(results_files)[:experiences_limit]:
    with open(results_file, 'rb') as file:
        current_experiences = pickle.load(file)
    for experience in current_experiences:
        experience['episode'] = n_episodes
    experiences.extend(current_experiences)
    n_episodes += 1

def extract_array(experiences, key):
    return np.asarray([experience[key] for experience in experiences])

episodes = extract_array(experiences, 'episode')
steps = extract_array(experiences, 'step')
obs = extract_array(experiences, 'ob')
states = extract_array(experiences, 'state')
actions = extract_array(experiences, 'action')
rewards = extract_array(experiences, 'reward')
next_obs = extract_array(experiences, 'next_ob')
next_states = extract_array(experiences, 'next_state')
dones = extract_array(experiences, 'done')
goals = extract_array(experiences, 'goal')

#%% ------------------ Load model ------------------
def torchify(x):
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    result = torch.as_tensor(np.moveaxis(x, -1, 1)).float().to(device) / 255
    return result

x = torchify(obs[0])
xp = torchify(next_obs[0])
if args.model_type == 'perfect':
    fnet = FeatureNet(args,
                      n_actions=5,
                      input_shape=x.squeeze(0).shape,
                      latent_dims=args.latent_dims,
                      device=device)
else:
    raise NotImplementedError
fnet.to(device)

#%% ------------------ Encode observations to latent states ------------------
n_training = len(states) // 2
n_test = 2000

if device.type == 'cpu':
    n_training = n_training // 10
    n_test = n_test // 10
    args.batch_size = 10
    args.n_updates = 40

with torch.no_grad():
    obs_batches = []
    n_batch_divisions = int(np.ceil(n_training / n_test) + 1)
    batch_divisions = np.linspace(0, n_training, n_batch_divisions).astype(int)
    batch_starts = batch_divisions[:-1]
    batch_ends = batch_divisions[1:]
    for low, high in zip(tqdm(batch_starts), batch_ends):
        obs_batches.append(torchify(obs[low:high]).to(device))
    obs_train = torch.cat(obs_batches)

    obs_test = torchify(obs[-n_test:]).to(device)

#%% ------------------ Define helper functions for training predictor ------------------
def get_batch(mode='train'):
    with torch.no_grad():
        if mode == 'test':
            batch_x = obs_test
            batch_s = states[-n_test:]
        elif mode == 'train':
            idx = np.random.choice(n_training, args.batch_size, replace=False)
            batch_x = obs_train[idx]
            batch_s = states[idx]
        else:
            raise RuntimeError('Invalid mode for get_batch: ' + str(mode))
        batch_s = torch.as_tensor(batch_s).long().to(device)
    return batch_x, batch_s

def convert_and_log_loss_info(log_file, loss_info, step):
    for loss_type, loss_value in loss_info.items():
        loss_info[loss_type] = loss_value.detach().cpu().numpy().tolist()
    loss_info['step'] = step

    json_str = json.dumps(loss_info)
    log_file.write(json_str + '\n')
    log_file.flush()

#%% ------------------ Train ground-truth state predictor ------------------
n_values_per_variable = [5, 5] + ([5, 5, 2] * args.n_passengers)
predictor = CategoricalPredictor(
    n_inputs=args.latent_dims,
    n_values=n_values_per_variable,
    learning_rate=0.001,
).to(device)

def process_batch(x, s, test=False):
    if not test:
        fnet.train()
        predictor.train()
        fnet.optimizer.zero_grad()
        predictor.optimizer.zero_grad()
    else:
        fnet.eval()
        predictor.eval()
    z = fnet.encode(x)
    loss = predictor.compute_loss(z, s)
    if not test:
        loss.backward()
        fnet.optimizer.step()
        predictor.optimizer.step()
    return loss

loss_infos = []
with open(log_dir + '/train-{}.txt'.format(args.seed), 'w') as logfile:
    for step in tqdm(range(args.n_updates)):
        loss_info = dict()
        loss_info['test'] = process_batch(*get_batch(mode='test'), test=True)
        loss_info['train'] = process_batch(*get_batch(mode='train'), test=False)
        convert_and_log_loss_info(logfile, loss_info, step)
        loss_infos.append(loss_info)

fnet.save('fnet-{}'.format(args.seed), models_dir)
fnet.phi.save('phi-{}'.format(args.seed), models_dir)
predictor.save('predictor-{}'.format(args.seed), models_dir)

#%% ------------------ Visualize training loss ------------------
data = pd.DataFrame(loss_infos).melt(id_vars=['step'],
                                     value_vars=['train', 'test'],
                                     var_name='mode',
                                     value_name='loss')
sns.lineplot(data=data, x='step', y='loss', hue='mode')
plt.savefig(os.path.join(output_dir, 'perfect_predictor_loss.png'),
            facecolor='white',
            edgecolor='white')

#%% ------------------ Predict ground-truth states ------------------
with torch.no_grad():
    # state_reconstructions_train = predictor.predict(fnet.encode(obs_train)).detach().cpu().numpy()
    state_reconstructions_test = predictor.predict(fnet.encode(obs_test)).detach().cpu().numpy()

#%% ------------------ Analyze state predictions ------------------
s_actual = states[-n_test:].astype(np.float32)
s_predicted = state_reconstructions_test

state_vars = ['taxi_row', 'taxi_col', 'passenger_row', 'passenger_col', 'in_taxi'][:len(states[0])]

fig, axes = plt.subplots(len(state_vars), 1, figsize=(3, 2 * len(state_vars)))

for (state_var_idx, state_var), ax in zip(enumerate(state_vars), axes):
    bins = len(np.unique(s_actual[:, state_var_idx]))
    h = ax.hist2d(x=s_predicted[:, state_var_idx], y=s_actual[:, state_var_idx], bins=bins)
    fig.colorbar(h[3], ax=ax)
    ax.set_title(state_var)
    ax.set_xlabel('predicted')
    ax.set_ylabel('actual')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'perfect_predictor_confusion_plots.png'),
            facecolor='white',
            edgecolor='white')
plt.show()
