import argparse
import glob
import pickle
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seeding
import torch
from tqdm import tqdm

from visgrid.gridworld import GridWorld
from visgrid.sensors import *
from factored_reps.models.factornet import FactorNet
from markov_abstr.gridworld.models.featurenet import FeatureNet
from factored_reps.utils import load_hyperparams_and_inject_args

args = argparse.Namespace()
args.quick = False
# args.quick = True
args.seed = 1
args.hyperparams = 'hyperparams/taxi.csv'
args.tag = 'exp49-markov-save-best__learningrate_0.001'
if args.quick:
    args.taxi_experiences = 'episodes-1000_steps-20_passengers-0'
args.latent_dims = 5
args.markov_dims = 5
args.other_args = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))


params = load_hyperparams_and_inject_args(args)
args = argparse.Namespace(**params)

model_file = 'results/models/{}/fnet-{}_latest.pytorch'.format(args.tag, args.seed)
coefs = {
    'L_inv': args.L_inv,
    'L_fwd': args.L_fwd,
    'L_rat': args.L_rat,
    'L_fac': args.L_fac,
    'L_dis': args.L_dis,
}
args.coefs = coefs

#%% ------------------ Load environment ------------------
results_dir = os.path.join('results', 'taxi-experiences', args.taxi_experiences)
filename_pattern = os.path.join(results_dir, 'seed-*.pkl')

results_files = glob.glob(filename_pattern)

experiences = []
n_episodes = 0
for results_file in sorted(results_files):
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
    result = torch.as_tensor(np.moveaxis(x, -1, 1)).float().to(device)
    return result

x = torchify(obs[0])
xp = torchify(next_obs[0])
fnet = FeatureNet(args, n_actions=5, input_shape=x.squeeze(0).shape, latent_dims=args.latent_dims, device=device)
fnet.to(device)
fnet.load(model_file, to=device)

#%%
n_training = len(states)//2
n_test = 2000

if args.quick:
    n_training = n_training // 100
    n_test = n_test // 100

def compute_accuracy(obs, actions, next_obs):
    a_hat = fnet.predict_a(torchify(obs).to(device), torchify(next_obs).to(device)).detach().cpu().numpy()
    n_correct = (actions == a_hat).sum()
    accuracy = 100 * n_correct / len(obs)
    return n_correct, accuracy, a_hat

results = compute_accuracy(obs[:n_training], actions[:n_training], next_obs[:n_training])
n_train_correct, train_accuracy, a_hat_train = results

results = compute_accuracy(obs[-n_test:], actions[-n_test:], next_obs[-n_test:])
n_test_correct, test_accuracy, a_hat_test = results

print('Inverse model accuracy:')
print('Training:', n_train_correct, 'correct out of', n_training, 'total = {}%'.format(train_accuracy))
print('Test:', n_test_correct, 'correct out of', n_test, 'total = {}%'.format(test_accuracy))

# Inverse model accuracy:
# Training: 10043 correct out of 50000 total = 20.086%
# Test: 401 correct out of 2000 total = 20.05%

#%%
fig, axes = plt.subplots(1, 2, figsize=(12,4))

a = actions[:n_training]
a_hat = a_hat_train
def action_histogram(a_actual, a_predicted, ax, title):
    dfs = []
    for a, label in zip([a_actual, a_predicted], ['actual', 'predicted']):
        df = pd.DataFrame(a, columns=['action'])
        df['mode'] = label
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    sns.histplot(data=data, x='action', hue='mode', discrete=True, label='train', multiple='dodge', shrink=0.8, ax=ax)
    ax.set_title(title)

for a, a_hat, mode, ax in zip(
        [actions[:n_training], actions[-n_test:]],
        [a_hat_train, a_hat_test],
        ['training', 'test'],
        axes,
    ):
    action_histogram(a, a_hat, ax=ax, title=mode)

plt.savefig('results/predicted_action.png')
plt.show()
