import argparse
import glob
import pickle
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seeding
import torch
from tqdm import tqdm

from visgrid.gridworld import GridWorld
from visgrid.sensors import *
from factored_reps.models.factornet import FactorNet
from markov_abstr.gridworld.models.featurenet import FeatureNet
from factored_reps.utils import load_hyperparams_and_inject_args

args = argparse.Namespace()
args.seed = 1
args.hyperparams = 'hyperparams/taxi.csv'
args.tag = 'exp49-markov-save-best__learningrate_0.001'
# args.taxi_experiences = 'episodes-1000_steps-20_passengers-0'
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
    return torch.as_tensor(np.moveaxis(x, -1, 0)).float().unsqueeze(0).to(device)

x = torchify(obs[0])
xp = torchify(next_obs[0])
fnet = FeatureNet(args, n_actions=5, input_shape=x.squeeze(0).shape, latent_dims=args.latent_dims, device=device)
fnet.load(model_file, to=device)

#%%
n_training = len(states)//2
n_test = 2000

a_hat = fnet.predict_a(x.to(device), xp.to(device)).detach().cpu().numpy()

n_train_correct = (actions[:n_training] == a_hat[:n_training]).sum()
n_test_correct = (actions[-n_test:] == a_hat[-n_test:]).sum()

train_accuracy = 100 * n_train_correct / n_training
test_accuracy = 100 * n_test_correct / n_test

print('Training:', n_train_correct, 'correct out of', n_training, 'total = {}%'.format(train_accuracy))
print('Training:', n_test_correct, 'correct out of', n_test, 'total = {}%'.format(test_accuracy))
