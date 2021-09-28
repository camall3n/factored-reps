from argparse import Namespace
import glob
import os
import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from visgrid.taxi import VisTaxi5x5
from visgrid.sensors import *
from visgrid.utils import get_parser

parser = get_parser()
parser.add_argument('-s', '--seed', type=int, default=1)
parser.add_argument('-e', '--experiment', type=int, default=56)
parser.add_argument("-f", "--fool_ipython", help="Dummy arg to fool ipython", default="1")
args = parser.parse_args()
del args.fool_ipython

filepaths = glob.glob('results/logs/exp{}*/args-{}.txt'.format(args.experiment, args.seed))
for filepath in filepaths:
    with open(filepath, 'r') as argsfile:
        line = argsfile.readline()
        args = eval(line)
    break
args.n_passengers = int(args.taxi_experiences.split('passengers-')[-1].replace('_plus', '').replace('_gray', ''))

#%% Load results
results_dir = os.path.join('results', 'taxi-experiences', args.taxi_experiences)
filename_pattern = os.path.join(results_dir, 'seed-*.pkl')
results_files = glob.glob(filename_pattern)

experiences_limit = 1000 #if device.type == 'cpu' else 5000

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

#%% Check state coverage histograms
taxi_row = 0
taxi_col = 1
passenger_row = 2
passenger_col = 3
in_taxi = 4

goal_row = 0
goal_col = 1
goal_in_taxi = 2

#%%
counts = plt.hist2d(states[:, taxi_col], states[:, taxi_row], bins=5)[0]
plt.colorbar()
plt.gca().invert_yaxis()
plt.title('taxi location')
plt.show()
np.min(counts*25/len(states))
np.max(counts*25/len(states))

#%%
counts = plt.hist2d(states[:, passenger_col], states[:, passenger_row], bins=5)[0]
plt.colorbar()
plt.gca().invert_yaxis()
plt.title('passenger location')
plt.show()
np.min(counts*25/len(states))
np.max(counts*25/len(states))

#%%
counts = plt.hist2d(goals[:, goal_col], goals[:, goal_row], bins=5)[0]
plt.colorbar()
plt.gca().invert_yaxis()
plt.title('goal')
plt.show()
np.min(counts[counts>0]*4/len(states))
np.max(counts*4/len(states))

#%%
sns.histplot(states[:, 4], discrete=True)
np.histogram(states[:, 4], bins=2)[0]*2/len(states)
plt.title('passenger in taxi?')
plt.show()

#%%
sns.histplot(actions[:], discrete=True)
plt.title('action')
plt.show()

#%% -------------- Compute statistics on experiences --------------

#%% Taxi moved
different_taxi_col = states[:, taxi_col] != next_states[:, taxi_col]
different_taxi_row = states[:, taxi_row] != next_states[:, taxi_row]
taxi_moved = (different_taxi_col ^ different_taxi_row)
taxi_moved.sum()/len(states)

#%% Passenger moved
different_passenger_col = states[:, passenger_col] != next_states[:, passenger_col]
different_passenger_row = states[:, passenger_row] != next_states[:, passenger_row]
passenger_moved = (different_passenger_col ^ different_passenger_row)
passenger_moved.sum()/len(states)

#%% Pickup/dropoff
pickup_or_dropoff = (states[:, in_taxi] != next_states[:, in_taxi])
pickup_or_dropoff.sum()/len(states)

#%% Taxi at passenger
taxi_at_passenger = ((states[:, passenger_col] == states[:, taxi_col]) & (states[:, passenger_row] == states[:, taxi_row]))
taxi_at_passenger.sum()/len(states)

#%% Taxi L1 distance from passenger
row_dist = np.abs(states[:, taxi_row] - states[:, passenger_row])
col_dist = np.abs(states[:, taxi_col] - states[:, passenger_col])
taxi_dist = (row_dist + col_dist)
sns.histplot(taxi_dist, discrete=True)
plt.title('Taxi L1 distance from passenger')
plt.show()

#%% Goal frequencies
np.unique(goals, axis=0, return_counts=True)[1]/len(states)

#%% Episode frequencies
episode_length = int(len(states)/n_episodes)

#%% Taxi L1 distance traveled
ep_start_states = states[0::episode_length]
ep_end_states = states[episode_length-1::episode_length]
row_dist = np.abs(ep_start_states[:, taxi_row] - ep_end_states[:, taxi_row])
col_dist = np.abs(ep_start_states[:, taxi_col] - ep_end_states[:, taxi_col])
taxi_dist = (row_dist + col_dist)
sns.histplot(taxi_dist, discrete=True)
plt.title('Taxi L1 distance traveled')
plt.show()

#%% Passenger L1 distance traveled
ep_start_states = states[0::episode_length]
ep_end_states = states[episode_length-1::episode_length]
row_dist = np.abs(ep_start_states[:, passenger_row] - ep_end_states[:, passenger_row])
col_dist = np.abs(ep_start_states[:, passenger_col] - ep_end_states[:, passenger_col])
passenger_dist = (row_dist + col_dist).astype(int)
sns.histplot(passenger_dist, discrete=True)
plt.title('Passenger L1 distance traveled')
plt.show()

#%% Analyze shuffled samples
idx = np.random.permutation(len(states))
shuffled_goals = np.copy(goals)[idx]
shuffled_next_states = np.copy(next_states)[idx]

goal_matches = np.all(shuffled_goals == goals, axis=1)

taxi_row_delta = np.abs(states[:, taxi_row] - shuffled_next_states[:, taxi_row])
taxi_col_delta = np.abs(states[:, taxi_col] - shuffled_next_states[:, taxi_col])
taxi_delta = (taxi_row_delta + taxi_col_delta).astype(int)
valid_taxi_delta = (taxi_delta <= 1)

passenger_row_delta = np.abs(states[:, passenger_row] - shuffled_next_states[:, passenger_row])
passenger_col_delta = np.abs(states[:, passenger_col] - shuffled_next_states[:, passenger_col])
passenger_delta = (passenger_row_delta + passenger_col_delta).astype(int)
valid_passenger_delta = (passenger_delta <= 1)

valid_delta = valid_taxi_delta & valid_passenger_delta

goal_matches.sum()/len(states)
valid_taxi_delta.sum()/len(states)
valid_passenger_delta.sum()/len(states)
valid_delta.sum()/len(states)
(goal_matches & valid_delta).sum()/len(states)
pass

# -----------------------------------------------------------------

#%% Check that observations look reasonable

for i in range(1):
    plt.imshow(obs[i], cmap='gray')
    plt.show()
