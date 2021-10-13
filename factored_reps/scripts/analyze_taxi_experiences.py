import glob
import os
import pickle
import platform
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from visgrid.taxi import VisTaxi5x5
from visgrid.sensors import *
from visgrid.utils import get_parser

if 'ipykernel' in sys.argv[0]:
    sys.argv += ["-t", 'foo']#, '--restrict_samples']
parser = get_parser()
parser.add_argument('-t', '--tag', type=str, required=True)
parser.add_argument("-f", "--fool_ipython", help="Dummy arg to fool ipython", default="1")
parser.add_argument('--restrict_samples', action='store_true',
                    help='Restrict training samples to more easily learn the in_taxi bit')
args = parser.parse_args()
del args.fool_ipython

if 'ipykernel' in sys.argv[0]:
    # args.tag = 'debugger-03_steps-1_passengers-1_rgb'
    args.tag = 'debugger-5step_steps-5_passengers-1_gray'
    # args.tag = 'debugger_steps-20_passengers-1_plus'
    # args.tag = 'debugger_steps-20_passengers-1_gray'
    # args.tag = 'episodes-5000_steps-20_passengers-0'
    # args.tag = 'episodes-5000_steps-20_passengers-1'
    # args.tag = 'episodes-5000_steps-20_passengers-1_plus'
    # args.tag = 'episodes-5000_steps-20_passengers-1_gray'
    # args.tag = 'episodes-5000_steps-20_passengers-1_rgb'
    # args.tag = 'episodes-10000_steps-10_passengers-1_rgb'
    # args.tag = 'episodes-20000_steps-5_passengers-1_rgb'
    # args.tag = 'iid100k_steps-1_passengers-1_gray'
    # args.tag = 'iid100k_steps-1_passengers-1_rgb'

args.n_passengers = int(args.tag.split('passengers-')[-1].replace('_plus', '').replace('_gray', '').replace('_rgb', ''))
args.n_steps_per_episode = int(args.tag.split('_')[1].replace('steps-', ''))
args.n_episodes_per_chunk = 1000 if 'iid100k' in args.tag else (100 if 'debugger' in args.tag else 1)

#%% Load results
prefix = os.path.expanduser('~/scratch/') if platform.system() == 'Linux' else ''
results_dir = os.path.join(prefix+'results', 'taxi-experiences', args.tag)
filename_pattern = os.path.join(results_dir, 'seed-*.pkl')
results_files = glob.glob(filename_pattern)

experiences_limit = 20000//args.n_steps_per_episode//args.n_episodes_per_chunk*4

n_chunks = 0
experiences = []
for results_file in sorted(results_files)[:experiences_limit]:
    with open(results_file, 'rb') as file:
        current_experiences = pickle.load(file)
    for experience in current_experiences:
        experience['chunk'] = n_chunks
    experiences.extend(current_experiences)
    n_chunks += 1

def extract_array(experiences, key, condition=None):
    if condition is None:
        condition = lambda _: True
    result = np.asarray([experience[key] for experience in experiences if condition(experience)])
    print(len(np.unique(result, axis=0)), 'unique', key, 'entries')
    return result

condition = None
if args.restrict_samples:
    def condition(experience):
        x = experience
        state_a = np.array_equal(x['state'][:2], [4, 0])
        state_b = np.array_equal(x['state'][:2], [3, 0])
        passenger_a = np.array_equal(x['state'][2:4], [4, 0])
        passenger_b = np.array_equal(x['state'][2:4], [3, 0])
        if not (passenger_a or passenger_b):
            return False
        if not (state_a or state_b):
            return False
        if ((state_a and x['action'] in [1, 2, 4])# >, ^, interact
            or (state_b and x['action'] in [1, 3, 4])):# >, v, interact
            return True
        return False

# episodes = extract_array(experiences, 'episode')
steps = extract_array(experiences, 'step', condition)
obs = extract_array(experiences, 'ob', condition)
states = extract_array(experiences, 'state', condition)
actions = extract_array(experiences, 'action', condition)
rewards = extract_array(experiences, 'reward', condition)
next_obs = extract_array(experiences, 'next_ob', condition)
next_states = extract_array(experiences, 'next_state', condition)
dones = extract_array(experiences, 'done', condition)
goals = extract_array(experiences, 'goal', condition)

n_samples = len(states)

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
def taxi_heatmap(x, y, title=''):
    counts = plt.hist2d(x, y, bins=5, range=((-.5,4.5),(-.5,4.5)))[0]
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()
    return counts

counts = taxi_heatmap(states[:, taxi_col], states[:, taxi_row], 'taxi location')
np.min(counts*25/len(states))
np.max(counts*25/len(states))

#%%
counts = taxi_heatmap(next_states[:, taxi_col], next_states[:, taxi_row], 'taxi location')
# counts = plt.hist2d( bins=5, range=((-.5,4.5),(-.5,4.5)))[0]
# plt.colorbar()
# plt.gca().invert_yaxis()
# plt.title('taxi location')
# plt.show()
np.min(counts*25/len(next_states))
np.max(counts*25/len(next_states))

#%%
if args.n_passengers > 0:
    counts = taxi_heatmap(states[:, passenger_col], states[:, passenger_row], 'passenger location')
    np.min(counts*25/len(states))
    np.max(counts*25/len(states))

#%%
if args.n_passengers > 0:
    counts = taxi_heatmap(goals[:, goal_col], goals[:, goal_row], 'goal')
    np.min(counts[counts>0]*4/len(states))
    np.max(counts*4/len(states))

#%%
if args.n_passengers > 0:
    sns.histplot(states[:, in_taxi], discrete=True)
    np.histogram(states[:, in_taxi], bins=2)[0]/len(states)
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
if args.n_passengers > 0:
    different_passenger_col = states[:, passenger_col] != next_states[:, passenger_col]
    different_passenger_row = states[:, passenger_row] != next_states[:, passenger_row]
    passenger_moved = (different_passenger_col ^ different_passenger_row)
    passenger_moved.sum()/len(states)

#%% Pickup/dropoff
if args.n_passengers > 0:
    pickup_or_dropoff = (states[:, in_taxi] != next_states[:, in_taxi])
    pickup_or_dropoff.sum()/len(states)

#%% Taxi at passenger
if args.n_passengers > 0:
    taxi_at_passenger = ((states[:, passenger_col] == states[:, taxi_col]) & (states[:, passenger_row] == states[:, taxi_row]))
    taxi_at_passenger.sum()/len(states)

    next_taxi_at_passenger = ((next_states[:, passenger_col] == next_states[:, taxi_col]) & (next_states[:, passenger_row] == next_states[:, taxi_row]))
    next_taxi_at_passenger.sum()/len(next_states)

#%% Taxi L1 distance from passenger
if args.n_passengers > 0:
    row_dist = np.abs(states[:, taxi_row] - states[:, passenger_row])
    col_dist = np.abs(states[:, taxi_col] - states[:, passenger_col])
    taxi_dist = (row_dist + col_dist)
    sns.histplot(taxi_dist, discrete=True)
    plt.title('Taxi L1 distance from passenger')
    plt.show()

#%% Goal frequencies
if args.n_passengers > 0:
    np.unique(goals, axis=0, return_counts=True)[1]/len(states)

#%% -------------- Sanity checks --------------

if args.n_passengers > 0:
    different_col = states[:, taxi_col] != states[:, passenger_col]
    different_row = states[:, taxi_row] != states[:, passenger_row]
    different_position = different_col | different_row
    fish_out_of_water = different_position & (states[:, in_taxi] == True)
    fish_out_of_water.sum()
    np.argwhere(fish_out_of_water > 0).squeeze()

    waiting = (1-states[:, in_taxi]) & (1-next_states[:, in_taxi])
    riding = states[:, in_taxi] & next_states[:, in_taxi]
    dropoff = states[:, in_taxi] & (1-next_states[:, in_taxi])
    pickup = (1-states[:, in_taxi]) & next_states[:, in_taxi]
    [waiting.sum(), riding.sum(), dropoff.sum(), pickup.sum()]

    tried_dropoff = (states[:, in_taxi] & (actions == 4))
    tried_pickup = (1 - states[:, in_taxi]) & (actions == 4) & taxi_at_passenger
    [tried_dropoff.sum(), tried_pickup.sum()]
    np.argwhere(~dropoff & tried_dropoff)[:10]
    np.argwhere(~pickup & tried_pickup)[:10]

#%% -------------------------------------------

#%% Episode frequencies
episode_length = args.n_steps_per_episode

#%% Taxi L1 distance traveled
ep_start_states = states[0::episode_length]
ep_end_states = next_states[episode_length-1::episode_length]
row_dist = np.abs(ep_start_states[:, taxi_row] - ep_end_states[:, taxi_row])
col_dist = np.abs(ep_start_states[:, taxi_col] - ep_end_states[:, taxi_col])
taxi_dist = (row_dist + col_dist)
sns.histplot(taxi_dist, discrete=True)
plt.title('Taxi L1 distance traveled')
plt.show()

#%% Passenger L1 distance traveled
if args.n_passengers > 0:
    ep_start_states = states[0::episode_length]
    ep_end_states = next_states[episode_length-1::episode_length]
    row_dist = np.abs(ep_start_states[:, passenger_row] - ep_end_states[:, passenger_row])
    col_dist = np.abs(ep_start_states[:, passenger_col] - ep_end_states[:, passenger_col])
    passenger_dist = (row_dist + col_dist).astype(int)
    sns.histplot(passenger_dist, discrete=True)
    plt.title('Passenger L1 distance traveled')
    plt.show()

#%% Check negative samples
# -----------------------------------------------------------------
def get_negatives(idx, max_idx=n_samples):
    # provide alternate x' (negative examples) for contrastive model

    # draw from multiple sources of negative examples, depending on RNG
    r = np.random.rand(*idx.shape)
    all = np.arange(len(idx))
    to_keep = np.argwhere(r < 1/3).squeeze()  # yapf:disable;  where to use the same obs again
    to_offset = np.argwhere((1/3 <= r) & (r < 2/3) & ((idx + 1) < max_idx)).squeeze()  # yapf:disable;  where to use the *subsequent* obs from the trajectory (i.e. x''), unless out of bounds
    # otherwise we'll draw a random obs from the buffer

    shuffled_idx = np.random.permutation(idx)
    state_negatives = next_states[shuffled_idx]  # replace x' samples with random samples
    state_negatives[to_keep] = states[idx[to_keep]]  # replace x' samples with x
    state_negatives[to_offset] = next_states[idx[to_offset] + 1]  # replace x' samples with x''
    # state_negatives = next_states[shuffled_idx]  # x' -> \tilde x'
    # state_negatives = states[idx]  # x' -> x
    # state_negatives = next_states[(idx+1) % n_samples]  # x' -> x''

    return state_negatives

idx = np.arange(n_samples)
negatives = get_negatives(idx)
print(len(np.unique(negatives, axis=0)), 'unique', 'negatives', 'entries')

#%% What's the difference between corresponding positive and negative samples?
positives = next_states

# number of state variables that changed
distinctions = negatives - positives
n_changes = np.sum(np.abs(distinctions) > 0, axis=1)
sns.histplot(n_changes, discrete=True)
plt.title('Number of state variables changed')
plt.show()

#%% When it's only 1 state variable, which variable is it?
single_change = np.argwhere(n_changes==1).squeeze()
changed_var = np.argmax(distinctions[single_change, :] != 0, axis=1)
sns.histplot(changed_var, discrete=True)
plt.title('Distribution of which single state variable changed')
plt.show()

#%% When it's 2 state variables, which variables are those?
changes_to_2_vars = np.argwhere(n_changes==2).squeeze()
changed_2_vars = distinctions[changes_to_2_vars, :] != 0
changed_2_var_counts = np.sum(changed_2_vars, axis=0)
plt.bar(np.arange(5), changed_2_var_counts)
plt.title('Distribution of which two state variables changed')
plt.show()

#%% When it's 2 state variables, how are those correlated?
split_vars = np.hsplit(changed_2_vars.astype(float), 5)
correlations = np.zeros((5,5))
for i, x in enumerate(split_vars):
    for j, y in enumerate(split_vars):
        correlations[i,j] = np.corrcoef(x, y, rowvar=False)[0, 1]

plt.imshow(correlations)
plt.title('Correlations between variables when 2 changed')
plt.colorbar()
plt.show()


#%% Analyze negative samples
if args.n_passengers > 0:
    goal_matches = np.all(shuffled_goals == goals, axis=1)

taxi_row_delta = np.abs(states[:, taxi_row] - negatives[:, taxi_row])
taxi_col_delta = np.abs(states[:, taxi_col] - negatives[:, taxi_col])
taxi_delta = (taxi_row_delta + taxi_col_delta).astype(int)
valid_taxi_delta = (taxi_delta <= 1)

if args.n_passengers > 0:
    passenger_row_delta = np.abs(states[:, passenger_row] - negatives[:, passenger_row])
    passenger_col_delta = np.abs(states[:, passenger_col] - negatives[:, passenger_col])
    passenger_delta = (passenger_row_delta + passenger_col_delta).astype(int)
    valid_passenger_delta = (passenger_delta <= 1)

    valid_delta = valid_taxi_delta & valid_passenger_delta

valid_taxi_delta.sum()/len(states)
if args.n_passengers > 0:
    valid_passenger_delta.sum()/len(states)
    valid_delta.sum()/len(states)
pass

# -----------------------------------------------------------------

#%% Check that observations look reasonable

for i in range(1):
    if obs[i].shape[-1] == 3:
        plt.imshow(obs[i])
    else:
        plt.imshow(obs[i], cmap='gray')
    plt.show()
