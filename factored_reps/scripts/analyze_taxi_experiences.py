import glob
import os
import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from visgrid.taxi import VisTaxi5x5
from visgrid.sensors import *
from visgrid.utils import get_parser

#%% Setup
if 'ipykernel' not in sys.argv[0]:
    parser = get_parser()
    parser.add_argument('-t','--tag', type=str, required=True, help='Name of experiment')
    args = parser.parse_args()
else:
    class Args: pass
    args=Args()
    args.tag = 'episodes-2000_steps-10_passengers-1'

#%% Load results
results_dir = os.path.join('results', 'taxi-experiences', args.tag)
filename_pattern = os.path.join(results_dir, 'seed-*.pkl')

results_files = glob.glob(filename_pattern)

experiences = []
for results_file in sorted(results_files):
    with open(results_file, 'rb') as file:
        current_experiences = pickle.load(file)
        experiences.extend(current_experiences)

def extract_array(experiences, key):
    return [experience[key] for experience in experiences]

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
s = np.asarray(states)

plt.hist2d(s[:, 0], s[:, 1], bins=5)
plt.colorbar()
plt.gca().invert_yaxis()
plt.title('agent location')
plt.show()

plt.hist2d(s[:, 2], s[:, 3], bins=5)
plt.colorbar()
plt.gca().invert_yaxis()
plt.title('passenger location')
plt.show()

plt.hist(s[:, 4], bins=2)
plt.title('passenger in taxi?')
plt.show()
