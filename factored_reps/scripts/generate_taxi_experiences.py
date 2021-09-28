import os
import pickle
import platform
import random
import sys

#!! do not import matplotlib unless you check input arguments
import numpy as np
import seeding
from tqdm import tqdm

from visgrid.taxi import VisTaxi5x5
from visgrid.sensors import *
from visgrid.utils import get_parser

#%% Setup
if 'ipykernel' in sys.argv[0]:
    import matplotlib.pyplot as plt
    sys.argv += ["-t", "debugger", "--rgb"]
else:
    import matplotlib
    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

parser = get_parser()
# yapf: disable
parser.add_argument('-e','--n_episodes', type=int, default=100, help='Number of episodes')
parser.add_argument('-n','--n_steps_per_episode', type=int, default=1, help='Number of steps per episode')
parser.add_argument('-p','--n_passengers', type=int, default=1, help='Number of passengers')
parser.add_argument('-s','--seed', type=int, default=5, help='Random seed')
parser.add_argument('-t','--tag', type=str, required=True, help='Name of experiment')
parser.add_argument('--grayscale', action='store_true', help='Grayscale observations (default)')
parser.add_argument('--rgb', action='store_true', help='RGB observations (overrides grayscale)')
parser.add_argument("-f", "--fool_ipython", help="Dummy arg to fool ipython", default="1")
# yapf: enable
args = parser.parse_args()
del args.fool_ipython

assert not (args.grayscale and args.rgb), 'Cannot specify both grayscale and RGB observations'
args.grayscale = True if args.grayscale else (not args.rgb)
del args.rgb
args.grayscale

if args.tag == 'debugger':
    args.tag += '_steps-{}'.format(args.n_steps_per_episode)
    args.tag += '_passengers-{}'.format(args.n_passengers)
    if args.grayscale:
        args.tag += '_gray'
    else:
        args.tag += '_rgb'

#%%
env = VisTaxi5x5(n_passengers=args.n_passengers, grayscale=args.grayscale)

sensor_list = [
    MultiplySensor(scale=1 / 255),
    NoisySensor(sigma=0.05),
    ClipSensor(0.0, 1.0),
    MultiplySensor(scale=255),
    AsTypeSensor(np.uint8)
]
sensor = SensorChain(sensor_list)

prefix = os.path.expanduser('~/scratch/') if platform.system() == 'Linux' else ''
results_dir = os.path.join(prefix + 'results', 'taxi-experiences', args.tag)
os.makedirs(results_dir, exist_ok=True)

#%% Generate experiences
experiences = []
for episode in tqdm(range(args.n_episodes)):
    seeding.seed(args.n_episodes * (args.seed - 1) + 1 + episode, np, random)
    ob = sensor.observe(env.reset(goal=False, explore=True))
    state = env.get_state()
    goal = env.get_goal_state()
    for step in range(args.n_steps_per_episode):
        action = random.choice(env.actions)
        next_ob = sensor.observe(env.step(action)[0])
        reward = 0
        done = False
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
            'goal': goal
        }
        experiences.append(experience)

        ob = next_ob
        state = next_state

#%% Save results to file
filepath = os.path.join(results_dir, 'seed-{:04d}.pkl'.format(args.seed))
with open(filepath, 'wb') as file:
    pickle.dump(experiences, file)
