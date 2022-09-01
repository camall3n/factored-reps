from argparse import Namespace
import glob
import gzip
import imageio
import itertools
import json
#!! do not import matplotlib until you check input arguments
import numpy as np
import os
import pickle
import platform
import random
import seeding
import sys
import torch
from tqdm import tqdm

from factored_rl.envs.monte.fakemonteenv import FakeMonteEnv
from factored_rl.models.markov.featurenet import FeatureNet
from factored_rl.agents.replaymemory import ReplayMemory
from factored_rl import utils
from pfrl.wrappers import atari_wrappers

def plot_ob(ob):
    plt.imshow(ob.squeeze(), cmap='gray', interpolation='nearest')
    plt.show()

#% ------------------ Parse args/hyperparameters ------------------
parser = utils.get_parser()
# yapf: disable
parser.add_argument('-s','--seed', type=int, default=1,
                    help='Random seed')
parser.add_argument('--replay_buffer_size', type=int, default=20000,
                    help='Number of experiences in training replay buffer')
parser.add_argument('--n_expected_times_to_sample_experience', type=int, default=10,
                    help='Expected number of times to sample each experience in the replay buffer before replacement')
parser.add_argument('-t','--tag', type=str, default='monte02-txr-trp',
                    help='Tag for identifying experiment')
parser.add_argument('--hyperparams', type=str, default='hyperparams/monte.csv',
                    help='Path to hyperparameters csv file')
parser.add_argument('--no_graphics', action='store_true',
                    help='Turn off graphics (e.g. for running on cluster)')
parser.add_argument('--quick', action='store_true',
                    help='Flag to reduce number of updates for quick testing')
parser.add_argument("-f", "--fool_ipython", help="Dummy arg to fool ipython", default="1")
# yapf: enable

args = utils.parse_args_and_load_hyperparams(parser)

if args.no_graphics:
    import matplotlib
    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

#%% --------------------------------------

if args.quick:
    args.latent_dims = 8
    args.batch_size = 32
#%%

# Move all loss coefficients to a sub-namespace
coefs = Namespace(**{name: value for (name, value) in vars(args).items() if name[:2] == 'L_'})
for coef_name in vars(coefs):
    delattr(args, coef_name)
args.coefs = coefs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

results_dir = 'monte-results/'
log_dir = results_dir + 'logs/' + str(args.tag)
models_dir = results_dir + 'models/' + str(args.tag)
os.makedirs(log_dir, exist_ok=True)

seeding.seed(args.seed, np, random)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False

#%% ------------------ Define MDP ------------------
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
    while True:
        with gzip.open(path, 'rb') as f:
            try:
                while True:
                    traj = pickle.load(f)
                    yield traj
            except EOFError:
                pass

prefix = os.path.expanduser('~/scratch/') if platform.system() == 'Linux' else ''
path = prefix + 'monte/monte_rnd_with_reward_actions_full_trajectories.pkl.gz'
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
            experience['next_state'] = env.parseRAM(experience['ram'])
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

#%% ------------------ Generate & store experiences ------------------
on_retrieve = {
    '_index_': lambda items: np.asarray(items),
    '*': lambda items: torch.as_tensor(np.asarray(items)).to(device),
    'ob': lambda items: items.float(),
    'next_ob': lambda items: items.float(),
    'action': lambda items: items.long()
}
replay_test = ReplayMemory(args.batch_size, on_retrieve)
if args.quick:
    args.replay_buffer_size = args.batch_size
replay_train = ReplayMemory(args.replay_buffer_size, on_retrieve)
print('Initializing replay buffer...')
for buffer in [replay_test, replay_train]:
    while len(buffer) < buffer.capacity:
        for exp in generate_experiences(n_episodes=1):
            buffer.push(exp)

#%% ------------------ Define models ------------------
fnet = FeatureNet(args,
                  n_actions=18,
                  input_shape=replay_train.retrieve(0, 'ob').shape[1:],
                  latent_dims=args.latent_dims,
                  device=device).to(device)
fnet.print_summary()
fnet.phi.load('monte-results/models/{}/phi-{}_best.pytorch'.format(args.tag, args.seed), to=device)

phi = fnet.phi

#%% ------------------ Compute state pair distances ------------------
args.batch_size = 32

def check_batch(test=False):
    buffer = replay_train if not test else replay_test
    batch_size = args.batch_size if not test else len(replay_test)
    fields = ['_index_', 'ob', 'action', 'next_ob']
    batch = buffer.sample(batch_size, fields) if not test else buffer.retrieve(fields=fields)
    idx, obs, actions, next_obs = batch

    negatives = fnet.get_negatives(buffer, idx)
    anchor = phi(obs).detach()
    pos = phi(next_obs).detach()
    neg = phi(negatives).detach()
    l2_pos = torch.sqrt(((pos - anchor)**2).sum(dim=-1))
    l2_neg = torch.sqrt(((neg - anchor)**2).sum(dim=-1))
    return l2_pos, l2_neg

l2_pos, l2_neg = check_batch()

#%% ------------------ Plot results ------------------
plt.scatter(l2_pos, l2_neg)
plt.axis('square')
M = max(l2_pos.max(), l2_neg.max())
plt.xlim(0, M)
plt.ylim(0, M)
plt.plot((0, M), (0, M), '--k')
plt.xlabel(r'$\|\| z^{+} - z_{A} \|\|_2$')
plt.ylabel(r'$\|\| z^{-} - z_{A} \|\|_2$')
plt.show()

#%% ------------------ Load goal states ------------------
def load_goal_state(dir_path, file):
    file_name = os.path.join(dir_path, file)
    with open(file_name, "rb") as f:
        goals = pickle.load(f)
    if isinstance(goals, (list, tuple)):
        goal = random.choice(goals)
    else:
        goal = goals
    if hasattr(goal, "frame"):
        return goal.frame
    if isinstance(goal, atari_wrappers.LazyFrames):
        return goal
    return goal.obs

goal_dir_path = os.path.expanduser('~/dev/deep-skill-graphs/goal_states/')
s0 = load_goal_state(goal_dir_path, file="start_states.pkl")
g0 = load_goal_state(goal_dir_path, file="bottom_right_states.pkl")
g1 = load_goal_state(goal_dir_path, file="top_bottom_right_ladder_states.pkl")
g2 = load_goal_state(goal_dir_path, file="left_door_goal.pkl")
g3 = load_goal_state(goal_dir_path, file="right_door_goal.pkl")
g4 = load_goal_state(goal_dir_path, file="bottom_mid_ladder_goal.pkl")
g5 = load_goal_state(goal_dir_path, file="bottom_left_goal.pkl")
goals = [np.asarray(g)[-1][None, ...] for g in [s0, g0, g1, g2, g3, g4, g5]]

obs.shape

[g.shape for g in goals]
goal_latents = phi(torch.as_tensor(goals).float()).detach()

def distance(start, goal):
    l2_dist = torch.sqrt(((goal - start)**2).sum(dim=-1))
    return l2_dist

def distance_np(start, goal):
    l2_dist = np.sqrt(((goal - start)**2).sum(axis=-1))
    return l2_dist

distances = torch.zeros((7, 7))
for row in range(7):
    for col in range(7):
        distances[row][col] = distance(goal_latents[row], goal_latents[col])

ground_truth_positions = [
    (77, 235), # start_state
    (123, 148), # bottom_right
    (132, 192), # top_bottom_right_ladder
    (24, 235), # left_door_goal
    (130, 235), # right_door_goal
    (77, 192), # bottom_mid_ladder_goal
    (23, 148), # bottom_left_goal
]

xy = np.asarray(ground_truth_positions)

#%%
for g in range(7):
    plt.scatter(xy[:, 0], xy[:, 1], c=distances[g].numpy(), vmin=0, vmax=10)
    plt.colorbar()
    plt.show()
#%%
min_distances = (distances + torch.eye(7) * distances.max() + 1).argmin(dim=-1)

distance_np(np.asarray((132, 192)), np.asarray((123, 148)))
distance_np(np.asarray((132, 192)), np.asarray((77, 192)))

# maps destination -> source
correct_nearest_neighbors = {
    0: [5],
    1: [2],
    2: [1, 5],
    3: [0],
    4: [0],
    5: [0],
    6: [1],
}

#%%
accuracy = np.asarray([(min_distances[0].item() in correct_nearest_neighbors[g])
                       for g in range(len(min_distances))]).mean()
