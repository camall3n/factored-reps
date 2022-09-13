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

def plot_ob(ob):
    plt.imshow(ob.squeeze(), cmap='gray', interpolation='nearest')
    plt.show()

#% ------------------ Parse args/hyperparameters ------------------
parser = utils.get_parser()
# yapf: disable
parser.add_argument('-s','--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--replay_buffer_size', type=int, default=20000,
                    help='Number of experiences in training replay buffer')
parser.add_argument('--n_expected_times_to_sample_experience', type=int, default=10,
                    help='Expected number of times to sample each experience in the replay buffer before replacement')
parser.add_argument('-t','--tag', type=str, required=True,
                    help='Tag for identifying experiment')
parser.add_argument('--hyperparams', type=str, default='factored_rl/hyperparams/monte.csv',
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

if args.quick:
    args.latent_dims = 8
    args.batch_size = 16

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

train_log = open(log_dir + '/train-{}.txt'.format(args.seed), 'w')
test_log = open(log_dir + '/test-{}.txt'.format(args.seed), 'w')
with open(log_dir + '/args-{}.txt'.format(args.seed), 'w') as arg_file:
    arg_file.write(repr(args))

seeding.seed(args.seed, np, random)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False

#% ------------------ Define MDP ------------------
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

#% ------------------ Generate & store experiences ------------------
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

#% ------------------ Define models ------------------
fnet = FeatureNet(args,
                  n_actions=18,
                  input_shape=replay_train.retrieve(0, 'ob').shape[1:],
                  latent_dims=args.latent_dims,
                  device=device).to(device)
fnet.print_summary()

#% ------------------ Define training/testing callbacks ------------------
def train_batch(test=False):
    buffer = replay_train if not test else replay_test
    batch_size = args.batch_size if not test else len(replay_test)
    fields = ['_index_', 'ob', 'action', 'next_ob']
    batch = buffer.sample(batch_size, fields) if not test else buffer.retrieve(fields=fields)
    idx, obs, actions, next_obs = batch

    negatives = fnet.get_negatives(buffer, idx)
    _, _, loss_info = fnet.train_batch(obs, actions, next_obs, negatives, test=test)
    return loss_info

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
        fnet.eval()
        with torch.no_grad():
            loss_info = train_batch(test=True)

    convert_and_log_loss_info(test_log, loss_info, step)
    is_best = False
    if (args.save_model_every_n_steps > 0 and step > 0
            and step % args.save_model_every_n_steps == 0):
        global best_model_test_loss
        current_loss = loss_info['L']
        if current_loss < best_model_test_loss:
            is_best = True
            best_model_test_loss = current_loss
        fnet.phi.save('phi-{}'.format(args.seed), models_dir, is_best=is_best)
        fnet.save('fnet-{}'.format(args.seed), models_dir, is_best=is_best)

#% ------------------ Run Experiment ------------------
n_steps_per_episode = 335 # 334.86 (average of first 100 episodes)
probability_of_sampling = args.batch_size / replay_train.capacity
n_updates_per_buffer_fill = args.n_expected_times_to_sample_experience / probability_of_sampling
n_steps_per_update = replay_train.capacity / n_updates_per_buffer_fill
n_episodes_per_update = int(min(1, n_steps_per_update // n_steps_per_episode))

data = []
print('Training model...')
if args.quick:
    args.n_updates = 10
for step in tqdm(range(args.n_updates)):
    test_rep(step)

    train_loss_info = train_batch()
    convert_and_log_loss_info(train_log, train_loss_info, step)

    for exp in generate_experiences(n_episodes=n_episodes_per_update):
        replay_train.push(exp)

#% ------------------ Save results ------------------
fnet.phi.save('phi-{}'.format(args.seed), models_dir)
fnet.save('fnet-{}'.format(args.seed), models_dir)

train_log.close()
test_log.close()
