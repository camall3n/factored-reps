from argparse import Namespace
import glob
import gym
import json
#!! do not import matplotlib until you check input arguments
import numpy as np
import os
import random
import seeding
import sys
import torch
from tqdm import tqdm
from factored_reps.agents.replaymemory import ReplayMemory

from factored_reps import utils
from visgrid.taxi.taxi import VisTaxi5x5
from visgrid.sensors import *
from markov_abstr.gridworld.models.featurenet import FeatureNet
from factored_reps.models.focused_autoenc import FocusedAutoencoder

#%% ------------------ Parse args/hyperparameters ------------------
if 'ipykernel' in sys.argv[0]:
    sys.argv += ["-t", 'exp01-test', "--quick"]

parser = utils.get_parser()
# yapf: disable
parser.add_argument('-s','--seed', type=int, default=None,
                    help='Random seed')
parser.add_argument('-t','--tag', type=str, required=True,
                    help='Tag for identifying experiment')
parser.add_argument('--hyperparams', type=str, default='hyperparams/factorize-taxi.csv',
                    help='Path to hyperparameters csv file')
parser.add_argument('--save', action='store_true',
                    help='Save final network weights')
parser.add_argument('--quick', action='store_true',
                    help='Flag to reduce number of updates for quick testing')
parser.add_argument("-f", "--fool_ipython", help="Dummy arg to fool ipython", default="1")
# yapf: enable

args = utils.parse_args_and_load_hyperparams(parser)
del args.fool_ipython

# Move all loss coefficients to a sub-namespace
coefs = Namespace(**{name: value for (name, value) in vars(args).items() if name[:2] == 'L_'})
for coef_name in vars(coefs):
    delattr(args, coef_name)
args.coefs = coefs

# markov_abstraction_tag = 'exp78-blast-markov_100__learningrate_0.001__latentdims_20'
markov_abstraction_tag = 'exp78-blast-markov_122__learningrate_0.001__latentdims_20'

args_filename = glob.glob('results/taxi/logs/{}/args-*.txt'.format(markov_abstraction_tag))[0]
with open(args_filename, 'r') as args_file:
    markov_args = eval(args_file.read())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

results_dir = 'results/focused-taxi/'
log_dir = results_dir + '/logs/' + str(args.tag)
models_dir = results_dir + '/models/' + str(args.tag)
os.makedirs(log_dir, exist_ok=True)

train_log = open(log_dir + '/train-{}.txt'.format(args.seed), 'w')
test_log = open(log_dir + '/test-{}.txt'.format(args.seed), 'w')
with open(log_dir + '/args-{}.txt'.format(args.seed), 'w') as arg_file:
    arg_file.write(repr(args))

if args.seed is None:
    args.seed = markov_args.seed

seeding.seed(args.seed, np, random)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False

#%% ------------------ Define MDP ------------------
class StateAbstractionWrapper(gym.Wrapper):
    def __init__(self, env, state_abstraction_model):
        super().__init__(env)
        self.sensor = MoveAxisSensor(-1, 0)
        self.state_abstraction_model = state_abstraction_model
        self.device = next(self.state_abstraction_model.parameters()).device

    def reset(self):
        obs = self.env.reset(goal=False, explore=True)
        return self.encode(obs)

    def step(self, action):
        obs, reward, done = self.env.step(action)
        return self.encode(obs), reward, done

    def encode(self, obs):
        obs = self.sensor.observe(obs)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).float().to(self.device)
            abstract_state = self.state_abstraction_model(obs_tensor).numpy()
        return abstract_state

env = VisTaxi5x5(grayscale=markov_args.grayscale)
example_obs = MoveAxisSensor(-1, 0).observe(env.reset(goal=False, explore=True))

featurenet = FeatureNet(markov_args,
                  n_actions=len(env.actions),
                  input_shape=example_obs.shape,
                  latent_dims=markov_args.latent_dims,
                  device=device).to(device)
model_file = 'results/taxi/models/{}/fnet-{}_best.pytorch'.format(markov_abstraction_tag, markov_args.seed)
featurenet.load(model_file, to=device)
phi = featurenet.phi
phi.freeze()
env = StateAbstractionWrapper(env, phi)

env.reset()

pass

#%% ------------------ Generate & store experiences ------------------
def generate_experiences(env, n_episodes, n_steps_per_episode, seed, quiet=False):
    experiences = []
    episodes = range(n_episodes)
    if not quiet:
        episodes = tqdm(episodes)
    for episode in episodes:
        seeding.seed(n_episodes * (seed - 1) + 1 + episode, np, random)
        ob = env.reset()
        state = env.get_state()
        for step in range(n_steps_per_episode):
            action = random.choice(env.actions)
            next_ob, reward, done = env.step(action)
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
            }
            experiences.append(experience)

            ob = next_ob
            state = next_state
            if done:
                break

    return experiences

on_retrieve = {
    '_index_': lambda items: np.asarray(items),
    '*': lambda items: torch.as_tensor(np.asarray(items)).to(device),
    'ob': lambda items: items.float(),
    'next_ob': lambda items: items.float(),
    'action': lambda items: items.long()
}
replay_test = ReplayMemory(args.batch_size, on_retrieve)
replay_train = ReplayMemory(args.replay_buffer_size, on_retrieve)
n_test_episodes = 100
n_train_episodes = int(np.ceil(args.replay_buffer_size / args.n_steps_per_episode))
if args.quick:
    n_train_episodes = n_test_episodes
test_seed = 1
train_seed = 2 + args.seed
print('Initializing replay buffer...')
for buffer, n_episodes, seed in zip([replay_train, replay_test],
                                    [n_train_episodes, n_test_episodes], [train_seed, test_seed]):
    for exp in generate_experiences(env,
                                    n_episodes,
                                    n_steps_per_episode=args.n_steps_per_episode,
                                    seed=seed):
        if buffer is replay_test:
            s = exp['state']
        buffer.push(exp)

#%% ------------------ Define models ------------------
facnet = FocusedAutoencoder(args,
                          n_actions=len(env.actions),
                          n_input_dims=markov_args.latent_dims,
                          n_latent_dims=args.latent_dims,
                          device=device).to(device)
facnet.print_summary()

#%% ------------------ Define training/testing callbacks ------------------
def train_batch(test=False):
    buffer = replay_train if not test else replay_test
    batch_size = args.batch_size if not test else len(replay_test)
    fields = ['_index_', 'ob', 'state', 'action', 'next_ob']
    batch = buffer.sample(batch_size, fields) if not test else buffer.retrieve(fields=fields)
    idx, obs, states, actions, next_obs = batch

    z, _, loss_info = facnet.train_batch(obs, actions, next_obs, test=test)
    z = z.detach()
    return z, loss_info

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
        facnet.eval()
        with torch.no_grad():
            z0, loss_info = train_batch(test=True)

    convert_and_log_loss_info(test_log, loss_info, step)
    is_best = False
    if (args.save and args.save_model_every_n_steps > 0 and step > 0
            and step % args.save_model_every_n_steps == 0):
        global best_model_test_loss
        current_loss = loss_info['L']
        if current_loss < best_model_test_loss:
            is_best = True
            best_model_test_loss = current_loss
        facnet.save('focused-autoenc-{}'.format(args.seed), models_dir, is_best=is_best)

    text = '\n'.join([key + ' = ' + str(val) for key, val in loss_info.items()])

    return (z0.cpu().numpy(), text), is_best

#%% ------------------ Run Experiment ------------------
args.n_expected_times_to_sample_experience = 5

probability_of_sampling = args.batch_size / replay_train.capacity
n_updates_per_buffer_fill = args.n_expected_times_to_sample_experience / probability_of_sampling
n_steps_per_update = replay_train.capacity / n_updates_per_buffer_fill
n_episodes_per_update = int(min(1, n_steps_per_update // args.n_steps_per_episode))

data = []
best_frame = 0
print('Training model...')
if args.quick:
    args.n_updates = 10
for step in tqdm(range(args.n_updates)):
    test_results, is_best = test_rep(step)

    train_loss_info = train_batch()[-1]
    convert_and_log_loss_info(train_log, train_loss_info, step)

    for exp in generate_experiences(env,
                                    n_episodes=n_episodes_per_update,
                                    n_steps_per_episode=args.n_steps_per_episode,
                                    seed=train_seed + step,
                                    quiet=True):
        replay_train.push(exp)

#%% ------------------ Save results ------------------
if args.save:
    facnet.save('focused-autoenc-{}'.format(args.seed), models_dir)

train_log.close()
test_log.close()

#%% ------------------ Analyze results ------------------
torchify = lambda x: on_retrieve['ob'](on_retrieve['*'](x)).squeeze()

dz_list = []
ds_list = []

if args.quick:
    args.n_samples = 100
done = True
for i in tqdm(range(args.n_samples)):
    if done:
        ob = env.reset()
        state = env.get_state()

    with torch.no_grad():
        z = facnet.encode(torchify(ob)).numpy()

    a = random.choice(env.actions)
    next_ob, reward, done = env.step(a)
    next_state = env.get_state()

    with torch.no_grad():
        next_z = facnet.encode(torchify(next_ob)).numpy()

    dz = next_z - z
    ds = next_state - state
    dz_list.append(dz)
    ds_list.append(ds)

    ob = next_ob.copy()
    state = next_state.copy()
    z = next_z.copy()

#%% ------------------ Plot dz vs. ds correlation ------------------
import matplotlib.pyplot as plt

z_deltas = np.stack(dz_list, axis=1)
s_deltas = np.stack(ds_list, axis=1)

n_factors = len(state)

all_deltas = np.concatenate((z_deltas, s_deltas))
correlation = np.corrcoef(all_deltas)[:args.latent_dims, -n_factors:]

plt.imshow(correlation, vmin=-1, vmax=1)
plt.yticks(np.arange(args.latent_dims))
plt.xticks(np.arange(n_factors))
plt.ylabel(r'Learned representation ($\Delta z$)')
plt.xlabel(r'Ground truth factor ($\Delta s$)')
plt.title('Correlation coefficients')
plt.colorbar()
images_dir = 'results/focused-taxi/images/{}/markov-seed-{}/'.format(markov_abstraction_tag, markov_args.seed)
os.makedirs(images_dir, exist_ok=True)
plt.savefig(images_dir + 'seed-{}-correlation-plot.png'.format(args.seed))
plt.show()
