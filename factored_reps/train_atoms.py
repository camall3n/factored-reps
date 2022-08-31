from argparse import Namespace
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
from factored_reps.envs.atoms import AtomsEnv
from factored_reps.models.factored.focused_autoenc import FocusedAutoencoder
from factored_reps.plotting import add_heatmap_labels, diagonalize

#%% ------------------ Parse args/hyperparameters ------------------
if 'ipykernel' in sys.argv[0]:
    sys.argv += ["-t", 'exp00-test']

parser = utils.get_parser()
# yapf: disable
parser.add_argument('-s','--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('-t','--tag', type=str, required=True,
                    help='Tag for identifying experiment')
parser.add_argument('--hyperparams', type=str, default='hyperparams/atoms.csv',
                    help='Path to hyperparameters csv file')
parser.add_argument('--save', action='store_true',
                    help='Save final network weights')
parser.add_argument('--headless', action='store_true',
                    help='Enable headless (no graphics) mode for running on cluster')
parser.add_argument('--quick', action='store_true',
                    help='Flag to reduce number of updates for quick testing')
parser.add_argument("-f", "--fool_ipython", help="Dummy arg to fool ipython", default="1")
# yapf: enable

args = utils.parse_args_and_load_hyperparams(parser)
del args.fool_ipython

if args.headless:
    import matplotlib
    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Move all loss coefficients to a sub-namespace
coefs = Namespace(**{name: value for (name, value) in vars(args).items() if name[:2] == 'L_'})
for coef_name in vars(coefs):
    delattr(args, coef_name)
args.coefs = coefs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

results_dir = 'results/atoms/'
log_dir = results_dir + '/logs/' + str(args.tag)
models_dir = results_dir + '/models/' + str(args.tag)
os.makedirs(log_dir, exist_ok=True)

train_log = open(log_dir + '/train-{}.txt'.format(args.seed), 'w')
test_log = open(log_dir + '/test-{}.txt'.format(args.seed), 'w')
with open(log_dir + '/args-{}.txt'.format(args.seed), 'w') as arg_file:
    arg_file.write(repr(args))

seeding.seed(args.seed, np, random)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False

#%% ------------------ Define MDP ------------------
env = AtomsEnv(n_factors=args.n_factors,
               n_atoms_per_factor=args.n_atoms_per_factor,
               permute_actions=False,
               permute_atoms=False,
               correlate_factors=args.correlate_factors,
               add_noop_actions=args.add_noop_actions)
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
        info = env._get_current_info()
        state = info['state']
        for step in range(n_steps_per_episode):
            action = env.action_space.sample()
            next_ob, reward, done, info = env.step(action)
            next_state = info['state']

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
fnet = FocusedAutoencoder(args,
                          n_actions=env.action_space.n,
                          n_input_dims=replay_train.retrieve(0, 'ob').shape[-1],
                          n_latent_dims=args.latent_dims,
                          device=device).to(device)
fnet.print_summary()

#%% ------------------ Define training/testing callbacks ------------------
def train_batch(test=False):
    buffer = replay_train if not test else replay_test
    batch_size = args.batch_size if not test else len(replay_test)
    fields = ['_index_', 'ob', 'state', 'action', 'next_ob']
    batch = buffer.sample(batch_size, fields) if not test else buffer.retrieve(fields=fields)
    idx, obs, states, actions, next_obs = batch

    z, _, loss_info = fnet.train_batch(obs, actions, next_obs, test=test)
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
        fnet.eval()
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
        fnet.phi.save('phi-{}'.format(args.seed), models_dir, is_best=is_best)
        fnet.save('fnet-{}'.format(args.seed), models_dir, is_best=is_best)

    text = '\n'.join([key + ' = ' + str(val) for key, val in loss_info.items()])

    return (z0.cpu().numpy(), text), is_best

#%% ------------------ Run Experiment ------------------
data = []
best_frame = 0
print('Training model...')
if args.quick:
    args.n_updates = 10
for step in tqdm(range(args.n_updates)):
    test_results, is_best = test_rep(step)

    train_loss_info = train_batch()[-1]
    convert_and_log_loss_info(train_log, train_loss_info, step)

#%% ------------------ Save results ------------------
if args.save:
    fnet.save('fnet-{}'.format(args.seed), models_dir)

train_log.close()
test_log.close()

#%% ------------------ Analyze results ------------------
torchify = lambda x: on_retrieve['ob'](on_retrieve['*'](x))

dz_list = []
ds_list = []

done = True
for i in tqdm(range(1000)):
    if done:
        ob = env.reset()
        info = env._get_current_info()
        state = info['state']

    with torch.no_grad():
        z = fnet.encode(torchify(ob)).numpy()

    a = env.action_space.sample()
    next_ob, reward, done, info = env.step(a)
    next_state = info['state']

    with torch.no_grad():
        next_z = fnet.encode(torchify(next_ob)).numpy()

    dz = next_z - z
    ds = next_state - state
    dz_list.append(dz)
    ds_list.append(ds)

    ob = next_ob.copy()
    state = next_state.copy()
    z = next_z.copy()

#%% ------------------ Plot dz vs. ds correlation ------------------
z_deltas = np.stack(dz_list, axis=1)
s_deltas = np.stack(ds_list, axis=1)

n_factors = len(state)
n_vars = len(z)

all_deltas = np.concatenate((z_deltas, s_deltas))
correlation = np.corrcoef(all_deltas)[:args.latent_dims, -args.n_factors:]
diag_correlation, y_ticks = diagonalize(np.abs(correlation))

plt.imshow(diag_correlation, vmin=0, vmax=1)
add_heatmap_labels(diag_correlation)
ax = plt.gca()

ax.set_yticks(np.arange(n_vars))
ax.set_yticklabels(y_ticks)

plt.xticks(np.arange(n_factors))
plt.ylabel(r'Learned representation ($\Delta z$)')
plt.xlabel(r'Ground truth factor ($\Delta s$)')
plt.title('Correlation Magnitude')
plt.colorbar()
plt.tight_layout()
images_dir = 'results/atoms/images/{}/'.format(args.tag)
os.makedirs(images_dir, exist_ok=True)
plt.savefig(images_dir + 'seed-{}-correlation-plot.png'.format(args.seed))
plt.show()
