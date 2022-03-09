from argparse import Namespace
import glob
import imageio
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
from factored_reps.agents.replaymemory import ReplayMemory

from factored_reps import utils
from factored_reps.scripts.generate_taxi_experiences import generate_experiences
from markov_abstr.gridworld.models.featurenet import FeatureNet
from factored_reps.models.categorical_predictor import CategoricalPredictor
from markov_abstr.gridworld.repvis import RepVisualization
from visgrid.taxi import VisTaxi5x5
from visgrid.sensors import *

#% ------------------ Parse args/hyperparameters ------------------
parser = utils.get_parser()
# yapf: disable
parser.add_argument('--model_type', type=str, default='markov',
                    choices=['factored-split', 'factored-combined', 'focused-autoenc', 'markov', 'autoencoder', 'pixel-predictor'],
                    help='Which type of representation learning method')
parser.add_argument('--load_markov', type=str, default=None,
                    help='Specifies a tag to load a pretrained Markov abstraction')
parser.add_argument('--freeze_markov', action='store_true',
                    help='Prevents Markov abstraction from training')
parser.add_argument('-s','--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--replay_buffer_size', type=int, default=20000,
                    help='Number of experiences in training replay buffer')
parser.add_argument('--n_steps_per_episode', type=int, default=5,
                    help='Reset environment after this many steps')
parser.add_argument('--n_expected_times_to_sample_experience', type=int, default=10,
                    help='Expected number of times to sample each experience in the replay buffer before replacement')
parser.add_argument('-t','--tag', type=str, required=True,
                    help='Tag for identifying experiment')
parser.add_argument('--hyperparams', type=str, default='hyperparams/taxi.csv',
                    help='Path to hyperparameters csv file')
parser.add_argument('-v','--video', action='store_true',
                    help="Save training video")
parser.add_argument('--no_graphics', action='store_true',
                    help='Turn off graphics (e.g. for running on cluster)')
parser.add_argument('--save', action='store_true',
                    help='Save final network weights')
parser.add_argument('--no_sigma', action='store_true',
                    help='Turn off sensors and just use true state; i.e. x=s')
parser.add_argument('--grayscale', action='store_true',
                    help='Grayscale observations (default is RGB)')
parser.add_argument('--quick', action='store_true',
                    help='Flag to reduce number of updates for quick testing')
# yapf: enable

args = utils.parse_args_and_load_hyperparams(parser)
if args.load_markov is not None:
    args.load_markov = os.path.join(args.load_markov, 'fnet-{}_best.pytorch'.format(args.seed))

# Move all loss coefficients to a sub-namespace
coefs = Namespace(**{name: value for (name, value) in vars(args).items() if name[:2] == 'L_'})
for coef_name in vars(coefs):
    delattr(args, coef_name)
args.coefs = coefs

if (args.markov_dims > 0 and args.model_type not in ['factored-split', 'focused-autoenc']):
    print("Warning: 'markov_dims' arg not valid for network type {}. Ignoring...".format(
        args.model_type))

if args.no_graphics:
    import matplotlib
    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

log_dir = 'results/taxi/logs/' + str(args.tag)
models_dir = 'results/taxi/models/' + str(args.tag)
vid_dir = 'results/taxi/videos/' + str(args.tag)
maze_dir = 'results/taxi/mazes/' + str(args.tag)
os.makedirs(log_dir, exist_ok=True)

if args.video:
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(maze_dir, exist_ok=True)
    video_filename = vid_dir + '/video-{}.mp4'.format(args.seed)
    final_image_filename = vid_dir + '/final-{}.png'.format(args.seed)
    best_image_filename = vid_dir + '/best-{}.png'.format(args.seed)
    maze_file = maze_dir + '/maze-{}.png'.format(args.seed)

train_log = open(log_dir + '/train-{}.txt'.format(args.seed), 'w')
test_log = open(log_dir + '/test-{}.txt'.format(args.seed), 'w')
with open(log_dir + '/args-{}.txt'.format(args.seed), 'w') as arg_file:
    arg_file.write(repr(args))

seeding.seed(args.seed, np, random)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False

#% ------------------ Define MDP ------------------
env = VisTaxi5x5(grayscale=args.grayscale)
env.reset()

sensor_list = []
if not args.no_sigma:
    sensor_list += [
        MoveAxisSensor(-1, 0)  # Move image channel (-1) to front (0)
    ]
sensor = SensorChain(sensor_list)

#% ------------------ Generate & store experiences ------------------
on_retrieve = {
    '_index_': lambda items: np.asarray(items),
    '*': lambda items: torch.as_tensor(np.asarray(items)).to(device),
    'ob': lambda items: items.float(),
    'next_ob': lambda items: items.float(),
    'action': lambda items: items.long()
}
replay_test = ReplayMemory(args.batch_size, on_retrieve)
replay_train = ReplayMemory(args.replay_buffer_size, on_retrieve)
n_test_episodes = int(np.ceil(args.batch_size / args.n_steps_per_episode))
n_train_episodes = int(np.ceil(args.replay_buffer_size / args.n_steps_per_episode))
if args.quick:
    n_train_episodes = n_test_episodes
test_seed = 1
train_seed = 2 + args.seed
print('Initializing replay buffer...')
for buffer, n_episodes, seed in zip([replay_train, replay_test],
                                    [n_train_episodes, n_test_episodes], [train_seed, test_seed]):
    for exp in generate_experiences(env,
                                    sensor,
                                    n_episodes,
                                    n_steps_per_episode=args.n_steps_per_episode,
                                    seed=seed):
        if buffer is replay_test:
            s = exp['state']
            exp['color'] = s[0] * env._cols + s[1]
        buffer.push(exp)

#% ------------------ Define models ------------------
fnet = FeatureNet(args,
                  n_actions=len(env.actions),
                  input_shape=replay_train.retrieve(0, 'ob').shape[1:],
                  latent_dims=args.latent_dims,
                  device=device).to(device)
fnet.print_summary()

n_values_per_variable = [5, 5] + ([5, 5, 2] * args.n_passengers)
predictor = CategoricalPredictor(
    n_inputs=args.latent_dims,
    n_values=n_values_per_variable,
    learning_rate=args.learning_rate,
).to(device)
predictor.print_summary()

#% ------------------ Define training/testing callbacks ------------------
def train_batch(test=False):
    buffer = replay_train if not test else replay_test
    batch_size = args.batch_size if not test else len(replay_test)
    fields = ['_index_', 'ob', 'state', 'action', 'next_ob']
    batch = buffer.sample(batch_size, fields) if not test else buffer.retrieve(fields=fields)
    idx, obs, states, actions, next_obs = batch

    negatives = fnet.get_negatives(buffer, idx)
    z, _, loss_info = fnet.train_batch(obs, actions, next_obs, negatives, test=test)
    z = z.detach()
    loss_info['predictor'] = predictor.process_batch(z, states, test=test)
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
        predictor.eval()
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
        predictor.save('predictor-{}'.format(args.seed), models_dir, is_best=is_best)

    text = '\n'.join([key + ' = ' + str(val) for key, val in loss_info.items()])

    return (z0.cpu().numpy(), text), is_best

if args.video:
    repvis = RepVisualization(env,
                              replay_test.retrieve(0, 'ob').cpu().numpy(),
                              batch_size=len(replay_test),
                              n_dims=args.latent_dims,
                              colors=replay_test.retrieve(fields='color').cpu().numpy())

#% ------------------ Run Experiment ------------------
n_updates_per_frame = 100

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

    if args.video:
        if step % n_updates_per_frame == 0:
            frame = repvis.update_plots(*test_results)
            data.append(frame)
            if is_best:
                best_frame = step

    train_loss_info = train_batch()[-1]
    convert_and_log_loss_info(train_log, train_loss_info, step)

    for exp in generate_experiences(env,
                                    sensor,
                                    n_episodes=n_episodes_per_update,
                                    n_steps_per_episode=args.n_steps_per_episode,
                                    seed=train_seed + step,
                                    quiet=True):
        replay_train.push(exp)

#% ------------------ Save results ------------------
if args.video:
    imageio.mimwrite(video_filename, data, fps=15)
    imageio.imwrite(final_image_filename, data[-1])
    imageio.imwrite(best_image_filename, data[best_frame])

if args.save:
    fnet.phi.save('phi-{}'.format(args.seed), models_dir)
    fnet.save('fnet-{}'.format(args.seed), models_dir)
    predictor.save('predictor-{}'.format(args.seed), models_dir)

train_log.close()
test_log.close()

#% ------------------ Analyze results ------------------
from factored_reps.analyze_online_results import analyze_results

output_dir = 'results/taxi/analyze_markov_accuracy/{}/seed-{}'.format(args.tag, args.seed)
analyze_results(output_dir, replay_test, fnet, predictor)
