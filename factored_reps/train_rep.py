from argparse import Namespace
import glob
import imageio
import json
#!! do not import matplotlib until you check input arguments
import numpy as np
import os
import pickle
import random
import seeding
import sys
import torch
from tqdm import tqdm

from factored_reps.models.factornet import FactorNet
from factored_reps.models.factored_fwd_model import FactoredFwdModel
from factored_reps.models.focused_autoenc import FocusedAutoencoder
from factored_reps import utils
from markov_abstr.gridworld.models.featurenet import FeatureNet
from markov_abstr.gridworld.models.autoencoder import AutoEncoder
from markov_abstr.gridworld.models.pixelpredictor import PixelPredictor
from markov_abstr.gridworld.repvis import RepVisualization, CleanVisualization
from visgrid.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld, MazeWorld, SpiralWorld, LoopWorld
from visgrid.taxi import VisTaxi5x5
from visgrid.sensors import *

parser = utils.get_parser()
# yapf: disable
parser.add_argument('--model_type', type=str, default='factored-split',
                    choices=['factored-split', 'factored-combined', 'focused-autoenc', 'markov', 'autoencoder', 'pixel-predictor'],
                    help='Which type of representation learning method')
# parser.add_argument('-w', '--walls', type=str, default='empty', choices=['empty', 'maze', 'spiral', 'loop', 'taxi'],
#                     help='The wall configuration mode of gridworld')
parser.add_argument('-s','--seed', type=int, default=0,
                    help='Random seed')
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
parser.add_argument('--cleanvis', action='store_true',
                    help='Switch to representation-only visualization')
parser.add_argument('--no_sigma', action='store_true',
                    help='Turn off sensors and just use true state; i.e. x=s')
parser.add_argument('--rearrange_xy', action='store_true',
                    help='Rearrange discrete x-y positions to break smoothness')
# yapf: enable

args = utils.parse_args_and_load_hyperparams(parser)

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

log_dir = 'results/logs/' + str(args.tag)
vid_dir = 'results/videos/' + str(args.tag)
maze_dir = 'results/mazes/' + str(args.tag)
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
if args.walls == 'maze':
    env = MazeWorld.load_maze(rows=args.rows, cols=args.cols, seed=args.seed)
elif args.walls == 'spiral':
    env = SpiralWorld(rows=args.rows, cols=args.cols)
elif args.walls == 'loop':
    env = LoopWorld(rows=args.rows, cols=args.cols)
elif args.walls == 'taxi':
    env = VisTaxi5x5()
    env.reset()
else:
    env = GridWorld(rows=args.rows, cols=args.cols)
# env = RingWorld(2,4)
# env = TestWorld()
# env.add_random_walls(10)

# cmap = 'Set3'
cmap = None

#% ------------------ Generate experiences ------------------
if args.walls != 'taxi':
    n_samples = 20000
    states = [env.get_state()]
    actions = []
    for t in range(n_samples):
        a = np.random.choice(env.actions)
        s, _, _ = env.step(a)
        states.append(s)
        actions.append(a)
    states = np.stack(states)
    s0 = np.asarray(states[:-1, :])
    c0 = s0[:, 0] * env._cols + s0[:, 1]
    s1 = np.asarray(states[1:, :])
    a = np.asarray(actions)

    ax = env.plot()
    xx = s0[:, 1] + 0.5
    yy = s0[:, 0] + 0.5
    ax.scatter(xx, yy, c=c0)
    if args.video:
        plt.savefig(maze_file)

    # Confirm that we're covering the state space relatively evenly
    # np.histogram2d(states[:,0], states[:,1], bins=6)

    #% ------------------ Define sensor ------------------
    sensor_list = []
    if args.rearrange_xy:
        sensor_list.append(RearrangeXYPositionsSensor((env._rows, env._cols)))
    if not args.no_sigma:
        sensor_list += [
            OffsetSensor(offset=(0.5, 0.5)),
            NoisySensor(sigma=0.05),
            ImageSensor(range=((0, env._rows), (0, env._cols)), pixel_density=3),
            # ResampleSensor(scale=2.0),
            BlurSensor(sigma=0.6, truncate=1.),
            NoisySensor(sigma=0.01)
        ]
    sensor = SensorChain(sensor_list)

    x0 = sensor.observe(s0)
    x1 = sensor.observe(s1)

    env.reset_agent()

else:
    experiences_dir = os.path.join('results', 'taxi-experiences', args.taxi_experiences)
    filename_pattern = os.path.join(experiences_dir, 'seed-*.pkl')

    experience_files = glob.glob(filename_pattern)

    experiences = []
    for experience_file in sorted(experience_files):
        with open(experience_file, 'rb') as file:
            current_experiences = pickle.load(file)
            experiences.extend(current_experiences)

    def extract_array(experiences, key):
        return [experience[key] for experience in experiences]

    n_samples = len(experiences)
    obs = extract_array(experiences, 'ob')
    states = extract_array(experiences, 'state')
    actions = extract_array(experiences, 'action')
    next_obs = extract_array(experiences, 'next_ob')
    next_states = extract_array(experiences, 'next_state')

    sensor_list = []
    if not args.no_sigma:
        sensor_list += [
            AsTypeSensor(np.float32),
            MultiplySensor(scale=1 / 255),
            MoveAxisSensor(-1, 1)  # Move image channels to front (after batch dim)
        ]
    sensor = SensorChain(sensor_list)

    s0 = np.stack(states)
    s1 = np.stack(next_states)
    a = np.asarray(actions)
    x0 = sensor.observe(np.stack(obs))
    x1 = sensor.observe(np.stack(next_obs))
    c0 = s0[:, 0] * env._cols + s0[:, 1]

#% ------------------ Setup experiment ------------------
n_updates_per_frame = 100
n_frames = args.n_updates // n_updates_per_frame

batch_size = args.batch_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

if args.model_type == 'factored-split':
    fnet = FactoredFwdModel(args,
                            n_actions=len(env.actions),
                            input_shape=x0.shape[1:],
                            device=device)
elif args.model_type == 'factored-combined':
    fnet = FactorNet(args, n_actions=len(env.actions), input_shape=x0.shape[1:], device=device)
elif args.model_type == 'focused-autoenc':
    fnet = FocusedAutoencoder(args,
                              n_actions=len(env.actions),
                              input_shape=x0.shape[1:],
                              device=device)
elif args.model_type == 'markov':
    fnet = FeatureNet(args,
                      n_actions=len(env.actions),
                      input_shape=x0.shape[1:],
                      latent_dims=args.latent_dims,
                      device=device)
elif args.model_type == 'autoencoder':
    fnet = AutoEncoder(args, n_actions=len(env.actions), input_shape=x0.shape[1:])
elif args.model_type == 'pixel-predictor':
    fnet = PixelPredictor(args, n_actions=len(env.actions), input_shape=x0.shape[1:])

fnet.to(device)

fnet.print_summary()

n_test_samples = 2000
test_s0 = s0[-n_test_samples:, :]
test_s1 = s1[-n_test_samples:, :]
test_x0 = torch.as_tensor(x0)[-n_test_samples:].float().to(device)
test_x1 = torch.as_tensor(x1)[-n_test_samples:].float().to(device)
test_a = torch.as_tensor(a)[-n_test_samples:].long().to(device)
test_i = torch.arange(n_test_samples).long().to(device)
test_c = c0[-n_test_samples:]

state = s0[0]
obs = x0[0]

if args.video:
    if not args.cleanvis:
        repvis = RepVisualization(env,
                                  obs,
                                  batch_size=n_test_samples,
                                  n_dims=args.latent_dims,
                                  colors=test_c,
                                  cmap=cmap)
    else:
        repvis = CleanVisualization(env,
                                    obs,
                                    batch_size=n_test_samples,
                                    n_dims=args.latent_dims,
                                    colors=test_c,
                                    cmap=cmap)

def get_batch(x0, x1, a, batch_size=batch_size):
    idx = np.random.choice(len(a), batch_size, replace=False)
    tx0 = torch.as_tensor(x0[idx]).float().to(device)
    tx1 = torch.as_tensor(x1[idx]).float().to(device)
    ta = torch.as_tensor(a[idx]).long().to(device)
    ti = torch.as_tensor(idx).long().to(device)
    return tx0, tx1, ta, idx

get_next_batch = (
    lambda: get_batch(x0[:n_samples // 2, :], x1[:n_samples // 2, :], a[:n_samples // 2]))

def convert_and_log_loss_info(log_file, loss_info, step):
    for loss_type, loss_value in loss_info.items():
        loss_info[loss_type] = loss_value.detach().cpu().numpy().tolist()
    loss_info['step'] = step

    json_str = json.dumps(loss_info)
    log_file.write(json_str + '\n')
    log_file.flush()

best_model_test_loss = np.inf

def test_rep(fnet, step):
    with torch.no_grad():
        fnet.eval()
        if args.model_type in ['markov', 'factored-combined', 'factored-split', 'focused-autoenc']:
            with torch.no_grad():
                z0, z1, loss_info = fnet.train_batch(test_x0, test_a, test_x1, test=True)
        elif args.model_type == 'autoencoder':
            z0 = fnet.encode(test_x0)
            z1 = fnet.encode(test_x1)

            loss_info = {
                'L': fnet.compute_loss(test_x0),
            }

        elif args.model_type == 'pixel-predictor':
            z0 = fnet.encode(test_x0)
            z1 = fnet.encode(test_x1)

            loss_info = {
                'L': fnet.compute_loss(test_x0, test_a, test_x1),
            }

    convert_and_log_loss_info(test_log, loss_info, step)
    is_best = False
    if (args.save and args.save_model_every_n_steps > 0 and step > 0
            and step % args.save_model_every_n_steps == 0):
        global best_model_test_loss
        current_loss = loss_info['L']
        if current_loss < best_model_test_loss:
            is_best = True
            best_model_test_loss = current_loss
        fnet.phi.save('phi-{}'.format(args.seed),
                      'results/models/{}'.format(args.tag),
                      is_best=is_best)
        fnet.save('fnet-{}'.format(args.seed),
                  'results/models/{}'.format(args.tag),
                  is_best=is_best)

    text = '\n'.join([key + ' = ' + str(val) for key, val in loss_info.items()])

    results = [z0, test_a, z1]
    return [r.cpu().numpy() for r in results] + [text], is_best

#% ------------------ Run Experiment ------------------
data = []
best_frame = 0
for step in range(args.n_updates):
    test_results, is_best = test_rep(fnet, step)

    if step % n_updates_per_frame == 0:
        if args.video:
            frame = repvis.update_plots(*test_results)
            data.append(frame)
            if is_best:
                best_frame = step

    tx0, tx1, ta, idx = get_next_batch()
    train_loss_info = fnet.train_batch(tx0, ta, tx1)[-1]
    convert_and_log_loss_info(train_log, train_loss_info, step)

if args.video:
    imageio.mimwrite(video_filename, data, fps=15)
    imageio.imwrite(final_image_filename, data[-1])
    imageio.imwrite(best_image_filename, data[best_frame])

if args.save:
    fnet.phi.save('phi-{}'.format(args.seed), 'results/models/{}'.format(args.tag))
    fnet.save('fnet-{}'.format(args.seed), 'results/models/{}'.format(args.tag))

train_log.close()
test_log.close()
