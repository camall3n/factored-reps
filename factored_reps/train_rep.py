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
from markov_abstr.gridworld.models.featurenet import FeatureNet
from markov_abstr.gridworld.models.autoencoder import AutoEncoder
from markov_abstr.gridworld.models.pixelpredictor import PixelPredictor
from markov_abstr.gridworld.repvis import RepVisualization, CleanVisualization
from visgrid.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld, MazeWorld, SpiralWorld, LoopWorld
from visgrid.taxi import VisTaxi5x5
from visgrid.utils import get_parser
from visgrid.sensors import *

parser = get_parser()
# parser.add_argument('-d','--dims', help='Number of latent dimensions', type=int, default=2)
# yapf: disable
parser.add_argument('--type', type=str, default='factored-split',
                    choices=['factored-split', 'factored-combined', 'focused-autoenc', 'markov', 'autoencoder', 'pixel-predictor'],
                    help='Which type of representation learning method')
parser.add_argument('-n','--n_updates', type=int, default=3000,
                    help='Number of training updates')
parser.add_argument('-r','--rows', type=int, default=6,
                    help='Number of gridworld rows')
parser.add_argument('-c','--cols', type=int, default=6,
                    help='Number of gridworld columns')
parser.add_argument('-w', '--walls', type=str, default='empty', choices=['empty', 'maze', 'spiral', 'loop', 'taxi'],
                    help='The wall configuration mode of gridworld')
parser.add_argument('--markov_dims', type=int, default=None,
                    help='Number of latent dimensions to use for Markov representation')
parser.add_argument('-l','--latent_dims', type=int, default=5,
                    help='Number of latent dimensions to use for representation')
parser.add_argument('--L_inv', type=float, default=1.0,
                    help='Coefficient for inverse-model-matching loss')
parser.add_argument('--L_rat', type=float, default=1.0,
                    help='Coefficient for ratio-matching loss')
parser.add_argument('--L_dis', type=float, default=1.0,
                    help='Coefficient for planning-distance loss')
parser.add_argument('--L_fwd', type=float, default=1.0,
                    help='Coefficient for forward dynamics loss')
parser.add_argument('--L_fac', type=float, default=0.0,
                    help='Coefficient for factorization loss')
parser.add_argument('--L_foc', type=float, default=0.003,
                    help='Coefficient for focused loss')
parser.add_argument('--L_rec', type=float, default=1.0,
                    help='Coefficient for reconstruction loss')
parser.add_argument('--max_dz', type=float, default=0.1,
                    help='Distance threshold for planning-distance loss')
parser.add_argument('-lr','--learning_rate', type=float, default=0.003,
                    help='Learning rate for Adam optimizer')
parser.add_argument('--batch_size', type=int, default=2048,
                    help='Mini batch size for training updates')
parser.add_argument('-s','--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('-t','--tag', type=str, required=True,
                    help='Tag for identifying experiment')
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
if 'ipykernel' in sys.argv[0]:
    arglist = [
        '--spiral', '--tag', 'test-spiral', '-r', '6', '-c', '6', '--L_ora', '1.0', '--video'
    ]
    args = parser.parse_args(arglist)
else:
    args = parser.parse_args()

assert (args.markov_dims is None
        or args.type in ['factored-split', 'focused-autoenc'
                         ]), "'markov_dims' arg not valid for network type {}".format(args.type)

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
    image_filename = vid_dir + '/final-{}.png'.format(args.seed)
    maze_file = maze_dir + '/maze-{}.png'.format(args.seed)

log = open(log_dir + '/train-{}.txt'.format(args.seed), 'w')
with open(log_dir + '/args-{}.txt'.format(args.seed), 'w') as arg_file:
    arg_file.write(repr(args))

seeding.seed(args.seed, np, random, torch)

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
    tag = 'episodes-1000_steps-20_passengers-1'
    results_dir = os.path.join('results', 'taxi-experiences', tag)
    filename_pattern = os.path.join(results_dir, 'seed-*.pkl')

    results_files = glob.glob(filename_pattern)

    experiences = []
    for results_file in sorted(results_files):
        with open(results_file, 'rb') as file:
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

    s0 = np.stack(states)
    s1 = np.stack(next_states)
    a = np.asarray(actions)
    x0 = np.stack(obs)
    x1 = np.stack(next_obs)
    c0 = s0[:, 0] * env._cols + s0[:, 1]

#% ------------------ Setup experiment ------------------
n_updates_per_frame = 100
n_frames = args.n_updates // n_updates_per_frame

batch_size = args.batch_size

coefs = {
    'L_inv': args.L_inv,
    'L_rat': args.L_rat,
    'L_dis': args.L_dis,
    'L_fwd': args.L_fwd,
    'L_fac': args.L_fac,
    'L_foc': args.L_foc,
    'L_rec': args.L_rec,
    'L_ora': 0.0,
    'L_coinv': 0.0,
}

if args.type == 'factored-split':
    fnet = FactoredFwdModel(n_actions=len(env.actions),
                            input_shape=x0.shape[1:],
                            n_markov_dims=args.markov_dims,
                            n_latent_dims=args.latent_dims,
                            n_hidden_layers=1,
                            n_units_per_layer=32,
                            lr=args.learning_rate,
                            coefs=coefs)
elif args.type == 'factored-combined':
    fnet = FactorNet(n_actions=len(env.actions),
                     input_shape=x0.shape[1:],
                     n_latent_dims=args.latent_dims,
                     n_hidden_layers=1,
                     n_units_per_layer=32,
                     lr=args.learning_rate,
                     max_dz=args.max_dz,
                     coefs=coefs)
elif args.type == 'focused-autoenc':
    fnet = FocusedAutoencoder(n_actions=len(env.actions),
                              input_shape=x0.shape[1:],
                              n_markov_dims=args.markov_dims,
                              n_latent_dims=args.latent_dims,
                              n_hidden_layers=1,
                              n_units_per_layer=32,
                              lr=args.learning_rate,
                              coefs=coefs)
elif args.type == 'markov':
    fnet = FeatureNet(n_actions=len(env.actions),
                      input_shape=x0.shape[1:],
                      n_latent_dims=args.latent_dims,
                      n_hidden_layers=1,
                      n_units_per_layer=32,
                      lr=args.learning_rate,
                      coefs=coefs)
elif args.type == 'autoencoder':
    fnet = AutoEncoder(n_actions=len(env.actions),
                       input_shape=x0.shape[1:],
                       n_latent_dims=args.latent_dims,
                       n_hidden_layers=1,
                       n_units_per_layer=32,
                       lr=args.learning_rate,
                       coefs=coefs)
elif args.type == 'pixel-predictor':
    fnet = PixelPredictor(n_actions=len(env.actions),
                          input_shape=x0.shape[1:],
                          n_latent_dims=args.latent_dims,
                          n_hidden_layers=1,
                          n_units_per_layer=32,
                          lr=args.learning_rate,
                          coefs=coefs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))
fnet.to(device)

fnet.print_summary()

n_test_samples = 2000
test_s0 = s0[-n_test_samples:, :]
test_s1 = s1[-n_test_samples:, :]
test_x0 = torch.as_tensor(x0[-n_test_samples:, :]).float().to(device)
test_x1 = torch.as_tensor(x1[-n_test_samples:, :]).float().to(device)
test_a = torch.as_tensor(a[-n_test_samples:]).long().to(device)
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

def test_rep(fnet, step):
    with torch.no_grad():
        fnet.eval()
        if args.type in ['markov', 'factored-combined', 'factored-split', 'focused-autoenc']:
            with torch.no_grad():
                z0, z1, loss_info = fnet.train_batch(test_x0, test_a, test_x1, test=True)
        elif args.type == 'autoencoder':
            z0 = fnet.encode(test_x0)
            z1 = fnet.encode(test_x1)

            loss_info = {
                'L': fnet.compute_loss(test_x0),
            }

        elif args.type == 'pixel-predictor':
            z0 = fnet.encode(test_x0)
            z1 = fnet.encode(test_x1)

            loss_info = {
                'L': fnet.compute_loss(test_x0, test_a, test_x1),
            }

        for loss_type, loss_value in loss_info.items():
            loss_info[loss_type] = loss_value.numpy().tolist()
        loss_info['step'] = step

    json_str = json.dumps(loss_info)
    log.write(json_str + '\n')
    log.flush()

    text = '\n'.join([key + ' = ' + str(val) for key, val in loss_info.items()])

    results = [z0, test_a, z1]
    return [r.numpy() for r in results] + [text]

#% ------------------ Run Experiment ------------------
data = []
for frame_idx in tqdm(range(n_frames + 1)):
    test_results = test_rep(fnet, frame_idx * n_updates_per_frame)
    if args.video:
        frame = repvis.update_plots(*test_results)
        data.append(frame)

    for _ in range(n_updates_per_frame):
        tx0, tx1, ta, idx = get_next_batch()
        fnet.train_batch(tx0, ta, tx1)

if args.video:
    imageio.mimwrite(video_filename, data, fps=15)
    imageio.imwrite(image_filename, data[-1])

if args.save:
    fnet.phi.save('phi-{}'.format(args.seed), 'results/models/{}'.format(args.tag))
    fnet.save('fnet-{}'.format(args.seed), 'results/models/{}'.format(args.tag))

log.close()
