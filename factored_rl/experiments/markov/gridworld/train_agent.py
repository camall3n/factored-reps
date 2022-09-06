import json
#!! do not import matplotlib until you check input arguments
import numpy as np
import os
import pandas as pd
import seaborn as sns
import seeding
from tqdm import tqdm

from factored_rl.models.nnutils import Reshape
from factored_rl.models.markov.nullabstraction import NullAbstraction
from factored_rl.models.markov.phinet import PhiNet
from factored_rl.agents.randomagent import RandomAgent
from factored_rl.agents.legacy.dqnagent import DQNAgent
from visgrid.envs import GridworldEnv
from visgrid.envs.components import Grid
from visgrid.utils import get_parser
from visgrid.sensors import *

parser = get_parser()
# parser.add_argument('-d','--dims', help='Number of latent dimensions', type=int, default=2)
# yapf: disable
parser.add_argument('-a','--agent', type=str, required=True,
                    choices=['random','dqn'], help='Type of agent to train')
parser.add_argument('-n','--n_trials', type=int, default=1,
                    help='Number of trials')
parser.add_argument('-e','--n_episodes', type=int, default=10,
                    help='Number of episodes per trial')
parser.add_argument('-m','--max_steps', type=int, default=1000,
                    help='Maximum number of steps per episode')
parser.add_argument('-r','--rows', type=int, default=6,
                    help='Number of gridworld rows')
parser.add_argument('-c','--cols', type=int, default=6,
                    help='Number of gridworld columns')
parser.add_argument('-w', '--walls', type=str, default='empty', choices=['empty', 'maze', 'spiral', 'loop'],
                    help='The wall configuration mode of gridworld')
parser.add_argument('-b','--batch_size', type=int, default=16,
                    help='Number of experiences to sample per batch')
parser.add_argument('-l','--latent_dims', type=int, default=2,
                    help='Number of latent dimensions to use for representation')
parser.add_argument('-lr','--learning_rate', type=float, default=0.003,
                    help='Learning rate for Adam optimizer')
parser.add_argument('-s','--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('-t','--tag', type=str, required=True,
                    help='Tag for identifying experiment')
parser.add_argument('--phi_path', type=str,
                    help='Load an existing abstraction network by tag')
parser.add_argument('--no_phi', action='store_true',
                    help='Turn off abstraction and just use observed state; i.e. ϕ(x)=x')
parser.add_argument('--train_phi', action='store_true',
                    help='Allow simultaneous training of abstraction')
parser.add_argument('--no_sigma', action='store_true',
                    help='Turn off sensors and just use true state; i.e. x=s')
parser.add_argument('--one_hot', action='store_true',
                    help='Bypass sensor and use one-hot representation instead')
parser.add_argument('--save', action='store_true',
                    help='Save final network weights')
parser.add_argument('-v','--video', action='store_true',
                    help='Show video of agent training')
parser.add_argument('--xy_noise', action='store_true',
                    help='Add truncated gaussian noise to x-y positions')
parser.add_argument('--rearrange_xy', action='store_true',
                    help='Rearrange discrete x-y positions to break smoothness')
# yapf: enable
args = parser.parse_args()
if args.train_phi and args.no_phi:
    assert False, '--no_phi and --train_phi are mutually exclusive'

if args.one_hot and args.no_sigma:
    assert False, '--one_hot and --no_sigma are mutually exclusive'

if args.video:
    import matplotlib.pyplot as plt

log_dir = 'results/gridworld/scores/' + str(args.tag)
os.makedirs(log_dir, exist_ok=True)
log = open(log_dir + '/scores-{}-{}.txt'.format(args.agent, args.seed), 'w')

#%% ------------------ Define MDP ------------------
if args.walls == 'maze':
    env = GridworldEnv.from_saved_maze(rows=args.rows, cols=args.cols, seed=args.seed)
else:
    env = GridworldEnv(rows=args.rows, cols=args.cols)
    if args.walls == 'spiral':
        env.grid = Grid.generate_spiral(rows=args.rows, cols=args.cols)
    elif args.walls == 'loop':
        env.grid = Grid.generate_spiral_with_shortcut(rows=args.rows, cols=args.cols)
gamma = 0.9

#%% ------------------ Define sensor ------------------
sensor_list = []
if args.xy_noise:
    sensor_list.append(NoiseSensor(sigma=0.2, truncation=0.4))
if args.rearrange_xy:
    sensor_list.append(RearrangeXYPositionsSensor((env.rows, env.cols)))
if not args.no_sigma:
    if args.one_hot:
        sensor_list += [
            OffsetSensor(offset=(0.5, 0.5)),
            ImageSensor(range=((0, env.rows), (0, env.cols)), pixel_density=1),
        ]
    else:
        sensor_list += [
            OffsetSensor(offset=(0.5, 0.5)),
            ImageSensor(range=((0, env.rows), (0, env.cols)), pixel_density=3),
            # ResampleSensor(scale=2.0),
            BlurSensor(sigma=0.6, truncate=1.),
            NoiseSensor(sigma=0.01)
        ]
sensor = SensorChain(sensor_list)

#%% ------------------ Define abstraction ------------------
if args.no_phi:
    phinet = NullAbstraction(-1, args.latent_dims)
else:
    x0 = sensor(env.get_state())
    phinet = PhiNet(input_shape=x0.shape,
                    n_latent_dims=args.latent_dims,
                    n_hidden_layers=1,
                    n_units_per_layer=32)
    if args.phi_path:
        modelfile = 'results/gridworld/models/{}/phi-{}_latest.pytorch'.format(
            args.phi_path, args.seed)
        phinet.load(modelfile)

seeding.seed(args.seed, np)

#%% ------------------ Load agent ------------------
n_actions = 4
if args.agent == 'random':
    agent = RandomAgent(n_actions=n_actions)
elif args.agent == 'dqn':
    agent = DQNAgent(n_features=args.latent_dims,
                     n_actions=n_actions,
                     phi=phinet,
                     lr=args.learning_rate,
                     batch_size=args.batch_size,
                     train_phi=args.train_phi,
                     gamma=gamma,
                     factored=False)
else:
    assert False, 'Invalid agent type: {}'.format(args.agent)

#%% ------------------ Train agent ------------------
if args.video:
    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()
    fig.show()

    def plot_value_function(ax):
        s = np.asarray([[np.asarray([x, y]) for x in range(args.cols)] for y in range(args.rows)])
        v = np.asarray(agent.q_values(s).detach().numpy()).max(-1)
        xy = OffsetSensor(offset=(0.5, 0.5))(s).reshape(args.cols, args.rows, -1)
        ax.contourf(np.arange(0.5, args.cols + 0.5),
                    np.arange(0.5, args.rows + 0.5),
                    v,
                    vmin=-10,
                    vmax=0)

    def plot_states(ax):
        data = pd.DataFrame(agent.replay.memory)
        data[['x.r', 'x.c']] = pd.DataFrame(data['x'].tolist(), index=data.index)
        data[['xp.r', 'xp.c']] = pd.DataFrame(data['xp'].tolist(), index=data.index)
        sns.scatterplot(data=data,
                        x='x.c',
                        y='x.r',
                        hue='done',
                        style='done',
                        markers=True,
                        size='done',
                        size_order=[1, 0],
                        ax=ax,
                        alpha=0.3,
                        legend=False)
        ax.invert_yaxis()

for trial in tqdm(range(args.n_trials), desc='trials'):
    env.reset_goal()
    agent.reset()
    total_reward = 0
    total_steps = 0
    losses = []
    rewards = []
    value_fn = []
    for episode in tqdm(range(args.n_episodes), desc='episodes'):
        env.reset_agent()
        ep_rewards = []
        for step in range(args.max_steps):
            s = env.get_state()
            x = sensor(s)

            a = agent.act(x)
            sp, r, done = env.step(a)
            xp = sensor(sp)
            ep_rewards.append(r)
            if args.video:
                value_fn.append(agent.v(x))
            total_reward += r

            loss = agent.train(x, a, r, xp, done)
            losses.append(loss)
            rewards.append(r)

            if done:
                break

        if args.video:
            [a.clear() for a in ax]
            plot_value_function(ax[0])
            env.plot(ax[0])
            ax[1].plot(value_fn)
            ax[2].plot(rewards, c='C3')
            ax[3].plot(losses, c='C1')
            # plot_states(ax[3])
            ax[1].set_ylim([-10, 0])
            fig.canvas.draw()
            fig.canvas.flush_events()

        total_steps += step
        score_info = {
            'trial': trial,
            'episode': episode,
            'reward': sum(ep_rewards),
            'total_reward': total_reward,
            'total_steps': total_steps,
            'steps': step
        }
        json_str = json.dumps(score_info)
        log.write(json_str + '\n')
        log.flush()
print('\n\n')

if args.save:
    agent.q.save('qnet-{}'.format(args.seed), 'results/gridworld/models/{}'.format(args.tag))
