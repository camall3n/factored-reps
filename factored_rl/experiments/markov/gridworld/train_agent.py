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
from visgrid.wrappers import transforms
from visgrid.wrappers.transforms import wrap_gridworld
from factored_rl.wrappers.permutation import ObservationPermutationWrapper

parser = get_parser()
# parser.add_argument('-d','--dims', help='Number of latent dimensions', type=int, default=2)
# yapf: disable
parser.add_argument('-a','--agent', type=str, required=True,
                    choices=['random','dqn'], help='Type of agent to train')
parser.add_argument('-e','--n_episodes', type=int, default=10,
                    help='Number of episodes')
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
parser.add_argument('--ground_truth', action='store_true',
                    help='Turn off transforms and just use true state; i.e. x=s')
parser.add_argument('--one_hot', action='store_true',
                    help='Bypass transforms and use one-hot representation instead')
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

if args.one_hot and args.ground_truth:
    assert False, '--one_hot and --ground_truth are mutually exclusive'

if args.video:
    import matplotlib.pyplot as plt

log_dir = 'results/gridworld/scores/' + str(args.tag)
os.makedirs(log_dir, exist_ok=True)
log = open(log_dir + '/scores-{}-{}.txt'.format(args.agent, args.seed), 'w')

#%% ------------------ Define MDP ------------------
dims = GridworldEnv.dimensions_6x6_to_18x18
if args.one_hot:
    dims['cell_width'] = 1
if args.walls == 'maze':
    env = GridworldEnv.from_saved_maze(rows=args.rows,
                                       cols=args.cols,
                                       seed=args.seed,
                                       should_render=False,
                                       dimensions=dims)
else:
    env = GridworldEnv(rows=args.rows, cols=args.cols, should_render=False, dimensions=dims)
    if args.walls == 'spiral':
        env.grid = Grid.generate_spiral(rows=args.rows, cols=args.cols)
    elif args.walls == 'loop':
        env.grid = Grid.generate_spiral_with_shortcut(rows=args.rows, cols=args.cols)
gamma = 0.9

#%% ------------------ Define transforms ------------------
if args.xy_noise:
    env = transforms.NoiseWrapper(env, sigma=0.2, truncation=0.4)
if args.rearrange_xy:
    env = ObservationPermutationWrapper(env)
if not args.ground_truth and not args.one_hot:
    env = wrap_gridworld(env)

#%% ------------------ Define abstraction ------------------
if args.no_phi:
    phinet = NullAbstraction(-1, args.latent_dims)
else:
    x0 = env.reset()[0]
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
        xy = (s + (0.5, 0.5)).reshape(args.cols, args.rows, -1)
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

x = env.reset()[0]
agent.reset()
total_reward = 0
total_steps = 0
losses = []
rewards = []
value_fn = []
for episode in tqdm(range(args.n_episodes), desc='episodes'):
    env.reset()
    ep_rewards = []
    for step in range(args.max_steps):
        s = env.get_state()

        a = agent.act(x)
        xp, r, done, _, info = env.step(a)
        ep_rewards.append(r)
        if args.video:
            value_fn.append(agent.v(x))
        total_reward += r

        loss = agent.train(x, a, r, xp, done)
        losses.append(loss)
        rewards.append(r)

        if done:
            break

        x = xp

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
