# Args & hyperparams
from dataclasses import dataclass
import hydra
from factored_rl import configs

# Filesystem & devices
import torch
import logging

# Agent
from factored_rl.agents.dqn import DQNAgent

# Env
from visgrid.envs import GridworldEnv
from factored_rl.wrappers import RotationWrapper
from factored_rl.wrappers import FactorPermutationWrapper, ObservationPermutationWrapper
from visgrid.wrappers import GrayscaleWrapper, InvertWrapper, NormalizedFloatWrapper, NoiseWrapper

# ----------------------------------------
# Args & hyperparameters
# ----------------------------------------

parser = configs.new_parser()
# yapf: disable
parser.add_argument('-e', '--experiment', type=str, default='rl_vs_disent', help='A name for the experiment')
parser.add_argument('-t', '--trial', type=str, default='trial', help='A name for the trial')
parser.add_argument('-s', '--seed', type=int, default=0, help='A seed for the random number generator')
parser.add_argument('--no-timestamp', action='store_true', help='Disable automatic trial timestamps')
parser.add_argument('--noise', action='store_true')
parser.add_argument('--rotate', action='store_true')
parser.add_argument('--permute', type=str, default=None, choices=[None, 'factors', 'states'])
parser.add_argument('--images', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('-f', '--fool-ipython', action='store_true',
    help='Dummy arg to make ipython happy')
# yapf: enable

# ----------------------------------------
# Environment & wrappers
# ----------------------------------------
def initialize_env(args, cfg: configs.EnvConfig):
    env = GridworldEnv(10,
                       10,
                       exploring_starts=cfg.exploring_starts,
                       terminate_on_goal=cfg.terminate_on_goal,
                       fixed_goal=cfg.fixed_goal,
                       hidden_goal=cfg.hidden_goal,
                       should_render=False,
                       dimensions=GridworldEnv.dimensions_6x6_to_18x18)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    if args.permute:
        assert not args.images
        if args.permute == 'factors':
            env = FactorPermutationWrapper(env)
        elif args.permute == 'states':
            env = ObservationPermutationWrapper(env)
    if args.images:
        env.set_rendering(enabled=args.images)
        env = InvertWrapper(GrayscaleWrapper(env))
    else:
        env = NormalizedFloatWrapper(env)
    if args.rotate:
        env = RotationWrapper(env)
    if args.noise:
        env = NoiseWrapper(env, cfg.noise_std)

    return env

# ----------------------------------------
# Agent
# ----------------------------------------

def initialize_agent(env, args, cfg: configs.AgentConfig):
    agent = DQNAgent(env.observation_space, env.action_space, cfg)
    return agent

args, cfg, log = configs.initialize_experiment(parser)
env = initialize_env(args, cfg.env)
agent = initialize_agent(env, args, cfg.agent)

ob, _ = env.reset()
if args.images:
    env.plot(ob)
else:
    print(ob)

# ----------------------------------------
# Disent metrics for representation
# ----------------------------------------

# TODO

# ----------------------------------------
# RL performance for representation
# ----------------------------------------

# TODO
