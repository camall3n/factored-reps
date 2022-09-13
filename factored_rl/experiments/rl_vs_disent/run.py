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
import gym
from gym.wrappers.flatten_observation import FlattenObservation
import numpy as np
from visgrid.envs import GridworldEnv, TaxiEnv
from factored_rl.wrappers import RotationWrapper
from factored_rl.wrappers import FactorPermutationWrapper, ObservationPermutationWrapper
from visgrid.wrappers import GrayscaleWrapper, InvertWrapper, FloatWrapper, NormalizeWrapper, NoiseWrapper, TransformWrapper

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
    if args.env == 'gridworld':
        env = GridworldEnv(10,
                           10,
                           exploring_starts=True,
                           terminate_on_goal=True,
                           fixed_goal=True,
                           hidden_goal=True,
                           should_render=False,
                           dimensions=GridworldEnv.dimensions_onehot)
    elif args.env == 'taxi':
        env = TaxiEnv(size=5,
                      n_passengers=1,
                      exploring_starts=True,
                      terminate_on_goal=True,
                      should_render=False,
                      dimensions=TaxiEnv.dimensions_5x5_to_48x48)
    else:
        env = gym.make(args.env)
        # TODO: wrap env to support disent protocol

    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    if args.transform == 'images':
        env.set_rendering(enabled=True)
        env = InvertWrapper(GrayscaleWrapper(env))
        env = FlattenObservation(env)
    else:
        if args.transform == 'permute_factors':
            env = FactorPermutationWrapper(env)
        elif args.transform == 'permute_states':
            env = ObservationPermutationWrapper(env)
        env = NormalizeWrapper(FloatWrapper(env), -1, 1)
        if args.transform == 'rotate':
            env = TransformWrapper(RotationWrapper(env), lambda x: x / np.sqrt(2))
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
