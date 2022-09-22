import os

from factored_rl.experiments import configs

# Env
import gym
from gym.wrappers import FlattenObservation, TimeLimit
import numpy as np
from visgrid.envs import GridworldEnv, TaxiEnv
from factored_rl.wrappers import RotationWrapper
from factored_rl.wrappers import FactorPermutationWrapper, ObservationPermutationWrapper
from visgrid.wrappers import GrayscaleWrapper, InvertWrapper, ToFloatWrapper, NormalizeWrapper, NoiseWrapper, TransformWrapper

# ----------------------------------------
# Environment & wrappers
# ----------------------------------------

def initialize_env(cfg: configs.Config, seed: int = None):
    if cfg.env.name == 'gridworld':
        env = GridworldEnv(10,
                           10,
                           exploring_starts=cfg.env.exploring_starts,
                           terminate_on_goal=True,
                           fixed_goal=cfg.env.fixed_goal,
                           hidden_goal=True,
                           should_render=False,
                           dimensions=GridworldEnv.dimensions_onehot)
    elif cfg.env.name == 'taxi':
        env = TaxiEnv(size=5,
                      n_passengers=1,
                      exploring_starts=cfg.env.exploring_starts,
                      terminate_on_goal=True,
                      fixed_goal=cfg.env.fixed_goal,
                      depot_dropoff_only=cfg.env.depot_dropoff_only,
                      should_render=False,
                      dimensions=TaxiEnv.dimensions_5x5_to_64x64)
    else:
        env = gym.make(cfg.env.name)
        # TODO: wrap env to support disent protocol

    env.reset(seed=seed)
    env.action_space.seed(seed)

    if cfg.transform.name == 'images':
        env.set_rendering(enabled=True)
        env = InvertWrapper(GrayscaleWrapper(env))
        if cfg.model.architecture == 'mlp':
            env = FlattenObservation(env)
    else:
        if cfg.transform.name == 'permute_factors':
            env = FactorPermutationWrapper(env)
        elif cfg.transform.name == 'permute_states':
            env = ObservationPermutationWrapper(env)
        env = NormalizeWrapper(ToFloatWrapper(env), -1, 1)
        if cfg.transform.name == 'rotate':
            env = RotationWrapper(env)
    if cfg.transform.noise:
        env = NoiseWrapper(env, cfg.transform.noise_std)
    env = ToFloatWrapper(env)
    return env

def cpu_count():
    # os.cpu_count()
    #     returns number of cores on machine
    # os.sched_getaffinity(pid)
    #     returns set of cores on which process is allowed to run
    #     if pid=0, results are for current process
    #
    # if os.sched_getaffinity doesn't exist, just return cpu_count and hope for the best
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()
