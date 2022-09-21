from factored_rl.experiments import configs

# Env
import gym
from gym.wrappers import FlattenObservation, TimeLimit
import numpy as np
from visgrid.envs import GridworldEnv, TaxiEnv
from factored_rl.wrappers import RotationWrapper
from factored_rl.wrappers import FactorPermutationWrapper, ObservationPermutationWrapper
from visgrid.wrappers import GrayscaleWrapper, InvertWrapper, FloatWrapper, NormalizeWrapper, NoiseWrapper, TransformWrapper

# ----------------------------------------
# Environment & wrappers
# ----------------------------------------

def initialize_env(cfg: configs.Config):
    if cfg.env.name == 'gridworld':
        env = GridworldEnv(10,
                           10,
                           exploring_starts=True,
                           terminate_on_goal=True,
                           fixed_goal=True,
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

    env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)

    if cfg.transform.name == 'images':
        env.set_rendering(enabled=True)
        env = InvertWrapper(GrayscaleWrapper(env))
        if cfg.agent.model.architecture == 'mlp':
            env = FlattenObservation(env)
    else:
        if cfg.transform.name == 'permute_factors':
            env = FactorPermutationWrapper(env)
        elif cfg.transform.name == 'permute_states':
            env = ObservationPermutationWrapper(env)
        env = NormalizeWrapper(FloatWrapper(env), -1, 1)
        if cfg.transform.name == 'rotate':
            env = RotationWrapper(env)
    if cfg.noise:
        env = NoiseWrapper(env, cfg.transform.noise_std)

    return env
