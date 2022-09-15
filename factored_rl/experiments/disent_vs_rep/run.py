import json
import logging

# Args & hyperparams
import hydra
from factored_rl import configs

# Env
import gym
from gym.wrappers.flatten_observation import FlattenObservation
import numpy as np
from visgrid.envs import GridworldEnv, TaxiEnv
from factored_rl.wrappers import RotationWrapper
from factored_rl.wrappers import FactorPermutationWrapper, ObservationPermutationWrapper
from visgrid.wrappers import GrayscaleWrapper, InvertWrapper, FloatWrapper, NormalizeWrapper, NoiseWrapper, TransformWrapper

# Disent
from disent import metrics
from disent.util.seeds import seed as disent_seed
from disent.dataset.data import GymEnvData
from disent.dataset import DisentDataset
from disent.dataset.sampling import SingleSampler

# ----------------------------------------
# Args & hyperparams
# ----------------------------------------

@hydra.main(config_path=None, config_name='rl_vs_rep', version_base=None)
def main(cfg):
    configs.initialize_experiment(cfg)
    env = initialize_env(cfg)
    data = GymEnvData(env)

    metric_scores = {}
    for metric in initialize_metrics():
        dataset = DisentDataset(dataset=data, sampler=SingleSampler())
        scores = metric(dataset, lambda x: x)
        metric_scores.update(scores)

    results = dict(**metric_scores)
    results.update({
        'experiment': cfg.experiment,
        'trial': cfg.trial,
        'seed': cfg.seed,
        'env': cfg.env.name,
        'noise': cfg.noise,
        'transform': cfg.transform.name,
    })

    # Save results
    filename = cfg.dir + 'results.json'
    with open(filename, 'w') as file:
        json.dump(results, file)
    logging.getLogger().info(f'Results logged to: {filename}')

# ----------------------------------------
# Environment & wrappers
# ----------------------------------------
def initialize_env(cfg: configs.RLvsRepConfig):
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
                      exploring_starts=True,
                      terminate_on_goal=True,
                      should_render=False,
                      dimensions=TaxiEnv.dimensions_5x5_to_48x48)
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
        env = NoiseWrapper(env, cfg.env.noise_std)

    return env

# ----------------------------------------
# Disent metrics
# ----------------------------------------

def initialize_metrics():
    return [
        metrics.metric_dci,
        metrics.metric_mig,
    ]

if __name__ == '__main__':
    main()
