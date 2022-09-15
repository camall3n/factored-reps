import json
import logging
import yaml

# Args & hyperparams
from factored_rl import configs
import hydra

# Env
import gym
from gym.wrappers import FlattenObservation, TimeLimit
import numpy as np
from visgrid.envs import GridworldEnv, TaxiEnv
from factored_rl.wrappers import RotationWrapper
from factored_rl.wrappers import FactorPermutationWrapper, ObservationPermutationWrapper
from visgrid.wrappers import GrayscaleWrapper, InvertWrapper, FloatWrapper, NormalizeWrapper, NoiseWrapper, TransformWrapper

# Agent
from factored_rl.agents.dqn import DQNAgent
from tqdm import tqdm

# ----------------------------------------
# Environment & wrappers
# ----------------------------------------
def initialize_env(args, cfg: configs.RLvsRepConfig):
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

    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    if args.transform == 'images':
        env.set_rendering(enabled=True)
        env = InvertWrapper(GrayscaleWrapper(env))
        if cfg.agent.model.architecture == 'mlp':
            env = FlattenObservation(env)
    else:
        if args.transform == 'permute_factors':
            env = FactorPermutationWrapper(env)
        elif args.transform == 'permute_states':
            env = ObservationPermutationWrapper(env)
        env = NormalizeWrapper(FloatWrapper(env), -1, 1)
        if args.transform == 'rotate':
            env = RotationWrapper(env)
    if args.noise:
        env = NoiseWrapper(env, cfg.env.noise_std)

    env = TimeLimit(env, max_episode_steps=cfg.env.n_steps_per_episode)
    return env

# ----------------------------------------
# Agent
# ----------------------------------------

def initialize_agent(env, args, cfg: configs.AgentConfig):
    agent = DQNAgent(env.observation_space, env.action_space, cfg)
    return agent

# ----------------------------------------
# Evaluate RL performance
# ----------------------------------------
def train_agent_on_env(agent, env, n_episodes, results_file=None):
    total_reward = 0
    total_steps = 0
    losses = []
    results = []
    for episode in tqdm(range(n_episodes), desc='episodes'):
        ob, info = env.reset()
        terminal, truncated = False, False
        ep_rewards = []
        ep_steps = 0
        while not (terminal or truncated):
            action = agent.act(ob)
            next_ob, reward, terminal, truncated, info = env.step(action)
            experience = {
                'ob': ob,
                'action': action,
                'reward': reward,
                'terminal': terminal,
                'truncated': truncated,
                'next_ob': next_ob,
            }
            agent.store(experience)

            loss = agent.update()
            losses.append(loss)
            ep_rewards.append(reward)
            ep_steps += 1
            ob = next_ob
            if terminal or truncated:
                break

        total_steps += ep_steps
        total_reward += sum(ep_rewards)
        episode_result = {
            'episode': episode,
            'reward': sum(ep_rewards),
            'steps': ep_steps,
            'total_reward': total_reward,
            'total_steps': total_steps,
        }
        results.append(episode_result)
        if results_file is not None:
            results_file.write(json.dumps(episode_result) + '\n')
        logging.getLogger().info('\n' + yaml.dump(episode_result, sort_keys=False))
    return results

# ----------------------------------------
# Run experiment & save results
# ----------------------------------------

@hydra.main(config_path=None, config_name='rl_vs_rep', version_base=None)
def main(cfg):
    configs.initialize_experiment(cfg)

    # env = initialize_env(args, cfg.env)
    # agent = initialize_agent(env, args, cfg.agent)
    # filename = cfg.experiment.dir + 'args.json'
    # with open(filename, 'w') as args_file:
    #     json.dump(
    #         {
    #             'experiment': args.experiment,
    #             'trial': args.trial,
    #             'seed': args.seed,
    #             'env': cfg.env.name,
    #             'noise': args.noise,
    #             'transform': args.transform,
    #             'agent': cfg.agent.name,
    #             'model': cfg.agent.model.architecture,
    #         }, args_file)

    # filename = cfg.experiment.dir + 'results.json'
    # with open(filename, 'w') as results_file:
    #     train_agent_on_env(agent, env, cfg.env.n_training_episodes, results_file)

main()
