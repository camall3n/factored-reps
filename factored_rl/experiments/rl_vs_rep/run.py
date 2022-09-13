import json
import yaml

# Args & hyperparams
from factored_rl import configs

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
# Args & hyperparameters
# ----------------------------------------

parser = configs.new_parser()
# yapf: disable
parser.add_argument('-e', '--experiment', type=str, default='rl_vs_rep', help='A name for the experiment')
parser.add_argument('-t', '--trial', type=str, default='trial', help='A name for the trial')
parser.add_argument('-s', '--seed', type=int, default=0, help='A seed for the random number generator')
parser.add_argument('--no-timestamp', action='store_true', help='Disable automatic trial timestamps')
parser.add_argument('--noise', action='store_true')
parser.add_argument('--transform', type=str, default='identity', choices=['identity', 'images', 'permute_factors', 'permute_states', 'rotate'])
parser.add_argument('--test', action='store_true')
parser.add_argument('-f', '--fool-ipython', action='store_true',
    help='Dummy arg to make ipython happy')
# yapf: enable

# ----------------------------------------
# Environment & wrappers
# ----------------------------------------
def initialize_env(args, env_cfg: configs.EnvConfig):
    if env_cfg.name == 'gridworld':
        env = GridworldEnv(10,
                           10,
                           exploring_starts=True,
                           terminate_on_goal=True,
                           fixed_goal=True,
                           hidden_goal=True,
                           should_render=False,
                           dimensions=GridworldEnv.dimensions_onehot)
    elif env_cfg.name == 'taxi':
        env = TaxiEnv(size=5,
                      n_passengers=1,
                      exploring_starts=True,
                      terminate_on_goal=True,
                      should_render=False,
                      dimensions=TaxiEnv.dimensions_5x5_to_48x48)
    else:
        env = gym.make(env_cfg.name)
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
        env = NoiseWrapper(env, env_cfg.noise_std)

    env = TimeLimit(env, max_episode_steps=env_cfg.n_steps_per_episode)
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
        if results_file is not None:
            json_str = json.dumps(episode_result)
            results_file.write(json_str + '\n')
            results_file.flush()
        log.info('\n' + yaml.dump(episode_result, sort_keys=False))

# ----------------------------------------
# Run experiment
# ----------------------------------------

args, cfg, log = configs.initialize_experiment(parser)
env = initialize_env(args, cfg.env)
agent = initialize_agent(env, args, cfg.agent)

filename = cfg.experiment.dir + 'results.json'
with open(filename, 'w') as results_file:
    results_file.write('[\n')
    train_agent_on_env(agent, env, cfg.env.n_training_episodes, results_file)
    results_file.write(']\n')
