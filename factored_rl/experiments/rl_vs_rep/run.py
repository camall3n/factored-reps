import json
import logging
import yaml

# Args & hyperparams
import hydra
from factored_rl import configs

# Env
from factored_rl.experiments.common import initialize_env, initialize_model, get_checkpoint_path
from gym.wrappers import TimeLimit

# Agent
from factored_rl.agents.dqn import DQNAgent
from factored_rl.agents.randomagent import RandomAgent
from visgrid.agents.expert import GridworldExpert, TaxiExpert
from tqdm import tqdm

# ----------------------------------------
# Args & hyperparams
# ----------------------------------------

@hydra.main(config_path="../conf", config_name='config', version_base=None)
def main(cfg: configs.Config):
    configs.initialize_experiment(cfg, 'rl_vs_rep')

    env = initialize_env(cfg, cfg.seed)
    if cfg.trainer.quick:
        cfg.env.n_steps_per_episode = 5
        cfg.env.n_training_episodes = 2
        cfg.agent.replay_warmup_steps = cfg.trainer.batch_size
    env = TimeLimit(env, max_episode_steps=cfg.env.n_steps_per_episode)

    agent = initialize_agent(env, cfg)

    filename = cfg.dir + 'results.json'
    with open(filename, 'w') as results_file:
        train_agent_on_env(agent, env, cfg.env.n_training_episodes, results_file)

    if not cfg.loader.should_load:
        ckpt_path = get_checkpoint_path(cfg, logs_dirname='pytorch_logs', create_new_version=True)
        agent.save('qnet', ckpt_path, is_best=False)

# ----------------------------------------
# Agent
# ----------------------------------------

def initialize_agent(env, cfg: configs.Config):
    if cfg.agent.name == 'expert':
        if cfg.env.name == 'gridworld':
            agent = GridworldExpert(env.unwrapped)
        elif cfg.env.name == 'taxi':
            agent = TaxiExpert(env.unwrapped)
    elif cfg.agent.name == 'dqn':
        model = initialize_model(env.observation_space.shape, env.action_space.n, cfg)
        agent = DQNAgent(env.action_space, model, cfg)
    elif cfg.agent.name == 'random':
        agent = RandomAgent(env.action_space)
    else:
        raise NotImplementedError
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
# Run experiment
# ----------------------------------------

if __name__ == '__main__':
    main()
