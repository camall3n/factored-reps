import glob
import json
from omegaconf import OmegaConf
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

experiment_name = 'rl_vs_rep_01'
os.makedirs(f'images/{experiment_name}', exist_ok=True)

results_list = []
experiment_dirs = glob.glob(f'results/factored_rl/{experiment_name}/exp*/*')
for experiment_dir in tqdm(sorted(experiment_dirs)):
    config_filename = experiment_dir + '/config.yaml'
    try:
        with open(config_filename, 'r') as config_file:
            cfg = OmegaConf.load(config_file)
    except FileNotFoundError:
        raise
        # continue
    results_filename = experiment_dir + '/results.json'
    if not os.path.exists(results_filename):
        print(f'Results missing: {results_filename}')
        continue
    results = pd.read_json(results_filename, lines=True)
    if len(results) == 0:
        print(f'Results empty: {results_filename}')
        continue
    args = {
        'experiment': cfg.experiment,
        'env': cfg.env.name,
        'transform': cfg.transform.name,
        'agent': cfg.agent.name,
        'arch': cfg.agent.model.architecture,
        'model': cfg.agent.model.get('name', cfg.agent.model.architecture),
        'seed': cfg.seed,
        'noise': cfg.noise,
        'max_steps': cfg.env.n_steps_per_episode,
    }
    for arg, value in args.items():
        results[arg] = value
    for column in ['steps', 'reward', 'total_steps', 'total_reward']:
        results['rolling_'+column] = results[column].rolling(10).mean()
    results_list.append(results)

data = pd.concat(results_list, ignore_index=True)

#%%
# subset = data
# subset = subset.query("env=='taxi'")# and noise and transform=='identity'")
# subset = subset.query("agent=='dqn' and episode==episode.max() and seed in [1,2,3]")
# subset
#%%
quick = False
for env in ['gridworld', 'taxi']:
    subset = data
    subset = subset.query(f"env=='{env}'")
    if env == 'taxi':
        subset = subset.query(f"episode%10 == 0")
    if quick:
        subset = subset.query(f"seed==1")
    subset = subset.query(f"not noise")

    dqn = subset.query(f"agent=='dqn'")
    random = subset.query(f"agent=='random'")
    expert = subset.query(f"agent=='expert'")

    max_steps = subset.max_steps.max()
    random_steps = random.steps.mean()
    expert_steps = expert.steps.mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    y_axis = 'rolling_steps' if env == 'taxi' else 'steps'
    x_axis = 'episode'
    ax = sns.lineplot(
        data=dqn,
        y=y_axis,
        x=x_axis,
        style='model',
        hue='transform',
        hue_order=['identity', 'rotate', 'permute_factors', 'permute_states', 'images'],
        # palette='colorblind',
    )
    text_offset = dqn.episode.max() / 100
    plt.hlines(expert_steps, dqn.episode.min(), dqn.episode.max(), colors='k', linestyles='dotted')
    plt.text(dqn.episode.max() + text_offset, expert_steps, 'expert', ha='left', va='center')
    plt.hlines(random_steps, dqn.episode.min(), dqn.episode.max(), colors='k', linestyles='dotted')
    plt.text(dqn.episode.max() + text_offset, random_steps, 'random', ha='left', va='center')
    if env == 'taxi':
        plt.hlines(max_steps, dqn.episode.min(), dqn.episode.max(), colors='k', linestyles='dotted')
        plt.text(dqn.episode.max() + text_offset, max_steps, 'timeout', ha='left', va='center')
    ax.set_xlim(xmax=ax.get_xlim()[-1]*1.05) # extend right end of plot to accomodate text

    if env == 'taxi':
        sns.move_legend(ax, "center left")
    plt.title(f'Learning curves ({env})')
    plt.xlabel('Episode')
    plt.ylabel('Steps to reach goal')
    plt.tight_layout()
    if not quick:
        pass # in case we comment out the following line
        plt.savefig(f'images/{experiment_name}/{env}.png', facecolor='white')
    plt.show()

#%%
