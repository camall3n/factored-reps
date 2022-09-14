import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

experiment_name = 'rl_vs_rep'
os.makedirs(f'images/{experiment_name}', exist_ok=True)

results_list = []
experiment_dirs = glob.glob(f'results/factored_rl/{experiment_name}/*/*')
for experiment_dir in experiment_dirs:
    args_filename = experiment_dir + '/args.json'
    with open(args_filename, 'r') as args_file:
        args = json.load(args_file)
    args['seed'] = int(experiment_dir.split('/')[-1])
    results_filename = experiment_dir + '/results.json'
    results = pd.read_json(results_filename, lines=True)
    # results['cost'] = -1 * results['steps']
    for arg, value in args.items():
        results[arg] = value
    results_list.append(results)

data = pd.concat(results_list, ignore_index=True)

# subset = data.query("env=='gridworld' and noise and transform=='identity'")

#%%
for env in ['gridworld', 'taxi']:
    fig = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=data.query(f"env=='{env}' and noise"),
        y='steps',
        x='episode',
        hue='transform',
        hue_order=['identity', 'rotate', 'permute_factors', 'permute_states', 'images'],
        # palette='colorblind',
    )
    plt.title(f'Learning curves ({env})')
    plt.xlabel('Episode')
    plt.ylabel('Steps to reach goal')
    plt.tight_layout()
    plt.savefig(f'images/{experiment_name}/{env}.png', facecolor='white')
    plt.show()

#%%
