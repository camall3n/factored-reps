import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

experiment_name = 'disent_vs_rep'
os.makedirs(f'images/{experiment_name}', exist_ok=True)

results = []
experiment_dirs = glob.glob(f'results/factored_rl/{experiment_name}/*/*')
for experiment_dir in experiment_dirs:
    filename = experiment_dir + '/results.json'
    if os.path.exists(filename):
        with open(filename) as file:
            results.append(json.load(file))

results = pd.DataFrame(results)
results = results.rename(
    columns={
        'dci.disentanglement': 'DCI:D',
        'dci.completeness': 'DCI:C',
        'dci.informativeness_train': 'DCI:I(train)',
        'dci.informativeness_test': 'DCI:I(test)',
        'mig.discrete_score': 'MIG',
    })
long_data = pd.melt(results,
                    id_vars=['experiment', 'trial', 'seed', 'env', 'noise', 'transform'],
                    var_name='metric',
                    value_name='score')

#%%
for env in ['gridworld', 'taxi']:
    fig = plt.subplots(figsize=(12, 6))
    sns.barplot(data=long_data.query(f"env=='{env}' and noise"),
                y='score',
                x='metric',
                hue='transform',
                hue_order=['identity', 'rotate', 'permute_factors', 'permute_states', 'images'],
                # palette='colorblind',
                )
    plt.title(f'Metric score vs. representation transform ({env})')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(f'images/{experiment_name}/{env}.png', facecolor='white')
    plt.show()

#%%
