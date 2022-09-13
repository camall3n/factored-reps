import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results = []
experiment_dirs = glob.glob('results/factored_rl/disent_vs_rep/*/*')
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
fig = plt.subplots(figsize=(12, 6))
sns.barplot(data=long_data.query("env=='gridworld' and noise"),
            y='score',
            x='metric',
            hue='transform',
            hue_order=['identity', 'permute_factors', 'permute_states', 'rotate', 'images'],
            palette='colorblind')
plt.title('Metric score vs. representation transform (10x10 gridworld)')
plt.xlabel('Metric')
plt.ylabel('Score')

#%%
