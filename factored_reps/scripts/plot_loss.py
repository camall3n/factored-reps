from argparse import Namespace
import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

experiment = 'exp49-markov-save-best__learningrate_0.001'
# experiment = 'exp00-debugger'
# filepaths = glob.glob('results/logs/{}/train-2.txt'.format(experiment))

modes = ['train', 'test']
dfs = []
for mode in modes:
    filepaths = glob.glob('results/logs/{}/{}-*.txt'.format(experiment, mode))
    for filepath in filepaths:
        df = pd.read_json(filepath, lines=True, orient='records')
        for loss in ['L', 'L_inv', 'L_rat', 'L_dis', 'L_foc', 'L_fac', 'L_rec', 'L_fwd']:
            if loss in df.columns:
                df['smoothed_' + loss] = df[loss].rolling(10, center=True).mean()
        argspath = filepath.replace(mode, 'args')
        with open(argspath, 'r') as argsfile:
            line = argsfile.readline()
            args = eval(line)
        df['seed'] = args.seed
        df['learning_rate'] = args.learning_rate
        df['mode'] = mode
        for name, value in vars(args).items():
            if name[:2] == 'L_':
                df['coef_' + name] = value
        dfs.append(df)
data = pd.concat(dfs, ignore_index=True)

def plot(seed=None):
    if seed is None:
        subset = data
        plot_suffix = ''
    else:
        subset = data.query("seed == {}".format(seed))
        plot_suffix = '-seed{}'.format(seed)

    subset = subset.query("step % 200 == 0")
    plot_suffix += '-mod200'

    y_labels = ['L', 'L_inv', 'L_rat', 'L_dis', 'L_rec', 'L_foc']
    y_labels = [label for label in y_labels if label in subset.columns]
    fig, axes = plt.subplots(len(y_labels), 1, sharex=True, sharey='row', figsize=(7, 12))
    p = sns.color_palette(n_colors=len(subset['mode'].unique()))
    for ax, y_label in zip(axes, y_labels):
        sns.lineplot(data=subset,
                        x='step',
                        y=y_label,
                        units='seed',
                        estimator=None,
                        style='seed',
                        hue='mode',
                        palette=p,
                        # legend=False,
                        ax=ax)

    results_dir = 'results/loss_plots/'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(results_dir + '{}{}.png'.format(experiment, plot_suffix), facecolor='white', edgecolor='white')
    plt.show()

# for seed in range(1,11):
#     plot(seed)

plot()


#%%

seed =
subset = data.query('seed == {} and step % 200 == 0 and mode == "test"'.format(seed))
idx = np.argmin(subset['L'])
subset.iloc[idx,:]['step']