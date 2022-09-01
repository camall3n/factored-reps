from argparse import Namespace, ArgumentParser
import glob
import json
import os
import platform

if platform.system() == 'Linux':
    # Force matplotlib to not use any Xwindows backend.
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

parser = ArgumentParser()
parser.add_argument('--exp_num', required=True, type=int)
args = parser.parse_args()

prefix = os.path.expanduser('~/data-gdk/csal/factored/') if platform.system() == 'Linux' else ''

experiments = [
    filename.split('/')[-1]
    for filename in glob.glob(prefix + 'results/focused-taxi/logs/exp{:02d}*'.format(args.exp_num))
]

for experiment in experiments:
    modes = ['train', 'test']
    dfs = []
    for mode in modes:
        filepaths = glob.glob(prefix +
                              'results/focused-taxi/logs/{}/{}-*.txt'.format(experiment, mode))
        for filepath in filepaths:
            dirpath = os.path.dirname(filepath)
            filename = os.path.basename(filepath)
            argsfilename = filename.replace(mode, 'args')
            argsfilepath = os.path.join(dirpath, argsfilename)
            with open(argsfilepath, 'r') as argsfile:
                line = argsfile.readline()
                args = eval(line)
            if args.seed > 10:
                continue
            df = pd.read_json(filepath, lines=True, orient='records')
            for loss in ['L', 'L_rec_x', 'L_rec_z', 'L_rec_z_aug', 'L_foc']:
                if loss in df.columns:
                    df['smoothed_' + loss] = df[loss].rolling(10, center=True).mean()
            df['seed'] = args.seed
            # df['lr_G'] = args.lr_G
            df['learning_rate'] = args.lr
            df['mode'] = mode
            for name, value in vars(args).items():
                if name[:2] == 'L_':
                    df['coef_' + name] = value
            dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    def plot(seed=None):
        if seed is None:
            subset = data.query("seed <= 10")
            plot_suffix = ''
        else:
            subset = data.query("seed == {}".format(seed))
            plot_suffix = '-seed{}'.format(seed)

        # subset = subset.query("step % 200 == 0")
        # plot_suffix += '-mod200'

        y_labels = ['L', 'L_rec_x', 'L_rec_z', 'L_rec_z_aug', 'L_foc']
        y_labels = [label for label in y_labels if label in subset.columns]
        fig, axes = plt.subplots(len(y_labels), 1, sharex=True, sharey='row', figsize=(7, 12))
        p = sns.color_palette(n_colors=len(subset['mode'].unique()))
        for i, (ax, y_label) in enumerate(zip(axes, y_labels)):
            g = sns.lineplot(data=subset,
                             x='step',
                             y=y_label,
                             units='seed',
                             estimator=None,
                             style='seed',
                             hue='mode',
                             palette=p,
                             legend=(i == 0),
                             ax=ax)
            if i == 0:
                h, l = g.get_legend_handles_labels()
                ax.legend(h[0:3], l[0:3])

        results_dir = prefix + 'results/focused-taxi/images/{}/'.format(experiment)
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(results_dir + 'losses{}.png'.format(plot_suffix),
                    facecolor='white',
                    edgecolor='white')
        # plt.show()
        plt.close()

    # for seed in range(1,11):
    #     plot(seed)

    plot()

    # #%%

    # for seed in range(1, 11):
    #     subset = data.query('seed == {} and step % 200 == 0 and mode == "test"'.format(seed))
    #     idx = np.argmin(subset['L'])
    #     record = subset.iloc[idx, :]
    #     print(seed, record['step'], record['L'])
