import glob
from omegaconf import OmegaConf
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

experiment_name = 'rl_vs_rep_02'

#%%
def load_results(experiment_name) -> pd.DataFrame:
#%%
    results_list = []
    experiment_dirs = glob.glob(f'results/factored_rl/{experiment_name}/exp*/*')
    sorted_experiment_dirs = list(sorted(experiment_dirs))
    # start = 400
    # end = 500
    start = 0
    end = len(sorted_experiment_dirs)
    for i in tqdm(range(start, end)):
        # for experiment_dir in tqdm(sorted(experiment_dirs)):
        experiment_dir = sorted_experiment_dirs[i]
        config_filename = experiment_dir + '/config.yaml'
        # if 'expert' in config_filename:
        #     break
        # else:
        #     continue
        try:
            with open(config_filename, 'r') as config_file:
                cfg = OmegaConf.load(config_file)
        except FileNotFoundError:
            print(f'Missing configs: {config_filename}')
            continue
        results_filename = experiment_dir + '/results.json'
        if not os.path.exists(results_filename):
            print(f'Results missing: {results_filename}')
            continue
        results = pd.read_json(results_filename, lines=True)
        if len(results) == 0:
            print(f'Results empty: {results_filename}')
            continue
        arch = cfg.agent.model.get('architecture', cfg.agent.model.arch.encoder)
        args = {
            'experiment': cfg.experiment,
            'env': cfg.env.name,
            'trial': cfg.trial,
            'transform': cfg.transform.name,
            'agent': cfg.agent.name,
            'arch': arch,
            'model': cfg.agent.model.get('name', arch),
            'seed': cfg.seed,
            'noise': cfg.transform.noise,
            'max_steps': cfg.env.n_steps_per_episode,
        }
        for arg, value in args.items():
            results[arg] = value
        for column in ['steps', 'reward', 'total_steps', 'total_reward']:
            results['rolling_' + column] = results[column].rolling(10).mean()
        results_list.append(results)

    data = pd.concat(results_list, ignore_index=True)
    return data

#%%
def plot_results(data):
#%%
    quick = False
    os.makedirs(f'images/{experiment_name}', exist_ok=True)

    for env in ['gridworld', 'taxi']:
        subset = data
        if quick:
            subset = subset.query(f"seed==1")
        subset = subset.query(f"env=='{env}'")
        downsample_every_n_steps = 100 if env == 'taxi' else 5
        subset = subset.query(f'episode % {downsample_every_n_steps} == 0')
        dqn = subset.query(f"agent=='dqn'")
        random = subset.query(f"agent=='random'")
        expert = subset.query(f"agent=='expert'")
        max_steps = subset.max_steps.max()
        random_steps = random.steps.mean()
        expert_steps = expert.steps.mean()
        for noise_val in [True, False]:
            subset = dqn.query(f"noise=={noise_val}")

            fig, ax = plt.subplots(figsize=(12, 6))
            y_axis = 'rolling_steps' # if env == 'taxi' else 'steps'
            x_axis = 'episode'
            ax = sns.lineplot(
                data=subset,
                y=y_axis,
                x=x_axis,
                style='arch',
                style_order=['mlp', 'cnn'],
                hue='transform',
                hue_order=['identity', 'rotate', 'permute_factors', 'permute_states', 'images'],
                # palette='colorblind',
            )
            text_offset = dqn.episode.max() / 100
            plt.hlines(expert_steps,
                       dqn.episode.min(),
                       dqn.episode.max(),
                       colors='k',
                       linestyles='dotted')
            plt.text(dqn.episode.max() + text_offset, expert_steps, 'expert', ha='left', va='center')
            plt.hlines(random_steps,
                       dqn.episode.min(),
                       dqn.episode.max(),
                       colors='k',
                       linestyles='dotted')
            plt.text(dqn.episode.max() + text_offset, random_steps, 'random', ha='left', va='center')
            plt.hlines(max_steps,
                       dqn.episode.min(),
                       dqn.episode.max(),
                       colors='k',
                       linestyles='dotted')
            plt.text(dqn.episode.max() + text_offset, max_steps, 'timeout', ha='left', va='center')
            ax.set_xlim(xmax=ax.get_xlim()[-1] * 1.05) # extend right end of plot to accomodate text

            # if env == 'taxi':
            #     sns.move_legend(ax, "center left")
            noise_str = '+noise' if noise_val else ''
            plt.title(f'Learning curves ({env}{noise_str})')
            plt.xlabel('Episode')
            plt.ylabel('Steps to reach goal')
            plt.tight_layout()
            if not quick:
                pass # in case we comment out the following line
                plt.savefig(f'images/{experiment_name}/{env}{noise_str}.png', facecolor='white')
            plt.show()

#%%

def main():
    data = load_results(experiment_name)
    plot_results(data)


if __name__ == '__main__':
    main()
