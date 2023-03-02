import glob
from omegaconf import OmegaConf, errors
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# experiment_name = 'rl_vs_rep_02'
experiment_names = ['gt_03', 'cnn_08', 'gt_nupdates']
# experiment_names = ['poly_3', 'poly_4', 'leg_3', 'leg_4', 'four_3', 'four_4']
# experiment_names = []

def try_load(load_fn, dirname, filename_list):
    for filename in filename_list:
        try:
            with open(dirname + '/' + filename, 'r') as file:
                result = load_fn(file)
        except FileNotFoundError:
            continue # check next filename
        except ValueError:
            print(f'Potentially clobbered output in {dirname}')
            continue
        break # already loaded
    else:
        print(f'Could not find {filename_list} in {dirname}')
        result = None
    return result

#%%
def load_results(experiment_name) -> pd.DataFrame:
    #%%
    results_list = []
    experiment_dirs = glob.glob(f'results/factored_rl/{experiment_name}/*/*')
    sorted_experiment_dirs = list(sorted(experiment_dirs))
    # start = 400
    # end = 500
    start = 0
    end = len(sorted_experiment_dirs)
    for i in tqdm(range(start, end)):
        experiment_dir = sorted_experiment_dirs[i]

        cfg = try_load(OmegaConf.load, experiment_dir, ['config_rl_vs_rep.yaml', 'config.yaml'])
        if cfg is None:
            continue

        results = try_load(lambda x: pd.read_json(x, lines=True), experiment_dir, ['results.jsonl', 'results.json'])
        if results is None:
            continue
        if len(results) == 0:
            print(f'Results empty: {experiment_dir}')
            continue

        try:
            arch = cfg.model.arch.get('type', cfg.model.arch)
        except errors.ConfigAttributeError:
            arch = cfg.agent.model.get('type', cfg.agent.model.architecture)
        noise = cfg.transform.get('noise', cfg.transform.noise_std) > 0
        args = {
            'experiment': cfg.experiment,
            'env': cfg.env.name,
            'trial': cfg.trial,
            'transform': cfg.transform.name,
            'agent': cfg.agent.name,
            'arch': arch,
            'rl_lr': cfg.trainer.get('rl_learning_rate', 0) if cfg.get('trainer', None) is not None else 0,
            'epsilon_half_life': cfg.agent.epsilon_half_life_steps if cfg.agent.name == 'dqn' else 0,
            'target_copy_alpha': cfg.agent.target_copy_alpha if cfg.agent.name == 'dqn' else 0,
            'seed': cfg.seed,
            'noise': noise,
            'max_steps': cfg.env.n_steps_per_episode,
        }
        try:
            args['updates_per_interaction'] = cfg.agent.updates_per_interaction
        except:
            args['updates_per_interaction'] = 1
        for arg, value in args.items():
            results[arg] = value
        for column in ['steps', 'reward', 'total_steps', 'total_reward']:
            results['rolling_' + column] = results[column].rolling(10).mean()
        results_list.append(results)

    if results_list == []:
        raise RuntimeError(f'No results for experiment "{experiment_name}"')
    data = pd.concat(results_list, ignore_index=True)
    return data

#%%
def plot_results(data):
    #%%
    quick = False
    os.makedirs(f'images/{experiment_name}', exist_ok=True)

    for env in ['taxi']:#['gridworld', 'taxi']:break
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
        for noise_val in [True]:#, False]:
            subset = dqn.query(f"noise=={noise_val}")

            fig, ax = plt.subplots(figsize=(12, 6))
            y_axis = 'rolling_steps' # if env == 'taxi' else 'steps'
            x_axis = 'episode'
            ax = sns.lineplot(
                data=subset,
                y=y_axis,
                x=x_axis,
                style='experiment',
                # units='trial',
                # units='seed',
                # estimator=None,
                # style_order=['mlp', 'cnn'],
                hue='experiment',
                # hue_order=['identity', 'rotate', 'permute_factors', 'permute_states', 'images'],
                # palette='colorblind',
            )
            # best_seed = dqn.query(f'trial=="trial_0140"')
            # ax.plot(best_seed[x_axis], best_seed[y_axis], 'r')
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
    baselines = load_results('rl_baselines')
    data = baselines.copy()
    for experiment_name in experiment_names:
        exp_data = load_results(experiment_name)
        data = pd.concat((exp_data, data), ignore_index=True)
    plot_results(data.query("agent!='dqn' or (experiment=='cnn_08' and rl_lr in [0.0002, 0.0003]) or (experiment == 'gt_nupdates' and (updates_per_interaction > 4 and rl_lr < 2e-4 and epsilon_half_life < 4400 and epsilon_half_life > 2500 and target_copy_alpha < 0.1)) or (experiment == 'gt_03')"))
    # plot_results(data.query("agent!='dqn' or (arch=='qnet' and (experiment=='rl_vs_rep_02' or (rl_lr < 0.001)))"))



#%%
# plot_experiments = ['gt_03', 'rl_vs_rep_02', 'mlp_tune02', 'cnn_08']
final_episode_data = data.query(f"episode==9999").sort_values(by='experiment')
final_episode_data['experiment'].unique()
# subset_data = final_episode_data.query("experiment=='mlp_tune02'")# or (rl_lr < 0.00065 and rl_lr > 0.00025 and epsilon_half_life < 5000 and epsilon_half_life > 2500 and target_copy_alpha > 0.03)")
#%%
subset = final_episode_data
subset = subset.query("experiment != 'gt_nupdates' or (updates_per_interaction > 20 and rl_lr < 2e-4 and epsilon_half_life < 4400 and epsilon_half_life > 2500 and target_copy_alpha < 0.1)")
ax = sns.scatterplot(data=subset, x='rl_lr', y='total_steps', hue='experiment', alpha=0.5)
# ax.set_xscale('log')
ax.set_xlim([-0.0002, 0.0032])
ax.legend(loc='upper right')
#%%


if __name__ == '__main__':
    main()
