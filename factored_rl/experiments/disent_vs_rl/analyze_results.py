import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from factored_rl.experiments.disent_vs_rep.analyze_results import load_results as load_disent_results
from factored_rl.experiments.rl_vs_rep.analyze_results import load_results as load_rl_results

#%% load results
experiment_name = 'disent_vs_rl'
disent_data = load_disent_results('disent_vs_rep', melt=False)
rl_data = load_rl_results('rl_vs_rep_02')

#%% define helper functions
def compute_means_and_keep_strings(data, trial):
    data_means = data.mean(numeric_only=True)
    non_numeric_data_cols = data.applymap(lambda x: isinstance(x, str)).all(0)
    non_numeric_data = data[data.columns[non_numeric_data_cols]]
    if (non_numeric_data.nunique() != 1).all(0):
        print(f'Skipping trial "{trial}", due to non-unique configs:')
        print(non_numeric_data.nunique())
        return None
    else:
        return pd.concat((non_numeric_data.iloc[0], data_means))

def tag_rows_for_merging(data):
    env = data['env']
    transform = data['transform']
    noise = pd.Series(np.where(data['noise'] == True, '+noise', '-noise'))
    data['tag'] = env + '__' + transform + '__' + noise

#%%
def summarize_disent_data(disent_data):
    #%%
    disent_trials = disent_data.trial.unique()
    disent_summaries = []
    for trial in tqdm(disent_trials):
        trial_data = disent_data.query(f'trial=="{trial}"')
        trial_summary = compute_means_and_keep_strings(trial_data, trial)
        if trial_summary is not None:
            disent_summaries.append(trial_summary)

    disent_summary_data = pd.concat(disent_summaries, axis=1, ignore_index=True).transpose()
    tag_rows_for_merging(disent_summary_data)
    return disent_summary_data

#%%
summarized_disent_data = summarize_disent_data(disent_data)

#%%
def summarize_rl_data(rl_data):
    #%%
    rl_trials = rl_data.trial.unique()
    rl_summaries = []
    for trial in tqdm(rl_trials):
        trial_data = rl_data.query(f'trial=="{trial}"')
        trial_seeds = trial_data.seed.unique()

        # grab the row for the final episode in each trial
        trial_final_episodes = []
        for seed in trial_seeds:
            seed_data = trial_data.query(f'seed=={seed}')
            final_episode_num = seed_data.episode.max()
            final_episode = seed_data.query(f'episode=={final_episode_num}')
            trial_final_episodes.append(final_episode)
        trial_final_ep_data = pd.concat(trial_final_episodes, ignore_index=True)

        # confirm that all trials were the same length
        longest_trial_ep_num = trial_final_ep_data.episode.max()
        filtered_final_ep_data = trial_final_ep_data.query(f'episode=={longest_trial_ep_num}')
        if len(filtered_final_ep_data) < len(trial_final_ep_data):
            incomplete_seeds = trial_final_ep_data.query(
                f'episode!={longest_trial_ep_num}').seed.unique()
            print(f'Trial "{trial}" has incomplete seeds:', incomplete_seeds)

        trial_summary = compute_means_and_keep_strings(filtered_final_ep_data, trial)
        if trial_summary is not None:
            rl_summaries.append(trial_summary)

    rl_summary_data = pd.concat(rl_summaries, axis=1, ignore_index=True).transpose()
    tag_rows_for_merging(rl_summary_data)
    return rl_summary_data

#%%
summarized_rl_data = summarize_rl_data(rl_data)

#%%
def merge_results(summarized_disent_data, summarized_rl_data):
    #%%
    # drop columns that are duplicated or irrelevant
    disent_cols_to_drop = ['experiment', 'trial', 'env', 'transform', 'noise', 'seed']
    rl_cols_to_keep = [
        'env', 'transform', 'noise', 'agent', 'arch', 'model', 'episode', 'total_reward',
        'total_steps', 'max_steps', 'tag'
    ]
    disent_data_to_merge = summarized_disent_data.drop(columns=disent_cols_to_drop)
    rl_data_to_merge = summarized_rl_data[rl_cols_to_keep]
    merged_data = rl_data_to_merge.merge(disent_data_to_merge, on='tag')
    melted_data = pd.melt(merged_data,
                          id_vars=rl_cols_to_keep,
                          var_name='metric',
                          value_name='disent_score')
    melted_data[
        'reward_per_step'] = melted_data['total_reward'] / melted_data['total_steps'].astype(float)
    for col in melted_data.columns:
        melted_data[col] = pd.to_numeric(melted_data[col], errors='ignore')
    return melted_data

#%%
a = summarized_disent_data.query('env=="gridworld" and noise==False')
b = summarized_rl_data.query('env=="gridworld" and noise==False')
merged_data = merge_results(summarized_disent_data, summarized_rl_data)

#%% plot all metrics together
os.makedirs(f'images/{experiment_name}', exist_ok=True)
for env in ['gridworld', 'taxi']:
    subset = merged_data.query(f'env=="{env}"')
    expert = subset.query('agent=="expert"')
    random = subset.query('agent=="random"')
    dqn = subset.query('agent=="dqn"')

    # expert_rl_performance = expert.reward_per_step.mean()
    random_rl_performance = random.reward_per_step.mean()
    worst_case_rl_performance = 0

    text_offset = (random_rl_performance - worst_case_rl_performance) / 10
    # xlim = (worst_case_rl_performance - 3 * text_offset, expert_rl_performance + 3 * text_offset)
    ylim = (-0.03, 1.03)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # for met, ls in zip(dqn['metric'].unique(),
    #                    ['solid', 'dashed', 'dashdot', (0, (3, 1, 1, 1, 1)), 'dotted']):
    #     sns.regplot(x="reward_per_step",
    #                 y="disent_score",
    #                 data=dqn.loc[dqn['metric'] == met],
    #                 scatter=False,
    #                 line_kws={"ls": ls},
    #                 color='k',
    #                 ax=ax,
    #                 label=met,
    #                 ci=None)
    for tfm, mkr, hue in zip(['identity', 'rotate', 'permute_factors', 'permute_states', 'images'],
                             ["o", "x", ".", "D", '+'], ['C0', 'C1', 'C2', 'C3', 'C4']):
        sns.scatterplot(x="reward_per_step",
                        y="disent_score",
                        data=dqn.loc[dqn['transform'] == tfm],
                        color=hue,
                        marker=mkr,
                        ax=ax,
                        label=tfm,
                        legend=False)
    ax.legend()
    ax.set_ylim(ylim)
    ax.set_xlabel('Reward per step')
    ax.set_ylabel('Disentanglement (any metric)')
    ax.set_title(f'Disentanglement vs. RL Performance ({env})')
    plt.tight_layout()
    plt.savefig(f'images/{experiment_name}/{env}.png', facecolor='white')
    plt.show()

#%% plot each metric separately
os.makedirs(f'images/{experiment_name}', exist_ok=True)
for env in ['gridworld', 'taxi']:
    subset = merged_data.query(f'env=="{env}"')
    expert = subset.query('agent=="expert"')
    random = subset.query('agent=="random"')
    dqn = subset.query('agent=="dqn"')

    # expert_rl_performance = expert.reward_per_step.mean()
    random_rl_performance = random.reward_per_step.mean()
    worst_case_rl_performance = 0

    text_offset = (random_rl_performance - worst_case_rl_performance) / 10
    # xlim = (worst_case_rl_performance - 3 * text_offset, expert_rl_performance + 3 * text_offset)
    ylim = (-0.03, 1.03)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, met, use_legend in zip(axes, dqn["metric"].unique(), [True] + [False] * 4):
        met_data = dqn.query(f"metric=='{met}'")
        for tfm, mkr, hue in zip(
            ['identity', 'rotate', 'permute_factors', 'permute_states', 'images'],
            ["o", "x", ".", "D", '+'], ['C0', 'C1', 'C2', 'C3', 'C4']):
            sns.scatterplot(x="reward_per_step",
                            y="disent_score",
                            data=met_data.loc[met_data['transform'] == tfm],
                            color=hue,
                            marker=mkr,
                            ax=ax,
                            label=tfm,
                            legend=use_legend)
        sns.regplot(x="reward_per_step",
                    y="disent_score",
                    data=met_data,
                    scatter=False,
                    color='k',
                    ax=ax,
                    label=met)
        ax.set_ylim(ylim)
        ax.set_xlabel('Reward per step')
        ax.set_ylabel('Disentanglement')
        ax.set_title(str(met))
    fig.suptitle(f'Disentanglement vs. RL Performance ({env})', y=1.05)
    plt.tight_layout()
    plt.savefig(f'images/{experiment_name}/{env}_by_metric.png', facecolor='white')
    plt.show()

#%%
