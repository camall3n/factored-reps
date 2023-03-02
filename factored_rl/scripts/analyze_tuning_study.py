#%%

import optuna
import seaborn as sns
import pandas as pd

from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice

#%%
experiment_name = 'gt_nupdates'
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"./factored_rl/hyperparams/tuning/{experiment_name}.journal"))
study = optuna.load_study(
    study_name=f'{experiment_name}'.replace('_reduced',''),
    storage=storage,
)
str(study.best_trial)
finished_trials = sorted([(t.values[0], t) for t in study.get_trials() if str(t.state) in ['TrialState.COMPLETE', 'TrialState.PRUNED']])
#%%
# both_sprites = sorted([258, 440, 253, 364, 484, 220, 198])
# one_sprite = sorted([326, 251, 398, 455, 441, 456, 135])

one_sprite.extend([t.number for value, t in finished_trials if str(t.state) == 'TrialState.PRUNED'])

both_entries = sorted([(t.number, t.values[0], t.params) for t in study.get_trials() if t.number in both_sprites])
[entry[-1].update({'sprites': 2,'trial': entry[0], 'loss/reconst': entry[1]}) for entry in both_entries]
both_entries = [entry[-1] for entry in both_entries]

one_entries = sorted([(t.number, t.values[0], t.params) for t in study.get_trials() if t.number in one_sprite])
[entry[-1].update({'sprites': 1,'trial': entry[0], 'loss/reconst': entry[1]}) for entry in one_entries]
one_entries = [entry[-1] for entry in one_entries]

both_entries.extend(one_entries)


df = pd.DataFrame(both_entries)
#%%
results = sorted([(t.number, t.values[0], t.params) for t in study.get_trials() if t.values is not None])
[entry[-1].update({'trial': entry[0], 'steps_per_episode': entry[1]}) for entry in results]
results = [entry[-1] for entry in results]
df = pd.DataFrame(results)

#%%
subset = df
subset = subset.query("rl_lr < 2e-4")
# subset = subset.query("epsilon_half_life < 4400 and epsilon_half_life > 2500")
# subset = subset.query("target_copy_alpha < 0.1")
subset = subset.query("1 < updates_per_interaction < 40")
ax = sns.scatterplot(data=subset, x='rl_lr', y='steps_per_episode', hue='updates_per_interaction')
# ax.set_xscale('log')
# ax.set_yscale('log')

#%%
sns.kdeplot(data=df, hue='sprites', x='loss.actions', log_scale=True, cut=0)
sns.kdeplot(data=df, hue='sprites', x='loss.effects', log_scale=True, cut=0)
sns.kdeplot(data=df, hue='sprites', x='loss.parents', log_scale=True, cut=0)
sns.histplot(data=df, hue='sprites', x='loss.sparsity.name')
sns.kdeplot(data=df, hue='sprites', x='loss/reconst', log_scale=False, clip=[0,0.04], cut=0)

import matplotlib as mpl
import matplotlib.pyplot as plt

x_var = 'loss.parents'
y_var = 'loss.actions'
z_var = 'loss/reconst'
ax = sns.scatterplot(data=df.sort_values(by=z_var, ascending=False), y=y_var, x=x_var, hue=z_var, palette=sns.color_palette("viridis", as_cmap=True), hue_norm=mpl.colors.LogNorm())
ax.set_xscale('log')
ax.set_yscale('log')
norm = plt.Normalize(df[z_var].min(), df[z_var].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
ax.get_legend().remove()
ax.figure.colorbar(sm, label=z_var)
sns.kdeplot(data=df.query('sprites==2'), color='k', y=y_var, x=x_var, log_scale=False, ax=ax, levels=4)
ax.set_title('Hyperparam distribution of "good" trials')


#%%
# experiment_name = 'wm_tune03'
# study = optuna.load_study(
#     study_name=f'{experiment_name}',
#     storage=f"sqlite:///factored_rl/hyperparams/tuning/{experiment_name}.db",
# )
#%%
plot_contour(study)
plot_edf(study)
plot_optimization_history(study)
plot_parallel_coordinate(study)
plot_param_importances(study)
plot_slice(study)
