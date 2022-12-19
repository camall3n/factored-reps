#%%

import optuna

from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice

#%%
experiment_name = 'wm_tune04'
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"./factored_rl/hyperparams/tuning/{experiment_name}.journal"))
study = optuna.load_study(
    study_name=f'{experiment_name}',
    storage=storage,
)
str(study.best_trial)
sorted([(t.values[0], str(t.params)) for t in study.get_trials() if str(t.state)=='TrialState.COMPLETE'])

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
