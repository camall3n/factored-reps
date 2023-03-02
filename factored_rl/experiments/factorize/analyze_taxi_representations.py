import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import seaborn as sns
import torch

from factored_rl import configs
from factored_rl.experiments.common import initialize_env, initialize_model
from factored_rl.experiments.factorize.run import initialize_dataloader

load_dir = '/Users/cam/dev/factored-reps/results/factored_rl/wm_tune08/trial_0182/0001/'
cfg = OmegaConf.load(load_dir + 'config_factorize.yaml')
cfg.dir = load_dir
cfg.script = 'factorize'
cfg.transform.basis = {'name': None}
cfg.model.qnet.basis = {'name': None}
cfg.loader.load_model = True
cfg.loader.eval_only = True
train_dl, input_shape, n_actions = initialize_dataloader(cfg, cfg.seed)
model = initialize_model(input_shape, n_actions, cfg)
env = initialize_env(cfg, cfg.seed)

def get_obs(s):
    obs = env.get_observation(s)
    wrapper_list = []
    env_iter = env
    while hasattr(env_iter, 'observation'):
        wrapper_list.append(env_iter)
        env_iter = env_iter.env
    for wrapper in reversed(wrapper_list):
        obs = wrapper.observation(obs)
    return obs

def encode(obs: np.ndarray):
    with torch.no_grad():
        return model.encode(torch.asarray(obs))

def decode(z: np.ndarray):
    with torch.no_grad():
        return model.decode(torch.asarray(z))

def plot_obs(obs):
    if obs.ndim == 4:
        assert obs.shape[0] == 1
        return plot_obs(obs[0])
    plt.imshow(np.moveaxis(obs, 0, -1))
#%%
data = []
for taxi_row in range(5):
    for taxi_col in range(5):
        for pass_row in range(5):
            for pass_col in range(5):
                for in_taxi in range(2):
                    for goal_id in range(4):
                        for obs_realization in range(10):
                            if in_taxi and (
                                pass_row != taxi_row
                                or pass_col != taxi_col
                            ):
                                continue
                            else:
                                data_item = {
                                    'taxi_row': taxi_row,
                                    'taxi_col': taxi_col,
                                    'pass_row': pass_row,
                                    'pass_col': pass_col,
                                    'in_taxi': in_taxi,
                                    'goal_id': goal_id,
                                    'obs_realization': obs_realization
                                }
                                s = np.asarray(list(data_item.values())[:-1])
                                obs = np.expand_dims(get_obs(s), 0)
                                z = encode(obs)[0]
                                for i, _ in enumerate(z):
                                    data_item[f'z_{i}'] = z[i].item()
                                data.append(data_item)
df = pd.DataFrame(data)
# df.to_pickle('representations-wm_tune08-trial_0182-0001-64k.pkl')

#%%
df = pd.read_pickle('representations-wm_tune08-trial_0182-0001-64k.pkl')

#%%
#%%
g = sns.PairGrid(df)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3, legend=False)

#%%
x_vars = [col for col in df.columns if 'z' in col and 'obs' not in col]
y_vars = [col for col in df.columns if 'z' not in col]
g = sns.PairGrid(df, x_vars=x_vars, y_vars=y_vars, hue='in_taxi')
g.map(sns.scatterplot)

#%%
x_vars = [col for col in df.columns if 'z' in col and 'obs' not in col]
y_vars = [col for col in df.columns if 'z' not in col]
g = sns.PairGrid(df, x_vars=x_vars, y_vars=y_vars)
g.map(sns.kdeplot)


#%%
x_vars = [col for col in df.columns if 'z' in col and 'obs' not in col]
y_vars = [col for col in df.columns if 'z' not in col]
g = sns.PairGrid(df, x_vars=x_vars, y_vars=y_vars, hue='in_taxi')
g.map(sns.violinplot, orient='h', split=True)
