import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm, trange

from factored_rl import configs
from factored_rl.experiments.common import initialize_env, initialize_model
from factored_rl.experiments.factorize.run import initialize_dataloader
from factored_rl.agents.replaymemory import ReplayMemory
from factored_rl.models.nnutils import one_hot, extract

load_dir = '/Users/cam/dev/factored-reps/results/factored_rl/wm_tune08/trial_0182/0001/'
cfg = OmegaConf.load(load_dir + 'config_factorize.yaml')
cfg.dir = load_dir
cfg.script = 'factorize'
cfg.transform.basis = {'name': None}
cfg.model.qnet.basis = {'name': None}
cfg.loader.load_model = True
cfg.loader.eval_only = True
cfg.trainer.log_every_n_steps = 0
cfg.tuner.tune_rep = False
train_dl, input_shape, n_actions = initialize_dataloader(cfg, cfg.seed)
model = initialize_model(input_shape, n_actions, cfg)
env = initialize_env(cfg, cfg.seed)

on_retrieve = {
    '*': lambda x: torch.as_tensor(np.asarray(x)).to(cfg.model.device),
    'ob': lambda x: x.float(),
    'action': lambda x: x.long(),
    'reward': lambda x: x.float(),
    'terminal': lambda x: x.bool(),
    'truncated': lambda x: x.bool(),
    'next_ob': lambda x: x.float(),
    'state': lambda x: x.numpy(),
    'next_state': lambda x: x.numpy(),
}
replay = ReplayMemory(cfg.agent.replay_buffer_size, on_retrieve=on_retrieve)

for _ in trange(10 * cfg.trainer.batch_size):
    obs_transition, state_transition = train_dl.dataset.get_ob_transition_pair()
    for key in obs_transition.keys():
        if key in state_transition.keys():
            assert np.allclose(state_transition[key], obs_transition[key])
    obs_transition.update(**state_transition)
    replay.push(obs_transition)

fields = ['ob', 'state', 'action', 'next_ob', 'next_state']
batch = {k: val for (k, val) in zip(fields, replay.retrieve(fields=fields))}

#%%
obs = batch['ob']
model.update_mean_input(obs)
actions = one_hot(batch['action'], model.n_actions)
next_obs = batch['next_ob']
model.update_mean_input(next_obs)
z = model.encode(obs)
effects, attn_weights = model.predict(z, actions)
next_z_hat = z + effects
obs_hat = model.decode(z)
next_obs_hat = model.decode(next_z_hat)
losses = {
    'actions': model.action_semantics_loss(actions, effects),
    'effects': model.effects_loss(effects),
    'parents': model.parents_loss(attn_weights),
    'reconst': model.reconstruction_loss(obs, next_obs, obs_hat, next_obs_hat),
}
loss = sum([losses[key] * model.cfg.loss[key] for key in losses.keys()])
print(loss.item())

#%%
cfg.transform.name = 'identity'
cfg.transform.noise = False
state_env = initialize_env(cfg, cfg.seed)
def wrap_state(s):
    wrapper_list = []
    env_iter = state_env
    while hasattr(env_iter, 'observation'):
        wrapper_list.append(env_iter)
        env_iter = env_iter.env
    for wrapper in reversed(wrapper_list):
        s = wrapper.observation(s)
    return torch.as_tensor(s).float()

obs = batch['ob']
next_obs = batch['next_ob']
actions = one_hot(batch['action'], model.n_actions)
z = wrap_state(batch['state'])
next_z_hat = wrap_state(batch['next_state'])
effects = next_z_hat - z

#%%
plt.figure()
plt.imshow(actions[:10])
plt.title('actions')
plt.xlabel('state var')
plt.ylabel('sample_id')

plt.figure()
plt.imshow(effects[:10], vmin=-2, vmax=2, cmap='RdYlBu')
plt.colorbar()
plt.title('effects')
plt.xlabel('state var')
plt.ylabel('sample_id')

plt.figure()
action_mask = actions.unsqueeze(-1) #(N,A,1)
action_effects = action_mask * effects.unsqueeze(dim=1) #(N,1,d)
mean_action_effects = (action_effects.sum(dim=0, keepdim=True) / (action_mask.sum(dim=0, keepdim=True) + 1e-9)) #(A,d)
plt.imshow(mean_action_effects[0, batch['action'][:10], :], vmin=-2, vmax=2, cmap='RdYlBu')
plt.colorbar()
plt.title('mean_action_effects')
plt.xlabel('state var')
plt.ylabel('action')

action_residuals = model._get_action_residuals(actions, effects)
plt.figure()
plt.imshow(action_residuals[:10], vmin=-2, vmax=2, cmap='RdYlBu')
plt.colorbar()
plt.title('action_residuals')
plt.xlabel('state_var')
plt.ylabel('sample_id')

magnitudes = torch.abs(action_residuals)
l1 = torch.sum(magnitudes, dim=-1)
lmax = torch.max(magnitudes, dim=-1)[0]
residual_scores = l1 / (lmax + 1e-9)
residual_scores.mean()

d = action_residuals.shape[-1]
magnitudes = torch.abs(action_residuals)
l1 = torch.sum(magnitudes, dim=-1, keepdim=True)
normalized_data = magnitudes / (l1 + 1e-9)
L = torch.sum(normalized_data**2, dim=-1) # ranges from 1/(d^(p-1)) to 1
Lmin = 1 / (d**(2 - 1))
residual_scores = (L * (1 - d) - Lmin + d) / (1 - Lmin) # ranges from 1 to d


#%%

obs_hat = batch['ob']
next_obs_hat = batch['next_ob']
oracle_losses = {
    'actions': model.action_semantics_loss(actions, effects),
    'effects': model.effects_loss(effects),
    'parents': 0,#model.parents_loss(attn_weights),
    'reconst': model.reconstruction_loss(obs, next_obs, obs_hat, next_obs_hat),
}
loss = sum([oracle_losses[key] * cfg.loss[key] for key in oracle_losses.keys()])
print(loss.item())
