import math

import numpy as np
import torch

from factored_rl import configs
from factored_rl.models.nnutils import extract, Sequential, Reshape
from factored_rl.models.mlp import MLP
from .replaymemory import ReplayMemory
from ..models.dqn_nature import NatureDQN

class DQNAgent():
    def __init__(self, observation_space, action_space, cfg: configs.AgentConfig):
        self.observation_space = observation_space
        self.action_space = action_space
        self.cfg = cfg

        self.replay = ReplayMemory(cfg.replay_buffer_size)

        self.n_training_steps = 0
        input_shape = self.observation_space.shape
        n_actions = self.action_space.n
        self.q = self._make_qnet(input_shape, n_actions, cfg.model).to(cfg.model.device)
        self.q_target = self._make_qnet(input_shape, n_actions, cfg.model).to(cfg.model.device)
        self.q_target.hard_copy_from(self.q)
        self.replay.reset()
        params = list(self.q.parameters())
        self.optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)

    def save(self, fname, dir, is_best):
        self.q.save(fname, dir, is_best)

    def act(self, observation, testing=False):
        if ((len(self.replay) < self.cfg.replay_warmup_steps
             or np.random.uniform() < self._get_epsilon(testing=testing))):
            action = self.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self._get_q_values_for_observation(observation)
                action = torch.argmax(q_values.squeeze())
        return action

    def store(self, experience: dict):
        self.replay.push(experience)

    def update(self):
        if len(self.replay) < self.cfg.replay_warmup_steps:
            return 0.0

        if len(self.replay) >= self.cfg.batch_size:
            fields = ['ob', 'state', 'action', 'reward', 'terminal', 'next_ob']
            batch = self.replay.sample(self.cfg.batch_size, fields)
            obs, states, actions, rewards, terminals, next_obs = batch
        else:
            return 0.0

        self.q.train()
        self.optimizer.zero_grad()
        q_values = self._get_q_predictions(obs)
        q_targets = self._get_q_targets(rewards, terminals, next_obs)
        loss = torch.nn.functional.smooth_l1_loss(input=q_values, target=q_targets)
        loss.backward()
        self.optimizer.step()
        self.n_training_steps += 1

        if self.cfg.target_copy_mode == 'soft':
            self.q_target.soft_copy_from(self.q, self.cfg.target_copy_alpha)
        else:
            if self.n_training_steps % self.cfg.target_copy_period == 0:
                self.q_target.hard_copy_from(self.q)

        return loss.detach().item()

    def _get_epsilon(self, testing=False):
        if testing:
            epsilon = self.cfg.epsilon_final
        else:
            t = self.n_training_steps - self.cfg.replay_warmup_steps
            scale = self.cfg.epsilon_initial - self.cfg.epsilon_final
            epsilon = self.cfg.epsilon_final + scale * math.pow(0.5,
                                                                t / self.cfg.epsilon_half_life)
        return epsilon

    def _get_q_targets(self, rewards, terminals, next_obs):
        with torch.no_grad():
            # Compute Double-Q targets
            next_action = torch.argmax(self.q(next_obs), dim=-1) # (-1, )
            next_q = self.q_target(next_obs) # (-1, A)
            next_v = next_q.gather(-1, next_action.unsqueeze(-1)).squeeze(-1) # (-1, )
            non_terminal_idx = ~terminals # (-1, )
            q_targets = rewards + non_terminal_idx * self.cfg.discount_factor * next_v # (-1, )
        return q_targets

    def _get_q_predictions(self, obs, action):
        q_values = self.q(obs) # (-1, A)
        q_acted = extract(q_values, idx=torch.stack(action).long(), idx_dim=-1) #(-1, )
        return q_acted

    def _get_q_values_for_observation(self, obs):
        return self.q(torch.as_tensor(obs).float().to(self.cfg.model.device))

    def _make_qnet(self, input_shape, n_actions, cfg: configs.ModelConfig):
        if cfg.architecture == 'mlp':
            n_features = np.prod(input_shape) if len(input_shape) > 1 else input_shape[0]
            mlp = MLP(n_inputs=n_features,
                      n_outputs=n_actions,
                      n_hidden_layers=cfg.n_hidden_layers,
                      n_units_per_layer=cfg.n_units_per_layer)
            if len(input_shape) > 1:
                return Sequential(*[Reshape(-1, n_features), mlp])
            else:
                return mlp
        elif cfg.architecture == 'nature_dqn':
            return NatureDQN(n_actions=n_actions).to(cfg.device)
        else:
            raise NotImplementedError('"{}" is not a known network architecture'.format(
                cfg.architecture))
