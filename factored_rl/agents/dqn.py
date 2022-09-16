import math

import hydra
import numpy as np
import torch

from factored_rl import configs
from factored_rl.models.nnutils import extract, Sequential, Reshape
from factored_rl.models import MLP, CNN
from .replaymemory import ReplayMemory
from ..models.dqn_nature import NatureDQN

class DQNAgent():
    def __init__(self, observation_space, action_space, cfg: configs.AgentConfig):
        self.observation_space = observation_space
        self.action_space = action_space
        self.cfg = cfg

        on_retrieve = {
            '*': lambda x: torch.as_tensor(np.asarray(x)).to(cfg.model.device),
            'ob': lambda x: x.float(),
            'reward': lambda x: x.float(),
            'terminal': lambda x: x.bool(),
            'truncated': lambda x: x.bool(),
            'next_ob': lambda x: x.float(),
            'action': lambda x: x.long()
        }
        self.replay = ReplayMemory(cfg.replay_buffer_size, on_retrieve=on_retrieve)

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
        return int(action)

    def store(self, experience: dict):
        self.replay.push(experience)

    def update(self):
        if len(self.replay) < self.cfg.replay_warmup_steps:
            return 0.0

        if len(self.replay) >= self.cfg.batch_size:
            fields = ['ob', 'action', 'reward', 'terminal', 'next_ob']
            batch = self.replay.sample(self.cfg.batch_size, fields)
            obs, actions, rewards, terminals, next_obs = batch
        else:
            return 0.0

        self.q.train()
        self.optimizer.zero_grad()
        q_values = self._get_q_predictions(obs, actions)
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
            training_faction = self.n_training_steps / self.cfg.epsilon_half_life_steps
            scale = self.cfg.epsilon_initial - self.cfg.epsilon_final
            epsilon = self.cfg.epsilon_final + scale * math.pow(0.5, training_faction)
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
        q_acted = extract(q_values, idx=action, idx_dim=-1) #(-1, )
        return q_acted

    def _get_q_values_for_observation(self, obs):
        return self.q(torch.as_tensor(obs).float().to(self.cfg.model.device))

    def _make_qnet(self, input_shape, n_actions, cfg: configs.ModelConfig):
        if 'mlp' in cfg.architecture:
            n_features = np.prod(input_shape) if len(input_shape) > 1 else input_shape[0]
            mlp = MLP(n_inputs=n_features,
                      n_outputs=n_actions,
                      n_hidden_layers=cfg.n_hidden_layers,
                      n_units_per_layer=cfg.n_units_per_layer,
                      activation=hydra.utils.instantiate(cfg.activation))
            if len(input_shape) > 1:
                return Sequential(*[Reshape(-1, n_features), mlp])
            else:
                return mlp
        elif 'cnn' in cfg.architecture:
            assert input_shape[-2:] == cfg.supported_2d_input_shape
            cnn = CNN(
                input_shape=input_shape,
                n_output_channels=cfg.n_output_channels,
                kernel_sizes=cfg.kernel_sizes,
                strides=cfg.strides,
                activation=hydra.utils.instantiate(cfg.activation),
            )
            n_features = np.prod(cnn.output_shape)
            mlp = MLP(
                n_inputs=n_features,
                n_outputs=n_actions,
                n_hidden_layers=cfg.n_hidden_layers,
                n_units_per_layer=cfg.n_units_per_layer,
                activation=hydra.utils.instantiate(cfg.activation),
            )
            return Sequential(*[cnn, Reshape(-1, n_features), mlp])
        else:
            raise NotImplementedError('"{}" is not a known network architecture'.format(
                cfg.architecture))
