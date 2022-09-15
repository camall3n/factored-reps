from dataclasses import dataclass, field
import datetime
import logging
import os
import platform
from typing import Any, List

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
import torch

def immutable(obj):
    return field(default_factory=lambda: obj)

cs = ConfigStore.instance()

@dataclass
class ModelConfig:
    architecture: str = MISSING
    device: str = MISSING

@dataclass
class MLPModelConfig(ModelConfig):
    architecture: str = 'mlp'
    flatten_input: bool = True
    n_hidden_layers: int = 1
    n_units_per_layer: int = 32

@dataclass
class CNNModelConfig(ModelConfig):
    architecture: str = 'cnn'
    flatten_input: bool = False

cs.store(group='agent/model', name='base', node=ModelConfig)
cs.store(group='agent/model', name='mlp', node=MLPModelConfig)
cs.store(group='agent/model', name='cnn', node=CNNModelConfig)

@dataclass
class AgentConfig:
    defaults: List[Any] = immutable([
        '_self_',
        {'model': 'base'},
    ]) # yapf: disable

    name: str = MISSING
    discount_factor: float = 0.99
    model: ModelConfig = MISSING

@dataclass
class ExpertAgentConfig(AgentConfig):
    name: str = 'expert'

@dataclass
class RandomAgentConfig(AgentConfig):
    name: str = 'random'

@dataclass
class DQNAgentConfig(AgentConfig):
    defaults: List[Any] = immutable([
        '_self_',
        {'model': 'mlp'},
    ]) # yapf: disable

    name: str = 'dqn'
    batch_size: int = 128
    epsilon_final: float = 0.01 # final / eval exploration probability
    epsilon_half_life_steps: int = 1000 # number of steps for epsilon to decrease by half
    epsilon_initial: float = 1.0 # initial exploration probability
    learning_rate: float = 0.001
    replay_buffer_size: int = 50000
    replay_warmup_steps: int = 500
    target_copy_alpha: float = 0.01 # per-step EMA contribution from online network
    target_copy_every_n_steps: int = MISSING
    target_copy_mode: str = 'soft'

cs.store(group='agent', name='base', node=AgentConfig)
cs.store(group='agent', name='random', node=RandomAgentConfig)
cs.store(group='agent', name='expert', node=ExpertAgentConfig)
cs.store(group='agent', name='dqn', node=DQNAgentConfig)

@dataclass
class EnvConfig:
    name: str = MISSING
    noise_std: float = 0.01
    n_training_episodes: int = 200
    n_steps_per_episode: int = 1000

@dataclass
class GridworldEnvConfig(EnvConfig):
    name: str = 'gridworld'

@dataclass
class TaxiEnvConfig(EnvConfig):
    name: str = 'taxi'

cs.store(group='env', name='base', node=EnvConfig)
cs.store(group='env', name='gridworld', node=GridworldEnvConfig)
cs.store(group='env', name='taxi', node=TaxiEnvConfig)

@dataclass
class TransformConfig:
    name: str = 'identity'

@dataclass
class RotateConfig:
    name: str = 'rotate'

@dataclass
class PermuteFactorsConfig:
    name: str = 'permute_factors'

@dataclass
class PermuteStatesConfig:
    name: str = 'permute_states'

@dataclass
class ImagesConfig:
    name: str = 'images'

cs.store(group='transform', name='identity', node=TransformConfig)
cs.store(group='transform', name='rotate', node=RotateConfig)
cs.store(group='transform', name='permute_factors', node=PermuteFactorsConfig)
cs.store(group='transform', name='permute_states', node=PermuteStatesConfig)
cs.store(group='transform', name='images', node=ImagesConfig)

@dataclass
class Config:
    defaults: List[Any] = immutable([
        '_self_',
        {'env': 'base'},
        {'agent': 'base'},
        {'transform': 'identity'},
    ]) # yapf: disable

    experiment: str = MISSING
    dir: str = MISSING
    env: EnvConfig = MISSING
    agent: AgentConfig = MISSING
    trial: str = 'trial' # A name for the trial
    seed: int = 0 # A seed for the random number generator
    timestamp: bool = True # Whether to add a timestamp to the experiment directory path
    noise: bool = False
    transform: TransformConfig = MISSING
    verbose: bool = False

@dataclass
class RLvsRepConfig(Config):
    experiment: str = 'rl_vs_rep'

@dataclass
class DisentvsRepConfig(Config):
    experiment: str = 'disent_vs_rep'

@dataclass
class DisentvsRLConfig(Config):
    experiment: str = 'disent_vs_rl'

cs.store(name='rl_vs_rep', node=RLvsRepConfig)
cs.store(name='disent_vs_rep', node=DisentvsRepConfig)
cs.store(name='disent_vs_rl', node=DisentvsRLConfig)

def get_config_yaml_str(cfg):
    return OmegaConf.to_yaml(cfg)

def _initialize_experiment_dir(cfg: Config) -> str:
    if cfg.timestamp:
        cfg.trial += '__' + datetime.datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')
    prefix = '~/data-gdk/csal/factored' if platform.system() == 'Linux' else '~/dev/factored-reps'
    prefix = os.path.expanduser(prefix)
    cfg.dir = f'{prefix}/results/factored_rl/{cfg.experiment}/{cfg.trial}/{cfg.seed:04d}/'
    os.makedirs(cfg.dir, exist_ok=True)
    return cfg.dir

def _initialize_device(cfg) -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.zeros(1).to(device)
    cfg.agent.model.device = device
    return device

def _initialize_logger(cfg: Config) -> logging.Logger:
    log = logging.getLogger()
    log_filename = cfg.dir + 'log.txt'
    log.addHandler(logging.FileHandler(log_filename, mode='w'))

def initialize_experiment(cfg):
    _initialize_experiment_dir(cfg)
    _initialize_device(cfg)
    _initialize_logger(cfg)
    log = logging.getLogger()
    log.info('\n' + get_config_yaml_str(cfg))
    log.info(f'Training on device: {cfg.agent.model.device}\n')
