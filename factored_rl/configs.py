from dataclasses import dataclass, field
import datetime
import logging
import os
import platform
from typing import Any, Dict, List, Optional, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
import torch

def immutable(obj):
    return field(default_factory=lambda: obj)

@dataclass
class ModelConfig:
    architecture: Optional[str] = None
    device: str = MISSING

@dataclass
class MLPModelConfig(ModelConfig):
    architecture: str = 'base_mlp'
    flatten_input: bool = True
    n_hidden_layers: int = MISSING
    n_units_per_layer: int = MISSING
    activation: Optional[Dict[str, str]] = None

@dataclass
class CNNModelConfig(MLPModelConfig):
    architecture: str = 'base_cnn'
    flatten_input: bool = False
    supported_2d_input_shape: Tuple[int, int] = MISSING # (H, W)
    n_output_channels: List[int] = MISSING
    kernel_sizes: Any = MISSING
    strides: Any = MISSING
    padding: Any = None
    dilations: Any = None

@dataclass
class AgentConfig:
    name: str = MISSING
    discount_factor: float = 0.99
    model: ModelConfig = ModelConfig()

@dataclass
class DQNAgentConfig(AgentConfig):
    batch_size: int = MISSING
    epsilon_final: float = MISSING # final / eval exploration probability
    epsilon_half_life_steps: int = MISSING # number of steps for epsilon to decrease by half
    epsilon_initial: float = MISSING # initial exploration probability
    learning_rate: float = MISSING
    replay_buffer_size: int = MISSING
    replay_warmup_steps: int = MISSING
    target_copy_alpha: float = MISSING # per-step EMA contribution from online network
    target_copy_every_n_steps: int = MISSING
    target_copy_mode: str = MISSING

@dataclass
class EnvConfig:
    name: str = MISSING
    noise_std: float = 0.01
    n_training_episodes: int = 200
    n_steps_per_episode: int = 1000

@dataclass
class TransformConfig:
    name: str = MISSING

@dataclass
class Config:
    experiment: str = MISSING
    dir: str = MISSING
    env: EnvConfig = MISSING
    transform: TransformConfig = MISSING
    agent: AgentConfig = MISSING
    trial: str = 'trial' # A name for the trial
    seed: int = 0 # A seed for the random number generator
    timestamp: bool = True # Whether to add a timestamp to the experiment directory path
    noise: bool = False
    test: bool = False
    verbose: bool = False

cs = ConfigStore.instance()
cs.store(name='base_config', node=Config)
cs.store(group='env', name='base_env', node=EnvConfig)
cs.store(group='transform', name='base_transform', node=TransformConfig)
cs.store(group='agent', name='base_agent', node=AgentConfig)
cs.store(group='agent', name='base_dqn_agent', node=DQNAgentConfig)
cs.store(group='agent/model', name='base_model', node=ModelConfig)
cs.store(group='agent/model', name='base_mlp', node=MLPModelConfig)
cs.store(group='agent/model', name='base_cnn', node=CNNModelConfig)

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

def initialize_experiment(cfg, experiment_name):
    if cfg.get('experiment', None) is None:
        cfg.experiment = experiment_name
    _initialize_experiment_dir(cfg)
    _initialize_device(cfg)
    _initialize_logger(cfg)
    log = logging.getLogger()
    log.info('\n' + get_config_yaml_str(cfg))
    log.info(f'Training on device: {cfg.agent.model.device}\n')

    filename = cfg.dir + 'config.yaml'
    with open(filename, 'w') as args_file:
        args_file.write(get_config_yaml_str(cfg))
