from dataclasses import dataclass, field
import datetime
import inspect
import logging
import os
import platform
from typing import Any, Dict, List, Optional, Tuple, Union

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
import torch

def immutable(obj):
    return field(default_factory=lambda: obj)

# -----------------------------------------------------------------------
# Register a resolver with OmegaConf to handle math within .yaml files
#
# example.yaml:
# ```
# one: 1
# two: 2
# three: "${eval: ${one} + ${two}}"  # => 3
# ```
OmegaConf.register_new_resolver("eval", eval, replace=True)
# -----------------------------------------------------------------------

@dataclass
class MLPConfig:
    n_hidden_layers: int = MISSING
    n_units_per_layer: int = MISSING
    activation: Optional[Dict[str, str]] = None
    final_activation: Optional[Dict[str, str]] = None

@dataclass
class CNNConfig:
    supported_2d_input_shape: Tuple[int, int] = MISSING # (H, W)
    n_output_channels: List[int] = MISSING
    kernel_sizes: Any = MISSING
    strides: Any = MISSING
    padding: Any = None
    dilations: Any = None
    activation: Optional[Dict[str, str]] = None
    final_activation: Optional[Dict[str, str]] = None

@dataclass
class AEConfig:
    n_latent_dims: int = MISSING

@dataclass
class VAEConfig:
    beta: float = MISSING
    loss_reduction: str = MISSING

@dataclass
class ModelConfig:
    name: Optional[str] = None
    architecture: Optional[str] = None
    device: str = MISSING
    flatten_input: bool = False
    mlp: MLPConfig = MLPConfig()
    cnn: CNNConfig = CNNConfig()
    ae: AEConfig = AEConfig()
    vae: VAEConfig = VAEConfig()

@dataclass
class AgentConfig:
    name: str = MISSING
    discount_factor: float = 0.99

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
    n_training_episodes: int = MISSING
    n_steps_per_episode: int = MISSING
    exploring_starts: bool = MISSING
    fixed_goal: bool = MISSING

@dataclass
class TaxiEnvConfig(EnvConfig):
    depot_dropoff_only: bool = MISSING

@dataclass
class TransformConfig:
    name: str = MISSING
    noise: bool = True
    noise_std: float = 0.01

@dataclass
class TrainerConfig:
    name: str = MISSING
    batch_size: int = MISSING
    learning_rate: float = MISSING
    log_every_n_steps: int = MISSING
    max_steps: int = MISSING
    num_dataloader_workers: int = MISSING
    optimizer: Optional[Dict[str, str]] = None
    overfit_batches: int = MISSING
    persistent_workers: bool = MISSING
    quick: bool = MISSING

@dataclass
class Config:
    experiment: str = MISSING
    trial: str = 'trial' # A name for the trial
    seed: int = 0 # A seed for the random number generator
    timestamp: bool = True # Whether to add a timestamp to the experiment directory path
    dir: str = MISSING
    agent: AgentConfig = MISSING
    env: EnvConfig = MISSING
    model: ModelConfig = MISSING
    trainer: TrainerConfig = MISSING
    transform: TransformConfig = MISSING
    test: bool = False
    verbose: bool = False
    disable_gpu: bool = False

cs = ConfigStore.instance()
cs.store(name='base_config', node=Config)
cs.store(group='agent', name='base_agent', node=AgentConfig)
cs.store(group='agent', name='base_dqn_agent', node=DQNAgentConfig)
cs.store(group='env', name='base_env', node=EnvConfig)
cs.store(group='env', name='taxi_env', node=TaxiEnvConfig)
cs.store(group='model', name='base_model', node=ModelConfig)
cs.store(group='trainer', name='base_trainer', node=TrainerConfig)
cs.store(group='transform', name='base_transform', node=TransformConfig)

def get_config_yaml_str(cfg):
    return OmegaConf.to_yaml(cfg, resolve=True)

def _initialize_experiment_dir(cfg: Config) -> str:
    if cfg.timestamp:
        cfg.trial += '__' + datetime.datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')
    prefix = '~/data-gdk/csal/factored' if platform.system() == 'Linux' else '~/dev/factored-reps'
    prefix = os.path.expanduser(prefix)
    cfg.dir = f'{prefix}/results/factored_rl/{cfg.experiment}/{cfg.trial}/{cfg.seed:04d}/'
    os.makedirs(cfg.dir, exist_ok=True)
    return cfg.dir

def _initialize_device(cfg) -> torch.device:
    device = torch.device('cuda' if (torch.cuda.is_available() and not cfg.disable_gpu) else 'cpu')
    torch.zeros(1).to(device)
    cfg.model.device = device
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
    log.info(f'Training on device: {cfg.model.device}\n')

    filename = cfg.dir + 'config.yaml'
    with open(filename, 'w') as args_file:
        args_file.write(get_config_yaml_str(cfg))
