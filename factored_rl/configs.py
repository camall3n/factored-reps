from dataclasses import dataclass, field
import datetime
import logging
import os
import platform
from typing import Any, Dict, List, Optional, Tuple

import hydra
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

ActivationType = List[Dict[str, Any]]
OptimizerType = Dict[str, Any]

def instantiate(obj):
    if obj is None:
        return obj
    elif isinstance(OmegaConf.to_object(obj), Dict) and '_target_' in obj.keys():
        return hydra.utils.instantiate(obj)
    elif isinstance(OmegaConf.to_object(obj), List):
        return [instantiate(o) for o in obj]
    else:
        raise RuntimeError(f'Cannot instantiate object {obj} with type: {type(obj)}')

@dataclass
class ArchitectureConfig:
    type: Optional[str] = None
    encoder: Optional[str] = None
    decoder: Optional[str] = None
    predictor: Optional[str] = None

@dataclass
class MLPConfig:
    n_hidden_layers: int = MISSING
    n_units_per_layer: int = MISSING
    activation: Optional[ActivationType] = None
    final_activation: Optional[ActivationType] = None

@dataclass
class CNNConfig:
    supported_2d_input_shape: Tuple[int, int] = MISSING # (H, W)
    n_output_channels: List[int] = MISSING
    kernel_sizes: Any = MISSING
    strides: Any = MISSING
    padding: Any = None
    dilations: Any = None
    activation: Optional[ActivationType] = None
    final_activation: Optional[ActivationType] = None

@dataclass
class AttnConfig:
    key_embed_dim: Optional[int] = None
    action_embed_dim: int = MISSING
    factor_embed_dim: int = MISSING
    dropout: float = 0.0

@dataclass
class WMConfig:
    mlp: MLPConfig = MLPConfig()
    attn: AttnConfig = AttnConfig()

@dataclass
class ModelConfig:
    name: Optional[str] = None
    arch: ArchitectureConfig = ArchitectureConfig()
    action_sampling: Optional[str] = MISSING # None, 'random' or 'valid'
    lib: Optional[str] = None # external library name (e.g. 'disent', 'dreamerv2')
    device: str = MISSING
    n_latent_dims: int = MISSING
    flatten_input: bool = False
    mlp: MLPConfig = MLPConfig()
    cnn: CNNConfig = CNNConfig()
    wm: WMConfig = WMConfig()
    qnet: Optional[MLPConfig] = None

@dataclass
class SparsityConfig:
    name: Optional[str] = None
    epsilon: float = 1.0e-9
    p_norm: float = 2.0
    sigma: float = MISSING

@dataclass
class LossConfig:
    actions: float = MISSING # consistent semantics
    effects: float = MISSING # sparse effects
    parents: float = MISSING # sparse dependencies
    reconst: float = MISSING # current observation
    distance: str = 'mse' # for predictions/reconstructions
    sparsity: SparsityConfig = MISSING
    vae_beta: float = MISSING

@dataclass
class LoaderConfig:
    load_model: bool = False # whether to load model from checkpoint
    load_config: bool = False # whether to load config.yaml from previous run
    eval_only: bool = False # whether to prevent loaded model from continuing to train
    experiment: Optional[str] = None # which experiment to load
    trial: Optional[str] = None # which trial to load
    seed: Optional[int] = None # which seed to load
    version: Optional[int] = None # which version to load (default is latest)
    checkpoint_path: Optional[str] = None # path to .ckpt file for loading

@dataclass
class AgentConfig:
    name: str = MISSING
    discount_factor: float = 0.99

@dataclass
class DQNAgentConfig(AgentConfig):
    epsilon_final: float = MISSING # final / eval exploration probability
    epsilon_half_life_steps: int = MISSING # number of steps for epsilon to decrease by half
    epsilon_initial: float = MISSING # initial exploration probability
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
    grayscale: bool = MISSING

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
    log_every_n_steps: Optional[int] = None
    rep_learning_rate: float = MISSING
    rl_learning_rate: float = MISSING
    optimizer: Optional[OptimizerType] = None
    quick: bool = False

@dataclass
class LightningTrainerConfig(TrainerConfig):
    max_steps: int = MISSING
    num_dataloader_workers: int = MISSING
    overfit_batches: int = MISSING
    persistent_workers: bool = MISSING

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
    loss: LossConfig = LossConfig()
    trainer: TrainerConfig = MISSING
    transform: TransformConfig = MISSING
    loader: LoaderConfig = LoaderConfig()
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
cs.store(group='loss', name='base_loss', node=LossConfig)
cs.store(group='loss/sparsity', name='base_sparsity', node=SparsityConfig)
cs.store(group='trainer', name='base_trainer', node=TrainerConfig)
cs.store(group='trainer', name='lightning_trainer', node=LightningTrainerConfig)
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
    _initialize_logger(cfg)
    _initialize_device(cfg)
    log = logging.getLogger()
    log.info('\n' + get_config_yaml_str(cfg))
    log.info(f'Training on device: {cfg.model.device}\n')

    filename = cfg.dir + f'config_{experiment_name}.yaml'
    if cfg.loader.load_config:
        old_cfg = cfg
        cfg = OmegaConf.load(filename)
        cfg.loader = old_cfg.loader
        # Override max steps
        max_steps = old_cfg.trainer.get('max_steps', -1)
        if max_steps > cfg.trainer.get('max_steps', 0):
            if not hasattr(cfg.trainer, 'max_steps'):
                raise KeyError('Invalid config key: max_steps')
            cfg.trainer.max_steps = max_steps
    else:
        with open(filename, 'w') as args_file:
            args_file.write(get_config_yaml_str(cfg))

    return cfg
