import argparse
from dataclasses import dataclass
import datetime
import logging
import os
import platform
from typing import Tuple
import yaml

import hydra
import hydra.core
from hydra.core.config_store import ConfigStore
from hydra.core.utils import configure_log
from omegaconf import OmegaConf, MISSING
import torch

from . import utils

@dataclass
class ModelConfig:
    architecture: str = MISSING
    device: str = MISSING

@dataclass
class MLPModelConfig(ModelConfig):
    architecture: str = 'mlp'
    n_hidden_layers: int = 1
    n_units_per_layer: int = 32

@dataclass
class NatureDQNModelConfig(ModelConfig):
    architecture: str = 'nature_dqn'

@dataclass
class AgentConfig:
    replay_buffer_size: int = 50000
    replay_warmup_steps: int = 500
    learning_rate: float = 0.001
    batch_size: int = 128
    discount_factor: float = 0.99
    model: ModelConfig = MLPModelConfig()

@dataclass
class DQNAgentConfig(AgentConfig):
    target_copy_mode: str = 'soft'
    target_copy_every_n_steps: int = None
    target_copy_alpha: float = 0.01 # per-step EMA contribution from online network
    epsilon_initial: float = 1.0 # initial exploration probability
    epsilon_final: float = 0.01 # final / eval exploration probability
    epsilon_half_life_steps: int = 1000 # number of steps for epsilon to decrease by half

@dataclass
class EnvConfig:
    noise_std: float = 0.01

@dataclass
class ExperimentConfig:
    name: str = MISSING
    trial: str = MISSING
    seed: int = MISSING
    dir: str = MISSING
    verbose: bool = False
    timestamp: bool = True

@dataclass
class Config:
    experiment: ExperimentConfig = ExperimentConfig()
    env: EnvConfig = EnvConfig()
    agent: AgentConfig = AgentConfig()

def get_config_yaml_str(cfg):
    return OmegaConf.to_yaml(cfg)

def new_parser():
    """Return a nicely formatted argument parser

    This function is a simple wrapper for the argument parser I like to use,
    which has a stupidly long argument that I always forget.
    """
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def _parse_args_and_overrides(parser) -> Tuple[argparse.ArgumentParser, list]:
    """
    1. Parse known args registered with the argparse parser
    2. Keep track of unknown args (e.g. '--some_hyperparameter')
    2. Load default hyperparameters from args.hyperparams file
    3. For known args, add them directly to the list of hyperparameters
    4. For unknown args, check if they match a valid hyperparameter, and update the value
    5. Return a namespace so we can access values easily (e.g. `args.some_hyperparameter`)
    """
    args, unknown = parser.parse_known_args()
    del args.fool_ipython
    unknown = [term for arg in unknown for term in arg.split('=')]
    overrides = [
        f"{utils.remove_prefix(key, '--')}={val}"
        for (key, val) in zip(unknown[::2], unknown[1::2])
    ]
    return args, overrides

def _initialize_hydra_config(args, overrides: list) -> Config:
    cs = ConfigStore.instance()
    cs.store(name='config', node=Config)
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(version_base=None, config_path=None, job_name=args.experiment)
    cfg = hydra.compose(config_name='config', overrides=overrides)
    return cfg

def _initialize_experiment_dir(args, cfg: ExperimentConfig) -> str:
    trial_name = args.trial
    if cfg.experiment.timestamp and not args.no_timestamp:
        trial_name += '__' + datetime.datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')
    cfg.experiment.name = args.experiment
    cfg.experiment.trial = trial_name
    cfg.experiment.seed = args.seed
    prefix = '~/data-gdk/csal/factored' if platform.system() == 'Linux' else '~/dev/factored-reps'
    cfg.experiment.dir = f'{prefix}/results/factored_rl/{args.experiment}/{trial_name}/{args.seed:04d}/'
    os.makedirs(cfg.experiment.dir, exist_ok=True)
    return cfg.experiment.dir

def _initialize_device(cfg) -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.zeros(1).to(device)
    cfg.agent.model.device = device
    return device

def _initialize_logger(cfg: ExperimentConfig) -> logging.Logger:
    configure_log(None, verbose_config=cfg.experiment.verbose)
    log = logging.getLogger()
    log_filename = cfg.experiment.dir + 'log.txt'
    log.addHandler(logging.FileHandler(log_filename, mode='w'))
    return log

def initialize_experiment(parser):
    args, overrides = _parse_args_and_overrides(parser)
    cfg = _initialize_hydra_config(args, overrides)
    _initialize_experiment_dir(args, cfg)
    _initialize_device(cfg)
    log = _initialize_logger(cfg)
    log.info('\n' + yaml.dump(vars(args), sort_keys=False))
    log.info('\n' + get_config_yaml_str(cfg))
    log.info(f'Training on device: {cfg.agent.model.device}\n')
    return args, cfg, log
