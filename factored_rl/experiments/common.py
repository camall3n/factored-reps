import os
import glob

from pytorch_lightning.loggers import TensorBoardLogger

from factored_rl import configs

# Env
import gym
from gym.wrappers import FlattenObservation
from visgrid.envs import GridworldEnv, TaxiEnv
from factored_rl.wrappers import RotationWrapper, FactorPermutationWrapper, ObservationPermutationWrapper, MoveAxisToCHW
from visgrid.wrappers import GrayscaleWrapper, InvertWrapper, ToFloatWrapper, NormalizeWrapper, NoiseWrapper

# Model
from factored_rl.models.ae import EncoderModel, AutoencoderModel, PairedAutoencoderModel
from factored_rl.models.wm import WorldModel
from factored_rl.models.disent import build_disent_model

# ----------------------------------------
# Environment & wrappers
# ----------------------------------------

def initialize_env(cfg: configs.Config, seed: int = None):
    if cfg.env.name == 'gridworld':
        env = GridworldEnv(10,
                           10,
                           exploring_starts=cfg.env.exploring_starts,
                           terminate_on_goal=True,
                           fixed_goal=cfg.env.fixed_goal,
                           hidden_goal=True,
                           should_render=False,
                           dimensions=GridworldEnv.dimensions_onehot)
    elif cfg.env.name == 'taxi':
        env = TaxiEnv(size=5,
                      n_passengers=1,
                      exploring_starts=cfg.env.exploring_starts,
                      terminate_on_goal=True,
                      fixed_goal=cfg.env.fixed_goal,
                      depot_dropoff_only=cfg.env.depot_dropoff_only,
                      should_render=False,
                      dimensions=TaxiEnv.dimensions_5x5_to_64x64)
    else:
        env = gym.make(cfg.env.name)
        # TODO: wrap env to support disent protocol
        raise NotImplementedError('Need to wrap env to support disent protocol')

    env.reset(seed=seed)
    env.action_space.seed(seed)

    if cfg.transform.name == 'images':
        env.set_rendering(enabled=True)
        if cfg.env.grayscale:
            env = GrayscaleWrapper(env, keep_dim=True)
        if cfg.env.name == 'gridworld':
            env = InvertWrapper(env)
        if cfg.model.arch.encoder == 'mlp':
            env = FlattenObservation(env)
        elif cfg.model.arch.encoder == 'cnn':
            env = MoveAxisToCHW(env)
    else:
        if cfg.transform.name == 'permute_factors':
            env = FactorPermutationWrapper(env)
        elif cfg.transform.name == 'permute_states':
            env = ObservationPermutationWrapper(env)
        env = NormalizeWrapper(ToFloatWrapper(env), -1, 1)
        if cfg.transform.name == 'rotate':
            env = RotationWrapper(env)
    if cfg.transform.noise:
        env = NoiseWrapper(env, cfg.transform.noise_std)
    env = ToFloatWrapper(env)
    return env

def initialize_model(input_shape, n_actions, cfg: configs.Config):
    if cfg.model.lib == 'disent':
        return build_disent_model(input_shape, cfg)
    elif cfg.model.name is not None:
        if cfg.model.arch.type == 'enc':
            module = EncoderModel
            module_args = {'input_shape': input_shape, 'cfg': cfg}
        elif cfg.model.arch.type == 'ae':
            module = AutoencoderModel
            module_args = {'input_shape': input_shape, 'cfg': cfg}
        elif cfg.model.arch.type == 'paired_ae':
            module = PairedAutoencoderModel
            module_args = {'input_shape': input_shape, 'n_actions': n_actions, 'cfg': cfg}
        elif cfg.model.arch.type == 'wm':
            module = WorldModel
            module_args = {'input_shape': input_shape, 'n_actions': n_actions, 'cfg': cfg}
        else:
            raise NotImplementedError(f'Unknown model architecture: {cfg.model.arch.type}')

        if cfg.loader.should_load:
            ckpt_path = get_checkpoint_path(cfg)
            model = module.load_from_checkpoint(ckpt_path, **module_args)
            if not cfg.loader.should_train:
                model.eval()
                model.encoder.freeze()
        else:
            model = module(**module_args)
    return model

def get_checkpoint_path(cfg, logs_dirname='lightning_logs', create_new_version=False):
    if cfg.loader.checkpoint_path is None:
        experiment = cfg.experiment if cfg.loader.experiment is None else cfg.loader.experiment
        trial = cfg.trial if cfg.loader.trial is None else cfg.loader.trial
        seed = f'{cfg.seed if cfg.loader.seed is None else cfg.loader.seed:04d}'
        load_dir = os.path.join('/', *cfg.dir.split('/')[:-4], experiment, trial, seed)
        logger = TensorBoardLogger(save_dir=load_dir, name=logs_dirname)
        if cfg.loader.version is None:
            version = logger.version - 1
            if create_new_version:
                version += 1
        else:
            version = cfg.loader.version
        checkpoint_dir = os.path.join(load_dir, f'{logs_dirname}/version_{version}/checkpoints/')
        checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        if len(checkpoints) > 1:
            raise RuntimeError(f'Multiple checkpoints detected in {checkpoint_dir}\n'
                               f'Please specify model.checkpoint_path')
        elif len(checkpoints) == 1:
            ckpt_path = checkpoints[0]
        elif create_new_version:
            ckpt_path = checkpoint_dir
        return ckpt_path

def cpu_count():
    # os.cpu_count(): returns number of cores on machine
    # os.sched_getaffinity(pid): returns set of cores on which process is allowed to run
    #                            if pid=0, results are for current process
    #
    # if os.sched_getaffinity doesn't exist, just return cpu_count and hope for the best
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()
