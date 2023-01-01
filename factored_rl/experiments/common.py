import os
import glob

from pytorch_lightning.loggers import TensorBoardLogger

from factored_rl import configs

# Env
import gym
from gym.wrappers import FlattenObservation
from visgrid.envs import GridworldEnv, TaxiEnv
from factored_rl.wrappers import RotationWrapper, FactorPermutationWrapper, ObservationPermutationWrapper, MoveAxisToCHW
from visgrid.wrappers import GrayscaleWrapper, InvertWrapper, ToFloatWrapper, NormalizeWrapper, NoiseWrapper, ClipWrapper

# Model
from factored_rl.models.ae import BaseModel, EncoderModel, AutoencoderModel, PairedAutoencoderModel
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
        if cfg.transform.name == 'images':
            env = ClipWrapper(env, 0., 1.)
        else:
            env = ClipWrapper(env, -1., 1.)
    env = ToFloatWrapper(env)
    return env

def initialize_model(input_shape, n_actions, cfg: configs.Config):
    if cfg.model.lib == 'disent':
        return build_disent_model(input_shape, cfg)
    elif cfg.model.name is not None:
        if cfg.model.param_scaling > 1:
            scale = cfg.model.param_scaling
            print(f'Scaling up model parameters by {scale}x')
            # scale MLP n_units_per_layer
            if cfg.model.mlp.get('n_units_per_layer', None) is not None:
                n_units = cfg.model.mlp.n_units_per_layer
                cfg.model.mlp.n_units_per_layer = scale * n_units
            # scale CNN n_output_channels
            if cfg.model.cnn.get('n_output_channels', None) is not None:
                n_chans = cfg.model.cnn.n_output_channels
                cfg.model.cnn.n_output_channels = [scale * n_chan for n_chan in n_chans]
            # reset param_scaling to avoid double-scaling confusion later
            cfg.model.param_scaling = 1

        if cfg.model.arch.type == 'qnet':
            module = BaseModel
        elif cfg.model.arch.type == 'enc':
            module = EncoderModel
        elif cfg.model.arch.type == 'ae':
            module = AutoencoderModel
        elif cfg.model.arch.type == 'paired_ae':
            module = PairedAutoencoderModel
        elif cfg.model.arch.type == 'wm':
            module = WorldModel
        else:
            raise NotImplementedError(f'Unknown model architecture: {cfg.model.arch.type}')

        module_args = {'input_shape': input_shape, 'n_actions': n_actions, 'cfg': cfg}
        if cfg.loader.load_model:
            cfg.loader.checkpoint_path = get_checkpoint_path(cfg)
            model = module.load_from_checkpoint(cfg.loader.checkpoint_path, **module_args)
            if cfg.loader.eval_only:
                model.eval()
                model.encoder.freeze()
        else:
            cfg.loader.checkpoint_path = None
            model = module(**module_args)

    model = model.to(cfg.model.device)
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
        checkpoints = glob.glob(os.path.join(checkpoint_dir, 'last.ckpt'))
        if checkpoints == []:
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
