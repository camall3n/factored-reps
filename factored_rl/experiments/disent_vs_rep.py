import json

# Args & hyperparams
from factored_rl import configs

# Env
import gym
from visgrid.envs import GridworldEnv, TaxiEnv
from factored_rl.wrappers import RotationWrapper
from factored_rl.wrappers import FactorPermutationWrapper, ObservationPermutationWrapper
from visgrid.wrappers import GrayscaleWrapper, InvertWrapper, NormalizedFloatWrapper, NoiseWrapper

# Disent
from disent import metrics
from disent.util.seeds import seed as disent_seed
from disent.dataset.data import GymEnvData
from disent.dataset import DisentDataset
from disent.dataset.sampling import SingleSampler

# ----------------------------------------
# Args & hyperparameters
# ----------------------------------------

parser = configs.new_parser()
# yapf: disable
parser.add_argument('-e', '--experiment', type=str, default='rl_vs_disent', help='A name for the experiment')
parser.add_argument('-t', '--trial', type=str, default='trial', help='A name for the trial')
parser.add_argument('-s', '--seed', type=int, default=0, help='A seed for the random number generator')
parser.add_argument('--env', type=str, default='gridworld', help="['gridworld', 'taxi', 'CartPole-v1', ...]")
parser.add_argument('--no-timestamp', action='store_true', help='Disable automatic trial timestamps')
parser.add_argument('--noise', action='store_true')
parser.add_argument('--rotate', action='store_true')
parser.add_argument('--permutation', type=str, default='identity', choices=['identity', 'factors', 'states'])
parser.add_argument('--images', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('-f', '--fool-ipython', action='store_true',
    help='Dummy arg to make ipython happy')
# yapf: enable

# ----------------------------------------
# Environment & wrappers
# ----------------------------------------
def initialize_env(args, cfg: configs.EnvConfig):
    if args.env == 'gridworld':
        env = GridworldEnv(10,
                           10,
                           exploring_starts=True,
                           terminate_on_goal=True,
                           fixed_goal=True,
                           hidden_goal=True,
                           should_render=False,
                           dimensions=GridworldEnv.dimensions_6x6_to_18x18)
    elif args.env == 'taxi':
        env = TaxiEnv(size=5,
                      n_passengers=1,
                      exploring_starts=True,
                      terminate_on_goal=True,
                      should_render=False,
                      dimensions=TaxiEnv.dimensions_5x5_to_64x64)
    else:
        env = gym.make(args.env)
        # TODO: wrap env to support disent protocol

    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    disent_seed(args.seed)

    if args.permutation != 'identity':
        assert not args.images
        if args.permutation == 'factors':
            env = FactorPermutationWrapper(env)
        elif args.permutation == 'states':
            env = ObservationPermutationWrapper(env)
    if args.images:
        assert not args.rotate
        env.set_rendering(enabled=args.images)
        env = InvertWrapper(GrayscaleWrapper(env))
    else:
        env = NormalizedFloatWrapper(env)
        if args.rotate:
            env = RotationWrapper(env)
    if args.noise:
        env = NoiseWrapper(env, cfg.noise_std)

    return env

# ----------------------------------------
# Disent metrics
# ----------------------------------------

def initialize_metrics():
    return [
        metrics.metric_dci,
        metrics.metric_mig,
    ]

# ----------------------------------------
# Run experiment trial
# ----------------------------------------

args, cfg, log = configs.initialize_experiment(parser)
env = initialize_env(args, cfg.env)
data = GymEnvData(env)

metric_scores = {}
for metric in initialize_metrics():
    dataset = DisentDataset(dataset=data, sampler=SingleSampler())
    scores = metric(dataset, lambda x: x)
    metric_scores.update(scores)

results = dict(**metric_scores)
results.update({
    'experiment': args.experiment,
    'trial': args.trial,
    'seed': args.seed,
    'env': args.env,
    'noise': args.noise,
    'rotate': args.rotate,
    'permute': args.permutation,
    'images': args.images,
})

# ----------------------------------------
# Save results
# ----------------------------------------
filename = cfg.experiment.dir + 'results.json'
with open(filename, 'w') as file:
    json.dump(results, file)
