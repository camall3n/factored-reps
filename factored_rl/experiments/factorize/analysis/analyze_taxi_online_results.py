from argparse import Namespace
import glob
import imageio
import json
#!! do not import matplotlib until you check input arguments
import numpy as np
import os
import pickle
import platform
import random
import seeding
import sys
import torch
from tqdm import tqdm
from factored_rl.agents.replaymemory import ReplayMemory

from factored_rl import utils
from factored_rl.experiments.markov.taxi.generate_taxi_experiences import generate_experiences
from factored_rl.models.markov.featurenet import FeatureNet
from factored_rl.models.debug.categorical_predictor import CategoricalPredictor
from factored_rl.experiments.markov.analysis.repvis import RepVisualization
from visgrid.envs import TaxiEnv
from visgrid.wrappers.transforms import TransformWrapper

def analyze_results(output_dir, replay_test, fnet, predictor):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    fields = ['_index_', 'ob', 'state', 'action', 'next_ob']
    test_batch = replay_test.retrieve(fields=fields)
    idx, obs, states, actions, next_obs = test_batch

    def compute_accuracy(labels, predictions):
        n_correct = (labels == predictions).sum()
        n_total = len(labels)
        accuracy = 100 * n_correct / n_total
        return n_correct, n_total, accuracy

    def record_model_accuracy(output_file, description, n_correct, n_total, accuracy):
        output_file.write('{}: {} correct out of {} total = {}%\n'.format(
            description, n_correct, n_total, accuracy))
        output_file.write('\n')

    with open(os.path.join(output_dir, 'results.txt'), 'w') as output_file:
        #% ------------------ Compute inverse model accuracy ------------------
        # predicted_actions = fnet.predict_a(obs, next_obs).detach().cpu().numpy()
        # inv_results = compute_accuracy(actions.detach().cpu().numpy(), predicted_actions)
        # record_model_accuracy(output_file, 'Inverse model accuracy', *inv_results)

        #% ------------------ Compute discriminator accuracy ------------------
        predicted_is_fake_on_positives = fnet.predict_is_fake_transition(
            obs, actions, next_obs).detach().cpu().numpy()
        discrim_results_positives = compute_accuracy(np.zeros_like(predicted_is_fake_on_positives),
                                                     predicted_is_fake_on_positives)
        record_model_accuracy(output_file, 'Discriminator accuracy (positives)',
                              *discrim_results_positives)

        for mode in ['random', 'following']:
            negatives = fnet.get_negatives(replay_test, idx, mode=mode)
            predicted_is_fake_on_negatives = fnet.predict_is_fake_transition(
                obs, actions, negatives).detach().cpu().numpy()
            discrim_results_negatives = compute_accuracy(
                np.ones_like(predicted_is_fake_on_negatives), predicted_is_fake_on_negatives)
            record_model_accuracy(output_file,
                                  'Discriminator accuracy ({}-state negatives)'.format(mode),
                                  *discrim_results_negatives)

    #% ------------------ Generate predictor confusion plots ------------------
    z0 = fnet.encode(obs)
    test_reconstructions = predictor.predict(z0).detach().cpu().numpy()

    def generate_confusion_plots(s_actual, s_predicted):
        state_vars = ['taxi_row', 'taxi_col', 'passenger_row', 'passenger_col',
                      'in_taxi'][:len(states[0])]
        n_passengers = 1
        n_values_per_variable = [5, 5] + ([5, 5, 2] * n_passengers)

        fig, axes = plt.subplots(len(state_vars), 1, figsize=(3, 2 * len(state_vars)))

        for state_var_idx, (state_var, n_values,
                            ax) in enumerate(zip(state_vars, n_values_per_variable, axes)):
            bins = n_values
            value_range = ((-0.5, n_values - 0.5), (0, n_values - 0.5))
            h = ax.hist2d(x=s_predicted[:, state_var_idx],
                          y=s_actual[:, state_var_idx],
                          bins=bins,
                          range=value_range)
            counts, xedges, yedges, im = h
            fig.colorbar(im, ax=ax)
            # sns.histplot(
            #     x=s_predicted[:, state_var_idx],
            #     y=s_actual[:, state_var_idx],
            #     bins=bins,
            #     discrete=True,
            #     cbar=True,
            #     stat='count',
            #     ax=ax,
            # )

            for i in range(len(yedges) - 1):
                for j in range(len(xedges) - 1):
                    ax.text(xedges[j] + 0.5,
                            yedges[i] + 0.4,
                            int(counts.T[i, j]),
                            color="w",
                            ha="center",
                            va="center",
                            fontweight="bold")
            ax.set_title(state_var)
            ax.set_xlabel('predicted')
            ax.set_ylabel('actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'predictor_confusion_plots.png'),
                    facecolor='white',
                    edgecolor='white')
        # plt.show()

    generate_confusion_plots(states.detach().cpu().numpy(), test_reconstructions)

if __name__ == '__main__':
    #% ------------------ Parse args/hyperparameters ------------------
    parser = utils.get_parser()
    # yapf: disable
    parser.add_argument('--model_type', type=str, default='markov',
                        choices=['factored-split', 'factored-combined', 'focused-autoenc', 'markov', 'autoencoder', 'pixel-predictor'],
                        help='Which type of representation learning method')
    parser.add_argument('--load_markov', type=str, default=None,
                        help='Specifies a tag to load a pretrained Markov abstraction')
    parser.add_argument('--freeze_markov', action='store_true',
                        help='Prevents Markov abstraction from training')
    parser.add_argument('-s','--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--replay_buffer_size', type=int, default=20000,
                        help='Number of experiences in training replay buffer')
    parser.add_argument('--n_steps_per_episode', type=int, default=5,
                        help='Reset environment after this many steps')
    parser.add_argument('--n_expected_times_to_sample_experience', type=int, default=10,
                        help='Expected number of times to sample each experience in the replay buffer before replacement')
    parser.add_argument('-t','--tag', type=str, required=True,
                        help='Tag for identifying experiment')
    parser.add_argument('--hyperparams', type=str, default='factored_rl/hyperparams/taxi.csv',
                        help='Path to hyperparameters csv file')
    parser.add_argument('-v','--video', action='store_true',
                        help="Save training video")
    parser.add_argument('--no_graphics', action='store_true',
                        help='Turn off graphics (e.g. for running on cluster)')
    parser.add_argument('--save', action='store_true',
                        help='Save final network weights')
    parser.add_argument('--no_sigma', action='store_true',
                        help='Turn off sensors and just use true state; i.e. x=s')
    parser.add_argument('--grayscale', action='store_true',
                        help='Grayscale observations (default is RGB)')
    parser.add_argument('--quick', action='store_true',
                        help='Flag to reduce number of updates for quick testing')
    # yapf: enable

    args = utils.parse_args_and_load_hyperparams(parser)
    if args.load_markov is not None:
        args.load_markov = os.path.join(args.load_markov, 'fnet-{}_best.pytorch'.format(args.seed))

    # Move all loss coefficients to a sub-namespace
    coefs = Namespace(**{name: value for (name, value) in vars(args).items() if name[:2] == 'L_'})
    for coef_name in vars(coefs):
        delattr(args, coef_name)
    args.coefs = coefs

    if (args.markov_dims > 0 and args.model_type not in ['factored-split', 'focused-autoenc']):
        print("Warning: 'markov_dims' arg not valid for network type {}. Ignoring...".format(
            args.model_type))

    if args.no_graphics:
        import matplotlib
        # Force matplotlib to not use any Xwindows backend.
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    log_dir = 'results/taxi/logs/' + str(args.tag)
    models_dir = 'results/taxi/models/' + str(args.tag)
    vid_dir = 'results/taxi/videos/' + str(args.tag)
    maze_dir = 'results/taxi/mazes/' + str(args.tag)
    os.makedirs(log_dir, exist_ok=True)

    if args.video:
        os.makedirs(vid_dir, exist_ok=True)
        os.makedirs(maze_dir, exist_ok=True)
        video_filename = vid_dir + '/video-{}.mp4'.format(args.seed)
        final_image_filename = vid_dir + '/final-{}.png'.format(args.seed)
        best_image_filename = vid_dir + '/best-{}.png'.format(args.seed)
        maze_file = maze_dir + '/maze-{}.png'.format(args.seed)

    train_log = open(log_dir + '/train-{}.txt'.format(args.seed), 'w')
    test_log = open(log_dir + '/test-{}.txt'.format(args.seed), 'w')
    with open(log_dir + '/args-{}.txt'.format(args.seed), 'w') as arg_file:
        arg_file.write(repr(args))

    seeding.seed(args.seed, np, random)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    #% ------------------ Define MDP ------------------
    env = TaxiEnv()
    env = TransformWrapper(env, lambda obs: np.moveaxis(obs, -1, 0))
    env.reset()

    #% ------------------ Generate & store experiences ------------------
    on_retrieve = {
        '_index_': lambda items: np.asarray(items),
        '*': lambda items: torch.as_tensor(np.asarray(items)).to(device),
        'ob': lambda items: items.float(),
        'next_ob': lambda items: items.float(),
        'action': lambda items: items.long()
    }
    replay_test = ReplayMemory(args.batch_size, on_retrieve)
    replay_train = ReplayMemory(args.replay_buffer_size, on_retrieve)
    n_test_episodes = int(np.ceil(args.batch_size / args.n_steps_per_episode))
    n_train_episodes = int(np.ceil(args.replay_buffer_size / args.n_steps_per_episode))
    if args.quick:
        n_train_episodes = n_test_episodes
    test_seed = 1
    train_seed = 2 + args.seed
    print('Initializing replay buffer...')
    for buffer, n_episodes, seed in zip([replay_train, replay_test],
                                        [n_train_episodes, n_test_episodes],
                                        [train_seed, test_seed]):
        for exp in generate_experiences(env,
                                        lambda x: x,
                                        n_episodes,
                                        n_steps_per_episode=args.n_steps_per_episode,
                                        seed=seed):
            if buffer is replay_test:
                s = exp['state']
                exp['color'] = s[0] * env.cols + s[1]
            buffer.push(exp)

    #% ------------------ Define models ------------------
    fnet = FeatureNet(args,
                      n_actions=env.action_space.n,
                      input_shape=replay_train.retrieve(0, 'ob').shape[1:],
                      latent_dims=args.latent_dims,
                      device=device).to(device)
    fnet.load(models_dir + '/fnet-{}_best.pytorch'.format(args.seed))
    fnet.print_summary()

    n_values_per_variable = [5, 5] + ([5, 5, 2] * args.n_passengers)
    predictor = CategoricalPredictor(
        n_inputs=args.latent_dims,
        n_values=n_values_per_variable,
        learning_rate=args.learning_rate,
    ).to(device)
    predictor.load(models_dir + '/predictor-{}_best.pytorch'.format(args.seed))
    predictor.print_summary()

    #% ------------------ Analyze results ------------------
    from factored_rl.experiments.factorize.analysis.analyze_taxi_online_results import analyze_results

    output_dir = 'results/taxi/analyze_markov_accuracy/{}/seed-{}'.format(args.tag, args.seed)
    analyze_results(output_dir, replay_test, fnet, predictor)
