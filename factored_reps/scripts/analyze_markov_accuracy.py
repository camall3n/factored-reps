from argparse import Namespace
import glob
import pickle
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seeding
import torch
from tqdm import tqdm

from visgrid.gridworld import GridWorld
from visgrid.sensors import *
from factored_reps.models.factornet import FactorNet
from markov_abstr.gridworld.models.featurenet import FeatureNet
from factored_reps.utils import get_parser, load_hyperparams_and_inject_args

parser = get_parser()
parser.add_argument('-s', '--seed', type=int, default=1)
parser.add_argument('-e', '--experiment', type=int, default=56)
parser.add_argument("-f", "--fool_ipython", help="Dummy arg to fool ipython", default="1")
args = parser.parse_args()
del args.fool_ipython

filepaths = glob.glob('results/logs/exp{}*/args-{}.txt'.format(args.experiment, args.seed))
for filepath in filepaths:
    with open(filepath, 'r') as argsfile:
        line = argsfile.readline()
        args = eval(line)
    break

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

output_dir = 'results/analyze_markov_accuracy/{}'.format('quick' if device.type ==
                                                         'cpu' else args.tag)
os.makedirs(output_dir, exist_ok=True)

model_file = 'results/models/{}/fnet-{}_latest.pytorch'.format(args.tag, args.seed)

#%% ------------------ Load environment ------------------
experiences_dir = os.path.join('results', 'taxi-experiences', args.taxi_experiences)
filename_pattern = os.path.join(experiences_dir, 'seed-*.pkl')

results_files = glob.glob(filename_pattern)

experiences_limit = 1000 if device.type == 'cpu' else 5000

experiences = []
n_episodes = 0
for results_file in sorted(results_files)[:experiences_limit]:
    with open(results_file, 'rb') as file:
        current_experiences = pickle.load(file)
    for experience in current_experiences:
        experience['episode'] = n_episodes
    experiences.extend(current_experiences)
    n_episodes += 1

def extract_array(experiences, key):
    return np.asarray([experience[key] for experience in experiences])

episodes = extract_array(experiences, 'episode')
steps = extract_array(experiences, 'step')
obs = extract_array(experiences, 'ob')
states = extract_array(experiences, 'state')
actions = extract_array(experiences, 'action')
rewards = extract_array(experiences, 'reward')
next_obs = extract_array(experiences, 'next_ob')
next_states = extract_array(experiences, 'next_state')
dones = extract_array(experiences, 'done')
goals = extract_array(experiences, 'goal')

#%% ------------------ Load model ------------------
def torchify(x):
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    result = torch.as_tensor(np.moveaxis(x, -1, 1)).float().to(device)
    return result

x = torchify(obs[0])
xp = torchify(next_obs[0])
fnet = FeatureNet(args,
                  n_actions=5,
                  input_shape=x.squeeze(0).shape,
                  latent_dims=args.latent_dims,
                  device=device)
fnet.to(device)
fnet.load(model_file, to=device)

#%%
n_training = len(states) // 2
n_test = 2000

if device.type == 'cpu':
    n_training = n_training // 10
    n_test = n_test // 10

def get_action_predictions(obs, next_obs):
    return fnet.predict_a(torchify(obs).to(device),
                          torchify(next_obs).to(device)).detach().cpu().numpy()

def get_discrim_predictions(obs, next_obs):
    return fnet.predict_is_fake(torchify(obs).to(device),
                                torchify(next_obs).to(device)).detach().cpu().numpy()

def compute_accuracy(labels, predictions):
    n_correct = (labels == predictions).sum()
    accuracy = 100 * n_correct / len(labels)
    return n_correct, accuracy

inv_predictions = []
discrim_predictions_on_real = []
discrim_predictions_on_fake = []
# divide training samples into batches, to save GPU memory
n_batch_divisions = int(np.ceil(n_training / n_test) + 1)
batch_divisions = np.linspace(0, n_training, n_batch_divisions).astype(int)
batch_starts = batch_divisions[:-1]
batch_ends = batch_divisions[1:]
for low, high in zip(tqdm(batch_starts), batch_ends):
    inv_predictions.append(get_action_predictions(obs[low:high], next_obs[low:high]))
    discrim_predictions_on_real.append(get_discrim_predictions(obs[low:high], next_obs[low:high]))
    discrim_predictions_on_fake.append(
        get_discrim_predictions(obs[low:high], np.random.permutation(next_obs[low:high])))

predicted_a_train = np.concatenate(inv_predictions)
predicted_a_test = get_action_predictions(obs[-n_test:], next_obs[-n_test:])
n_correct_inv_train, inv_train_accuracy = compute_accuracy(actions[:n_training], predicted_a_train)
n_correct_inv_test, inv_test_accuracy = compute_accuracy(actions[-n_test:], predicted_a_test)

predicted_is_fake_on_real_train = np.concatenate(discrim_predictions_on_real)
predicted_is_fake_on_fake_train = np.concatenate(discrim_predictions_on_fake)
predicted_is_fake_on_real_test = get_discrim_predictions(obs[-n_test:], next_obs[-n_test:])
predicted_is_fake_on_fake_test = get_discrim_predictions(obs[-n_test:],
                                                         np.random.permutation(next_obs[-n_test:]))

n_correct_discrim_on_real_train, discrim_train_accuracy_on_real = compute_accuracy(
    np.zeros_like(predicted_is_fake_on_real_train), predicted_is_fake_on_real_train)
n_correct_discrim_on_fake_train, discrim_train_accuracy_on_fake = compute_accuracy(
    np.ones_like(predicted_is_fake_on_real_train), predicted_is_fake_on_fake_train)
n_correct_discrim_on_real_test, discrim_test_accuracy_on_real = compute_accuracy(
    np.zeros_like(predicted_is_fake_on_real_test), predicted_is_fake_on_real_test)
n_correct_discrim_on_fake_test, discrim_test_accuracy_on_fake = compute_accuracy(
    np.ones_like(predicted_is_fake_on_fake_test), predicted_is_fake_on_fake_test)

with open(os.path.join(output_dir, 'results.txt'), 'w') as output_file:
    output_file.write('Inverse model accuracy:\n')
    output_file.write('Training: {} correct out of {} total = {}%\n'.format(
        n_correct_inv_train, n_training, inv_train_accuracy))
    output_file.write('Test: {} correct out of {} total = {}%\n'.format(
        n_correct_inv_test, n_test, inv_test_accuracy))

    output_file.write('\n')

    output_file.write('Discriminator accuracy:\n')
    output_file.write('Training (on real pairs): {} correct out of {} total = {}%\n'.format(
        n_correct_discrim_on_real_train, n_training, discrim_train_accuracy_on_real))
    output_file.write('Training (on fake pairs): {} correct out of {} total = {}%\n'.format(
        n_correct_discrim_on_fake_train, n_training, discrim_train_accuracy_on_fake))
    output_file.write('Test (on real pairs): {} correct out of {} total = {}%\n'.format(
        n_correct_discrim_on_real_test, n_test, discrim_test_accuracy_on_real))
    output_file.write('Test (on fake pairs): {} correct out of {} total = {}%\n'.format(
        n_correct_discrim_on_fake_test, n_test, discrim_test_accuracy_on_fake))

#%%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

def action_histogram(a_actual, a_predicted, ax, title):
    dfs = []
    for a, label in zip([a_actual, a_predicted], ['actual', 'predicted']):
        df = pd.DataFrame(a, columns=['action'])
        df['mode'] = label
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    sns.histplot(data=data,
                 x='action',
                 hue='mode',
                 discrete=True,
                 label='train',
                 multiple='dodge',
                 shrink=0.8,
                 ax=ax)
    ax.set_title(title)

for a, a_hat, mode, ax in zip(
        [actions[:n_training], actions[-n_test:]],
        [predicted_a_train, predicted_a_test],
        ['training', 'test'],
        axes,
    ):
    action_histogram(a, a_hat, ax=ax, title=mode)

plt.savefig(os.path.join(output_dir, 'predicted_action.png'))
plt.show()

#%%

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

def action_confusion_matrix(a_actual, a_predicted, ax, title):
    confusion_counts = list(
        zip(*np.unique(
            np.stack([a_actual[:n_training], a_predicted], 1), axis=0, return_counts=True)))
    heatmap = np.zeros((5, 5))
    for (a, ahat), count in confusion_counts:
        heatmap[a, ahat] = count

    im = ax.imshow(heatmap, interpolation='nearest')
    ax.set_xlabel('predicted action')
    ax.set_ylabel('actual action')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.75)

for a, a_hat, mode, ax in zip(
        [actions[:n_training], actions[-n_test:]],
        [predicted_a_train, predicted_a_test],
        ['training', 'test'],
        axes,
    ):
    action_confusion_matrix(a, a_hat, ax=ax, title=mode)

fig.suptitle('Inverse model classifications')
plt.tight_layout()
plt.subplots_adjust(top=1.0)
plt.savefig(os.path.join(output_dir, 'action_confusion_matrix.png'))
plt.show()
