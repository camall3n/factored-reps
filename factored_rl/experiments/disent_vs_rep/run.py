import json
import logging

# Args & hyperparams
import hydra
from factored_rl import configs

# Env
from factored_rl.experiments.common import initialize_env, initialize_model
import torch

# Disent
from disent import metrics
from disent.util.seeds import seed as disent_seed
from disent.dataset.data import GymEnvData
from disent.dataset import DisentIterDataset
from disent.dataset.transform import ToImgTensorF32

# ----------------------------------------
# Args & hyperparams
# ----------------------------------------

@hydra.main(config_path="../conf", config_name='config', version_base=None)
def main(cfg):
    configs.initialize_experiment(cfg, 'disent_vs_rep')
    env = initialize_env(cfg, cfg.seed)
    input_shape = env.observation_space.shape
    dataset = GymEnvData(env, cfg.seed, transform=None)
    encode = get_encode_fn(input_shape, env.action_space.n, cfg)

    metric_scores = {}
    for metric in initialize_metrics(cfg):
        disent_dataset = DisentIterDataset(dataset=dataset)
        scores = metric(disent_dataset, encode)
        metric_scores.update(scores)

    results = dict(**metric_scores)
    results.update({
        'experiment': cfg.experiment,
        'trial': cfg.trial,
        'seed': cfg.seed,
        'env': cfg.env.name,
        'noise': cfg.transform.noise,
        'transform': cfg.transform.name,
    })

    # Save results
    filename = cfg.dir + 'results.json'
    with open(filename, 'w') as file:
        json.dump(results, file)
        file.write('\n')
    logging.getLogger().info(f'Results logged to: {filename}')

def get_encode_fn(input_shape, n_actions, cfg):
    if cfg.model.name is None:
        return lambda x: x
    else:
        model = initialize_model(input_shape, n_actions, cfg)
        return lambda x: model.encoder(torch.as_tensor(x).float().to(cfg.model.device))

# ----------------------------------------
# Disent metrics
# ----------------------------------------

def initialize_metrics(cfg):
    disent_seed(cfg.seed)
    if cfg.trainer.quick:
        return [
            lambda data, fn: metrics.metric_dci(data, fn, num_train=20, num_test=10, batch_size=2),
            lambda data, fn: metrics.metric_mig(data, fn, num_train=20, batch_size=2),
        ]
    else:
        return [
            metrics.metric_dci,
            metrics.metric_mig,
        ]

if __name__ == '__main__':
    main()
