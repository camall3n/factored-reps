import json
import logging

# Args & hyperparams
import hydra
from factored_rl.experiments import configs

# Env
from factored_rl.experiments.common import initialize_env, initialize_model

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
    transform = ToImgTensorF32() if cfg.model.name is not None else None
    dataset = GymEnvData(env, cfg.seed, transform=transform)
    encode = get_encode_fn(dataset.x_shape, cfg)

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
    logging.getLogger().info(f'Results logged to: {filename}')

def get_encode_fn(input_shape, cfg):
    if cfg.model.name is None:
        return lambda x: x
    else:
        model = initialize_model(input_shape, cfg)
        return lambda x: model.encoder(x)

# ----------------------------------------
# Disent metrics
# ----------------------------------------

def initialize_metrics(cfg):
    disent_seed(cfg.seed)
    return [
        metrics.metric_dci,
        metrics.metric_mig,
    ]

if __name__ == '__main__':
    main()
