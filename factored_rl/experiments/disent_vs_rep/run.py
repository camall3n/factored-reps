import json
import logging

# Args & hyperparams
import hydra
from factored_rl.experiments import configs

# Env
from factored_rl.experiments.common import initialize_env

# Disent
from disent import metrics
from disent.util.seeds import seed as disent_seed
from disent.dataset.data import GymEnvData
from disent.dataset import DisentDataset
from disent.dataset.sampling import SingleSampler

# ----------------------------------------
# Args & hyperparams
# ----------------------------------------

@hydra.main(config_path=None, config_name='disent_vs_rep', version_base=None)
def main(cfg):
    configs.initialize_experiment(cfg)
    env = initialize_env(cfg, cfg.seed)
    data = GymEnvData(env)

    metric_scores = {}
    for metric in initialize_metrics(cfg):
        dataset = DisentDataset(dataset=data, sampler=SingleSampler())
        scores = metric(dataset, lambda x: x)
        metric_scores.update(scores)

    results = dict(**metric_scores)
    results.update({
        'experiment': cfg.experiment,
        'trial': cfg.trial,
        'seed': cfg.seed,
        'env': cfg.env.name,
        'noise': cfg.noise,
        'transform': cfg.transform.name,
    })

    # Save results
    filename = cfg.dir + 'results.json'
    with open(filename, 'w') as file:
        json.dump(results, file)
    logging.getLogger().info(f'Results logged to: {filename}')

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
