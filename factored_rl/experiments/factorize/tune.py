import os

import hydra
import optuna

# Args & hyperparams
import hydra
from hydra.core.global_hydra import GlobalHydra
from factored_rl import configs

from factored_rl.experiments.factorize.run import main as factorize
from factored_rl.experiments.rl_vs_rep.run import main as rl_vs_rep

def get_config(name, path, overrides):
    GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path=path):
        cfg = hydra.compose(config_name=name, overrides=overrides)
    return cfg

@hydra.main(config_path="../conf", config_name='config', version_base=None)
def main(cfg: configs.Config):
    os.makedirs('factored_rl/hyperparams/tuning/', exist_ok=True)

    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(
            f"./factored_rl/hyperparams/tuning/{cfg.experiment}.journal"))

    if not (cfg.tuner.tune_rep or cfg.tuner.tune_rl):
        raise RuntimeError('No variables to tune. Enable tuner.tune_rep and/or tuner.tune_rl')
    if cfg.tuner.tune_metric not in ['rl', 'reconst']:
        raise RuntimeError(f'Unknown tuning metric: {cfg.tuner.tune_metric}')

    if cfg.tuner.should_prune:
        if not cfg.tuner.tune_rep:
            raise RuntimeError(f'Pruning is only implemented when tune_rep==True')
        if cfg.tuner.tune_metric not in ['reconst']:
            raise RuntimeError(
                f'Pruning is incompatible with tuning metric "{cfg.tuner.tune_metric}"')
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=2000, # every trial gets at least min_resource steps
            reduction_factor=2, # only promote the best 1/reduction_factor of trials in each rung
        )

    else:
        pruner = None

    study = optuna.create_study(
        study_name=cfg.experiment,
        storage=storage,
        pruner=pruner,
        load_if_exists=True,
        direction='minimize',
    )

    def objective(trial: optuna.Trial):
        cfg.trial = f'trial_{trial.number:04d}'
        arch = cfg.model.arch.type
        if cfg.tuner.tune_rep:
            if arch not in ['wm', 'paired_ae', 'ae']:
                raise RuntimeError(f'Cannot tune representation for architecture "{arch}"')
            cfg.trainer.rep_learning_rate = trial.suggest_float(
                'rep_lr',
                low=2e-4,
                high=5e-3,
                log=True,
            )

            cfg.model.param_scaling = trial.suggest_categorical('param_scaling', [2, 4, 8, 16])

            if arch in ['wm', 'paired_ae']:
                cfg.loss.sparsity.name = trial.suggest_categorical('sparsity',
                                                                   ['sum_div_max', 'unit_pnorm'])
                cfg.loss.actions = trial.suggest_float('L_actions', low=1e-5, high=1e-3, log=True)
                cfg.loss.effects = trial.suggest_float('L_effects', low=1e-1, high=3e2, log=True)

            if arch == 'wm':
                cfg.loss.parents = trial.suggest_float('L_parents', low=3e-5, high=4e-2, log=True)

            if cfg.env.name == 'taxi':
                cfg.env.depot_dropoff_only = True

            callback_metrics = factorize(cfg, trial)

            cfg.loader.load_config = False
            cfg.loader.load_model = True
            cfg.loader.eval_only = True
            quick = cfg.trainer.quick
            cfg.trainer = get_config(name='trainer/rl', path='../conf', overrides=[]).trainer
            cfg.trainer.quick = quick

        if cfg.tuner.tune_rl:
            cfg.trainer.rl_learning_rate = trial.suggest_float(
                'rl_lr',
                low=3e-5,
                high=3e-3,
                log=True,
            )
            if cfg.tuner.tune_dqn:
                if cfg.agent.name != 'dqn':
                    raise RuntimeError(f"Cannot tune DQN with agent '{cfg.agent.name}'")
                # cfg.model.qnet.n_hidden_layers = trial.suggest_int(
                #     'n_hidden_layers',
                #     low=1,
                #     high=2,
                # )
                cfg.agent.target_copy_alpha = trial.suggest_float(
                    'target_copy_alpha',
                    low=1e-4,
                    high=1.0,
                    log=True,
                )
                cfg.agent.epsilon_half_life_steps = trial.suggest_int(
                    'epsilon_half_life',
                    low=1000,
                    high=20000,
                    log=True,
                )
                cfg.agent.updates_per_interaction = trial.suggest_int(
                    'updates_per_interaction',
                    low=1,
                    high=128,
                    log=True,
                )

        if cfg.env.name == 'taxi':
            cfg.env.depot_dropoff_only = True

        results = rl_vs_rep(cfg)

        if cfg.tuner.tune_metric == 'rl':
            final_ep_results = results[-1]
            score = final_ep_results['total_steps'] / (final_ep_results['total_reward'] + 1e-5)
        elif cfg.tuner.tune_metric == 'reconst':
            score = callback_metrics['loss/reconst']
        else:
            raise NotImplementedError(f'Unknown tuning metric: {cfg.tuner.tune_metric}')
        return score

    study.optimize(
        objective,
        n_trials=1,
    )

if __name__ == "__main__":
    main()
