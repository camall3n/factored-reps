import hydra
import optuna
import sqlite3

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
    try:
        study = optuna.create_study(
            study_name=cfg.experiment,
            storage=f"sqlite:///factored_rl/hyperparams/tuning/{cfg.experiment}.db",
            load_if_exists=True,
            direction='maximize')
    except sqlite3.OperationalError:
        # Fixes race condition where two nodes try to create at the same time
        study = optuna.load_study(
            study_name=cfg.experiment,
            storage=f"sqlite:///factored_rl/hyperparams/tuning/{cfg.experiment}.db")

    def objective(trial: optuna.Trial):
        cfg.trial = f'trial_{trial.number:04d}'
        cfg.trainer.rep_learning_rate = trial.suggest_float(
            'trainer.rep_learning_rate',
            low=1e-5,
            high=1e-2,
            log=True,
        )
        cfg.loss.sparsity.name = trial.suggest_categorical(
            'loss.sparsity.name',
            ['sum_div_max', 'unit_pnorm'],
        )
        cfg.loss.actions = trial.suggest_float('loss.actions', low=1e-5, high=10.0, log=True)
        cfg.loss.effects = trial.suggest_float('loss.effects', low=1e-5, high=10.0, log=True)
        cfg.loss.parents = trial.suggest_float('loss.parents', low=1e-5, high=10.0, log=True)

        factorize(cfg)

        cfg.loader.should_load = True
        cfg.loader.should_train = False
        quick = cfg.trainer.quick
        cfg.trainer = get_config(name='trainer/rl', path='../conf', overrides=[]).trainer
        cfg.trainer.quick = quick
        cfg.trainer.rl_learning_rate = trial.suggest_float(
            'trainer.rl_learning_rate',
            low=1e-5,
            high=1e-2,
            log=True,
        )

        results = rl_vs_rep(cfg)
        final_ep_results = results[-1]
        score = final_ep_results['total_reward'] / final_ep_results['total_steps']
        return score

    study.optimize(
        objective,
        n_trials=1,
    )

if __name__ == "__main__":
    main()
