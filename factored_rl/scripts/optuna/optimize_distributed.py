import argparse
import optuna

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2)**2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=str)
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    study = optuna.create_study(
        study_name="distributed-example",
        storage="sqlite:///factored_rl/scripts/optuna/distributed-example.db",
        load_if_exists=True)

    if not args.summary:
        study.optimize(objective, n_trials=10)
    else:
        print(f'Best param: {study.best_params}')
        print(f'Best value: {study.best_value}')
        print()
        print(f'Best trial: {study.best_trials}')
