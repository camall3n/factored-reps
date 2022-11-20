import optuna

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2)**2

if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
