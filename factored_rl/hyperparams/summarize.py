import argparse
import os

import optuna

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--study-name', type=str, default=None)
    parser.add_argument('-p', '--storage-path', type=str)
    args = parser.parse_args()

    if args.study_name is None:
        args.study_name = os.path.splitext(os.path.basename(args.storage_path))[0]

    study = optuna.load_study(study_name=args.study_name, storage=f"sqlite:///{args.storage_path}")

print(f'Best param: {study.best_params}')
print(f'Best value: {study.best_value}')
print()
print(f'Best trial: {study.best_trial}')
