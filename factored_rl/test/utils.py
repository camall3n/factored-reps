import os
import shutil

import hydra

from factored_rl.configs import _initialize_experiment_dir

def get_config(overrides):
    # the config_path in hydra.initialize is relative to the locaiton of THIS FILE
    with hydra.initialize(version_base=None, config_path='../experiments/conf'):
        cfg = hydra.compose(config_name='config', overrides=overrides)
    return cfg

def cleanup():
    cfg = get_config(["experiment=pytest"])
    dirname = _initialize_experiment_dir(cfg)
    shutil.rmtree(dirname)

    journalfile = 'factored_rl/hyperparams/tuning/pytest.journal'
    if os.path.isfile(journalfile):
        os.remove(journalfile)
