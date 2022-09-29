import hydra

from factored_rl.experiments.disent_vs_rep.run import main as disent_vs_rep
from factored_rl.experiments.rl_vs_rep.run import main as rl_vs_rep
from factored_rl.experiments.factorize.run import main as factorize

def get_config(overrides):
    with hydra.initialize(version_base=None, config_path='../../experiments/conf'):
        cfg = hydra.compose(config_name='config', overrides=overrides)
    return cfg

def test_disent_vs_rep():
    configurations = [
        ["env=gridworld", "transform=rotate"],
        ["env=taxi", "transform=images", "model=ae/ae_cnn_64"],
    ] # yapf: disable
    for overrides in configurations:
        overrides.extend([
            "experiment=pytest",
            "timestamp=false",
            "trainer.quick=true",
        ])
        cfg = get_config(overrides)
        disent_vs_rep(cfg)

def test_rl_vs_rep():
    overrides = [
        "experiment=pytest",
        "env=taxi",
        "timestamp=false",
        "transform=images",
        "agent=dqn",
        "model=cnn_64",
        "trainer=rl.quick",
    ]
    cfg = get_config(overrides)
    rl_vs_rep(cfg)

def test_factorize():
    configurations = [
        ["env=taxi", "transform=images", "model=ae/betavae", "loss@losses.vae=betavae"],
        ["env=gridworld", "transform=permute_factors", "model=ae/ae_mlp", "loss@losses.sparsity=sparsity/unit_pnorm"],
        ["env=gridworld", "transform=permute_factors", "model=factored/ae_mlp", "losses.effects=0.003", "losses.reconst=1.0", "loss@losses.sparsity=sparsity/sum_div_max"],
    ] # yapf: disable
    for overrides in configurations:
        overrides.extend([
            "experiment=pytest",
            "timestamp=false",
            "trainer=rep.quick",
        ])
        cfg = get_config(overrides)
        factorize(cfg)

def test_save_and_load():
    common = ["experiment=pytest", "timestamp=false", "trainer=rep.quick"]
    train_and_save = ["env=taxi", "transform=images", "model=ae/ae_cnn_64"]
    load_and_check = [
        "env=taxi", "transform=images", "model=ae/ae_cnn_64", "loader.should_load=true",
        "loader.experiment=pytest"
    ]
    train_and_save.extend(common)
    load_and_check.extend(common)
    factorize(get_config(train_and_save))
    disent_vs_rep(get_config(load_and_check))
