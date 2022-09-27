import hydra

from factored_rl.experiments.disent_vs_rep.run import main as disent_vs_rep
from factored_rl.experiments.rl_vs_rep.run import main as rl_vs_rep
from factored_rl.experiments.factorize.run import main as factorize

def get_config(overrides):
    with hydra.initialize(version_base=None, config_path='../../experiments/conf'):
        cfg = hydra.compose(config_name='config', overrides=overrides)
    return cfg

def test_disent_vs_rep():
    configurations = [["env=gridworld", "transform=rotate"],
                      ["env=taxi", "transform=images", "model=ae/ae_cnn_64"]]
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
        ["env=taxi", "transform=images", "model=ae/ae_cnn_64"],
        ["env=taxi", "transform=images", "model=ae/betavae", "loss@losses.vae=betavae"],
        ["env=taxi", "transform=images", "model=ae/ae_cnn_64", "loss@losses.sparsity=sparsity/sum_div_max"],
        ["env=gridworld", "transform=permute_factors", "model=ae/ae_mlp", "loss@losses.sparsity=sparsity/unit_pnorm"],
    ] # yapf: disable
    for overrides in configurations:
        overrides.extend([
            "experiment=pytest",
            "timestamp=false",
            "trainer=rep.quick",
        ])
        cfg = get_config(overrides)
        factorize(cfg)
