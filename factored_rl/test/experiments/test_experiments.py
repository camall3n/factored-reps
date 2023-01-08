import pytest

from ..utils import get_config, cleanup
from factored_rl.experiments.disent_vs_rep.run import main as disent_vs_rep
from factored_rl.experiments.rl_vs_rep.run import main as rl_vs_rep
from factored_rl.experiments.factorize.run import main as factorize
from factored_rl.experiments.factorize.tune import main as tune_factorize

@pytest.fixture(scope="module")
def setup():
    yield None
    cleanup()

@pytest.mark.parametrize("model,scale", [
    ('ae/ae_cnn_64', 2),
    ('factored/ae_cnn_64', 3),
    ('factored/wm_cnn_64_attn', 4),
])
def test_param_scaling(setup, model, scale):
    cfg = get_config([
        "experiment=pytest",
        "timestamp=false",
        "env=taxi",
        "transform=images",
        "agent=dqn",
        f"model={model}",
        f"model.param_scaling={scale}",
        "trainer=rep.quick",
        "tuner.tune_rep=true",
        "tuner.tune_rl=false",
        "tuner.should_prune=true",
        "tuner.tune_metric=reconst",
    ])
    tune_factorize(cfg)

@pytest.mark.parametrize("tune_rl,prune,metric", [
    ('false', 'true', 'reconst'),
    ('true', 'false', 'rl'),
])
def test_tune_factorize(setup, tune_rl: bool, prune: bool, metric: str):
    cfg = get_config([
        "experiment=pytest",
        "timestamp=false",
        "env=taxi",
        "transform=images",
        "model=factored/wm_cnn_64_attn",
        "agent=dqn",
        "trainer=rep.quick",
        "tuner.tune_rep=true",
        f"tuner.tune_rl={tune_rl}",
        f"tuner.should_prune={prune}",
        f"tuner.tune_metric={metric}",
    ])
    tune_factorize(cfg)

@pytest.mark.parametrize(
    "tune_rep,tune_rl,prune,metric",
    [
        ('false', 'false', 'false', 'rl'), # nothing to tune
        ('true', 'false', 'false', 'actions'), # unknown metric
        ('false', 'true', 'true', 'rep'), # pruning requires tune_rep
        ('true', 'true', 'true', 'rl'), # pruning incompatible with tuning via RL metric
    ])
def test_tune_factorize_errors(setup, tune_rep, tune_rl, prune, metric):
    cfg = get_config([
        "experiment=pytest",
        "timestamp=false",
        "env=taxi",
        "transform=images",
        "model=factored/wm_cnn_64_attn",
        "agent=dqn",
        "trainer=rep.quick",
        f"tuner.tune_rep={tune_rep}",
        f"tuner.tune_rl={tune_rl}",
        f"tuner.should_prune={prune}",
        f"tuner.tune_metric={metric}",
    ])
    with pytest.raises(RuntimeError):
        tune_factorize(cfg)

@pytest.mark.parametrize("env,transform,model_override", [
    ('gridworld', 'rotate', []),
    ('taxi', 'images', ['model=ae/ae_cnn_64']),
])
def test_disent_vs_rep(setup, env, transform, model_override):
    cfg = get_config([
        f"env={env}",
        f"transform={transform}",
        "experiment=pytest",
        "timestamp=false",
        "trainer.quick=true",
    ] + model_override)
    disent_vs_rep(cfg)

@pytest.mark.parametrize('transform,model', [
    ('images', 'cnn_64'),
    ('identity', 'qnet'),
])
def test_rl_vs_rep(setup, transform, model):
    cfg = get_config([
        "experiment=pytest",
        "timestamp=false",
        "env=taxi",
        f"transform={transform}",
        f"model={model}",
        "agent=dqn",
        "trainer=rl.quick",
    ])
    rl_vs_rep(cfg)

def test_factorize_betavae(setup):
    cfg = get_config([
        "experiment=pytest",
        "timestamp=false",
        "env=taxi",
        "transform=images",
        "model=ae/betavae",
        "loss=betavae",
        "trainer=rep.quick",
    ])
    factorize(cfg)

@pytest.mark.parametrize("env,transform,model", [
    ('taxi', 'images', 'ae/ae_cnn_64'),
    ('gridworld', 'permute_factors', 'ae/ae_mlp'),
])
def test_factorize_ae(setup, env, transform, model):
    cfg = get_config([
        "experiment=pytest",
        "timestamp=false",
        f"env={env}",
        f"transform={transform}",
        f"model={model}",
        "trainer=rep.quick",
    ])
    factorize(cfg)

def test_factorize_ae_losses(setup):
    cfg = get_config([
        "experiment=pytest",
        "timestamp=false",
        "env=gridworld",
        "transform=permute_factors",
        "model=factored/ae_mlp",
        "loss.actions=0.003",
        "loss.effects=0.003",
        "loss.reconst=1.0",
        "trainer=rep.quick",
    ])
    factorize(cfg)

def test_factorize_wm_losses(setup):
    cfg = get_config([
        "experiment=pytest",
        "timestamp=false",
        "env=taxi",
        "transform=images",
        "model=factored/wm_cnn_64_attn",
        "loss.actions=0.003",
        "loss.effects=0.003",
        "loss.reconst=1.0",
        "loss.parents=1.0",
        "loss/sparsity=unit_pnorm",
        "trainer=rep.quick",
    ])
    factorize(cfg)

def test_save_and_load_ae(setup):
    common = [
        "experiment=pytest", "timestamp=false", "env=taxi", "transform=images",
        "model=ae/ae_cnn_64", "trainer=rep.quick"
    ]
    train_and_save = []
    load_and_check = ["loader.load_model=true", "loader.experiment=pytest"]
    train_and_save.extend(common)
    load_and_check.extend(common)
    factorize(get_config(train_and_save))
    disent_vs_rep(get_config(load_and_check))
    rl_vs_rep(get_config(load_and_check + ["agent=dqn", "trainer=rl.quick"]))

def test_save_and_load_wm(setup):
    common = ["experiment=pytest", "timestamp=false", "trainer=rep.quick"]
    train_and_save = [
        "env=taxi", "transform=images", "model=factored/wm_cnn_64_attn", "loss.actions=0.003",
        "loss.effects=0.003", "loss.reconst=1.0", "loss.parents=1.0", "loss/sparsity=sum_div_max"
    ]
    load_and_check = [
        "env=taxi", "transform=images", "model=factored/wm_cnn_64_attn", "loader.load_model=true",
        "loader.experiment=pytest"
    ]
    train_and_save.extend(common)
    load_and_check.extend(common)
    factorize(get_config(train_and_save))
    disent_vs_rep(get_config(load_and_check))
    rl_vs_rep(get_config(load_and_check + ["trainer=rl.quick", "agent=dqn"]))
