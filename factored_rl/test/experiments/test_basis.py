import pytest

from ..utils import get_config, cleanup
from factored_rl.experiments.rl_vs_rep.run import main as rl_vs_rep
from factored_rl.experiments.factorize.run import main as factorize

def test_no_basis():
    overrides = [
        "transform=identity",
        "experiment=pytest",
        "env=taxi",
        "timestamp=false",
        "agent=dqn",
        "trainer=rl.quick",
        "model=qnet",
    ]
    cfg = get_config(overrides)
    rl_vs_rep(cfg)

@pytest.mark.parametrize("basis,rank", [('polynomial', 3), ('legendre', 3), ('fourier', 2)])
def test_transform_basis(basis, rank):
    overrides = [
        "experiment=pytest",
        "timestamp=false",
        "env=taxi",
        "transform=identity",
        f"transform.basis.name={basis}",
        f"transform.basis.rank={rank}",
        "model=qnet",
        "agent=dqn",
        "trainer=rl.quick",
    ]
    cfg = get_config(overrides)
    rl_vs_rep(cfg)

def test_model_basis():
    common = [
        "experiment=pytest",
        "timestamp=false",
        "env=taxi",
        "transform=images",
        "model=ae/ae_cnn_64",
        "model.qnet.n_hidden_layers=0",
        "model.qnet.basis.name=polynomial",
        "model.qnet.basis.rank=1",
    ]
    train_rep_and_save = ["trainer=rep.quick"]
    load_and_train_rl = [
        "loader.load_model=true",
        "loader.experiment=pytest",
        "loader.eval_only=true",
        "agent=dqn",
        "trainer=rl.quick",
    ]
    factorize(get_config(common + train_rep_and_save))
    rl_vs_rep(get_config(common + load_and_train_rl))

cleanup()
