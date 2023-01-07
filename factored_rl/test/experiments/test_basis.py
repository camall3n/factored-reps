import pytest

from ..utils import get_config, cleanup
from factored_rl.experiments.rl_vs_rep.run import main as rl_vs_rep

cleanup()

def test_no_basis():
    configurations = [
        ["transform=identity"],
    ]
    for overrides in configurations:
        overrides.extend([
            "experiment=pytest",
            "env=taxi",
            "timestamp=false",
            "agent=dqn",
            "trainer=rl.quick",
            "model=qnet",
        ])
    cfg = get_config(overrides)
    rl_vs_rep(cfg)

def test_basis_override():
    configurations = [
        ["transform=identity", "transform.basis.name=polynomial"],
        ["transform=identity", "transform.basis.name=legendre"],
        ["transform=identity", "transform.basis.name=fourier", "transform.basis.rank=1"],
        ["transform=identity", "model.basis.name=polynomial", "model.basis.rank=1"],
    ]
    for overrides in configurations:
        overrides.extend([
            "experiment=pytest",
            "env=taxi",
            "timestamp=false",
            "agent=dqn",
            "trainer=rl.quick",
            "model=qnet",
        ])
    cfg = get_config(overrides)
    rl_vs_rep(cfg)
