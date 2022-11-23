from factored_rl.models.losses import unit_pnorm_loss, sum_div_max_loss
from factored_rl.models.nnutils import one_hot
import torch
import numpy as np

def test_unit_pnorm_loss_range():
    for p in [1.1, 2, 3, 4, 10]:
        for d in [2, 3, 10]:
            x = one_hot(torch.tensor(0), d)
            y = torch.ones(d)
            assert np.isclose(unit_pnorm_loss(x, p=p).item(), 1)
            assert np.isclose(unit_pnorm_loss(y, p=p).item(), d)

def test_sum_div_max_loss_range():
    for d in [2, 3, 10]:
        x = one_hot(torch.tensor(0), d)
        y = torch.ones(d)
        assert np.isclose(unit_pnorm_loss(x).item(), 1)
        assert np.isclose(unit_pnorm_loss(y).item(), d)
