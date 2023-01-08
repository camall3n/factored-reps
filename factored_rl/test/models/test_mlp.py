import pytest

import torch

from factored_rl.models.mlp import MLP

@pytest.mark.parametrize('n_hidden_layers', [10, 128])
def test_mlp_layers(n_hidden_layers):
    mlp = MLP(n_inputs=10,
              n_outputs=4,
              n_hidden_layers=n_hidden_layers,
              n_units_per_layer=32,
              activation=None,
              final_activation=None)
    assert mlp.n_layers == n_hidden_layers + 1

def test_mlp_linear():
    mlp = MLP(n_inputs=10,
              n_outputs=4,
              n_hidden_layers=0,
              n_units_per_layer=32,
              activation=None,
              final_activation=None)
    assert mlp.n_layers == 1
    assert len(mlp.layers) == 1
    assert isinstance(mlp.layers[0], torch.nn.Linear)
