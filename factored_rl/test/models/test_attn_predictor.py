import pytest

from omegaconf import OmegaConf
import torch

from factored_rl.models.nnutils import one_hot
from factored_rl.models.wm import AttnPredictor

@pytest.fixture
def cfg():
    return OmegaConf.create("""
    mlp:
      n_hidden_layers: 1
      n_units_per_layer: 64
      activation:
        - _target_: torch.nn.ReLU
      final_activation:
        - _target_: torch.nn.Tanh
    attn:
      key_embed_dim: 6
      factor_embed_dim: 4
      action_embed_dim: 2
      dropout: 0.0
    """)

def test_batch_shapes(cfg):
    batch_size = 5
    n_latent_dims = 3
    n_actions = 2
    z = torch.arange(batch_size * n_latent_dims).reshape(batch_size, n_latent_dims).float() # (N,d)
    a = one_hot(torch.arange(batch_size) % n_actions, n_actions) # (N, A)

    model = AttnPredictor(n_latent_dims, n_actions, cfg)
    effect, attn_weights = model(z, a)

    assert attn_weights.shape == (5, 3, 3)
    assert effect.shape == (5, 3)

def test_single_shapes(cfg):
    n_latent_dims = 3
    n_actions = 2
    z = torch.arange(n_latent_dims).reshape(n_latent_dims).float() # (d,)
    a = one_hot(torch.tensor(0), n_actions) # (A,)

    model = AttnPredictor(n_latent_dims, n_actions, cfg)
    effect, attn_weights = model(z, a)

    assert attn_weights.shape == (3, 3)
    assert effect.shape == (3, )
