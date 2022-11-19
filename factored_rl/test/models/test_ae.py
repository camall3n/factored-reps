import pytest

import hydra
from omegaconf import OmegaConf
import torch

from factored_rl.models.nnutils import one_hot
from factored_rl.models.ae import AutoencoderModel, PairedAutoencoderModel

@pytest.fixture
def model():
    with hydra.initialize(version_base=None, config_path='../../experiments/conf'):
        cfg = hydra.compose(config_name='config', overrides=['model=ae/ae_cnn_64'])
    cfg = cfg
    input_shape = tuple((3, ) + cfg.model.cnn.supported_2d_input_shape)
    model = AutoencoderModel(input_shape, cfg)
    return model

def test_transposed_flag(model):
    cnn = model.encoder.model[0]
    tcnn = model.decoder.model[-1]
    assert cnn.transposed == False
    assert tcnn.transposed == True

def test_shapes_single(model):
    x = torch.zeros(model.input_shape)
    z = model.encoder(x)
    assert z.shape == (model.n_latent_dims, )

    x_hat = model.decoder(z)
    assert x_hat.shape == x.shape

def test_shapes_batch(model):
    x = torch.zeros((10, ) + model.input_shape)
    z = model.encoder(x)
    assert z.shape == (10, model.n_latent_dims)

    x_hat = model.decoder(z)
    assert x_hat.shape == x.shape

def test_action_residuals():
    dummy_ae = OmegaConf.create("""
        n_actions: 5
        cfg:
          loss:
            actions: 1.0
        """)

    actions = torch.tensor([2, 3, 1, 2, 0, 0, 3])
    actions = one_hot(actions, 5)
    effects = torch.tensor([
        [ 0.4,  0.3],
        [ 0.2, -0.1],
        [-0.7,  0.6],
        [-0.2, -0.4],
        [-0.1,  0.1],
        [ 0.3,  0.5],
        [ 0.9, -0.4],
    ]) # yapf: disable

    action_residuals = PairedAutoencoderModel._get_action_residuals(dummy_ae, actions, effects)
    expected_result = torch.tensor([
        [ 0.3000,  0.3500],
        [-0.3500,  0.1500],
        [ 0.0000,  0.0000],
        [-0.3000, -0.3500],
        [-0.2000, -0.2000],
        [ 0.2000,  0.2000],
        [ 0.3500, -0.1500],
    ]) # yapf: disable
    assert torch.allclose(action_residuals, expected_result)
