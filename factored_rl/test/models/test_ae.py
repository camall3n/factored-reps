import pytest

import hydra
import torch

from factored_rl.models.ae import Autoencoder

@pytest.fixture
def model():
    with hydra.initialize(version_base=None, config_path='../../experiments/conf'):
        cfg = hydra.compose(config_name='config', overrides=['model=ae_64'])
    cfg = cfg.model
    input_shape = tuple((3, ) + cfg.cnn.supported_2d_input_shape)
    model = Autoencoder(input_shape, cfg)
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
