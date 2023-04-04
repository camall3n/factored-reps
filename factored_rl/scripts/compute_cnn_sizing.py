from factored_rl.models import Network, CNN, MLP, nnutils
import numpy as np
import hydra

from factored_rl import configs

with hydra.initialize(version_base=None, config_path='factored_rl/experiments/conf'):
    cfg = hydra.compose(config_name='config', overrides=['model=cnn_11'])

cfg = cfg.model
model = Network((3, ) + cfg.cnn.supported_2d_input_shape, 10, cfg)
model.print_summary()


#%%
cnn = CNN(
    input_shape=(1, 64, 64),
    n_output_channels=[32, 32, 64, 64],
    kernel_sizes=4,
    strides=2,
    activation=None,
    final_activation=None,
)
n_features = np.prod(cnn.output_shape)
mlp = MLP(
    n_inputs=n_features,
    n_outputs=5,
    n_hidden_layers=1,
    n_units_per_layer=256,
)
model = nnutils.Sequential(*[cnn, nnutils.Reshape(-1, n_features), mlp])
model.print_summary()
model[0].print_layers()

#%%
mlp = MLP(
    n_inputs=3 * 64 * 64,
    n_outputs=5,
    n_hidden_layers=1,
    n_units_per_layer=64,
)
mlp.print_summary()
