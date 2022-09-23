import numpy as np
import torch
import torch.nn

from ..nnutils import Module, one_hot, extract
from ..mlp import MLP

class FwdNet(Module):
    def __init__(self,
                 n_actions,
                 n_latent_dims=4,
                 n_hidden_layers=1,
                 n_units_per_layer=32,
                 factored=False):
        super().__init__()
        self.n_actions = n_actions
        self.factored = factored
        self.frozen = False

        n_inputs = n_latent_dims + self.n_actions

        if not self.factored:
            self.fwd_model = MLP(n_inputs, n_latent_dims, n_hidden_layers, n_units_per_layer)
        else:
            self.fwd_models = torch.nn.ModuleList([
                MLP(n_inputs, 1, n_hidden_layers, n_units_per_layer) for _ in range(n_latent_dims)
            ])

    def forward(self, z, a, parent_dependencies):
        a_onehot = one_hot(a, depth=self.n_actions)
        if not self.factored:
            z_masked = z * parent_dependencies
            context = torch.cat((z_masked, a_onehot), -1)
            z_hat = self.fwd_model(context)
        else:
            contexts = torch.stack(
                [torch.cat((z * mask, a_onehot), -1) for mask in parent_dependencies], dim=0)
            z_hat = torch.stack(
                [model(context).squeeze(-1) for model, context in zip(self.fwd_models, contexts)],
                dim=1)
        return z_hat
