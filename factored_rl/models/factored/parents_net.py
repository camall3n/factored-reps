import torch
from torch import nn
from torch.nn import functional

from factored_rl.models.nnutils import Network, one_hot
from factored_rl.models.mlp import MLP

class ParentsNet(Network):
    def __init__(self,
                 n_actions,
                 n_latent_dims,
                 n_units_per_layer,
                 n_hidden_layers,
                 include_parent_actions=False,
                 factored=False):
        super().__init__()
        self.n_actions = n_actions
        self.n_latent_dims = n_latent_dims
        self.include_parent_actions = include_parent_actions
        self.factored = factored

        n_inputs = n_latent_dims + n_actions
        n_outputs = n_latent_dims + (n_actions if include_parent_actions else 0)

        if not self.factored:
            self.model = MLP(n_inputs, n_outputs, n_hidden_layers, n_units_per_layer)
        else:
            self.models = nn.ModuleList([
                MLP(n_inputs, n_outputs, n_hidden_layers, n_units_per_layer)
                for _ in range(n_latent_dims)
            ])

    def forward(self, z, a):
        a_onehot = one_hot(a, depth=self.n_actions)
        context = torch.cat((z, a_onehot), -1)
        if not self.factored:
            parent_logits = self.model(context)
        else:
            parent_logits = torch.stack([model(context) for model in self.models], dim=0)
        soft_decisions = torch.sigmoid(parent_logits)

        # TODO: How should hard_decisions be computed?

        # 1. threshold?
        # with torch.no_grad():
        #     hard_decisions = (soft_decisions > 0.5).float()

        # 2. sampling?
        with torch.no_grad():
            hard_decisions = torch.bernoulli(soft_decisions).float()

        # 3. ???
        # ??????

        # Compute parents with straight-through gradients:
        # - activations act like hard_decisions
        # - gradients act like soft_decisions
        parents = hard_decisions + soft_decisions - soft_decisions.detach()

        return parents, soft_decisions
