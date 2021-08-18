import torch
from torch import nn
from torch.nn import functional

from markov_abstr.gridworld.models.nnutils import Network, one_hot
from markov_abstr.gridworld.models.simplenet import SimpleNet

class ParentsNet(Network):
    def __init__(self,
                 n_actions,
                 n_latent_dims,
                 n_units_per_layer,
                 n_hidden_layers,
                 include_parent_actions=False):
        super().__init__()
        self.n_actions = n_actions
        self.n_latent_dims = n_latent_dims
        self.include_parent_actions = include_parent_actions
        self.model = SimpleNet(
            n_inputs=(n_latent_dims + n_actions),
            n_outputs=(n_latent_dims + (n_actions if include_parent_actions else 0)),
            n_units_per_layer=n_units_per_layer,
            n_hidden_layers=n_hidden_layers,
        )

    def forward(self, z, a):
        a_onehot = one_hot(a, depth=self.n_actions)
        context = torch.cat((z, a_onehot), -1)
        parent_logits = self.model(context)
        soft_decisions = torch.tanh(parent_logits)

        # TODO: How should hard_decisions be computed?

        # 1. saturate?
        with torch.no_grad():
            hard_decisions = torch.sign(soft_decisions)

        # 2. sampling?
        # ???

        # 3. ???
        # ??????

        # Compute parents with straight-through gradients:
        # - activations act like hard_decisions
        # - gradients act like soft_decisions
        parents = hard_decisions + soft_decisions - soft_decisions.detach()

        return parents, soft_decisions
