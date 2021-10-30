from collections import Sequence

import torch
import torch.nn.functional as F

from markov_abstr.gridworld.models.nnutils import Network, one_hot
from markov_abstr.gridworld.models.simplenet import SimpleNet

class CategoricalPredictor(Network):
    def __init__(self, n_inputs, n_values, learning_rate=1e-3):
        """
        Output categorical distributions over multiple variables

        n_inputs: int
            Number of inputs (assumed to be flattened)
        n_values: int or sequence of ints
            Number of values per variable
        """
        super().__init__()

        self.n_inputs = n_inputs
        self.n_values = n_values
        try:
            self.n_vars = len(self.n_values)
        except TypeError:
            self.n_vars = 1
            self.n_values = list(n_values)

        self.predictors = torch.nn.ModuleList([
            SimpleNet(
                n_inputs=self.n_inputs,
                n_outputs=n_val,
                n_hidden_layers=2,
                n_units_per_layer=32,
                activation=torch.nn.ReLU,
                final_activation=torch.nn.Softmax(dim=-1),
            ) for n_val in self.n_values
        ])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def predict(self, x):
        soft_predictions = self(x)
        hard_predictions = [torch.argmax(p, dim=-1) for p in soft_predictions]
        return torch.stack(hard_predictions, dim=-1)

    def forward(self, x):
        soft_predictions = [model(x) for model in self.predictors]
        return soft_predictions

    def compute_loss(self, input, target):
        soft_predictions = self(input)
        losses = [
            F.nll_loss(input=pred, target=targ.squeeze(-1), weight=weights)
            for pred, targ, weights in zip(
                soft_predictions,
                torch.split(target, 1, dim=-1),
                [None] * 4 + [torch.tensor((0.5 / 0.75, 0.5 / 0.25)).float().to(target.device)],
            )
        ]
        return torch.stack(losses).mean()

    def process_batch(self, z, s, test=False):
        if not test:
            self.train()
            self.optimizer.zero_grad()
        else:
            self.eval()
        loss = self.compute_loss(z, s)
        if not test:
            loss.backward()
            self.optimizer.step()
        return loss
