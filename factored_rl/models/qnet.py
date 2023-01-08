import torch
from torch import nn

from factored_rl.models.nnutils import Module
from factored_rl.wrappers.basis import BasisFunction

class QNetModule(Module):
    def __init__(self, encoder: nn.Module, basis: BasisFunction, q_head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.basis = basis
        self.q_head = q_head

        if (self.basis is not None) and not self.encoder.frozen:
            raise RuntimeError('Cannot backprop through basis function')

    def forward(self, x):
        z: torch.Tensor = self.encoder(x)
        if self.basis is None:
            features = z
        else:
            features = torch.as_tensor(self.basis(z.detach().numpy())).to(z.device)
        return self.q_head(features)
