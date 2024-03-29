from functools import partial
import logging
import math
import os
import shutil
from typing import Union, List

import numpy as np
import torch
import torch.nn
from torch.nn.functional import dropout

Activation = Union[torch.nn.Module, type]
ActivationType = Union[Activation, List[Activation]]

def conv2d_size_out(size, kernel_size, stride):
    ''' Adapted from pytorch tutorials:
        https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    '''
    return ((size[-2] - (kernel_size[-2] - 1) - 1) // stride + 1,
            (size[-1] - (kernel_size[-1] - 1) - 1) // stride + 1)

def build_activation(ac, layer_shape):
    if isinstance(ac, partial):
        ac = ac(layer_shape)
    elif isinstance(ac, type):
        ac = ac()
    return ac

class Reshape(torch.nn.Module):
    """Module that returns a view of the input which has a different size

    Parameters
    ----------
    args : int...
        The desired size
    """
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def __repr__(self):
        s = self.__class__.__name__
        s += '{}'.format(self.shape)
        return s

    def forward(self, input):
        try:
            return input.view(*self.shape)
        except RuntimeError:
            return input.reshape(*self.shape)

class Module(torch.nn.Module):
    """Module that, when printed, shows its total number of parameters
    """
    def __init__(self):
        super().__init__()
        self.frozen = False

    def __str__(self):
        s = super().__str__() + '\n'
        n_params = 0
        for p in self.parameters():
            n_params += np.prod(p.size())
        s += 'Total params: {}'.format(n_params)
        return s

    def print_summary(self):
        s = str(self)
        print(s)

    def save(self, name, model_dir, is_best=False):
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, '{}_latest.ckpt'.format(name))
        torch.save(self.state_dict(), model_file)
        logging.info('Model saved to {}'.format(model_file))
        if is_best:
            best_file = os.path.join(model_dir, '{}_best.ckpt'.format(name))
            shutil.copyfile(model_file, best_file)
            logging.info('New best model! Model copied to {}'.format(best_file))

    def load(self, model_file, to: torch.device = None):
        logging.info('Loading model from {}...'.format(model_file))
        state_dict = torch.load(model_file, map_location=to)
        self.load_state_dict(state_dict)

    def freeze(self):
        if not self.frozen:
            for param in self.parameters():
                param.requires_grad = False
            self.frozen = True

    def unfreeze(self):
        if self.frozen:
            for param in self.parameters():
                param.requires_grad = True
            self.frozen = False

    def hard_copy_from(self, other: torch.nn.Module):
        self.load_state_dict(other.state_dict())

    def soft_copy_from(self, other: torch.nn.Module, alpha: float = 0.1):
        """
        Soft update current network towards weights of other network
        using an exponential moving average.

        θ_dest = alpha * θ_src + (1 - alpha) * θ_dest

        Args:
            local_model: weights will be copied from
            dest_model: weights will be copied to
            alpha: interpolation parameter, usually small (e.g. 0.0001)
        """
        for theta_dest, theta_src in zip(self.parameters(), other.parameters()):
            theta_dest.data.copy_(alpha * theta_src.data + (1.0 - alpha) * theta_dest.data)

class Sequential(torch.nn.Sequential, Module):
    pass

class Identity(Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.freeze()

    def forward(self, args):
        return args

def attention(query: torch.Tensor,
              key: torch.Tensor,
              value: torch.Tensor,
              attn_mask: torch.Tensor = None,
              dropout_p: float = 0.0):
    """
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

    Code adapted from:
        - "The Annotated Transformer" by Sasha Rush
        - torch.nn.functional._scaled_dot_product_attention
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, -1e9)
    attn = scores.softmax(dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    return torch.matmul(attn, value), attn

def one_hot(x: torch.Tensor, depth: int, dtype=torch.float32):
    """Convert a tensor of indices to a tensor of one-hot vectors, adding a dimension at the end

    Parameters
    ----------
    depth : int
        The length of each output vector
    """
    is_batch = (x.dim() > 0)
    if not is_batch:
        x = x.unsqueeze(0)
    i = x.unsqueeze(-1).expand(*x.shape, depth)
    result = torch.zeros_like(i, dtype=dtype).scatter_(-1, i, 1)
    if not is_batch:
        result = result.squeeze(0)
    return result

def extract(input, idx, idx_dim, batch_dim=0):
    '''
Extracts slices of input tensor along idx_dim at positions
specified by idx.

Notes:
    idx must have the same size as input.shape[batch_dim].
    Output tensor has the shape of input with idx_dim removed.

Args:
    input (Tensor): the source tensor
    idx (LongTensor): the indices of slices to extract
    idx_dim (int): the dimension along which to extract slices
    batch_dim (int): the dimension to treat as the batch dimension

Example::

    >>> t = torch.arange(24, dtype=torch.float32).view(3,4,2)
    >>> i = torch.tensor([1, 3, 0], dtype=torch.int64)
    >>> extract(t, i, idx_dim=1, batch_dim=0)
        tensor([[ 2.,  3.],
                [14., 15.],
                [16., 17.]])
'''
    if idx_dim == batch_dim:
        raise RuntimeError('idx_dim cannot be the same as batch_dim')
    if len(idx) != input.shape[batch_dim]:
        raise RuntimeError(
            "idx length '{}' not compatible with batch_dim '{}' for input shape '{}'".format(
                len(idx), batch_dim, list(input.shape)))
    viewshape = [
        1,
    ] * input.ndimension()
    viewshape[batch_dim] = input.shape[batch_dim]
    idx = idx.view(*viewshape).expand_as(input)
    result = torch.gather(input, idx_dim, idx).mean(dim=idx_dim)
    return result
