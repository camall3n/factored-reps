from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn
from torch.nn.functional import assert_int_or_pair

from .nnutils import Network

IntOrPairType = Union[int, Tuple[int]]
PairListType = Union[IntOrPairType, List[IntOrPairType]]

class CNN(Network):
    def __init__(
            self,
            input_shape: Tuple[int],
            n_output_channels: List[int],
            kernel_sizes: PairListType,
            strides: Optional[PairListType] = None, # default=1
            padding: Optional[Union[str, PairListType]] = None, # default=0
            dilations: Optional[PairListType] = None, # default=1
            activation: Optional[Union[torch.nn.Module, List[torch.nn.Module]]] = torch.nn.ReLU,
            final_activation: Optional[torch.nn.Module] = torch.nn.ReLU,
            **kwargs):
        super().__init__()
        self.input_shape = input_shape

        if len(input_shape) == 2:
            self.n_input_channels = 0
            self.output_shape = (1, ) + input_shape # will be adjusted as network gets built
        elif len(input_shape) == 3:
            self.n_input_channels = input_shape[0]
            self.output_shape = input_shape # will be adjusted as network gets built
            if input_shape[0] == input_shape[1] and input_shape[1] > input_shape[2]:
                warnings.warn('Input shape {input_shape} might be backwards. Should be (C, H, W).')
        else:
            raise ValueError('input_shape must be either 2D or 3D')

        self.n_layers = len(kernel_sizes)

        assert len(n_output_channels) == self.n_layers
        n_channels = (self.n_input_channels, ) if self.n_input_channels > 0 else (1, )
        n_channels = n_channels + tuple(n_output_channels)

        strides = self._list_of_pairs(strides, 'strides', default_value=1)
        dilations = self._list_of_pairs(dilations, 'dilations', default_value=1)
        if isinstance(padding, str):
            padding = [padding] * self.n_layers
        else:
            padding = self._list_of_pairs(padding, 'padding', default_value=0)

        if activation is None or isinstance(activation, (torch.nn.Module, type)):
            activations = [activation] * (self.n_layers - 1) + [final_activation]
        else:
            activations = activation + [final_activation]

        self.layers = []
        self.layer_shapes = [(1, ) + input_shape if self.n_input_channels == 0 else input_shape]
        for i in range(self.n_layers):
            conv = torch.nn.Conv2d(
                in_channels=n_channels[i],
                out_channels=n_channels[i + 1],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=padding[i],
                dilation=dilations[i],
                **kwargs,
            )
            self.layers.append(conv)
            out_shape = self._conv2d_size(input_shape=self.layer_shapes[-1], layer=conv)
            self.layer_shapes.append(out_shape)
            if activations[i] is not None:
                try:
                    self.layers.append(activations[i]())
                except TypeError:
                    self.layers.append(activations[i])

        self.output_shape = self.layer_shapes[-1]
        if not all(np.array(self.output_shape) >= 1):
            raise ValueError(
                f'The specified CNN has invalid output shape: {self.output_shape}\n'
                f'\n'
                f'Layer sizes:\n' +
                f'\n'.join([f'  {i:2d}. {shape}' for i, shape in enumerate(self.layer_shapes)]))

        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        if self.n_input_channels == 0:
            x = torch.unsqueeze(x, -3)
        return self.model(x)

    def _list_of_pairs(self, pair, argname: str, default_value):
        """
        Check if pair is a valid PairListType and output a list of pairs with length self.n_layers
        """
        if pair is None:
            pair_list = [default_value] * self.n_layers
        elif len(pair) != self.n_layers:
            assert_int_or_pair(pair, argname, message='Invalid {} value: ' + str(pair))
            pair_list = [pair] * self.n_layers
        elif self.n_layers == 2 and pair[0] != pair[1]:
            alternate_pair = (pair[0], pair[0]), (pair[1], pair[1])
            warnings.warn(
                f'The {argname} value of {pair} is ambiguous for a CNN with 2 layers.\n'
                f'Could mean either:\n'
                f'  1. Both layers use {argname} value of {pair}'
                f'  2. The different layers use {argname} {pair[0]} and {pair[1]} respectively'
                f'Assuming option 1. To achieve option 2, use {argname}={alternate_pair}')
            pair_list = [pair] * self.n_layers
        else:
            pair_list = list(pair)
        assert len(pair_list) == self.n_layers
        return pair_list

    def _conv2d_size(self, input_shape: Tuple[int], layer: torch.nn.Conv2d):
        """
        Compute the output shape after applying the Conv2d layer to the input shape
        """
        _, h_in, w_in = input_shape
        k = layer.kernel_size
        s = layer.stride
        d = layer.dilation
        p = layer.padding
        c_out = layer.out_channels
        h_out = self._conv1d_size(h_in, k[0], s[0], d[0], p[0])
        w_out = self._conv1d_size(w_in, k[1], s[1], d[1], p[1])
        return (c_out, h_out, w_out)

    @staticmethod
    def _conv1d_size(v_in: int,
                     kernel_size: int,
                     stride: int = 1,
                     dilation: int = 1,
                     padding: int = 0):
        """
        Compute the linear size associated with a 1D convolution operation

        Code shared by David Tao
        """
        numerator = v_in + 2 * padding - dilation * (kernel_size - 1) - 1
        float_out = (numerator / stride) + 1
        return int(np.floor(float_out))
