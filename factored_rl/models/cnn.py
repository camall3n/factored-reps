from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn

from .nnutils import Module

class CNN(Module):
    def __init__(
            self,
            input_shape: Tuple[int],
            n_output_channels: List[int],
            kernel_sizes: Union[int, List[int]],
            strides: Optional[Union[int, List[int]]] = None, # default=1
            padding: Optional[Union[int, List[int]]] = None, # default=0
            dilations: Optional[Union[int, List[int]]] = None, # default=1
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

        self.n_layers = len(n_output_channels)
        n_channels = (self.n_input_channels, ) if self.n_input_channels > 0 else (1, )
        n_channels = n_channels + tuple(n_output_channels)

        kernel_sizes = self._list_of_values(kernel_sizes, 'kernel_sizes', default_value=None)
        strides = self._list_of_values(strides, 'strides', default_value=1)
        dilations = self._list_of_values(dilations, 'dilations', default_value=1)
        if isinstance(padding, str):
            padding = [padding] * self.n_layers
        else:
            padding = self._list_of_values(padding, 'padding', default_value=0)

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
                f'Layer shapes:\n' +
                f'\n'.join([f'  {i:2d}. {shape}' for i, shape in enumerate(self.layer_shapes)]))

        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        if self.n_input_channels == 0:
            x = torch.unsqueeze(x, -3)
        return self.model(x)

    def print_layers(self):
        print(f'Layer shapes:\n' +
              f'\n'.join([f'  {i:2d}. {shape}' for i, shape in enumerate(self.layer_shapes)]))

    def _list_of_values(self, value: Union[int, List[int]], argname: str, default_value: int):
        """
        Ensure value is a list of ints, repeating it if necessary, to a length self.n_layers
        """
        if value is None:
            value_list = [default_value] * self.n_layers
        elif isinstance(value, int):
            value_list = [value] * self.n_layers
        elif len(value) != self.n_layers:
            assert isinstance(value, int), f'Invalid {argname} value: {value}'
            value_list = [value] * self.n_layers
        else:
            value_list = list(value)
            for i, val in enumerate(value_list):
                assert isinstance(
                    val,
                    int), f'Expected {argname} list to be integers, but item {i} was not: {val}'
        assert len(value_list) == self.n_layers
        return value_list

    def _conv2d_size(self, input_shape: Tuple[int], layer: torch.nn.Conv2d):
        """
        Compute the output shape after applying the Conv2d layer to the input shape
        """
        _, h_in, w_in = input_shape
        k_h, k_w = self._unpack(layer.kernel_size)
        s_h, s_w = self._unpack(layer.stride)
        d_h, d_w = self._unpack(layer.dilation)
        p_h, p_w = self._unpack(layer.padding)
        c_out = layer.out_channels
        h_out = self._conv1d_size(h_in, k_h, s_h, d_h, p_h)
        w_out = self._conv1d_size(w_in, k_w, s_w, d_w, p_w)
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

    @staticmethod
    def _unpack(size):
        try:
            size_h, size_w = size
        except TypeError:
            size_h, size_w = size, size
        return size_h, size_w