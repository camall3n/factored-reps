from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn

from factored_rl import configs
from .nnutils import Module, ActivationType, build_activation

class CNN(Module):
    def __init__(
            self,
            input_shape: Tuple[int],
            n_output_channels: List[int],
            kernel_sizes: Union[int, List[int]],
            strides: Optional[Union[int, List[int]]] = None, # default=1
            padding: Optional[Union[int, List[int]]] = None, # default=0
            dilations: Optional[Union[int, List[int]]] = None, # default=1
            activation: Optional[ActivationType] = torch.nn.ReLU, # activation or list thereof for internal layers
            final_activation: Optional[ActivationType] = torch.nn.ReLU, # activation or list thereof for final layer
            transpose: bool = False,
            layer_shapes: Optional[List[Tuple[int]]] = None,
        **kwargs): # yapf: disable
        super().__init__()
        input_shape = tuple(input_shape)
        self.input_shape = input_shape
        self.transposed = transpose

        if len(input_shape) == 2:
            self.n_input_channels = 0
        elif len(input_shape) == 3:
            self.n_input_channels = input_shape[0]
            if input_shape[0] == input_shape[1] and input_shape[1] > input_shape[2]:
                warnings.warn('Input shape {input_shape} might be backwards. Should be (C, H, W).')
        else:
            raise ValueError('input_shape must be either 2D or 3D')

        self.n_layers = len(n_output_channels)
        n_channels = (self.n_input_channels, ) if self.n_input_channels > 0 else (1, )
        n_channels = n_channels + tuple(n_output_channels)
        self.n_output_channels = n_channels[-1]

        kernel_sizes = self._list_of_values(kernel_sizes, 'kernel_sizes', default_value=None)
        strides = self._list_of_values(strides, 'strides', default_value=1)
        dilations = self._list_of_values(dilations, 'dilations', default_value=1)
        if isinstance(padding, str):
            padding = [padding] * self.n_layers
        else:
            padding = self._list_of_values(padding, 'padding', default_value=0)

        # build list of lists of activations
        if not isinstance(activation, List):
            activation = [activation]
        if not isinstance(final_activation, List):
            final_activation = [final_activation]
        activations = [activation] * (self.n_layers - 1) + [final_activation]

        if transpose:
            Conv = torch.nn.ConvTranspose2d
            if self.n_output_channels == 0:
                n_channels[-1] = 1
        else:
            Conv = torch.nn.Conv2d

        self.layers = []
        conv_layers = []
        if layer_shapes is None:
            self.layer_shapes = [(1, ) + input_shape if self.n_input_channels == 0 else input_shape] # yapf:disable
        else:
            self.layer_shapes = layer_shapes
        for i in range(self.n_layers):
            conv = Conv(
                in_channels=n_channels[i],
                out_channels=n_channels[i + 1],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=padding[i],
                dilation=dilations[i],
                **kwargs,
            )
            self.layers.append(conv)
            conv_layers.append(conv)
            if transpose and layer_shapes is not None:
                out_shape = self.layer_shapes[i + 1]
            else:
                out_shape = self._conv2d_size(self.layer_shapes[i], conv, transpose)
                self.layer_shapes.append(out_shape)
            for ac in activations[i]:
                if ac is not None:
                    self.layers.append(build_activation(ac, out_shape))

        self.output_shape = self.layer_shapes[-1]
        assert self.output_shape[0] == self.n_output_channels
        if not all(np.array(self.output_shape) >= 1):
            raise ValueError(
                f'The specified CNN has invalid output shape: {self.output_shape}\n'
                f'\n'
                f'Layer shapes:\n' +
                f'\n'.join([f'  {i:2d}. {shape}' for i, shape in enumerate(self.layer_shapes)]))

        self.model = torch.nn.Sequential(*self.layers)

    @classmethod
    def from_config(cls, input_shape, cfg: configs.CNNConfig):
        assert len(input_shape) in [2, 3], 'CNN input_shape must be 2D or 3D'
        if input_shape[-2:] != cfg.cnn.supported_2d_input_shape:
            raise ValueError(f'Input shape {input_shape} does not match supported 2D input shape: '
                             f'{cfg.cnn.supported_2d_input_shape}')
        return cls(
            input_shape=input_shape,
            n_output_channels=cfg.cnn.n_output_channels,
            kernel_sizes=cfg.cnn.kernel_sizes,
            strides=cfg.cnn.strides,
            activation=configs.instantiate(cfg.cnn.activation),
            final_activation=configs.instantiate(cfg.cnn.final_activation),
        )

    def forward(self, x):
        if self.n_input_channels == 0:
            x = torch.unsqueeze(x, -3)
        desired_layer_shapes = iter([shape[1:] for shape in self.layer_shapes[1:]])
        for layer in self.model:
            if type(layer) == torch.nn.ConvTranspose2d:
                desired_shape = next(desired_layer_shapes)
                x = layer(x, output_size=desired_shape)
            else:
                x = layer(x)
        if self.n_output_channels == 0:
            x = torch.squeeze(x, -3)
        return x

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

    def _conv2d_size(self, input_shape: Tuple[int], layer: torch.nn.Conv2d, transpose=False):
        """
        Compute the output shape after applying the Conv2d layer to the input shape
        """
        if transpose:
            sizer = self._conv1d_size_T
        else:
            sizer = self._conv1d_size
        _, h_in, w_in = input_shape
        k_h, k_w = self._unpack(layer.kernel_size)
        s_h, s_w = self._unpack(layer.stride)
        d_h, d_w = self._unpack(layer.dilation)
        p_h, p_w = self._unpack(layer.padding)
        c_out = layer.out_channels
        h_out = sizer(h_in, k_h, s_h, d_h, p_h)
        w_out = sizer(w_in, k_w, s_w, d_w, p_w)
        return (c_out, h_out, w_out)

    @staticmethod
    def _conv1d_size(v_in: int, kernel: int, stride: int = 1, dilation: int = 1, pad: int = 0):
        """
        Compute the linear size associated with a 1D convolution operation

        Code shared by David Tao
        """
        numerator = v_in + 2 * pad - dilation * (kernel - 1) - 1
        float_out = (numerator / stride) + 1
        return int(np.floor(float_out))

    @staticmethod
    def _conv1d_size_T(v_out: int, kernel: int, stride: int = 1, dilation: int = 1, pad: int = 0):
        """
        Compute the linear size associated with a 1D transposed convolution operation
        """
        numerator = (v_out - 1) * stride
        v_in = numerator - (2 * pad - dilation * (kernel - 1) - 1)
        return v_in

    @staticmethod
    def _unpack(size):
        try:
            size_h, size_w = size
        except TypeError:
            size_h, size_w = size, size
        return size_h, size_w
