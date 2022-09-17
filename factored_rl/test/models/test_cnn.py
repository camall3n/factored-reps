import pytest

import torch

from factored_rl.models.cnn import CNN

def test_2d_input():
    cnn = CNN(input_shape=(48, 48),
              n_output_channels=[32, 64, 64],
              kernel_sizes=[11, 9, 6],
              strides=[2, 2, 1],
              activation=None,
              final_activation=None)
    assert len(cnn.layers) == 3
    assert cnn.layers[0].weight.shape == (32, 1, 11, 11)
    assert cnn.layers[1].weight.shape == (64, 32, 9, 9)
    assert cnn.layers[2].weight.shape == (64, 64, 6, 6)

@pytest.fixture()
def cnn48x48():
    cnn = CNN(input_shape=(3, 48, 48),
              n_output_channels=[32, 64, 64],
              kernel_sizes=[11, 9, 6],
              strides=[2, 2, 1],
              activation=None,
              final_activation=None)
    return cnn

def test_3d_input(cnn48x48):
    cnn = cnn48x48
    assert len(cnn.layers) == 3
    assert cnn.layers[0].weight.shape == (32, 3, 11, 11)
    assert cnn.layers[1].weight.shape == (64, 32, 9, 9)
    assert cnn.layers[2].weight.shape == (64, 64, 6, 6)

def test_warning_for_HWC():
    with pytest.warns():
        with pytest.raises(ValueError):
            CNN(input_shape=(48, 48, 3),
                n_output_channels=[32, 64, 64],
                kernel_sizes=[11, 9, 6],
                strides=[2, 2, 1])

def test_output_shape(cnn48x48):
    cnn = cnn48x48
    x = torch.zeros(cnn.input_shape)
    output = cnn(x)
    assert output.shape == (64, 1, 1)
    assert cnn.output_shape == output.shape

def test_bias_arg_passed_to_conv2d(cnn48x48):
    assert cnn48x48.layers[0].bias is not None
    cnn = CNN(input_shape=(3, 48, 48),
              n_output_channels=[32, 64, 64],
              kernel_sizes=[11, 9, 6],
              strides=[2, 2, 1],
              bias=False)
    assert cnn.layers[0].bias is None

def test_1d_input():
    with pytest.raises(ValueError):
        CNN(input_shape=(48, ), kernel_sizes=[11], n_output_channels=[32])

def test_incorrect_arg_lengths():
    with pytest.raises(AssertionError):
        CNN(input_shape=(3, 48, 48), kernel_sizes=[11, 9, 6], n_output_channels=[32, 64])
    with pytest.raises(AssertionError):
        CNN(input_shape=(3, 48, 48),
            n_output_channels=[32, 64, 64],
            kernel_sizes=[11, 9, 6],
            strides=[1, 1, 1, 1])
    with pytest.raises(AssertionError):
        CNN(input_shape=(3, 48, 48),
            n_output_channels=[32, 64, 64],
            kernel_sizes=[11, 9, 6],
            dilations=[0])

def test_not_list_of_ints():
    with pytest.raises(AssertionError):
        CNN(input_shape=(3, 48, 48),
            n_output_channels=[32, 64, 64],
            kernel_sizes=[[5, 5], [4, 4], [3, 3]],
            strides=[1, 2, 3])
    with pytest.raises(AssertionError):
        CNN(input_shape=(3, 48, 48),
            n_output_channels=[32, 64, 64],
            kernel_sizes=2,
            strides=[[5, 5], [4, 4], [3, 3]])

def test_known_output_shapes():
    nature_dqn_cnn = CNN(
        input_shape=(4, 84, 84),
        n_output_channels=[32, 64, 64],
        kernel_sizes=[8, 4, 3],
        strides=[4, 2, 1],
    ), (64, 7, 7)

    # Examples from Dumoulin & Visin, "A guide to convolution arithmetic
    # for deep learning", ArXiv [1603.07285], 2016
    #
    # https://github.com/vdumoulin/conv_arithmetic
    no_padding_no_stride = CNN(
        input_shape=(4, 4),
        n_output_channels=[1],
        kernel_sizes=[3],
    ), (1, 2, 2)
    padding_no_stride = CNN(
        input_shape=(5, 5),
        n_output_channels=[1],
        kernel_sizes=[4],
        padding=[2],
    ), (1, 6, 6)
    half_padding_no_stride = CNN(
        input_shape=(5, 5),
        n_output_channels=[1],
        kernel_sizes=[3],
        padding=[1],
    ), (1, 5, 5)
    full_padding_no_stride = CNN(
        input_shape=(5, 5),
        n_output_channels=[1],
        kernel_sizes=[3],
        padding=[2],
    ), (1, 7, 7)
    no_padding_strided = CNN(
        input_shape=(5, 5),
        n_output_channels=[1],
        kernel_sizes=[3],
        strides=[2],
    ), (1, 2, 2)
    padding_strided = CNN(
        input_shape=(5, 5),
        n_output_channels=[1],
        kernel_sizes=[3],
        strides=[2],
        padding=[1],
    ), (1, 3, 3)
    padding_strided_odd = CNN(
        input_shape=(6, 6),
        n_output_channels=[1],
        kernel_sizes=[3],
        strides=[2],
        padding=[1],
    ), (1, 3, 3)
    no_padding_no_stride_dilation = CNN(
        input_shape=(7, 7),
        n_output_channels=[1],
        kernel_sizes=[3],
        dilations=[2],
    ), (1, 3, 3)

    cnn_shape_pairs = [
        nature_dqn_cnn,
        no_padding_no_stride,
        padding_no_stride,
        half_padding_no_stride,
        full_padding_no_stride,
        no_padding_strided,
        padding_strided,
        padding_strided_odd,
        no_padding_no_stride_dilation,
    ]

    for cnn, output_shape in cnn_shape_pairs:
        assert cnn.output_shape == output_shape
        x = torch.zeros(cnn.input_shape)
        assert cnn(x).shape == output_shape
