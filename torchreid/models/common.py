# Copyright (c) 2018-2021 Oleg Sémery
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from inspect import isfunction

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchreid.ops import Dropout

_VALID_MODEL_TYPES = ['classification', 'multilabel', 'multihead']

class ModelInterface(nn.Module):
    def __init__(self,
                 type,
                 feature_dim,
                 num_classes=1000,
                 pretrained=False,
                 loss='softmax',
                 mix_precision=False,
                 similarity_adjustment=False,
                 amb_t = 1.,
                 scale = 1.,
                 compute_scale=False,
                 **kwargs):
        super().__init__()

        assert type in _VALID_MODEL_TYPES
        self.type = type
        self.num_classes = num_classes
        if compute_scale:
            self.scale = self.compute_s(num_classes)
            print(f"LOG:: Computed margin scale for the dataset: {self.scale}\n")
        else:
            assert scale > 0.
            self.scale = scale
        self.pretrained = pretrained
        self.classification_classes = {}
        self.is_ie_model = False
        self.similarity_adjustment = similarity_adjustment
        self.amb_t = amb_t
        self.loss = loss
        self.num_head_features = feature_dim
        self.num_features = feature_dim
        self.mix_precision = mix_precision
        self.use_angle_simple_linear = True if loss in ['am_softmax', 'am_binary'] else False

    def get_num_features(self):
        return self.num_features

    def get_num_head_features(self):
        return self.num_head_features

    @staticmethod
    def sym_adjust(z, t):
        return 2 * torch.pow((z + 1)/2, t) - 1

    @staticmethod
    def compute_s(num_class: int):
        return float(min(max(np.sqrt(5) * np.log(num_class - 1), 7), 30.))

    @staticmethod
    def _glob_feature_vector(x, mode, reduce_dims=True):
        if mode == 'avg':
            out = F.adaptive_avg_pool2d(x, 1)
        elif mode == 'max':
            out = F.adaptive_max_pool2d(x, 1)
        elif mode == 'avg+max':
            avg_pool = F.adaptive_avg_pool2d(x, 1)
            max_pool = F.adaptive_max_pool2d(x, 1)
            out = avg_pool + max_pool
        else:
            raise ValueError(f'Unknown pooling mode: {mode}')

        if reduce_dims:
            return out.view(x.size(0), -1)
        return out

    @staticmethod
    def _construct_fc_layer(input_dim, output_dim, dropout=None):
        layers = []

        if dropout:
            layers.append(Dropout(**dropout))

        layers.extend([
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        ])

        return nn.Sequential(*layers)

    def get_config_optim(self, lrs):
        parameters = [
            {'params': self.named_parameters()},
        ]
        if isinstance(lrs, list):
            assert len(lrs) == len(parameters)
            parameters[0]['lr'] = lrs[0]
        else:
            assert isinstance(lrs, float)
            parameters[0]['lr'] = lrs

        return parameters


def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def depthwise_conv3x3(channels,
                      stride):
    """
    Depthwise convolution 3x3 layer.
    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    """
    return nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=channels,
        bias=False)

def get_activation_layer(activation):
    """
    Create activation layer from string/function.
    Parameters:
    ----------
    activation : function, or str, or nn.Module
        Activation function or name of activation function.
    Returns
    -------
    nn.Module
        Activation layer.
    """
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "relu6":
            return nn.ReLU6(inplace=True)
        elif activation == "prelu":
            return nn.PReLU(init=0.25)
        elif activation == "swish":
            return Swish()
        elif activation == "hswish":
            return HSwish(inplace=True)
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "hsigmoid":
            return HSigmoid()
        elif activation == "identity":
            return Identity()
        else:
            raise NotImplementedError()
    else:
        assert (isinstance(activation, nn.Module))
        return activation

class Concurrent(nn.Sequential):
    """
    A container for concatenation of modules on the base of the sequential container.
    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    stack : bool, default False
        Whether to concatenate tensors along a new dimension.
    """
    def __init__(self,
                 axis=1,
                 stack=False):
        super(Concurrent, self).__init__()
        self.axis = axis
        self.stack = stack

    def forward(self, x):
        out = []
        for module in self._modules.values():
            out.append(module(x))
        if self.stack:
            out = torch.stack(tuple(out), dim=self.axis)
        else:
            out = torch.cat(tuple(out), dim=self.axis)
        return out

class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and activation.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 IN_conv = False,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(ConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_pad = (isinstance(padding, (list, tuple)) and (len(padding) == 4))
        self.use_in = IN_conv

        if self.use_pad:
            self.pad = nn.ZeroPad2d(padding=padding)
            padding = 0
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.use_bn and not self.use_in:
            self.bn = nn.BatchNorm2d(
                num_features=out_channels,
                eps=bn_eps)
        elif self.use_in:
            self.ins = nn.InstanceNorm2d(
                num_features=out_channels,
                eps=bn_eps)
        if self.activate:
            self.activ = get_activation_layer(activation)

    def forward(self, x):
        if self.use_pad:
            x = self.pad(x)
        x = self.conv(x)
        if self.use_bn and not self.use_in:
            x = self.bn(x)
        elif self.use_in:
             x = self.ins(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=0,
                  groups=1,
                  bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=(lambda: nn.ReLU(inplace=True))):
    """
    1x1 version of the standard convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 0
        Padding value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)

def conv1x1(in_channels,
            out_channels,
            stride=1,
            groups=1,
            bias=False):
    """
    Convolution 1x1 layer.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias)

def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=(lambda: nn.ReLU(inplace=True)),
                  IN_conv = False):
    """
    3x3 version of the standard convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
        IN_conv = IN_conv)

def dwconv_block(in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True))):
    """
    Depthwise version of the standard convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


def dwconv3x3_block(in_channels,
                    out_channels,
                    stride=1,
                    padding=1,
                    dilation=1,
                    bias=False,
                    bn_eps=1e-5,
                    activation=(lambda: nn.ReLU(inplace=True))):
    """
    3x3 depthwise version of the standard convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        bn_eps=bn_eps,
        activation=activation)


def dwconv5x5_block(in_channels,
                    out_channels,
                    stride=1,
                    padding=2,
                    dilation=1,
                    bias=False,
                    bn_eps=1e-5,
                    activation=(lambda: nn.ReLU(inplace=True))):
    """
    5x5 depthwise version of the standard convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 2
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        bn_eps=bn_eps,
        activation=activation)


class DwsConvBlock(nn.Module):
    """
    Depthwise separable convolution block with BatchNorms and activations at each convolution layers.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    dw_use_bn : bool, default True
        Whether to use BatchNorm layer (depthwise convolution block).
    pw_use_bn : bool, default True
        Whether to use BatchNorm layer (pointwise convolution block).
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    dw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the depthwise convolution block.
    pw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the pointwise convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 bias=False,
                 dw_use_bn=True,
                 pw_use_bn=True,
                 bn_eps=1e-5,
                 dw_activation=(lambda: nn.ReLU(inplace=True)),
                 pw_activation=(lambda: nn.ReLU(inplace=True))):
        super(DwsConvBlock, self).__init__()
        self.dw_conv = dwconv_block(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            use_bn=dw_use_bn,
            bn_eps=bn_eps,
            activation=dw_activation)
        self.pw_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            use_bn=pw_use_bn,
            bn_eps=bn_eps,
            activation=pw_activation)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


def dwsconv3x3_block(in_channels,
                     out_channels,
                     stride=1,
                     padding=1,
                     dilation=1,
                     bias=False,
                     bn_eps=1e-5,
                     dw_activation=(lambda: nn.ReLU(inplace=True)),
                     pw_activation=(lambda: nn.ReLU(inplace=True)),
                     **kwargs):
    """
    3x3 depthwise separable version of the standard convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    dw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the depthwise convolution block.
    pw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the pointwise convolution block.
    """
    return DwsConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        bn_eps=bn_eps,
        dw_activation=dw_activation,
        pw_activation=pw_activation,
        **kwargs)

def round_channels(channels,
                   divisor=8):
    """
    Round weighted channel number (make divisible operation).
    Parameters:
    ----------
    channels : int or float
        Original number of channels.
    divisor : int, default 8
        Alignment value.
    Returns
    -------
    int
        Weighted number of channels.
    """
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels

class Identity(nn.Module):
    """
    Identity block.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Swish(nn.Module):
    """
    Swish activation function from 'Searching for Activation Functions,' https://arxiv.org/abs/1710.05941.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class HSigmoid(nn.Module):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


class HSwish(nn.Module):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    inplace : bool
        Whether to use inplace version of the module.
    """
    def __init__(self, inplace=False):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    mid_channels : int or None, default None
        Number of middle channels.
    round_mid : bool, default False
        Whether to round middle channel number (make divisible by 8).
    use_conv : bool, default True
        Whether to convolutional layers instead of fully-connected ones.
    activation : function, or str, or nn.Module, default 'relu'
        Activation function after the first convolution.
    out_activation : function, or str, or nn.Module, default 'sigmoid'
        Activation function after the last convolution.
    """
    def __init__(self,
                 channels,
                 reduction=16,
                 mid_channels=None,
                 round_mid=False,
                 use_conv=True,
                 mid_activation=(lambda: nn.ReLU(inplace=True)),
                 out_activation=(lambda: nn.Sigmoid())):
        super(SEBlock, self).__init__()
        self.use_conv = use_conv
        if mid_channels is None:
            mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        if use_conv:
            self.conv1 = conv1x1(
                in_channels=channels,
                out_channels=mid_channels,
                bias=True)
        else:
            self.fc1 = nn.Linear(
                in_features=channels,
                out_features=mid_channels)
        self.activ = get_activation_layer(mid_activation)
        if use_conv:
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=channels,
                bias=True)
        else:
            self.fc2 = nn.Linear(
                in_features=mid_channels,
                out_features=channels)
        self.sigmoid = get_activation_layer(out_activation)

    def forward(self, x):
        w = self.pool(x)
        if not self.use_conv:
            w = w.view(x.size(0), -1)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        if not self.use_conv:
            w = w.unsqueeze(2).unsqueeze(3)
        x = x * w
        return x
