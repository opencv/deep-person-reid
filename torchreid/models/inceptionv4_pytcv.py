"""
    InceptionV4 for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.
"""

__all__ = ['InceptionV4', 'inceptionv4_pytcv']

import os

import torch
import torch.nn as nn
import torch.nn.init as init

from torchreid.losses import AngleSimpleLinear
from torchreid.ops import Dropout, EvalModeSetter, rsc
from .common import Concurrent, ModelInterface


class InceptConv(nn.Module):
    """
    InceptionV4 specific convolution block.
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
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(InceptConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            eps=1e-3,
            momentum=0.1)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


def incept_conv1x1(in_channels,
                   out_channels,
                   stride=1,
                   padding=0):
    """
    1x1 version of the InceptionV4 specific convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    return InceptConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding)


def incept_conv3x3(in_channels,
                   out_channels,
                   stride,
                   padding=1):
    """
    3x3 version of the InceptionV4 specific convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    """
    return InceptConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding)


class MaxPoolBranch(nn.Module):
    """
    InceptionV4 specific max pooling branch block.
    """
    def __init__(self, kernel_size=3, stride=2, padding=0):
        super(MaxPoolBranch, self).__init__()
        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

    def forward(self, x):
        x = self.pool(x)
        return x


class AvgPoolBranch(nn.Module):
    """
    InceptionV4 specific average pooling branch block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1
                 ):
        super(AvgPoolBranch, self).__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            count_include_pad=False)
        self.conv = incept_conv1x1(
            in_channels=in_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Conv1x1Branch(nn.Module):
    """
    InceptionV4 specific convolutional 1x1 branch block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(Conv1x1Branch, self).__init__()
        self.conv = incept_conv1x1(
            in_channels=in_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv3x3Branch(nn.Module):
    """
    InceptionV4 specific convolutional 3x3 branch block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 padding):
        super(Conv3x3Branch, self).__init__()
        self.conv = incept_conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvSeqBranch(nn.Module):
    """
    InceptionV4 specific convolutional sequence branch block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of tuple of int
        List of numbers of output channels.
    kernel_size_list : list of tuple of int or tuple of tuple/list of 2 int
        List of convolution window sizes.
    strides_list : list of tuple of int or tuple of tuple/list of 2 int
        List of strides of the convolution.
    padding_list : list of tuple of int or tuple of tuple/list of 2 int
        List of padding values for convolution layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list):
        super(ConvSeqBranch, self).__init__()
        assert (len(out_channels_list) == len(kernel_size_list))
        assert (len(out_channels_list) == len(strides_list))
        assert (len(out_channels_list) == len(padding_list))

        self.conv_list = nn.Sequential()
        for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                out_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.add_module("conv{}".format(i + 1), InceptConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding))
            in_channels = out_channels

    def forward(self, x):
        x = self.conv_list(x)
        return x


class ConvSeq3x3Branch(nn.Module):
    """
    InceptionV4 specific convolutional sequence branch block with splitting by 3x3.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels_list : list of tuple of int
        List of numbers of output channels for middle layers.
    kernel_size_list : list of tuple of int or tuple of tuple/list of 2 int
        List of convolution window sizes.
    strides_list : list of tuple of int or tuple of tuple/list of 2 int
        List of strides of the convolution.
    padding_list : list of tuple of int or tuple of tuple/list of 2 int
        List of padding values for convolution layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list):
        super(ConvSeq3x3Branch, self).__init__()
        self.conv_list = nn.Sequential()
        for i, (mid_channels, kernel_size, strides, padding) in enumerate(zip(
                mid_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.add_module("conv{}".format(i + 1), InceptConv(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding))
            in_channels = mid_channels
        self.conv1x3 = InceptConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1))
        self.conv3x1 = InceptConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0))

    def forward(self, x):
        x = self.conv_list(x)
        y1 = self.conv1x3(x)
        y2 = self.conv3x1(x)
        x = torch.cat((y1, y2), dim=1)
        return x


class InceptionAUnit(nn.Module):
    """
    InceptionV4 type Inception-A unit.
    """
    def __init__(self):
        super(InceptionAUnit, self).__init__()
        in_channels = 384

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=96))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(64, 96),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 1)))
        self.branches.add_module("branch3", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(64, 96, 96),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 1),
            padding_list=(0, 1, 1)))
        self.branches.add_module("branch4", AvgPoolBranch(
            in_channels=in_channels,
            out_channels=96))

    def forward(self, x):
        x = self.branches(x)
        return x


class ReductionAUnit(nn.Module):
    """
    InceptionV4 type Reduction-A unit.
    """
    def __init__(self):
        super(ReductionAUnit, self).__init__()
        in_channels = 384

        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(384,),
            kernel_size_list=(3,),
            strides_list=(2,),
            padding_list=(0,)))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 224, 256),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 2),
            padding_list=(0, 1, 0)))
        self.branches.add_module("branch3", MaxPoolBranch())

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptionBUnit(nn.Module):
    """
    InceptionV4 type Inception-B unit.
    """
    def __init__(self):
        super(InceptionBUnit, self).__init__()
        in_channels = 1024

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=384))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 224, 256),
            kernel_size_list=(1, (1, 7), (7, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 3), (3, 0))))
        self.branches.add_module("branch3", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 192, 224, 224, 256),
            kernel_size_list=(1, (7, 1), (1, 7), (7, 1), (1, 7)),
            strides_list=(1, 1, 1, 1, 1),
            padding_list=(0, (3, 0), (0, 3), (3, 0), (0, 3))))
        self.branches.add_module("branch4", AvgPoolBranch(
            in_channels=in_channels,
            out_channels=128))

    def forward(self, x):
        x = self.branches(x)
        return x


class ReductionBUnit(nn.Module):
    """
    InceptionV4 type Reduction-B unit.
    """
    def __init__(self):
        super(ReductionBUnit, self).__init__()
        in_channels = 1024

        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(192, 192),
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0)))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(256, 256, 320, 320),
            kernel_size_list=(1, (1, 7), (7, 1), 3),
            strides_list=(1, 1, 1, 2),
            padding_list=(0, (0, 3), (3, 0), 0)))
        self.branches.add_module("branch3", MaxPoolBranch())

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptionCUnit(nn.Module):
    """
    InceptionV4 type Inception-C unit.
    """
    def __init__(self):
        super(InceptionCUnit, self).__init__()
        in_channels = 1536

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=256))
        self.branches.add_module("branch2", ConvSeq3x3Branch(
            in_channels=in_channels,
            out_channels=256,
            mid_channels_list=(384,),
            kernel_size_list=(1,),
            strides_list=(1,),
            padding_list=(0,)))
        self.branches.add_module("branch3", ConvSeq3x3Branch(
            in_channels=in_channels,
            out_channels=256,
            mid_channels_list=(384, 448, 512),
            kernel_size_list=(1, (3, 1), (1, 3)),
            strides_list=(1, 1, 1),
            padding_list=(0, (1, 0), (0, 1))))
        self.branches.add_module("branch4", AvgPoolBranch(
            in_channels=in_channels,
            out_channels=256))

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptBlock3a(nn.Module):
    """
    InceptionV4 type Mixed-3a block.
    """
    def __init__(self, stride=2, padding=0):
        super(InceptBlock3a, self).__init__()
        self.branches = Concurrent()
        self.branches.add_module("branch1", MaxPoolBranch(stride=stride, padding=padding))
        self.branches.add_module("branch2", Conv3x3Branch(
            in_channels=64,
            out_channels=96,
            stride=stride,
            padding=padding))

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptBlock4a(nn.Module):
    """
    InceptionV4 type Mixed-4a block.
    """
    def __init__(self, padding_list_branch1=(0,0), padding_list_branch2=(0, (0, 3), (3, 0), 0)):
        super(InceptBlock4a, self).__init__()
        self.branches = Concurrent()
        self.branches.add_module("branch1", ConvSeqBranch(
            in_channels=160,
            out_channels_list=(64, 96),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=padding_list_branch1))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=160,
            out_channels_list=(64, 64, 64, 96),
            kernel_size_list=(1, (1, 7), (7, 1), 3),
            strides_list=(1, 1, 1, 1),
            padding_list=padding_list_branch2))

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptBlock5a(nn.Module):
    """
    InceptionV4 type Mixed-5a block.
    """
    def __init__(self, stride=2, padding=0):
        super(InceptBlock5a, self).__init__()
        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv3x3Branch(
            in_channels=192,
            out_channels=192,
            stride=stride,
            padding=padding))
        self.branches.add_module("branch2", MaxPoolBranch(stride=stride, padding=padding))

    def forward(self, x):
        x = self.branches(x)
        return x


class InceptInitBlock(nn.Module):
    """
    InceptionV4 specific initial block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels, IN_conv1, in_size):
        super(InceptInitBlock, self).__init__()
        self.conv1 = InceptConv(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1 if in_size < 128 else 2,
            padding=0)
        self.conv2 = InceptConv(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1 if in_size < 128 else 0)
        self.conv3 = InceptConv(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1)

        self.block1 = InceptBlock3a(stride=1 if in_size < 128 else 2,
                                    padding=1 if in_size < 128 else 0)

        self.block2 = InceptBlock4a(padding_list_branch1=(1,1) if in_size < 128 else (0,0),
                                    padding_list_branch2=(1, (0, 3), (3, 0), 1) if in_size < 128 else (0, (0, 3), (3, 0), 0))
        self.block3 = InceptBlock5a(stride=1 if in_size < 128 else 2,
                                    padding=1 if in_size < 128 else 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class InceptionV4(ModelInterface):
    """
    InceptionV4 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.
    Parameters:
    ----------
    dropout_cfg : float, default 0.0
        Fraction of the input units in backbone to drop. Must be a number between 0 and 1.
    dropout_cls : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (299, 299)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000,
                 dropout_cls = None,
                 dropout_cfg = None,
                 pooling_type='avg',
                 bn_eval=False,
                 bn_frozen=False,
                 feature_dim=1536,
                 loss='softmax',
                 IN_first=False,
                 IN_conv1=False,
                 self_challenging_cfg=False,
                 lr_finder = None,
                 **kwargs):

        super().__init__(**kwargs)
        self.in_size = in_size
        self.num_classes = num_classes
        self.input_IN = nn.InstanceNorm2d(3, affine=True) if IN_first else None
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.pooling_type = pooling_type
        self.loss = loss
        self.feature_dim = feature_dim
        self.self_challenging_cfg = self_challenging_cfg
        self.lr_finder = lr_finder

        layers = [4, 8, 4]
        normal_units = [InceptionAUnit, InceptionBUnit, InceptionCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        self.features = nn.Sequential()
        self.features.add_module("init_block", InceptInitBlock(
            in_channels=in_channels, IN_conv1=IN_conv1, in_size=in_size[0]))

        for i, layers_per_stage in enumerate(layers):
            stage = nn.Sequential()
            for j in range(layers_per_stage):
                if (j == 0) and (i != 0):
                    unit = reduction_units[i - 1]
                else:
                    unit = normal_units[i]
                stage.add_module("unit{}".format(j + 1), unit())
            self.features.add_module("stage{}".format(i + 1), stage)

        self.output = nn.Sequential()
        if dropout_cls:
            self.output.add_module("dropout", Dropout(**dropout_cls))
        if self.loss in ['softmax', 'asl']:
            self.output.add_module("fc", nn.Linear(
                in_features=self.feature_dim,
                out_features=num_classes))
        else:
            assert self.loss in ['am_softmax', 'am_asl']
            self.output.add_module("asl", AngleSimpleLinear(
                in_features=self.feature_dim,
                out_features=num_classes))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x, return_featuremaps=False, get_embeddings=False, gt_labels=None):
        if self.input_IN is not None:
            x = self.input_IN(x)

        y = self.features(x)
        if return_featuremaps:
            return y

        glob_features = self._glob_feature_vector(y, self.pooling_type, reduce_dims=False)

        logits = self.output(glob_features.view(x.shape[0], -1))

        if self.training and self.self_challenging_cfg.enable and gt_labels is not None:
            glob_features = rsc(
                features = glob_features,
                scores = logits,
                labels = gt_labels,
                retain_p = 1.0 - self.self_challenging_cfg.drop_p,
                retain_batch = 1.0 - self.self_challenging_cfg.drop_batch_p
            )

            with EvalModeSetter([self.output], m_type=(nn.BatchNorm1d, nn.BatchNorm2d)):
                logits = self.output(glob_features.view(x.shape[0], -1))

        if not self.training and self.is_classification():
            return [logits]

        if get_embeddings:
            out_data = [logits, glob_features]
        elif self.loss in ['softmax', 'am_softmax', 'asl', 'am-asl']:
            if self.lr_finder.enable and self.lr_finder.lr_find_mode == 'automatic':
                out_data = logits
            else:
                out_data = [logits]
        elif self.loss in ['triplet']:
            out_data = [logits, glob_features]
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

        if self.lr_finder.enable and self.lr_finder.lr_find_mode == 'automatic':
            return out_data
        return tuple(out_data)


def get_inceptionv4(model_name=None,
                    pretrained=False,
                    root=os.path.join("~", ".torch", "models"),
                    **kwargs):
    """
    Create InceptionV4 model with specific parameters.
    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    net = InceptionV4(**kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def inceptionv4_pytcv(**kwargs):
    """
    InceptionV4 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_inceptionv4(model_name="inceptionv4", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        inceptionv4_pytcv,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != InceptionV4 or weight_count == 42679816)

        x = torch.randn(1, 3, 299, 299)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
