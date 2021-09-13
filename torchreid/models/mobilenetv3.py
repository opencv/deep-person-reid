import math

import torch
import torch.nn as nn

from torchreid.losses import AngleSimpleLinear
from torchreid.ops import Dropout, EvalModeSetter, rsc
from .common import HSigmoid, HSwish, ModelInterface, make_divisible
import timm

from torchreid.integration.nncf.compression import get_no_nncf_trace_context_manager, nullcontext

__all__ = ['mobilenetv3_large', 'mobilenetv3_large_075', 'mobilenetv3_small', 'mobilenetv3_large_150',
            'mobilenetv3_large_125', "mobilenetv3_large_21k"]

pretrained_urls = {
    'mobilenetv3_small':
    'https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/mobilenetv3-small-55df8e1f.pth?raw=true',
    'mobilenetv3_large':
    'https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/mobilenetv3-large-1cd25616.pth?raw=true',
    'mobilenetv3_large_075':
    'https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/mobilenetv3-large-0.75-9632d2a8.pth?raw=true',
    'mobilenetv3_large_21k':
    'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/mobilenetv3_large_100_miil_21k.pth'
}


SHOULD_NNCF_SKIP_SE_LAYERS = False
SHOULD_NNCF_SKIP_HEAD = False
no_nncf_se_layer_context = get_no_nncf_trace_context_manager() if SHOULD_NNCF_SKIP_SE_LAYERS else nullcontext
no_nncf_head_context = get_no_nncf_trace_context_manager() if SHOULD_NNCF_SKIP_HEAD else nullcontext

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(make_divisible(channel // reduction, 8), channel),
                HSigmoid()
        )

    def forward(self, x):
        with no_nncf_se_layer_context():
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride, IN_conv1=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup) if not IN_conv1 else nn.InstanceNorm2d(oup, affine=True),
        HSwish()
    )


def conv_1x1_bn(inp, oup, loss='softmax'):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        HSwish() if loss == 'softmax' else nn.PReLU()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                HSwish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                HSwish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                HSwish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3Base(ModelInterface):
    def __init__(self,
                num_classes=1000,
                width_mult=1.,
                in_channels=3,
                input_size=(224, 224),
                dropout_cls = None,
                pooling_type='avg',
                IN_first=False,
                self_challenging_cfg=False,
                **kwargs):

        super().__init__(**kwargs)
        self.in_size = input_size
        self.num_classes = num_classes
        self.input_IN = nn.InstanceNorm2d(in_channels, affine=True) if IN_first else None
        self.pooling_type = pooling_type
        self.self_challenging_cfg = self_challenging_cfg
        self.width_mult = width_mult
        self.dropout_cls = dropout_cls

    def infer_head(self, x, skip_pool=False):
        raise NotImplementedError

    def extract_features(self, x):
        raise NotImplementedError

    def forward(self, x, return_featuremaps=False, get_embeddings=False, gt_labels=None):
        if self.input_IN is not None:
            x = self.input_IN(x)

        y = self.extract_features(x)
        if return_featuremaps:
            return y

        with no_nncf_head_context():
            glob_features, logits = self.infer_head(y, skip_pool=False)
        if self.training and self.self_challenging_cfg.enable and gt_labels is not None:
            glob_features = rsc(
                features = glob_features,
                scores = logits,
                labels = gt_labels,
                retain_p = 1.0 - self.self_challenging_cfg.drop_p,
                retain_batch = 1.0 - self.self_challenging_cfg.drop_batch_p
            )

            with EvalModeSetter([self.output], m_type=(nn.BatchNorm1d, nn.BatchNorm2d)):
                _, logits = self.infer_head(x, skip_pool=True)

        if not self.training and self.is_classification():
            return [logits]

        if get_embeddings:
            out_data = [logits, glob_features]
        elif self.loss in ['softmax', 'am_softmax', 'asl', 'am_binary']:
                out_data = [logits]
        elif self.loss in ['triplet']:
            out_data = [logits, glob_features]
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

        return tuple(out_data)


class MobileNetV3(MobileNetV3Base):
    def __init__(self,
                 cfgs,
                 mode,
                 IN_conv1=False,
                 **kwargs):

        super().__init__(**kwargs)
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']
        # building first layer
        input_channel = make_divisible(16 * self.width_mult, 8)
        stride = 1 if self.in_size[0] < 100 else 2
        layers = [conv_3x3_bn(3, input_channel, stride, IN_conv1)]
        # building inverted residual blocks
        block = InvertedResidual
        flag = True
        for k, t, c, use_se, use_hs, s in self.cfgs:
            if (self.in_size[0] < 100) and (s == 2) and flag:
                s = 1
                flag = False
            output_channel = make_divisible(c * self.width_mult, 8)
            exp_size = make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self.num_features = exp_size
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size, self.loss)
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = make_divisible(output_channel[mode] * self.width_mult, 8) if self.width_mult > 1.0 else output_channel[mode]

        if self.loss == 'softmax' or self.loss == 'asl':
            self.classifier = nn.Sequential(
                nn.Linear(exp_size, output_channel),
                nn.BatchNorm1d(output_channel),
                HSwish(),
                Dropout(**self.dropout_cls),
                nn.Linear(output_channel, self.num_classes),
            )
        else:
            assert self.loss in ['am_softmax', 'am_binary']
            self.classifier = nn.Sequential(
                nn.Linear(exp_size, output_channel),
                nn.BatchNorm1d(output_channel),
                nn.PReLU(),
                Dropout(**self.dropout_cls),
                AngleSimpleLinear(output_channel, self.num_classes),
            )

        self._initialize_weights()

    def extract_features(self, x):
        y = self.conv(self.features(x))
        return y

    def infer_head(self, x, skip_pool=False):
        if not skip_pool:
            glob_features = self._glob_feature_vector(x, self.pooling_type, reduce_dims=False)
        else:
            glob_features = x

        logits = self.classifier(glob_features.view(x.shape[0], -1))
        return glob_features, logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MobileNetV3LargeTimm(MobileNetV3Base):
    def __init__(self,
                pretrained=False,
                **kwargs):

        super().__init__(**kwargs)
        self.model = timm.create_model('mobilenetv3_large_100_miil_in21k',
                                        pretrained=pretrained,
                                        num_classes=self.num_classes)
        self.dropout = Dropout(**self.dropout_cls)
        self.num_features = self.model.conv_head.in_channels
        assert self.loss in ['softmax', 'asl'], "mobilenetv3_large_100_miil_in21k supports only softmax aor ASL losses"

    def extract_features(self, x):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        y = self.model.blocks(x)
        return y

    def infer_head(self, x, skip_pool=False):
        glob_features = self.model.global_pool(x) if not skip_pool else x
        x = self.model.conv_head(glob_features)
        x = self.model.act2(x)
        x = x.flatten(1)
        x = self.dropout(x)
        logits = self.model.classifier(x)
        return glob_features, logits


def init_pretrained_weights(model, key='', **kwargs):
    """Initializes model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown

    from torchreid.utils import load_pretrained_weights

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    filename = key + '_imagenet.pth'
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        gdown.download(pretrained_urls[key], cached_file)
    model = load_pretrained_weights(model, cached_file, **kwargs)

def mobilenetv3_large_21k(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV3-Large_timm model
    """
    net = MobileNetV3LargeTimm(pretrained=pretrained, **kwargs)
    return net

def mobilenetv3_large_075(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]

    net = MobileNetV3(cfgs, mode='large', width_mult =.75, **kwargs)
    if pretrained:
        init_pretrained_weights(net, key='mobilenetv3_large_075')

    return net

def mobilenetv3_large(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]

    net = MobileNetV3(cfgs, mode='large', width_mult = 1., **kwargs)
    if pretrained:
        init_pretrained_weights(net, key='mobilenetv3_large')

    return net

def mobilenetv3_large_150(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]

    net = MobileNetV3(cfgs, mode='large', width_mult = 1.5, **kwargs)
    if pretrained:
        raise NotImplementedError("The weights for this configuration are not available")

    return net

def mobilenetv3_large_125(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]

    net = MobileNetV3(cfgs, mode='large', width_mult = 1.25, **kwargs)
    if pretrained:
        raise NotImplementedError("The weights for this configuration are not available")

    return net

def mobilenetv3_small(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]
    net = MobileNetV3(cfgs, mode='small', width_mult = 1., **kwargs)
    if pretrained:
        init_pretrained_weights(net, key='mobilenetv3_small')

    return net
