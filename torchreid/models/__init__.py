from __future__ import absolute_import
from torchreid.models.transformer import Transformer

from .efficient_net_pytcv import *
from .inceptionv4_pytcv import *
from .mobile_face_net_se import *
from .mobilenetv3 import *
from .osnet import *
from .osnet_ain import *
from .osnet_fpn import *
from .ptcv_wrapper import *
from .tresnet import *
from .resnet import *
from .q2l import *
from .transformer import *

__model_factory = {
    # image classification models
    'inceptionv4': inceptionv4_pytcv,
    'mobilenetv3_small': mobilenetv3_small,
    'mobilenetv3_large': mobilenetv3_large,
    'mobilenetv3_large_075': mobilenetv3_large_075,
    'mobilenetv3_large_150': mobilenetv3_large_150,
    'mobilenetv3_large_125': mobilenetv3_large_125,
    'mobilenetv3_large_21k': mobilenetv3_large_21k,
    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b1': efficientnet_b1,
    'efficientnet_b2': efficientnet_b2b,
    'efficientnet_b3': efficientnet_b3b,
    'efficientnet_b4': efficientnet_b4b,
    'efficientnet_b5': efficientnet_b5b,
    'efficientnet_b6': efficientnet_b6b,
    'efficientnet_b7': efficientnet_b7b,
    'tresnet': tresnet,
    'resnet101': resnet101,
    'resnet101D': resnet101D,
    'wide_resnet101': wide_resnet101,
    'resnext101': resnext101,

    # reid-specific models
    'osnet_x1_0': osnet_x1_0,
    'osnet_x0_75': osnet_x0_75,
    'osnet_x0_5': osnet_x0_5,
    'osnet_x0_25': osnet_x0_25,
    'osnet_ibn_x1_0': osnet_ibn_x1_0,
    'osnet_ain_x1_0': osnet_ain_x1_0,
    'osnet_ain2_x1_0': osnet_ain2_x1_0,
    'fpn_osnet_x1_0': fpn_osnet_x1_0,
    'fpn_osnet_x0_75': fpn_osnet_x0_75,
    'fpn_osnet_x0_5': fpn_osnet_x0_5,
    'fpn_osnet_x0_25': fpn_osnet_x0_25,
    'fpn_osnet_ibn_x1_0': fpn_osnet_ibn_x1_0,

    # face reid models
    'mobile_face_net_se_1x': mobile_face_net_se_1x,
    'mobile_face_net_se_2x': mobile_face_net_se_2x,
}

q2l_models = {'q2l_' + name : model for name, model in __model_factory.items()}
__model_factory = {**__model_factory, **wrapped_models, **q2l_models}


def build_model(name, **kwargs):
    """A function wrapper for building a model.

    Args:
        name (str): model name.

    Returns:
        nn.Module

    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(name, avai_models))
    if name.startswith('q2l'):
        backbone_name = name[4:]
        backbone = __model_factory[backbone_name](**kwargs)
        transformer = build_transformer(**kwargs)
        model = build_q2l(backbone, transformer, **kwargs)
    else:
        model = __model_factory[name](**kwargs)
    return model
