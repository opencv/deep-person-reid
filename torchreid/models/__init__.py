from __future__ import absolute_import

from .efficient_net_pytcv import *
from .inceptionv4_pytcv import *
from .mobile_face_net_se import *
from .mobilenetv3 import *
from .osnet import *
from .osnet_ain import *
from .osnet_fpn import *
from .ptcv_wrapper import *
from .tresnet import *

__model_factory = {
    # image classification models
    'inceptionv4_pytcv': inceptionv4_pytcv,
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

__model_factory = {**__model_factory, **wrapped_models}


def show_avai_models():
    """Displays available models.

    Examples::
        >>> from torchreid import models
        >>> models.show_avai_models()
    """
    print(list(__model_factory.keys()))


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
    return __model_factory[name](**kwargs)
