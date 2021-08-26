from __future__ import absolute_import, division, print_function

from .am_softmax import AMSoftmaxLoss, AngleSimpleLinear
from .asl import AsymmetricLoss, AMBinaryLoss
from .cross_entropy_loss import CrossEntropyLoss, PseudoCrossEntropyLoss
from .entropy import MaxEntropyLoss, MinEntropyLoss, entropy
from .fmix import FMixBase, sample_mask
from .hard_mine_triplet_loss import TripletLoss
from .info_nce import InfoNCELoss
from .kullback_leibler_div import kl_div, set_kl_div, symmetric_kl_div
from .metric import CenterLoss, MetricLosses
from .regularizers import (ConvRegularizer, NormRegularizer, SVMORegularizer,
                           get_regularizer)


def DeepSupervision(criterion, xs, y, **kwargs):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    total_loss = 0.0
    num_losses = 0
    for x in xs:
        loss_val = criterion(x, y, **kwargs)
        if loss_val > 0.0:
            total_loss += loss_val
            num_losses += 1
    total_loss /= float(num_losses if num_losses > 0 else 1)

    return total_loss
