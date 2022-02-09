# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import, division, print_function

import torch

@torch.no_grad()
def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.

    Examples::
        >>> from torchreid import metrics
        >>> metrics.accuracy(output, target)
    """
    maxk = max(topk)
    batch_size = max(1, target.size(0))

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


@torch.no_grad()
def accuracy_multilabel(output, target, threshold=0.5):
    batch_size = max(1, target.size(0))

    if isinstance(output, (tuple, list)):
        output = torch.sigmoid(output[0])

    pred_idx = output > threshold
    num_correct = (pred_idx == target).sum(dim=-1)
    num_correct = (num_correct == target.size(1)).sum()

    return num_correct / batch_size
