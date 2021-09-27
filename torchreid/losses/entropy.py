"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np
import torch
from  torch import nn
import torch.nn.functional as F


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0.0, p * p.log(), torch.zeros_like(p)).sum(dim=dim, keepdim=keepdim)


class EntropyLoss(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale
        assert self.scale > 0.0

    def forward(self, cos_theta, scale=None):
        scale = scale if scale is not None and scale > 0.0 else self.scale

        probs = F.softmax(scale * cos_theta, dim=-1)
        entropy_values = entropy(probs, dim=-1)

        losses = self._calc_losses(cos_theta, entropy_values)

        return losses.mean() if losses.numel() > 0 else losses.sum()

    def _calc_losses(self, cos_theta, entropy_values):
        raise NotImplementedError


class MinEntropyLoss(EntropyLoss):
    def _calc_losses(self, cos_theta, entropy_values):
        return entropy_values


class MaxEntropyLoss(EntropyLoss):
    def _calc_losses(self, cos_theta, entropy_values):
        return np.log(cos_theta.size(-1)) - entropy_values
