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

import torch
from torch import nn
import torch.nn.functional as F


def gumbel(x, eps=1e-20):
    return -torch.log(-torch.log(torch.rand_like(x) + eps) + eps)


class HardGumbelSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_prob, t=1.0, forward_hard_map=False):
        g1 = gumbel(log_prob)
        g2 = gumbel(log_prob)
        values = log_prob + g1 - g2

        scale = 1. / t
        soft_map = torch.sigmoid(scale * values)

        ctx.save_for_backward(soft_map)
        ctx.scale = scale

        if forward_hard_map:
            return (values > 0.).float()

        return soft_map

    @staticmethod
    def backward(ctx, grad_output):
        soft_map, = ctx.saved_tensors
        scale = ctx.scale

        out = scale * soft_map * (1. - soft_map) * grad_output

        return out, None, None


gumbel_sigmoid = HardGumbelSigmoid.apply


class GumbelSoftmax(nn.Module):
    def __init__(self, scale=1.0, dim=1):
        super().__init__()

        self.scale = float(scale)
        self.dim = dim

    def forward(self, logits):
        y = logits + gumbel(logits) if self.training else logits

        return F.softmax(self.scale * y, dim=self.dim)


class GumbelSigmoid(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = float(scale)

    def forward(self, logits):
        y = logits + gumbel(logits) if self.training else logits

        return torch.sigmoid(self.scale * y)
