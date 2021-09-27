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
from torch import nn
import torch.nn.functional as F


class LocalContrastNormalization(nn.Module):
    def __init__(self, num_channels, kernel_size=5, affine=False, eps=1e-4):
        super().__init__()

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2, (kernel_size - 1) // 2

        filter_weights = np.ones([1, num_channels, kernel_size, kernel_size], dtype=np.float32)
        filter_weights /= num_channels * kernel_size * kernel_size
        self.register_buffer('filter', torch.from_numpy(filter_weights))

        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(self.num_channels))
            self.bias = nn.Parameter(torch.Tensor(self.num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.eps = eps

    def forward(self, x):
        local_mean = F.conv2d(x, self.filter, None, 1, self.padding)
        centered_x = x - local_mean

        sum_sqr_image = F.conv2d(centered_x.pow(2), self.filter, None, 1, self.padding)
        denominator = sum_sqr_image.sqrt()
        per_img_mean = torch.mean(denominator, dim=(2, 3), keepdim=True)

        divisor = torch.max(per_img_mean, denominator)
        y = centered_x / divisor.clamp_min(self.eps)

        if self.affine:
            y = y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

        return y
