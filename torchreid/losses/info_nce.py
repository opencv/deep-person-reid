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

from __future__ import absolute_import, division

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, use_gpu=True, s=30, end_s=None, duration_s=None, skip_steps_s=None):
        super().__init__()
        self.use_gpu = use_gpu

        assert s > 0
        self.start_s = s
        assert self.start_s > 0.0
        self.end_s = end_s
        self.duration_s = duration_s
        self.skip_steps_s = skip_steps_s
        self.last_scale = self.start_s

    @staticmethod
    def get_last_info():
        return {}

    def get_last_scale(self):
        return self.last_scale

    @staticmethod
    def _get_scale(start_scale, end_scale, duration, skip_steps, iteration, power=1.2):
        def _invalid(_v):
            return _v is None or _v <= 0

        if not _invalid(skip_steps) and iteration < skip_steps:
            return start_scale

        if _invalid(iteration) or _invalid(end_scale) or _invalid(duration):
            return start_scale

        skip_steps = skip_steps if not _invalid(skip_steps) else 0
        steps_to_end = duration - skip_steps
        if iteration < duration:
            factor = (end_scale - start_scale) / (1.0 - power)
            var_a = factor / (steps_to_end ** power)
            var_b = -factor * power / float(steps_to_end)

            iteration -= skip_steps
            out_value = var_a * np.power(iteration, power) + var_b * iteration + start_scale
        else:
            out_value = end_scale

        return out_value

    def forward(self, embd, iteration=None):
        self.last_scale = self._get_scale(self.start_s, self.end_s, self.duration_s, self.skip_steps_s, iteration)

        norm_embd = F.normalize(embd, p=2, dim=1)
        num_samples = norm_embd.size(0)

        similarities = torch.mm(norm_embd, torch.t(norm_embd)).clamp(-1, 1)
        all_scores = torch.exp(self.last_scale * similarities)

        with torch.no_grad():
            batch_ids = torch.arange(embd.size(0), device=embd.device)

            top_matched = batch_ids.view(-1, 1) + 1 == batch_ids.view(1, -1)
            bottom_matched = batch_ids.view(-1, 1) == batch_ids.view(1, -1) + 1
            pos_mask = torch.where((batch_ids % 2 == 0).view(-1, 1), top_matched, bottom_matched)

            non_diagonal = batch_ids.view(-1, 1) != batch_ids.view(1, -1)
            neg_mask = ~pos_mask & non_diagonal

        pos_scores = all_scores[pos_mask].view(num_samples)
        neg_scores = all_scores[neg_mask].view(num_samples, num_samples - 2)

        losses = torch.log(pos_scores / (pos_scores + neg_scores.sum(dim=1))).neg()

        return losses.mean()
