"""
 Copyright (c) 2019 Intel Corporation

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

from __future__ import division, absolute_import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import math

from .fmix import FMixBase, sample_mask


class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""

    def __init__(self, in_features, out_features):
        super(AngleSimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.normal_().renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cos_theta = F.normalize(x.view(x.shape[0], -1), dim=1).mm(F.normalize(self.weight, p=2, dim=0))
        return cos_theta.clamp(-1, 1)

    def get_centers(self):
        return torch.t(self.weight)


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class AMSoftmaxLoss(nn.Module):
    margin_types = ['cos', 'arc']

    def __init__(self, use_gpu=True, margin_type='cos', gamma=0.0, m=0.5, t=1.0,
                 s=30, end_s=None, duration_s=None, skip_steps_s=None, conf_penalty=0.0,
                 label_smooth=0, epsilon=0.1, aug_type='', pr_product=False,
                 symmetric_ce=False, class_counts=None, adaptive_margins=False,
                 class_weighting=False):
        super(AMSoftmaxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.conf_penalty = conf_penalty
        self.label_smooth = label_smooth
        self.aug_type = aug_type
        self.pr_product = pr_product

        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert m >= 0
        self.m = m
        assert s > 0
        self.start_s = s
        assert self.start_s > 0.0
        self.end_s = end_s
        self.duration_s = duration_s
        self.skip_steps_s = skip_steps_s
        self.last_scale = self.start_s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        assert t >= 1
        self.t = t
        self.symmetric_ce = symmetric_ce

        if adaptive_margins and class_counts is not None:
            class_ids = list(sorted(class_counts.keys()))
            flat_class_counts = np.array([class_counts[class_id] for class_id in class_ids], dtype=np.float32)

            margins = (self.m / np.power(flat_class_counts, 1. / 4.)).reshape((1, -1))
            self.register_buffer('class_margins', torch.from_numpy(margins).cuda())
            print('[INFO] Enabled adaptive margins for AM-Softmax loss, avg_m=' + str(round(np.mean(margins), 2)))
        else:
            self.class_margins = self.m

        if class_weighting and class_counts is not None:
            weights = self._estimate_class_weights(class_counts)
            self.register_buffer('class_weights', torch.from_numpy(weights).cuda())
        else:
            self.class_weights = None

    @staticmethod
    def get_last_info():
        return {}

    def get_last_scale(self):
        return self.last_scale

    @staticmethod
    def _estimate_class_weights(class_sizes, num_steps=1000, num_samples=32, scale=1.0, eps=1e-4):
        class_ids = np.array(list(class_sizes.keys()), dtype=np.int32)
        counts = np.array(list(class_sizes.values()), dtype=np.float32)

        frequencies = counts / np.sum(counts)
        init_weights = np.reciprocal(frequencies + eps)

        average_weights = list()
        for _ in range(num_steps):
            ids = np.random.choice(class_ids, num_samples, p=frequencies)
            values = class_ids[ids]
            average_weights.append(np.mean(values))

        weights = scale / np.median(average_weights) * init_weights

        out_weights = np.zeros([len(class_sizes)], dtype=np.float32)
        for class_id, class_weight in zip(class_ids, weights):
            out_weights[class_id] = class_weight

        return out_weights

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

    def _reweight(self, losses, labels):
        with torch.no_grad():
            loss_weights = torch.gather(self.class_weights, 0, labels.view(-1))

        weighted_losses = loss_weights * losses

        return weighted_losses

    def forward(self, cos_theta, target, aug_index=None, lam=None, iteration=None, scale=None):
        """
        Args:
            cos_theta (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            target (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
            iteration (int): current iteration
        """
        if (self.aug_type and aug_index is not None and lam is not None):
            targets1 = torch.zeros(cos_theta.size(), device=target.device).scatter_(1, target.detach().unsqueeze(1), 1)
            targets2 = targets1[aug_index]
            new_targets = targets1 * lam + targets2 * (1 - lam)
            # in case if target label changed
            target = new_targets.argmax(dim=1)

        if self.pr_product:
            pr_alpha = torch.sqrt(1.0 - cos_theta.pow(2.0))
            cos_theta = pr_alpha.detach() * cos_theta + cos_theta.detach() * (1.0 - pr_alpha)

        one_hot_target = torch.zeros_like(cos_theta, dtype=torch.uint8)
        one_hot_target.scatter_(1, target.data.view(-1, 1), 1)
        # change margins accordingly
        self.class_margins *= one_hot_target

        if self.margin_type == 'cos':
            phi_theta = cos_theta - self.class_margins
        else:
            self.cos_m *= one_hot_target
            self.sin_m *= one_hot_target
            sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi_theta = cos_theta * self.cos_m - sine * self.sin_m
            phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - self.sin_m * self.class_margins)

        output = phi_theta
        scale = scale if scale else self.start_s
        self.last_scale = self._get_scale(scale, self.end_s, self.duration_s, self.skip_steps_s, iteration)

        if self.gamma == 0.0 and self.t == 1.0:
            output *= self.last_scale

            if self.label_smooth > 0:
                assert not self.aug_type
                target = torch.zeros(output.size(), device=target.device).scatter_(1, target.detach().unsqueeze(1), 1)
                num_classes = output.size(1)
                target = (1.0 - self.label_smooth) * target + self.label_smooth / float(num_classes)
                losses = (- target * F.log_softmax(output, dim=1)).sum(dim=1)

            elif (self.aug_type and aug_index is not None and lam is not None):
                losses = (- new_targets * F.log_softmax(output, dim=1)).sum(dim=1)

            else:
                losses = F.cross_entropy(output, target, reduction='none')

            if self.symmetric_ce:
                all_probs = F.softmax(output, dim=-1)
                target_probs = all_probs[torch.arange(target.size(0), device=target.device), target]
                losses += 4.0 * (1.0 - target_probs)

            if self.conf_penalty > 0.0:
                probs = F.softmax(output, dim=1)
                log_probs = F.log_softmax(output, dim=1)
                entropy = torch.sum(-probs * log_probs, dim=1)

                losses = F.relu(losses - self.conf_penalty * entropy)

            if self.class_weights is not None:
                with torch.no_grad():
                    loss_weights = torch.gather(self.class_weights, 0, target.view(-1))

                losses = loss_weights * losses

            with torch.no_grad():
                nonzero_count = max(losses.nonzero().size(0), 1)

            return losses.sum() / nonzero_count

        if self.t > 1:
            h_theta = self.t - 1 + self.t*cos_theta
            support_vecs_mask = (1 - index) * \
                torch.lt(torch.masked_select(phi_theta, index).view(-1, 1).repeat(1, h_theta.shape[1]) - cos_theta, 0)
            output = torch.where(support_vecs_mask, h_theta, output)

            return F.cross_entropy(self.last_scale * output, target)

        return focal_loss(F.cross_entropy(self.last_scale * output, target, reduction='none'), self.gamma)
