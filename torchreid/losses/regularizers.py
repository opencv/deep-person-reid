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

import torch
from torch import nn
import torch.nn.functional as F


class ConvRegularizer(nn.Module):
    def __init__(self, reg_class, exclude=None, **kwargs):
        super().__init__()

        self.reg_instance = reg_class(**kwargs)

        self.exclude = exclude
        if self.exclude is None:
            self.exclude = tuple()
        else:
            self.exclude = tuple(self.exclude)

    def forward(self, net):
        num_losses = 0
        accumulator = torch.tensor(0.0).cuda()
        for name, module in net.module.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue

            valid_name = True
            for exclude_name in self.exclude:
                if exclude_name not in name:
                    valid_name = False
                    break

            if not valid_name:
                continue

            loss = self.reg_instance(module.weight)
            if loss > 0:
                accumulator += loss.to(accumulator.device)
                num_losses += 1

        return accumulator / float(max(1, num_losses))


class SVMORegularizer(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    @staticmethod
    def dominant_eigenvalue(A):  # A: 'N x N'
        N, _ = A.size()
        x = torch.rand(N, 1, device='cuda')
        Ax = (A @ x)
        AAx = (A @ Ax)
        return AAx.permute(1, 0) @ Ax / (Ax.permute(1, 0) @ Ax)

    def get_singular_values(self, A):  # A: 'M x N, M >= N'
        ATA = A.permute(1, 0) @ A
        N, _ = ATA.size()
        largest = self.dominant_eigenvalue(ATA)
        I = torch.eye(N, device='cuda')  # noqa
        I = I * largest  # noqa
        tmp = self.dominant_eigenvalue(ATA - I)
        return tmp + largest, largest

    def forward(self, W):  # W: 'S x C x H x W'
        # old_W = W
        old_size = W.size()
        if old_size[0] == 1:
            return 0
        W = W.view(old_size[0], -1).permute(1, 0)  # (C x H x W) x S
        smallest, largest = self.get_singular_values(W)
        return (
            self.beta * 10 * (largest - smallest)**2
        ).squeeze()


class HardDecorrelationRegularizer(nn.Module):
    def __init__(self, max_score, scale):
        super().__init__()

        self.max_score = float(max_score)
        assert 0.0 < self.max_score < 1.0
        self.scale = float(scale)
        assert self.scale > 0.0

    def forward(self, W):
        num_filters = W.size(0)
        if num_filters == 1:
            return 0.0

        W = W.view(num_filters, -1)

        dim = W.size(1)
        if dim < num_filters:
            return 0.0

        W = F.normalize(W, p=2, dim=-1)
        similarities = torch.matmul(W, W.t())

        all_losses = (similarities.abs().clamp_min(self.max_score) - self.max_score) ** 2
        valid_losses = torch.triu(all_losses, diagonal=1)
        filtered_losses = valid_losses[valid_losses > 0.0]
        loss = filtered_losses.mean() if filtered_losses.numel() > 0 else filtered_losses.sum()

        return self.scale * loss


class NormRegularizer(nn.Module):
    def __init__(self, scale, max_ratio, min_norm=0.5, eps=1e-5):
        super().__init__()

        self.max_ratio = max_ratio
        self.scale = scale
        self.min_norm = min_norm
        self.eps = eps

    def forward(self, net):
        conv_layers = self._collect_conv_layers(net, self.eps)

        num_losses = 0
        accumulator = torch.tensor(0.0).cuda()
        for conv in conv_layers:
            loss = self._loss(conv['weight'], self.max_ratio, self.min_norm)
            if loss > 0:
                accumulator += loss.to(accumulator.device)
                num_losses += 1

        return self.scale * accumulator / float(max(1, num_losses))

    @staticmethod
    def _collect_conv_layers(net, eps):
        conv_layers = []
        for name, m in net.named_modules():
            if isinstance(m, nn.Conv2d):
                conv_layers.append(dict(
                    name=name,
                    weight=m.weight,
                    updated=False,
                ))
            elif isinstance(m, nn.BatchNorm2d):
                assert len(conv_layers) > 0

                last_conv = conv_layers[-1]
                assert not last_conv['updated']

                alpha = m.weight
                running_var = m.running_var.detach()

                scales = (alpha / torch.sqrt(running_var + eps)).view(-1, 1, 1, 1)
                last_conv['weight'] = scales * last_conv['weight']
                last_conv['updated'] = True

        return conv_layers

    @staticmethod
    def _loss(W, max_ratio, min_norm):
        num_filters = W.size(0)
        if num_filters == 1:
            return 0.0

        W = W.view(num_filters, -1)
        norms = torch.sqrt(torch.sum(W ** 2, dim=-1))

        norm_ratio = torch.max(norms.detach()) / torch.min(norms.detach())
        median_norm = torch.median(norms.detach())
        if norm_ratio < max_ratio and median_norm > min_norm:
            return 0.0

        trg_norm = max(float(min_norm), median_norm)
        losses = (norms - trg_norm) ** 2
        loss = losses.mean()

        return loss


class ComposeRegularizer(nn.Module):
    def __init__(self, regularizers):
        super().__init__()

        self.regularizers = regularizers
        assert len(regularizers) > 0

    def forward(self, net):
        loss = 0
        for regularizer in self.regularizers:
            loss += regularizer(net)

        return loss


def get_regularizer(cfg_reg):
    regularizers = []

    if cfg_reg.ow:
        regularizers.append(ConvRegularizer(
            SVMORegularizer,
            beta=cfg_reg.ow_beta
        ))

    if cfg_reg.nw:
        regularizers.append(NormRegularizer(
            max_ratio=cfg_reg.nw_max_ratio,
            scale=cfg_reg.nw_scale
        ))

    if cfg_reg.hd:
        regularizers.append(ConvRegularizer(
            HardDecorrelationRegularizer,
            max_score=cfg_reg.hd_max_score,
            scale=cfg_reg.hd_scale,
            exclude='gate.fc',
        ))

    if len(regularizers) > 0:
        return ComposeRegularizer(regularizers)

    return None
