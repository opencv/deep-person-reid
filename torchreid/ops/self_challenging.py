'''
Copyright (c) 2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 '''

import numpy as np
import torch
from torch import nn
from torch.autograd import grad


class RSC(nn.Module):
    def __init__(self, retain_p):
        super().__init__()

        self.retain_p = float(retain_p)
        assert 0. < self.retain_p < 1.

    def forward(self, features, scores, labels):
        return rsc(features, scores, labels, self.retain_p)


def rsc(features, scores, labels, retain_p=0.67, retain_batch=0.67):
    """Representation Self-Challenging module (RSC).
       Based on the paper: https://arxiv.org/abs/2007.02454
    """

    batch_range = torch.arange(scores.size(0), device=scores.device)
    gt_scores = scores[batch_range, labels.view(-1)]
    z_grads = grad(outputs=gt_scores,
                   inputs=features,
                   grad_outputs=torch.ones_like(gt_scores),
                   create_graph=True)[0]

    with torch.no_grad():
        z_grads_cpu = z_grads.cpu().numpy()
        non_batch_axis = tuple(range(1, len(z_grads_cpu.shape)))
        z_grad_thresholds_cpu = np.quantile(z_grads_cpu, retain_p, axis=non_batch_axis, keepdims=True)
        zero_mask = z_grads > torch.from_numpy(z_grad_thresholds_cpu).to(z_grads.device)

        unchanged_mask = torch.randint(2, [z_grads.size(0)], dtype=torch.bool, device=z_grads.device)
        unchanged_mask = unchanged_mask.view((-1,) + (1,) * len(non_batch_axis))

    scale = 1.0 / float(retain_p)
    filtered_features = scale * torch.where(zero_mask, torch.zeros_like(features), features)
    out_features = torch.where(unchanged_mask, features, filtered_features)
    # filter batch
    random_uniform = torch.rand(size=(features.size(0), 1, 1, 1), device=features.device)
    random_mask = random_uniform >= retain_batch
    out_features = torch.where(random_mask, out_features, features)

    return out_features
