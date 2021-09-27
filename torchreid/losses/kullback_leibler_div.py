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
import torch.nn.functional as F


def kl_div(p_logits, q_logits):
    p_dist = F.softmax(p_logits, dim=-1)
    p_log_dist = F.log_softmax(p_logits, dim=-1)
    q_log_dist = F.log_softmax(q_logits, dim=-1)
    return (p_dist * (p_log_dist - q_log_dist)).sum(dim=-1)


def symmetric_kl_div(p_logits, q_logits):
    return kl_div(p_logits, q_logits) + kl_div(p_logits=q_logits, q_logits=p_logits)


def set_kl_div(logits):
    logits = logits.view(-1, logits.size(-1))
    size = logits.size(0)

    out_value = torch.zeros([], dtype=logits.dtype, device=logits.device)
    for i in range(size):
        for j in range(i + 1, size):
            out_value += symmetric_kl_div(logits[i], logits[j])

    num_pairs = size * (size - 1)
    out_value /= float(num_pairs)

    return out_value
