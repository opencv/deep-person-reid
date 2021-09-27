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


class SemiHardTripletLoss(nn.Module):
    def __init__(self, margin=0.35):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        if torch.unique(labels).numel() <= 1:
            return torch.zeros([], dtype=features.dtype, device=features.device)

        embeddings = F.normalize(features, p=2, dim=1)

        similarities = torch.mm(embeddings, torch.t(embeddings)).clamp(-1.0, 1.0)

        with torch.no_grad():
            same_class_pairs = labels.view(-1, 1) == labels.view(1, -1)
            different_class_pairs = ~same_class_pairs

            ids_range = torch.arange(labels.size(0), device=labels.device)
            non_diagonal_pairs = ids_range.view(-1, 1) != ids_range.view(1, -1)

            pos_pairs = same_class_pairs & non_diagonal_pairs
            neg_pairs = different_class_pairs

        s_pos, _ = torch.where(pos_pairs, similarities, torch.full_like(similarities, 1.0)).min(dim=1)
        s_neg, _ = torch.where(neg_pairs, similarities, torch.full_like(similarities, -1.0)).max(dim=1)

        losses = F.relu(self.margin + s_neg - s_pos)
        loss = losses.sum()

        num_valid = (losses > 0.0).sum().float()
        if num_valid > 0.0:
            loss /= num_valid

        return loss


class InvDistanceTripletLoss(nn.Module):
    def __init__(self, margin=0.35):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        if torch.unique(labels).numel() <= 1:
            return torch.zeros([], dtype=features.dtype, device=features.device)

        dim = features.size(1)
        embeddings = F.normalize(features, p=2, dim=1)

        similarities = torch.mm(embeddings, torch.t(embeddings)).clamp(-1.0, 1.0)

        with torch.no_grad():
            same_class_pairs = labels.view(-1, 1) == labels.view(1, -1)
            different_class_pairs = ~same_class_pairs

            batch_ids = torch.arange(labels.size(0), device=labels.device)
            non_diagonal_pairs = batch_ids.view(-1, 1) != batch_ids.view(1, -1)

            pos_pairs = same_class_pairs & non_diagonal_pairs
            pos_ids = torch.multinomial(pos_pairs.float(), 1).view(-1)

            distances = 1.0 - similarities
            log_q_d_inv = float(2 - dim) * torch.log(distances) + \
                          0.5 * float(3 - dim) * torch.log(1.0 - 0.25 * distances.pow(2))
            log_q_d_inv = torch.where(different_class_pairs, log_q_d_inv, torch.zeros_like(log_q_d_inv))
            q_d_inv = torch.exp(log_q_d_inv - log_q_d_inv.max(dim=1, keepdim=True)[0])
            neg_ids_weights = torch.where(different_class_pairs, q_d_inv, torch.zeros_like(q_d_inv))
            neg_ids = torch.multinomial(neg_ids_weights, 1).view(-1)

        s_pos = similarities[batch_ids, pos_ids]
        s_neg = similarities[batch_ids, neg_ids]

        losses = F.relu(self.margin + s_neg - s_pos)
        loss = losses.sum()

        num_valid = (losses > 0.0).sum().float()
        if num_valid > 0.0:
            loss /= num_valid

        return loss


class CenterLoss(nn.Module):
    """Implementation of the Center loss from https://ydwen.github.io/papers/WenECCV16.pdf"""

    def __init__(self, num_classes, embed_size):
        super().__init__()
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(self.num_classes, embed_size).cuda())
        self.embed_size = embed_size

    def get_centers(self):
        """Returns estimated centers"""
        return self.centers

    def normalize_centers(self):
        self.centers.data = F.normalize(self.centers.data, p=2, dim=1)

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        batch_size = labels.size(0)
        features_dim = features.size(1)
        assert features_dim == self.embed_size

        centers = F.normalize(self.centers, p=2, dim=1)
        centers_batch = centers[labels, :]

        cos_diff = 1.0 - torch.sum(features * centers_batch, dim=1).clamp(-1, 1)
        center_loss = torch.sum(cos_diff) / batch_size

        return center_loss


class CentersPush(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()

        self.margin = margin

    def forward(self, centers, labels):
        centers = F.normalize(centers, p=2, dim=1)

        unique_labels = torch.unique(labels)
        unique_centers = centers[unique_labels, :]

        distances = 1.0 - torch.mm(unique_centers, torch.t(unique_centers)).clamp(-1, 1)
        losses = self.margin - distances

        different_class_pairs = unique_labels.view(-1, 1) != unique_labels.view(1, -1)
        pairs_valid_mask = different_class_pairs & (losses > 0.0)

        losses = torch.where(pairs_valid_mask, losses, torch.zeros_like(losses))

        num_valid = pairs_valid_mask.sum().float()
        loss = losses.sum()
        if num_valid > 0.0:
            loss /= num_valid

        return loss


class LocalPushLoss(nn.Module):
    def __init__(self, margin=0.1, smart_margin=True):
        super().__init__()
        self.margin = margin
        assert self.margin >= 0.0
        self.smart_margin = smart_margin

    def forward(self, features, cos_theta, target):
        normalized_embeddings = F.normalize(features, p=2, dim=1)
        similarity = normalized_embeddings.matmul(normalized_embeddings.permute(1, 0))

        with torch.no_grad():
            pairs_mask = target.view(-1, 1) != target.view(1, -1)

            if self.smart_margin:
                center_similarity = cos_theta[torch.arange(cos_theta.size(0), device=target.device), target]
                threshold = center_similarity.clamp(min=self.margin).view(-1, 1) - self.margin
            else:
                threshold = self.margin

            similarity_mask = similarity > threshold
            mask = pairs_mask & similarity_mask

        filtered_similarity = torch.where(mask, similarity - threshold, torch.zeros_like(similarity))
        losses, _ = filtered_similarity.max(dim=-1)

        return losses.mean()


class MetricLosses:
    """Class-aggregator for metric-learning losses"""

    def __init__(self, num_classes, embed_size,
                 center_coeff=1.0, triplet_coeff=1.0, local_push_coeff=1.0,
                 center_margin=0.1, triplet_margin=0.35, local_push_margin=0.1,
                 loss_balancing=True, centers_lr=0.5, balancing_lr=0.01,
                 smart_margin=True, triplet='semihard', name='ml'):
        self.name = name
        self.total_losses_num = 0
        self.losses_map = {}

        self.center_coeff = center_coeff
        if self.center_coeff is not None and self.center_coeff > 0:
            self.center_loss = CenterLoss(num_classes, embed_size)
            self.center_optimizer = torch.optim.SGD(self.center_loss.parameters(), lr=centers_lr)
            self.losses_map['center'] = self.total_losses_num
            self.total_losses_num += 1

            self.centers_push_loss = CentersPush(margin=center_margin)
            self.losses_map['push_center'] = self.total_losses_num
            self.total_losses_num += 1

        self.local_push_coeff = local_push_coeff
        if self.local_push_coeff is not None and self.local_push_coeff > 0:
            self.local_push_loss = LocalPushLoss(margin=local_push_margin, smart_margin=smart_margin)
            self.losses_map['local_push'] = self.total_losses_num
            self.total_losses_num += 1

        self.triplet_coeff = triplet_coeff
        if self.triplet_coeff is not None and self.triplet_coeff > 0:
            assert triplet in ['semihard', 'invdist']
            triplet_instance = SemiHardTripletLoss if triplet == 'semihard' else InvDistanceTripletLoss
            self.triplet_loss = triplet_instance(margin=triplet_margin)
            self.losses_map['triplet'] = self.total_losses_num
            self.total_losses_num += 1

        self.loss_balancing = loss_balancing and self.total_losses_num > 1
        if self.loss_balancing:
            self.loss_weights = nn.Parameter(torch.FloatTensor(self.total_losses_num).cuda())
            self.balancing_optimizer = torch.optim.SGD([self.loss_weights], lr=balancing_lr)
            for i in range(self.total_losses_num):
                self.loss_weights.data[i] = 0.0

    def _balance_losses(self, losses, scale=0.1):
        assert len(losses) == self.total_losses_num

        num_valid_losses = 0
        for i, loss_val in enumerate(losses):
            if loss_val > 0.0:
                weight = torch.exp(-self.loss_weights[i])
                weighted_loss_val = weight * loss_val + scale * self.loss_weights[i]

                losses[i] = weighted_loss_val.clamp_min(0.0)

                num_valid_losses += 1
            else:
                losses[i] = loss_val.clamp_min(0.0)

        scale = float(len(losses)) / float(num_valid_losses if num_valid_losses > 0 else 1)
        loss = scale * sum(losses)

        return loss

    def __call__(self, features, cos_theta, labels, iteration):
        all_loss_values = []
        loss_summary = {}

        center_loss_val = 0
        centers_push_loss_val = 0
        if self.center_coeff > 0.:
            center_loss_val = self.center_loss(features, labels)
            all_loss_values.append(center_loss_val)
            loss_summary[f'{self.name}/center'] = center_loss_val.item()

            centers_push_loss_val = self.centers_push_loss(self.center_loss.get_centers(), labels)
            all_loss_values.append(centers_push_loss_val)
            loss_summary[f'{self.name}/push_center'] = centers_push_loss_val.item()

        triplet_loss_val = 0
        if self.triplet_coeff > 0.0:
            triplet_loss_val = self.triplet_loss(features, labels)
            all_loss_values.append(triplet_loss_val)
            loss_summary[f'{self.name}/triplet'] = triplet_loss_val.item()

        local_push_loss_val = 0
        if self.local_push_coeff > 0.0:
            local_push_loss_val = self.local_push_loss(features, cos_theta, labels)
            all_loss_values.append(local_push_loss_val)
            loss_summary[f'{self.name}/local_push'] = local_push_loss_val.item()

        if self.loss_balancing and self.total_losses_num > 1:
            loss_value = self._balance_losses(all_loss_values)
        else:
            loss_value = self.center_coeff * (center_loss_val + centers_push_loss_val) + \
                         self.triplet_coeff * triplet_loss_val +\
                         self.local_push_coeff * local_push_loss_val
        self.last_loss_value = loss_value

        return loss_value, loss_summary

    def init_iteration(self):
        """Initializes a training iteration"""

        if self.center_coeff > 0.:
            self.center_optimizer.zero_grad()

        if self.loss_balancing:
            self.balancing_optimizer.zero_grad()

    def end_iteration(self, do_backward=True):
        """Finalizes a training iteration"""

        if do_backward:
            self.last_loss_value.backward()

        if self.center_coeff > 0.:
            self.center_optimizer.step()
            self.center_loss.normalize_centers()

        if self.loss_balancing:
            self.balancing_optimizer.step()
