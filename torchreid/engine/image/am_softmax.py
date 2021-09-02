"""
 MIT License

 Copyright (c) 2018 Kaiyang Zhou

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

from __future__ import absolute_import, division, print_function
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchreid import metrics
from torchreid.engine import Engine
from torchreid.utils import get_model_attr
from torchreid.losses import (AMSoftmaxLoss, CrossEntropyLoss, MetricLosses,
                              get_regularizer, sample_mask)
from torchreid.optim import SAM

class ImageAMSoftmaxEngine(Engine):
    r"""AM-Softmax-loss engine for image-reid.
    """

    def __init__(self, datamanager, models, optimizers, reg_cfg, metric_cfg, schedulers, use_gpu, save_chkpt,
                 train_patience, early_stoping, lr_decay_factor, loss_name, label_smooth,
                 margin_type, aug_type, decay_power, alpha, size, lr_finder, max_soft,
                 reformulate, aug_prob, conf_penalty, pr_product, m, s, compute_s, end_s, clip_grad,
                 duration_s, skip_steps_s, enable_masks, adaptive_margins, class_weighting,
                 attr_cfg, base_num_classes, symmetric_ce, mix_weight, enable_rsc, enable_sam,
                 should_freeze_aux_models, nncf_metainfo, initial_lr, target_metric, use_ema_decay, ema_decay, **kwargs):
        super(ImageAMSoftmaxEngine, self).__init__(datamanager,
                                                   models=models,
                                                   optimizers=optimizers,
                                                   schedulers=schedulers,
                                                   use_gpu=use_gpu,
                                                   save_chkpt=save_chkpt,
                                                   train_patience=train_patience,
                                                   lr_decay_factor=lr_decay_factor,
                                                   early_stoping=early_stoping,
                                                   should_freeze_aux_models=should_freeze_aux_models,
                                                   nncf_metainfo=nncf_metainfo,
                                                   initial_lr=initial_lr,
                                                   target_metric=target_metric,
                                                   lr_finder=lr_finder,
                                                   use_ema_decay=use_ema_decay,
                                                   ema_decay=ema_decay)

        assert loss_name in ['softmax', 'am_softmax']
        self.loss_name = loss_name
        assert s > 0.0
        if loss_name == 'am_softmax':
            assert m >= 0.0

        self.regularizer = get_regularizer(reg_cfg)
        self.enable_metric_losses = metric_cfg.enable
        self.enable_masks = enable_masks
        self.mix_weight = mix_weight
        self.clip_grad = clip_grad
        self.enable_rsc = enable_rsc
        self.enable_sam = enable_sam
        self.aug_type = aug_type
        self.aug_prob = aug_prob
        self.aug_index = None
        self.lam = None
        self.alpha = alpha
        self.decay_power = decay_power
        self.size =  size
        self.max_soft = max_soft
        self.reformulate = reformulate

        num_classes = self.datamanager.num_train_pids
        if not isinstance(num_classes, (list, tuple)):
            num_classes = [num_classes]
        self.num_classes = num_classes
        scales = dict()
        if compute_s:
            scale = self.compute_s(num_classes[0])
            s = scale
            print(f"computed margin scale for dataset: {scale}")
        else:
            scale = s

        for model_name, model in self.models.items():
            if get_model_attr(model, 'use_angle_simple_linear'):
                scales[model_name] = scale
            else:
                scales[model_name] = 1.
        self.scales = scales
        self.num_targets = len(self.num_classes)

        self.main_losses = nn.ModuleList()
        self.ml_losses = list()

        for trg_id, trg_num_classes in enumerate(self.num_classes):
            if base_num_classes <= 1:
                scale_factor = 1.0
            else:
                scale_factor = np.log(trg_num_classes - 1) / np.log(base_num_classes - 1)

            if loss_name == 'softmax':
                self.main_losses.append(CrossEntropyLoss(
                    use_gpu=self.use_gpu,
                    label_smooth=label_smooth,
                    augmentations=self.aug_type,
                    conf_penalty=conf_penalty,
                    scale=scale_factor * s
                ))
            elif loss_name == 'am_softmax':
                trg_class_counts = datamanager.data_counts[trg_id]
                assert len(trg_class_counts) == trg_num_classes

                self.main_losses.append(AMSoftmaxLoss(
                    use_gpu=self.use_gpu,
                    label_smooth=label_smooth,
                    margin_type=margin_type,
                    aug_type=aug_type,
                    conf_penalty=conf_penalty,
                    m=m,
                    s=scale_factor * s,
                    end_s=scale_factor * end_s if self._valid(end_s) else None,
                    duration_s=duration_s * self.num_batches if self._valid(duration_s) else None,
                    skip_steps_s=skip_steps_s * self.num_batches if self._valid(skip_steps_s) else None,
                    pr_product=pr_product,
                    symmetric_ce=symmetric_ce,
                    class_counts=trg_class_counts,
                    adaptive_margins=adaptive_margins,
                    class_weighting=class_weighting
                ))

            if self.enable_metric_losses:
                trg_ml_losses = dict()
                for model_name, model in self.models.items():
                    feature_dim = model.module.feature_dim
                    if hasattr(model.module, 'out_feature_dims'):
                        feature_dim = model.module.out_feature_dims[trg_id]

                    ml_cfg = copy.deepcopy(metric_cfg)
                    ml_cfg.pop('enable')
                    ml_cfg['name'] = 'ml_{}/{}'.format(trg_id, model_name)
                    trg_ml_losses[model_name] = MetricLosses(trg_num_classes, feature_dim, **ml_cfg)

                self.ml_losses.append(trg_ml_losses)

        self.enable_attr = attr_cfg is not None
        self.attr_losses = {}
        if self.enable_attr:
            self.attr_losses = nn.ModuleDict()
            for attr_name, attr_size in zip(attr_cfg.names, attr_cfg.num_classes):
                if attr_size is None or attr_size <= 0:
                    continue

                self.attr_losses[attr_name] = AMSoftmaxLoss(
                    use_gpu=self.use_gpu,
                    label_smooth=attr_cfg.label_smooth,
                    conf_penalty=attr_cfg.conf_penalty,
                    m=attr_cfg.m,
                    s=attr_cfg.s,
                    end_s=attr_cfg.end_s if self._valid(attr_cfg.end_s) else None,
                    duration_s=attr_cfg.duration_s * self.num_batches if self._valid(attr_cfg.duration_s) else None,
                    skip_steps_s=attr_cfg.skip_steps_s * self.num_batches if self._valid(attr_cfg.skip_steps_s) else None,
                    pr_product=attr_cfg.pr_product
                )

            if len(self.attr_losses) == 0:
                self.enable_attr = False
            else:
                self.attr_name_map = {attr_name: attr_id for attr_id, attr_name in enumerate(attr_cfg.names)}

    @staticmethod
    def _valid(value):
        return value is not None and value > 0

    @staticmethod
    def compute_s(num_class: int):
        return float(max(np.sqrt(2) * np.log(num_class - 1), 3))

    def forward_backward(self, data):
        n_iter = self.epoch * self.num_batches + self.batch_idx

        train_records = self.parse_data_for_train(data, True, self.enable_masks, self.use_gpu)
        imgs = train_records['img']
        obj_ids = train_records['obj_id']
        num_packages = 1
        if len(imgs.size()) != 4:
            assert len(imgs.size()) == 5

            b, num_packages, c, h, w = imgs.size()
            imgs = imgs.view(b * num_packages, c, h, w)
            obj_ids = obj_ids.view(-1, 1).repeat(1, num_packages).view(-1)
            train_records['dataset_id'] = train_records['dataset_id'].view(-1, 1).repeat(1, num_packages).view(-1)

        imgs, obj_ids = self._apply_batch_augmentation(imgs, obj_ids)

        model_names = self.get_model_names()
        num_models = len(model_names)
        steps = [1,2] if self.enable_sam and not self.lr_finder else [1]
        for step in steps:
            # if sam is enabled then statistics will be written each step, but will be saved only the second time
            # this is made just for convinience
            avg_acc = 0.0
            out_logits = [[] for _ in range(self.num_targets)]
            total_loss = torch.zeros([], dtype=imgs.dtype, device=imgs.device)
            loss_summary = dict()

            for model_name in model_names:
                self.optims[model_name].zero_grad()

                model_loss, model_loss_summary, model_avg_acc, model_logits = self._single_model_losses(
                    self.models[model_name], train_records, imgs, obj_ids, n_iter, model_name, num_packages
                )
                avg_acc += model_avg_acc / float(num_models)
                total_loss += model_loss / float(num_models)
                loss_summary.update(model_loss_summary)

                for trg_id in range(self.num_targets):
                    if model_logits[trg_id] is not None:
                        out_logits[trg_id].append(model_logits[trg_id])

            if len(model_names) > 1:
                num_mutual_losses = 0
                mutual_loss = torch.zeros([], dtype=imgs.dtype, device=imgs.device)
                for trg_id in range(self.num_targets):
                    if len(out_logits[trg_id]) <= 1:
                        continue

                    with torch.no_grad():
                        trg_probs = torch.softmax(torch.stack(out_logits[trg_id]), dim=2).mean(dim=0)

                    for model_id, logits in enumerate(out_logits[trg_id]):
                        log_probs = torch.log_softmax(logits, dim=1)
                        m_loss = (trg_probs * log_probs).sum(dim=1).mean().neg()

                        mutual_loss += m_loss
                        loss_summary['mutual_{}/{}'.format(trg_id, model_names[model_id])] = m_loss.item()
                        num_mutual_losses += 1

                should_turn_off_mutual_learning = self._should_turn_off_mutual_learning(self.epoch)
                coeff_mutual_learning = int(not should_turn_off_mutual_learning)

                total_loss += coeff_mutual_learning * mutual_loss / float(num_mutual_losses)


            total_loss.backward(retain_graph=self.enable_metric_losses)

            for model_name in model_names:
                if self.clip_grad != 0:
                    torch.nn.utils.clip_grad_norm_(self.models[model_name].parameters(), self.clip_grad)
                for trg_id in range(self.num_targets):
                    if self.enable_metric_losses:
                        ml_loss_module = self.ml_losses[trg_id][model_name]
                        ml_loss_module.end_iteration(do_backward=False)
                if isinstance(self.optims[model_name], SAM) and step == 1:
                    self.optims[model_name].first_step()
                elif isinstance(self.optims[model_name], SAM) and step == 2:
                    self.optims[model_name].second_step()
                elif not isinstance(self.optims[model_name], SAM) and step == 1:
                    self.optims[model_name].step()

            loss_summary['loss'] = total_loss.item()

        return loss_summary, avg_acc

    def _single_model_losses(self, model, train_records, imgs, obj_ids, n_iter, model_name, num_packages):
        run_kwargs = self._prepare_run_kwargs(obj_ids)
        model_output = model(imgs, **run_kwargs)
        all_logits, all_embeddings, extra_data = self._parse_model_output(model_output)

        total_loss = torch.zeros([], dtype=imgs.dtype, device=imgs.device)
        out_logits = []
        loss_summary = dict()

        num_trg_losses = 0
        avg_acc = 0

        for trg_id in range(self.num_targets):
            trg_mask = train_records['dataset_id'] == trg_id

            trg_obj_ids = obj_ids[trg_mask]
            trg_num_samples = trg_obj_ids.numel()
            if trg_num_samples == 0:
                out_logits.append(None)
                continue

            trg_logits = all_logits[trg_id][trg_mask]
            main_loss = self.main_losses[trg_id](trg_logits, trg_obj_ids, aug_index=self.aug_index,
                                                lam=self.lam, iteration=n_iter, scale=self.scales[model_name])
            avg_acc += metrics.accuracy(trg_logits, trg_obj_ids)[0].item()
            loss_summary['main_{}/{}'.format(trg_id, model_name)] = main_loss.item()

            scaled_trg_logits = self.main_losses[trg_id].get_last_scale() * trg_logits
            out_logits.append(scaled_trg_logits)

            trg_loss = main_loss
            if self.enable_metric_losses:
                ml_loss_module = self.ml_losses[trg_id][model_name]
                embd = all_embeddings[trg_id][trg_mask]

                ml_loss_module.init_iteration()
                ml_loss, ml_loss_summary = ml_loss_module(embd, trg_logits, trg_obj_ids, n_iter)

                loss_summary['ml_{}/{}'.format(trg_id, model_name)] = ml_loss.item()
                loss_summary.update(ml_loss_summary)
                trg_loss += ml_loss

            if num_packages > 1 and self.mix_weight > 0.0:
                mix_all_logits = scaled_trg_logits.view(-1, num_packages, scaled_trg_logits.size(1))
                mix_log_probs = torch.log_softmax(mix_all_logits, dim=2)

                with torch.no_grad():
                    trg_mix_probs = torch.softmax(mix_all_logits, dim=2).mean(dim=1, keepdim=True)

                mixing_loss = (trg_mix_probs * mix_log_probs).sum(dim=2).neg().mean()

                loss_summary['mix_{}/{}'.format(trg_id, model_name)] = mixing_loss.item()
                trg_loss += self.mix_weight * mixing_loss

            total_loss += trg_loss
            num_trg_losses += 1
        total_loss /= float(num_trg_losses)
        avg_acc /= float(num_trg_losses)

        if self.enable_attr and train_records['attr'] is not None:
            attributes = train_records['attr']
            all_attr_logits = extra_data['attr_logits']

            num_attr_losses = 0
            total_attr_loss = 0
            for attr_name, attr_loss_module in self.attr_losses.items():
                attr_labels = attributes[self.attr_name_map[attr_name]]
                valid_attr_mask = attr_labels >= 0

                attr_labels = attr_labels[valid_attr_mask]
                if attr_labels.numel() == 0:
                    continue

                attr_logits = all_attr_logits[attr_name][valid_attr_mask]

                attr_loss = attr_loss_module(attr_logits, attr_labels, iteration=n_iter)
                loss_summary['{}/{}'.format(attr_name, model_name)] = attr_loss.item()

                total_attr_loss += attr_loss
                num_attr_losses += 1

            total_loss += total_attr_loss / float(max(1, num_attr_losses))

        if self.enable_masks and train_records['mask'] is not None:
            att_loss_val = 0.0
            for att_map in extra_data['att_maps']:
                if att_map is not None:
                    with torch.no_grad():
                        att_map_size = att_map.size()[2:]
                        pos_float_mask = F.interpolate(train_records['mask'], size=att_map_size, mode='nearest')
                        pos_mask = pos_float_mask > 0.0
                        neg_mask = ~pos_mask

                        trg_mask_values = torch.where(pos_mask,
                                                      torch.ones_like(pos_float_mask),
                                                      torch.zeros_like(pos_float_mask))
                        num_positives = trg_mask_values.sum(dim=(1, 2, 3), keepdim=True)
                        num_negatives = float(att_map_size[0] * att_map_size[1]) - num_positives

                        batch_factor = 1.0 / float(att_map.size(0))
                        pos_weights = batch_factor / num_positives.clamp_min(1.0)
                        neg_weights = batch_factor / num_negatives.clamp_min(1.0)

                    att_errors = torch.abs(att_map - trg_mask_values)
                    att_pos_errors = (pos_weights * att_errors)[pos_mask].sum()
                    att_neg_errors = (neg_weights * att_errors)[neg_mask].sum()

                    att_loss_val += 0.5 * (att_pos_errors + att_neg_errors)

            if att_loss_val > 0.0:
                loss_summary['att/{}'.format(model_name)] = att_loss_val.item()
                total_loss += att_loss_val

        if self.regularizer is not None and (self.epoch + 1) > self.fixbase_epoch:
            reg_loss = self.regularizer(model)

            loss_summary['reg/{}'.format(model_name)] = reg_loss.item()
            total_loss += reg_loss

        return total_loss, loss_summary, avg_acc, out_logits

    def _prepare_run_kwargs(self, gt_labels):
        run_kwargs = dict()
        if self.enable_metric_losses:
            run_kwargs['get_embeddings'] = True
        if self.enable_attr or self.enable_masks:
            run_kwargs['get_extra_data'] = True
        if self.enable_rsc:
            run_kwargs['gt_labels'] = gt_labels


        return run_kwargs

    def _parse_model_output(self, model_output):
        if self.enable_metric_losses:
            all_logits, all_embeddings = model_output[:2]
            all_embeddings = all_embeddings if isinstance(all_embeddings, (tuple, list)) else [all_embeddings]
        else:
            all_logits = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
            all_embeddings = None

        all_logits = all_logits if isinstance(all_logits, (tuple, list)) else [all_logits]

        if self.enable_attr or self.enable_masks:
            extra_data = model_output[-1]
        else:
            extra_data = None

        return all_logits, all_embeddings, extra_data

    def _apply_batch_augmentation(self, imgs, obj_ids):
        if self.aug_type == 'fmix':
            r = np.random.rand(1)
            if self.alpha > 0 and r[0] <= self.aug_prob:
                lam, fmask = sample_mask(self.alpha, self.decay_power, self.size,
                                        self.max_soft, self.reformulate)
                index = torch.randperm(imgs.size(0), device=imgs.device)
                fmask = torch.from_numpy(fmask).float().to(imgs.device)
                # Mix the images
                x1 = fmask * imgs
                x2 = (1 - fmask) * imgs[index]
                self.aug_index = index
                self.lam = lam
                imgs = x1 + x2
            else:
                self.aug_index = None
                self.lam = None

        elif self.aug_type == 'mixup':
            r = np.random.rand(1)
            if self.alpha > 0 and r <= self.aug_prob:
                lam = np.random.beta(self.alpha, self.alpha)
                index = torch.randperm(imgs.size(0), device=imgs.device)

                imgs = lam * imgs + (1 - lam) * imgs[index, :]
                self.lam = lam
                self.aug_index = index
            else:
                self.aug_index = None
                self.lam = None

        elif self.aug_type == 'cutmix':
            r = np.random.rand(1)
            if self.alpha > 0 and r <= self.aug_prob:
                # generate mixed sample
                lam = np.random.beta(self.alpha, self.alpha)
                rand_index = torch.randperm(imgs.size(0), device=imgs.device)

                bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
                imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
                self.lam = lam
                self.aug_index = rand_index
            else:
                self.aug_index = None
                self.lam = None

        return imgs, obj_ids

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
