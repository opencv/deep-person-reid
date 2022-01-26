# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from torchreid import metrics
from torchreid.engine import Engine
from torchreid.losses import AMSoftmaxLoss, CrossEntropyLoss
from torchreid.losses import AsymmetricLoss, AMBinaryLoss
from torchreid.optim import SAM


class MultiheadEngine(Engine):
    r"""AM-Softmax-loss engine for image-reid.
    """
    def __init__(self, datamanager, models, optimizers, schedulers, use_gpu, save_all_chkpts,
                 train_patience, early_stopping, lr_decay_factor, loss_name, label_smooth,
                 margin_type, aug_type, decay_power, alpha, size, lr_finder, aug_prob,
                 conf_penalty, pr_product, m, amb_k, amb_t, clip_grad, symmetric_ce, enable_rsc,
                 should_freeze_aux_models, nncf_metainfo, compression_ctrl, initial_lr,
                 target_metric, use_ema_decay, ema_decay,  asl_gamma_pos, asl_gamma_neg, asl_p_m,
                 mix_precision, **kwargs):

        super().__init__(datamanager,
                         models=models,
                         optimizers=optimizers,
                         schedulers=schedulers,
                         use_gpu=use_gpu,
                         save_all_chkpts=save_all_chkpts,
                         train_patience=train_patience,
                         lr_decay_factor=lr_decay_factor,
                         early_stopping=early_stopping,
                         should_freeze_aux_models=should_freeze_aux_models,
                         nncf_metainfo=nncf_metainfo,
                         compression_ctrl=compression_ctrl,
                         initial_lr=initial_lr,
                         target_metric=target_metric,
                         lr_finder=lr_finder,
                         use_ema_decay=use_ema_decay,
                         ema_decay=ema_decay)

        loss_names = loss_name.split(',')
        assert len(loss_names) == 2
        if loss_names[0] in ['softmax', 'am_softmax']:
            sm_loss_name, multilabel_loss_name = loss_names[0], loss_names[1]
        else:
            sm_loss_name, multilabel_loss_name = loss_names[1], loss_names[0]
        assert sm_loss_name in ['softmax', 'am_softmax']
        assert multilabel_loss_name in ['am_binary', 'bce', 'asl']
        if sm_loss_name == 'am_softmax' or loss_name == 'am_binary':
            assert m >= 0.0

        self.clip_grad = clip_grad
        self.enable_rsc = enable_rsc
        self.enable_sam = isinstance(self.optims[self.main_model_name], SAM)
        for model_name in self.get_model_names():
            assert isinstance(self.optims[model_name], SAM) == self.enable_sam, "SAM must be enabled \
                                                                                 for all models or none of them"
        self.prev_smooth_metric = 0.
        self.mix_precision = mix_precision
        self.scaler = GradScaler(enabled=mix_precision)

        self.ml_losses = list()
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')

        self.mixed_cls_heads_info = self.datamanager.train_dataset.mixed_cls_heads_info
        self.multiclass_loss = None
        self.multilabel_loss = None

        if self.mixed_cls_heads_info['num_multiclass_heads'] > 0:
            if sm_loss_name == 'softmax':
                self.multiclass_loss = CrossEntropyLoss(
                    use_gpu=self.use_gpu,
                    label_smooth=label_smooth,
                    conf_penalty=conf_penalty,
                    scale=self.am_scale
                )
            elif sm_loss_name == 'am_softmax':
                self.multiclass_loss = AMSoftmaxLoss(
                    use_gpu=self.use_gpu,
                    label_smooth=label_smooth,
                    margin_type=margin_type,
                    aug_type=aug_type,
                    conf_penalty=conf_penalty,
                    m=m,
                    s=self.am_scale,
                    pr_product=pr_product,
                    symmetric_ce=symmetric_ce,
                )

        if self.mixed_cls_heads_info['num_multilabel_classes'] > 0:
            if multilabel_loss_name == 'asl':
                self.multilabel_loss = AsymmetricLoss(
                    gamma_neg=asl_gamma_neg,
                    gamma_pos=asl_gamma_pos,
                    probability_margin=asl_p_m,
                    label_smooth=label_smooth,
                )
            elif multilabel_loss_name == 'bce':
                self.multilabel_loss = AsymmetricLoss(
                    gamma_neg=0,
                    gamma_pos=0,
                    probability_margin=0,
                    label_smooth=label_smooth,
                )
            elif multilabel_loss_name == 'am_binary':
                self.multilabel_loss = AMBinaryLoss(
                    m=m,
                    k=amb_k,
                    t=amb_t,
                    s=self.am_scale,
                    gamma_neg=asl_gamma_neg,
                    gamma_pos=asl_gamma_pos,
                    label_smooth=label_smooth,
                )

    @staticmethod
    def _valid(value):
        return value is not None and value > 0

    def forward_backward(self, data):
        n_iter = self.epoch * self.num_batches + self.batch_idx
        imgs, targets = self.parse_data_for_train(data, self.use_gpu)

        model_names = self.get_model_names()
        num_models = len(model_names)
        assert num_models == 1 # mutual learning is not supported in case of multihead training

        steps = [1, 2] if self.enable_sam and not self.lr_finder else [1]
        for step in steps:
            # if sam is enabled then statistics will be written each step, but will be saved only the second time
            # this is made just for convenience
            loss_summary = dict()

            for i, model_name in enumerate(model_names):
                loss, model_loss_summary, acc, scaled_logits = self._single_model_losses(
                    self.models[model_name], imgs, targets, n_iter, model_name
                    )
                loss_summary.update(model_loss_summary)
                if i == 0: # main model
                    main_acc = acc

            for i, model_name in enumerate(model_names):
                self.optims[model_name].zero_grad()

                if self.compression_ctrl:
                    compression_loss = self.compression_ctrl.loss()
                    loss_summary['compression_loss'] = compression_loss
                    loss += compression_loss

                # backward pass
                self.scaler.scale(loss).backward()
                if not self.models[model_name].training:
                    continue

                if self.clip_grad != 0 and step == 1:
                    self.scaler.unscale_(self.optims[model_name])
                    torch.nn.utils.clip_grad_norm_(self.models[model_name].parameters(), self.clip_grad)
                if not self.enable_sam and step == 1:
                    self.scaler.step(self.optims[model_name])
                    self.scaler.update()
                elif step == 1:
                    assert self.enable_sam
                    if self.clip_grad == 0:
                        # if self.clip_grad == 0  this means that unscale_ wasn't applied,
                        # so we manually unscale the parameters to perform SAM manipulations
                        self.scaler.unscale_(self.optims[model_name])
                    overflow = self.optims[model_name].first_step()
                    self.scaler.update() # update scaler after first step
                    if overflow:
                        print("Overflow occurred. Skipping step ...")
                        loss_summary['loss'] = loss.item()
                        # skip second step  if overflow occurred
                        return loss_summary, main_acc
                else:
                    assert self.enable_sam and step==2
                    # unscale the parameters to perform SAM manipulations
                    self.scaler.unscale_(self.optims[model_name])
                    self.optims[model_name].second_step()
                    self.scaler.update()

            loss_summary['loss'] = loss.item()

        return loss_summary, main_acc

    def _single_model_losses(self, model, imgs, targets, n_iter, model_name):
        with autocast(enabled=self.mix_precision):
            model_output = model(imgs)
            all_logits = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
            loss_summary = dict()
            acc = 0
            trg_num_samples = targets.numel()
            if trg_num_samples == 0:
                raise RuntimeError("There is no samples in a batch!")

            loss = 0.
            for i in range(self.mixed_cls_heads_info['num_multiclass_heads']):
                head_gt = targets[:,i]
                head_logits = all_logits[:,self.mixed_cls_heads_info['head_idx_to_logits_range'][i][0] :
                                           self.mixed_cls_heads_info['head_idx_to_logits_range'][i][1]]
                valid_mask = head_gt >= 0
                head_gt = head_gt[valid_mask].long()
                head_logits = head_logits[valid_mask,:]
                loss += self.multiclass_loss(head_logits, head_gt, scale=self.scales[model_name])
                acc += metrics.accuracy(head_logits, head_gt)[0].item()

            if self.multilabel_loss:
                head_gt = targets[:,self.mixed_cls_heads_info['num_multiclass_heads']:]
                head_logits = all_logits[:,self.mixed_cls_heads_info['num_single_label_classes']:]
                valid_mask = head_gt >= 0
                head_gt = head_gt[valid_mask].view(*valid_mask.shape)
                head_logits = head_logits[valid_mask].view(*valid_mask.shape)
                loss += self.multilabel_loss(head_logits, head_gt, scale=self.scales[model_name])
                acc += metrics.accuracy_multilabel(head_logits * self.scales[model_name], head_gt).item()

            acc /= self.mixed_cls_heads_info['num_multiclass_heads'] + int(self.multilabel_loss != None)

            loss_summary[f'main_{model_name}'] = loss.item()

            scaled_logits = self.scales[model_name] * all_logits

            return loss, loss_summary, acc, scaled_logits

    def exit_on_plateau_and_choose_best(self, accuracy):
        '''
        The function returns a pair (should_exit, is_candidate_for_best).

        The function sets this checkpoint as a candidate for best if either it is the first checkpoint
        for this LR or this checkpoint is better then the previous best.

        The function sets should_exit = True if the LR is the minimal allowed
        LR (i.e. self.lb_lr) and the best checkpoint is not changed for self.train_patience
        epochs.
        '''

        # Note that we take LR of the previous iter, not self.get_current_lr(),
        # since typically the method exit_on_plateau_and_choose_best is called after
        # the method update_lr, so LR drop happens before.
        # If we had used the method self.get_current_lr(), the last epoch
        # before LR drop would be used as the first epoch with the new LR.
        should_exit = False
        is_candidate_for_best = False
        current_metric = np.round(accuracy, 4)
        if self.best_metric >= current_metric:
            # one drop has been done -> start early stopping
            if round(self.current_lr, 8) < round(self.initial_lr, 8):
                self.iter_to_wait += 1
                if self.iter_to_wait >= self.train_patience:
                    print("LOG:: The training should be stopped due to no improvements for {} epochs".format(self.train_patience))
                    should_exit = True
        else:
            self.best_metric = current_metric
            self.iter_to_wait = 0
            is_candidate_for_best = True

        return should_exit, is_candidate_for_best
