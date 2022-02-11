# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from torchreid import metrics
from torchreid.losses import AsymmetricLoss, AMBinaryLoss
from torchreid.optim import SAM
from torchreid.engine.engine import Engine

class MultilabelEngine(Engine):
    r"""Multilabel classification engine. It supports ASL, BCE and Angular margin loss with binary classification."""
    def __init__(self, datamanager, models, optimizers, schedulers, use_gpu, save_all_chkpts,
                 train_patience, early_stopping, lr_decay_factor, loss_name, label_smooth,
                 lr_finder, m, amb_k, amb_t, clip_grad,
                 should_freeze_aux_models, nncf_metainfo, compression_ctrl, initial_lr,
                 target_metric, use_ema_decay, ema_decay, asl_gamma_pos, asl_gamma_neg, asl_p_m,
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
                        lr_finder=lr_finder,
                        target_metric=target_metric,
                        use_ema_decay=use_ema_decay,
                        ema_decay=ema_decay)

        self.clip_grad = clip_grad
        self.aug_index = None
        self.lam = None

        if loss_name == 'asl':
            self.main_loss = AsymmetricLoss(
                gamma_neg=asl_gamma_neg,
                gamma_pos=asl_gamma_pos,
                probability_margin=asl_p_m,
                label_smooth=label_smooth,
            )
        elif loss_name == 'bce':
            self.main_loss = AsymmetricLoss(
                gamma_neg=0,
                gamma_pos=0,
                probability_margin=0,
                label_smooth=label_smooth,
            )
        elif loss_name == 'am_binary':
            self.main_loss = AMBinaryLoss(
                m=m,
                k=amb_k,
                t=amb_t,
                s=self.am_scale,
                gamma_neg=asl_gamma_neg,
                gamma_pos=asl_gamma_pos,
                label_smooth=label_smooth,
            )


        self.enable_sam = isinstance(self.optims[self.main_model_name], SAM)

        for model_name in self.get_model_names():
            assert isinstance(self.optims[model_name], SAM) == self.enable_sam, "SAM must be enabled \
                                                                                 for all models or none of them"
        self.scaler = GradScaler(enabled=mix_precision)
        self.mix_precision = mix_precision
        self.prev_smooth_accuracy = 0.

    def forward_backward(self, data):
        imgs, targets = self.parse_data_for_train(data, use_gpu=self.use_gpu)

        model_names = self.get_model_names()
        num_models = len(model_names)
        steps = [1,2] if self.enable_sam and not self.lr_finder else [1]
        # forward pass
        for step in steps:
            # if sam is enabled then statistics will be written each step, but will be saved only the second time
            # this is made just for convenience
            loss_summary = {}
            all_models_logits = []
            num_models = len(self.models)

            for i, model_name in enumerate(model_names):
                unscaled_model_logits = self._forward_model(self.models[model_name], imgs)
                all_models_logits.append(unscaled_model_logits)

            for i, model_name in enumerate(model_names):
                should_turn_off_mutual_learning = self._should_turn_off_mutual_learning(self.epoch)
                mutual_learning = num_models > 1 and not should_turn_off_mutual_learning
                self.optims[model_name].zero_grad()
                loss, model_loss_summary, acc = self._single_model_losses(all_models_logits[i],
                                                                          targets,
                                                                          model_name)
                loss_summary.update(model_loss_summary)
                if i == 0: # main model
                    main_acc = acc
                # compute mutual loss
                if mutual_learning:
                    mutual_loss = 0

                    trg_probs = torch.sigmoid(all_models_logits[i] * self.scales[model_name])
                    for j in range(num_models):
                        if i != j:
                            with torch.no_grad():
                                aux_probs = torch.sigmoid(all_models_logits[j] * self.scales[model_names[j]])
                            mutual_loss += self.kl_div_binary(trg_probs, aux_probs)

                    loss += mutual_loss / (num_models - 1)

                if self.compression_ctrl:
                    compression_loss = self.compression_ctrl.loss()
                    loss += compression_loss

                # backward pass
                self.scaler.scale(loss).backward()
                if self.clip_grad != 0 and step == 1:
                    self.scaler.unscale_(self.optims[model_name])
                    torch.nn.utils.clip_grad_norm_(self.models[model_name].parameters(), self.clip_grad)
                if not self.enable_sam and step == 1:
                    self.scaler.step(self.optims[model_name])
                    self.scaler.update()
                elif step == 1:
                    assert self.enable_sam
                    if self.clip_grad == 0:
                        # if self.clip_grad == 0  this means that unscale_ wasn't applied
                        # unscale parameters to perform SAM manipulations with parameters
                        self.scaler.unscale_(self.optims[model_name])
                    overflow = self.optims[model_name].first_step(self.scaler)
                    self.scaler.update() # update scaler after first step
                    if overflow:
                        print("Overflow occurred. Skipping step ...")
                        # skip second step  if overflow occurred
                        return loss_summary, main_acc
                else:
                    assert self.enable_sam and step==2
                    if self.clip_grad == 0:
                        self.scaler.unscale_(self.optims[model_name])
                    self.optims[model_name].second_step()
                    self.scaler.update()

        # record loss
        loss_summary['loss'] = loss.item()
        if mutual_learning:
            loss_summary[f'mutual_{model_name}'] = mutual_loss.item()
        if self.compression_ctrl:
            loss_summary['compression_loss'] = compression_loss

        return loss_summary, main_acc

    def _forward_model(self, model, imgs):
        with autocast(enabled=self.mix_precision):
            model_output = model(imgs)
            all_unscaled_logits = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
            return all_unscaled_logits

    def _single_model_losses(self, logits,  targets, model_name):
        with autocast(enabled=self.mix_precision):
            loss_summary = {}
            acc = 0
            trg_num_samples = logits.numel()
            if trg_num_samples == 0:
                raise RuntimeError("There is no samples in a batch!")

            loss = self.main_loss(logits, targets, aug_index=self.aug_index,
                                                lam=self.lam, scale=self.scales[model_name])
            acc += metrics.accuracy_multilabel(logits, targets).item()
            loss_summary[f'main_{model_name}'] = loss.item()

            return loss, loss_summary, acc

    def kl_div_binary(self, x, y):
        ''' compute KL divergence between two tensors represented
        independent binary distributions'''
        # get binary distributions for two models with shape = (BxCx2)
        p = torch.stack((x, (1-x))).permute(1,2,0)
        q = torch.stack((y, (1-y))).permute(1,2,0)
        # log probabilities
        p_log = torch.log(p.add_(1e-8))
        # compute true KLDiv for each sample, than do the batchmean reduction
        return F.kl_div(p_log, q, reduction='none').sum(2).div_(x.size(1)).sum().div_(x.size(0))

    def exit_on_plateau_and_choose_best(self, accuracy):
        '''
        The function returns a pair (should_exit, is_candidate_for_best).

        The function sets this checkpoint as a candidate for best if either it is the first checkpoint
        for this LR or this checkpoint is better then the previous best.

        The function sets should_exit = True if the overfitting is observed or the metric
        doesn't improves for a predetermined number of epochs.
        '''

        should_exit = False
        is_candidate_for_best = False
        current_metric = round(accuracy, 4)
        is_best = current_metric <= self.prev_smooth_accuracy
        # if current metric less than an average
        if is_best and self.warmup_finished:
            self.iter_to_wait += 1
            if self.iter_to_wait >= self.train_patience:
                print(f"LOG:: The training should be stopped due to no improvements for {self.train_patience} epochs")
                should_exit = True
        elif is_best:
            self.ema_smooth(accuracy)
            self.iter_to_wait = 0

        if current_metric >= self.best_metric:
            self.best_metric = current_metric
            is_candidate_for_best = True

        return should_exit, is_candidate_for_best

    def ema_smooth(self, cur_metric, alpha=0.8):
        """Exponential smoothing with factor `alpha`.
        """
        assert 0 < alpha <= 1
        self.prev_smooth_accuracy = alpha * cur_metric + (1. - alpha) * self.prev_smooth_accuracy

    @torch.no_grad()
    def _evaluate(self, model, epoch, data_loader, model_name, topk, lr_finder):
        mAP, mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o = metrics.evaluate_multilabel_classification(data_loader,
                                                                                                      model,
                                                                                                      self.use_gpu)

        if self.writer is not None and not lr_finder:
            self.writer.add_scalar(f'Val/{model_name}/mAP', mAP, epoch + 1)

        if not lr_finder:
            print(f'** Results ({model_name}) **')
            print(f'mAP: {mAP:.2%}')
            print(f'P_O: {p_o:.2%}')
            print(f'R_O: {r_o:.2%}')
            print(f'F_O: {f_o:.2%}')
            print(f'mean_P_C: {mean_p_c:.2%}')
            print(f'mean_R_C: {mean_r_c:.2%}')
            print(f'mean_F_C: {mean_f_c:.2%}')

        return mAP
