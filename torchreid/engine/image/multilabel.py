from __future__ import absolute_import, division, print_function
from enum import auto

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from torchreid import metrics
from torchreid.losses import AsymmetricLoss, AMBinaryLoss
from torchreid.metrics.accuracy import accuracy
from torchreid.optim import SAM
from torchreid.utils import get_model_attr
from torchreid.engine import Engine

class MultilabelEngine(Engine):
    r"""Multilabel classification engine. It supports ASL, BCE and Angular margin loss with binary classification."""
    def __init__(self, datamanager, models, optimizers, schedulers, use_gpu, save_all_chkpts,
                 train_patience, early_stoping, lr_decay_factor, loss_name, label_smooth,
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
                        early_stoping=early_stoping,
                        should_freeze_aux_models=should_freeze_aux_models,
                        nncf_metainfo=nncf_metainfo,
                        compression_ctrl=compression_ctrl,
                        initial_lr=initial_lr,
                        lr_finder=lr_finder,
                        target_metric=target_metric,
                        use_ema_decay=use_ema_decay,
                        ema_decay=ema_decay)

        self.main_losses = nn.ModuleList()
        self.clip_grad = clip_grad
        num_classes = self.datamanager.num_train_pids
        if not isinstance(num_classes, (list, tuple)):
            num_classes = [num_classes]
        self.num_classes = num_classes

        for _ in enumerate(self.num_classes):
            if loss_name == 'asl':
                self.main_losses.append(AsymmetricLoss(
                    gamma_neg=asl_gamma_neg,
                    gamma_pos=asl_gamma_pos,
                    probability_margin=asl_p_m,
                    label_smooth=label_smooth,
                ))
            elif loss_name == 'bce':
                self.main_losses.append(AsymmetricLoss(
                    gamma_neg=0,
                    gamma_pos=0,
                    probability_margin=0,
                    label_smooth=label_smooth,
                ))
            elif loss_name == 'am_binary':
                self.main_losses.append(AMBinaryLoss(
                    m=m,
                    k=amb_k,
                    t=amb_t,
                    s=self.am_scale,
                    gamma_neg=asl_gamma_neg,
                    gamma_pos=asl_gamma_pos,
                    label_smooth=label_smooth,
                ))

        num_classes = self.datamanager.num_train_pids
        if not isinstance(num_classes, (list, tuple)):
            num_classes = [num_classes]
        self.num_classes = num_classes
        self.num_targets = len(self.num_classes)
        self.enable_sam = isinstance(self.optims[self.main_model_name], SAM)

        for model_name in self.get_model_names():
            assert isinstance(self.optims[model_name], SAM) == self.enable_sam, "SAM must be enabled \
                                                                                 for all models or none of them"
        self.scaler = GradScaler(enabled=mix_precision)
        self.mix_precision = mix_precision
        self.prev_smooth_accuracy = 0.

    def forward_backward(self, data):
        n_iter = self.epoch * self.num_batches + self.batch_idx

        train_records = self.parse_data_for_train(data, output_dict=True, use_gpu=self.use_gpu)
        imgs, obj_ids = train_records['img'], train_records['obj_id']

        model_names = self.get_model_names()
        num_models = len(model_names)
        steps = [1,2] if self.enable_sam and not self.lr_finder else [1]
        # forward pass
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
                    model=self.models[model_name], train_records=train_records,
                    imgs=imgs, obj_ids=obj_ids, n_iter=n_iter, model_name=model_name)

                avg_acc += model_avg_acc / float(num_models)
                total_loss += model_loss / float(num_models)
                loss_summary.update(model_loss_summary)

                for trg_id in range(self.num_targets):
                    if model_logits[trg_id] is not None:
                        out_logits[trg_id].append(model_logits[trg_id])
            model_num = len(model_names)
            # compute mutual loss
            if len(model_names) > 1:
                mutual_loss = torch.zeros([], dtype=imgs.dtype, device=imgs.device)
                for trg_id in range(self.num_targets):
                    if len(out_logits[trg_id]) <= 1:
                        continue
                    for model_i, logits_i in enumerate(out_logits[trg_id]):
                        probabilities_i = torch.sigmoid(logits_i)
                        kl_loss = 0
                        for model_j, logits_j in enumerate(out_logits[trg_id]):
                            if model_i != model_j:
                                probabilities_j = torch.sigmoid(logits_j)
                                kl_loss += self.kl_div_binary(probabilities_i, probabilities_j)
                        mutual_loss += kl_loss / (model_num - 1)
                        loss_summary['mutual_{}/{}'.format(trg_id, model_names[model_i])] = mutual_loss.item()

                should_turn_off_mutual_learning = self._should_turn_off_mutual_learning(self.epoch)
                coeff_mutual_learning = int(not should_turn_off_mutual_learning)

                total_loss += coeff_mutual_learning * mutual_loss
                if self.compression_ctrl:
                    compression_loss = self.compression_ctrl.loss()
                    loss_summary['compression_loss'] = compression_loss
                    total_loss += compression_loss

            # backward pass
            self.scaler.scale(total_loss).backward(retain_graph=False)
            for model_name in model_names:
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
                        loss_summary['loss'] = total_loss.item()
                        # skip second step  if overflow occurred
                        return loss_summary, avg_acc
                else:
                    assert self.enable_sam and step==2
                    if self.clip_grad == 0:
                        self.scaler.unscale_(self.optims[model_name])
                    self.optims[model_name].second_step()
                    self.scaler.update()

            loss_summary['loss'] = total_loss.item()

        return loss_summary, avg_acc

    def _single_model_losses(self, model, train_records, imgs, obj_ids, n_iter, model_name):
        with autocast(enabled=self.mix_precision):
            model_output = model(imgs)
            all_logits = self._parse_model_output(model_output)

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
                main_loss = self.main_losses[trg_id](trg_logits, trg_obj_ids, scale=self.scales[model_name])
                avg_acc += metrics.accuracy_multilabel(trg_logits, trg_obj_ids).item()
                loss_summary['main_{}/{}'.format(trg_id, model_name)] = main_loss.item()

                scaled_trg_logits = self.main_losses[trg_id].get_last_scale() * trg_logits
                out_logits.append(scaled_trg_logits)

                total_loss += main_loss
                num_trg_losses += 1

            total_loss /= float(num_trg_losses)
            avg_acc /= float(num_trg_losses)

        return total_loss, loss_summary, avg_acc, out_logits

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

    def _parse_model_output(self, model_output):
        all_logits = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        all_logits = all_logits if isinstance(all_logits, (tuple, list)) else [all_logits]

        return all_logits

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
        # if current metric less than an average
        if current_metric <= self.prev_smooth_accuracy:
            self.iter_to_wait += 1
            if self.iter_to_wait >= self.train_patience:
                print("LOG:: The training should be stopped due to no improvements for {} epochs".format(self.train_patience))
                should_exit = True
        else:
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
