from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
from torch import nn

from torchreid import metrics
from torchreid.losses import AsymmetricLoss, AMBinaryLoss
from torchreid.optim import SAM
from ..engine import Engine

class MultilabelEngine(Engine):
    r"""Multilabel classification engine. It supports ASL, BCE and Angular margin loss with binary classification."""

    def __init__(self, datamanager, models, optimizers, schedulers, use_gpu, save_chkpt,
                 train_patience, early_stoping, lr_decay_factor, loss_name, label_smooth,
                 lr_finder, m, s, sym_adjustment, auto_balance, amb_k, amb_t,
                 enable_sam, should_freeze_aux_models, nncf_metainfo, initial_lr,
                 use_ema_decay, ema_decay, asl_gamma_pos, asl_gamma_neg, asl_p_m, **kwargs):

        super().__init__(datamanager,
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
                        lr_finder=lr_finder,
                        use_ema_decay=use_ema_decay,
                        ema_decay=ema_decay)
        self.main_losses = nn.ModuleList()
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
                ))
            elif loss_name == 'bce':
                self.main_losses.append(AsymmetricLoss(
                    gamma_neg=0,
                    gamma_pos=0,
                    probability_margin=asl_p_m,
                ))
            elif loss_name == 'am_binary':
                self.main_losses.append(AMBinaryLoss(
                    m=m,
                    k=amb_k,
                    t=amb_t,
                    s=s,
                    sym_adjustment=sym_adjustment,
                    auto_balance=auto_balance,
                    gamma_neg=asl_gamma_neg,
                    gamma_pos=asl_gamma_pos,
                ))

        num_classes = self.datamanager.num_train_pids
        if not isinstance(num_classes, (list, tuple)):
            num_classes = [num_classes]
        self.num_classes = num_classes
        self.num_targets = len(self.num_classes)
        self.enable_sam = enable_sam


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
                    self.models[model_name], train_records, imgs, obj_ids, n_iter, model_name)
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
            # backward pass
            total_loss.backward(retain_graph=False)

            for model_name in model_names:
                if not isinstance(self.optims[model_name], SAM) and step == 1:
                    self.optims[model_name].step()
                elif step == 1:
                    assert isinstance(self.optims[model_name], SAM)
                    self.optims[model_name].first_step()
                else:
                    assert isinstance(self.optims[model_name], SAM) and step==2
                    self.optims[model_name].second_step()

            loss_summary['loss'] = total_loss.item()

        return loss_summary, avg_acc

    def _single_model_losses(self, model, train_records, imgs, obj_ids, n_iter, model_name):
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
            main_loss = self.main_losses[trg_id](trg_logits, trg_obj_ids)
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
        p_log = torch.log(p)
        # mean over dists for each sample, than do the batchmean reduction
        return F.kl_div(p_log, q, reduction='none').mean(2).sum().div_(x.size(0))

    def _parse_model_output(self, model_output):
        all_logits = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        all_logits = all_logits if isinstance(all_logits, (tuple, list)) else [all_logits]

        return all_logits