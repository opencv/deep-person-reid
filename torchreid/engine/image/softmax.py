from __future__ import absolute_import, division, print_function

import torch

from torchreid import metrics
from torchreid.losses import AsymmetricLoss, AMBinaryLoss
from ..engine import Engine


class MultilabelEngine(Engine):
    r"""Multilabel classification engine. It supports ASL, BCE and Angular margin loss with binary classification."""

    def __init__(self, datamanager, models, optimizers, reg_cfg, metric_cfg, schedulers=None, use_gpu=False, save_chkpt=True,
                 train_patience=10, early_stoping = False, lr_decay_factor = 1000, softmax_type='softmax', label_smooth=False,
                 margin_type='cos', epsilon=0.1, aug_type=None, decay_power=3, alpha=1., size=(224, 224), lr_finder=None, max_soft=0.0,
                 reformulate=False, aug_prob=1., conf_penalty=False, pr_product=False, m=0.35, s=10, compute_s=False, end_s=None,
                 duration_s=None, skip_steps_s=None, enable_masks=False, adaptive_margins=False, class_weighting=False,
                 attr_cfg=None, base_num_classes=-1, symmetric_ce=False, mix_weight=1.0, enable_rsc=False, enable_sam=False,
                 should_freeze_aux_models=False, nncf_metainfo=None, initial_lr=None, use_ema_decay=False, ema_decay=0.999,
                 asl_gamma_pos=0.0, asl_gamma_neg=4.0, asl_p_m=0.05):
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

        self.main_loss = CrossEntropyLoss(
            use_gpu=self.use_gpu,
            label_smooth=label_smooth,
            conf_penalty=conf_penalty
        )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)
        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        model_names = self.get_model_names()
        num_models = len(model_names)

        all_losses, all_logits = [], []
        loss_summary = dict()

        for model_name in model_names:
            self.optims[model_name].zero_grad()

            logits = self.models[model_name](imgs)
            all_logits.append(logits)

            loss = self.compute_loss(self.main_real_loss, logits, pids)
            all_losses.append(loss / float(num_models))

            loss_summary['{}/main'.format(model_name)] = loss.item()
            loss_summary['{}/acc'.format(model_name)] = metrics.accuracy(logits, pids)[0].item()

        if len(all_logits) > 1:
            with torch.no_grad():
                trg_probs = torch.softmax(torch.stack(all_logits), dim=2).mean(dim=0)

            mix_loss = 0.0
            for logits in all_logits:
                log_probs = torch.log_softmax(logits, dim=1)
                mix_loss += (trg_probs * log_probs).mean().neg()
            mix_loss /= float(len(all_logits))

            all_losses.append(mix_loss)
            loss_summary['mix'] = mix_loss.item()

        total_loss = sum(all_losses)
        total_loss.backward()

        for model_name in model_names:
            self.optims[model_name].step()

        loss_summary['loss'] = total_loss.item()

        return loss_summary
