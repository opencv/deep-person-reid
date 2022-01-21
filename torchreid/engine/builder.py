# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .image import (ImageAMSoftmaxEngine, MultilabelEngine, MultiheadEngine)


def build_engine(cfg, datamanager, model, optimizer, scheduler,
                 should_freeze_aux_models=False,
                 nncf_metainfo=None,
                 compression_ctrl=None,
                 initial_lr=None):
    if should_freeze_aux_models or nncf_metainfo:
        if cfg.loss.name not in ['softmax', 'am_softmax', 'am_binary']:
            raise NotImplementedError('Freezing of aux models or NNCF compression are supported only for '
                                      'softmax, am_softmax and am_binary losses for data.type = image')
    initial_lr = initial_lr if initial_lr else cfg.train.lr
    classification_params = dict(
            datamanager=datamanager,
            models=model,
            optimizers=optimizer,
            schedulers=scheduler,
            use_gpu=cfg.use_gpu,
            save_all_chkpts = cfg.model.save_all_chkpts,
            lr_finder = cfg.lr_finder.enable,
            train_patience = cfg.train.train_patience,
            early_stopping = cfg.train.early_stopping,
            lr_decay_factor = cfg.train.lr_decay_factor,
            conf_penalty=cfg.loss.softmax.conf_penalty,
            label_smooth=cfg.loss.softmax.label_smooth,
            aug_type=cfg.loss.softmax.augmentations.aug_type,
            aug_prob=cfg.loss.softmax.augmentations.aug_prob,
            decay_power=cfg.loss.softmax.augmentations.fmix.decay_power,
            alpha=cfg.loss.softmax.augmentations.alpha,
            size=(cfg.data.height, cfg.data.width),
            pr_product=cfg.loss.softmax.pr_product,
            loss_name=cfg.loss.name,
            clip_grad=cfg.train.clip_grad,
            m=cfg.loss.softmax.m,
            s=cfg.loss.softmax.s,
            compute_s=cfg.loss.softmax.compute_s,
            margin_type=cfg.loss.softmax.margin_type,
            symmetric_ce=cfg.loss.softmax.symmetric_ce,
            enable_rsc=cfg.model.self_challenging_cfg.enable,
            should_freeze_aux_models=should_freeze_aux_models,
            nncf_metainfo=nncf_metainfo,
            compression_ctrl=compression_ctrl,
            initial_lr=initial_lr,
            target_metric=cfg.train.target_metric,
            use_ema_decay=cfg.train.ema.enable,
            ema_decay=cfg.train.ema.ema_decay,
            asl_gamma_neg=cfg.loss.asl.gamma_neg,
            asl_gamma_pos=cfg.loss.asl.gamma_pos,
            asl_p_m=cfg.loss.asl.p_m,
            amb_k = cfg.loss.am_binary.amb_k,
            amb_t=cfg.loss.am_binary.amb_t,
            mix_precision=cfg.train.mix_precision)
    cfg.model.type

    if cfg.model.type == 'classification':
        engine = ImageAMSoftmaxEngine(
            **classification_params
        )
    elif cfg.model.type == 'multilabel':
        engine = MultilabelEngine(
            **classification_params
        )
    elif cfg.model.type == 'multihead':
        engine = MultiheadEngine(
            **classification_params
        )

    return engine
