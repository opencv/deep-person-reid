from torchreid.optim import lr_finder
from torchreid.engine import (ImageAMSoftmaxEngine, ImageContrastiveEngine,
                              ImageTripletEngine, MultilabelEngine)


def build_engine(cfg, datamanager, model, optimizer, scheduler,
                 should_freeze_aux_models=False,
                 nncf_metainfo=None,
                 initial_lr=None):
    if should_freeze_aux_models or nncf_metainfo:
        if cfg.loss.name not in ['softmax', 'am_softmax']:
            raise NotImplementedError('Freezing of aux models or NNCF compression are supported only for '
                                      'softmax and am_softmax losses for data.type = image')
    initial_lr = initial_lr if initial_lr else cfg.train.lr
    classification_params = dict(
            datamanager=datamanager,
            models=model,
            optimizers=optimizer,
            reg_cfg=cfg.reg,
            metric_cfg=cfg.metric_losses,
            schedulers=scheduler,
            use_gpu=cfg.use_gpu,
            save_chkpt = cfg.model.save_chkpt,
            lr_finder = cfg.lr_finder.enable,
            train_patience = cfg.train.train_patience,
            early_stoping = cfg.train.early_stoping,
            lr_decay_factor = cfg.train.lr_decay_factor,
            conf_penalty=cfg.loss.softmax.conf_penalty,
            label_smooth=cfg.loss.softmax.label_smooth,
            aug_type=cfg.loss.softmax.augmentations.aug_type,
            aug_prob=cfg.loss.softmax.augmentations.aug_prob,
            decay_power=cfg.loss.softmax.augmentations.fmix.decay_power,
            alpha=cfg.loss.softmax.augmentations.alpha,
            size=(cfg.data.height, cfg.data.width),
            max_soft=cfg.loss.softmax.augmentations.fmix.max_soft,
            reformulate=cfg.loss.softmax.augmentations.fmix.reformulate,
            pr_product=cfg.loss.softmax.pr_product,
            loss_name=cfg.loss.name,
            epsilon=0.1,
            m=cfg.loss.softmax.m,
            s=cfg.loss.softmax.s,
            compute_s=cfg.loss.softmax.compute_s,
            margin_type=cfg.loss.softmax.margin_type,
            end_s=cfg.loss.softmax.end_s,
            duration_s=cfg.loss.softmax.duration_s,
            skip_steps_s=cfg.loss.softmax.skip_steps_s,
            enable_masks=cfg.data.enable_masks,
            adaptive_margins=cfg.loss.softmax.adaptive_margins,
            class_weighting=cfg.loss.softmax.class_weighting,
            attr_cfg=cfg.attr_loss,
            base_num_classes=cfg.loss.softmax.base_num_classes,
            symmetric_ce=cfg.loss.softmax.symmetric_ce,
            mix_weight=cfg.mixing_loss.enable * cfg.mixing_loss.weight,
            enable_rsc=cfg.model.self_challenging_cfg.enable,
            enable_sam=cfg.sam.enable,
            should_freeze_aux_models=should_freeze_aux_models,
            nncf_metainfo=nncf_metainfo,
            initial_lr=initial_lr,
            use_ema_decay=cfg.train.ema.enable,
            ema_decay=cfg.train.ema.ema_decay,
            asl_gamma_neg=cfg.loss.asl.gamma_neg,
            asl_gamma_pos=cfg.loss.asl.gamma_pos,
            asl_p_m=cfg.loss.asl.p_m,
            sym_adjustment=cfg.loss.am_binary.sym_adjustment,
            auto_balance=cfg.loss.am_binary.auto_balance,
            amb_k = cfg.loss.am_binary.amb_k,
            amb_t=cfg.loss.am_binary.amb_t)

    if cfg.loss.name in ['softmax', 'am_softmax']:
        engine = ImageAMSoftmaxEngine(
            **classification_params
        )
    elif cfg.loss.name in ['asl', 'bce', 'am_binary']:
        engine = MultilabelEngine(
            **classification_params
        )

    elif cfg.loss.name == 'contrastive':
        engine = ImageContrastiveEngine(
            datamanager,
            model,
            optimizer=optimizer,
            reg_cfg=cfg.reg,
            scheduler=scheduler,
            use_gpu=cfg.use_gpu,
            s=cfg.loss.softmax.s,
            end_s=cfg.loss.softmax.end_s,
            duration_s=cfg.loss.softmax.duration_s,
            skip_steps_s=cfg.loss.softmax.skip_steps_s,
        )
    else:
        engine = ImageTripletEngine(
            datamanager,
            model,
            optimizer=optimizer,
            margin=cfg.loss.triplet.margin,
            weight_t=cfg.loss.triplet.weight_t,
            weight_x=cfg.loss.triplet.weight_x,
            scheduler=scheduler,
            use_gpu=cfg.use_gpu,
            label_smooth=cfg.loss.softmax.label_smooth,
            conf_penalty=cfg.loss.softmax.conf_penalty
        )

    return engine
