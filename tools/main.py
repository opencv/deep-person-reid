# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
import sys
import time

from torch.utils.tensorboard import SummaryWriter
import torch
from ptflops import get_model_complexity_info

from scripts.default_config import (get_default_config,
                                    lr_scheduler_kwargs, model_kwargs,
                                    optimizer_kwargs,
                                    merge_from_files_with_base)
from scripts.script_utils import (build_base_argparser, reset_config,
                                  check_classification_classes,
                                  build_datamanager,
                                  put_main_model_on_the_device)

import torchreid
from torchreid.apis.training import run_lr_finder, run_training
from torchreid.utils import (Logger, check_isfile, resume_from_checkpoint,
                             set_random_seed, load_pretrained_weights, get_git_revision)
from torchreid.integration.nncf.compression_script_utils import (make_nncf_changes_in_config,
                                                                 make_nncf_changes_in_training)


def main():
    parser = build_base_argparser()
    parser.add_argument('-e', '--auxiliary-models-cfg', type=str, nargs='*', default='',
                        help='path to extra config files')
    parser.add_argument('--split-models', action='store_true',
                        help='whether to split models on own gpu')
    parser.add_argument('--enable_quantization', action='store_true',
                        help='Enable NNCF quantization algorithm')
    parser.add_argument('--enable_pruning', action='store_true',
                        help='Enable NNCF pruning algorithm')
    parser.add_argument('--aux-config-opts', nargs='+', default=None,
                        help='Modify aux config options using the command-line')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available() and args.gpu_num > 0
    if args.config_file:
        merge_from_files_with_base(cfg, args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)

    is_nncf_used = args.enable_quantization or args.enable_pruning
    if is_nncf_used:
        print('Using NNCF -- making NNCF changes in config')
        cfg = make_nncf_changes_in_config(cfg,
                                          args.enable_quantization,
                                          args.enable_pruning,
                                          args.opts)

    set_random_seed(cfg.train.seed, cfg.train.deterministic)

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))
    sha_commit, branch_name = get_git_revision()
    print(f'HEAD is: {branch_name}')
    print(f'commit SHA is: {sha_commit}\n')
    print(f'Show configuration\n{cfg}\n')

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    num_aux_models = len(cfg.mutual_learning.aux_configs)
    datamanager = build_datamanager(cfg, args.classes)
    num_train_classes = datamanager.num_train_ids

    print(f'Building main model: {cfg.model.name}')
    model = torchreid.models.build_model(**model_kwargs(cfg, num_train_classes))
    macs, num_params = get_model_complexity_info(model, (3, cfg.data.height, cfg.data.width),
                                                 as_strings=False, verbose=False, print_per_layer_stat=False)
    print(f'Main model complexity: params={num_params:,} flops={macs * 2:,}')

    aux_lr = cfg.train.lr # placeholder, needed for aux models, may be filled by nncf part below
    if is_nncf_used:
        print('Begin making NNCF changes in model')
        if cfg.use_gpu:
            model.cuda()

        compression_ctrl, model, cfg, aux_lr, nncf_metainfo = \
            make_nncf_changes_in_training(model, cfg,
                                          args.classes,
                                          args.opts)

        should_freeze_aux_models = True
        print(f'should_freeze_aux_models = {should_freeze_aux_models}')
        print('End making NNCF changes in model')
    else:
        compression_ctrl = None
        should_freeze_aux_models = False
        nncf_metainfo = None
    # creating optimizer and scheduler -- it should be done after NNCF part, since
    # NNCF could change some parameters
    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))

    if cfg.lr_finder.enable and not cfg.model.resume:
        scheduler = None
    else:
        scheduler = torchreid.optim.build_lr_scheduler(optimizer=optimizer,
                                                       num_iter=datamanager.num_iter,
                                                       **lr_scheduler_kwargs(cfg))
    # Loading model (and optimizer and scheduler in case of resuming training).
    # Note that if NNCF is used, loading is done inside NNCF part, so loading here is not required.
    if cfg.model.resume and check_isfile(cfg.model.resume) and not is_nncf_used:
        device_ = 'cuda' if cfg.use_gpu else 'cpu'
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler, device=device_
            )
    elif cfg.model.load_weights and not is_nncf_used:
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.model.type == 'classification':
        check_classification_classes(model, datamanager, args.classes, test_only=cfg.test.evaluate)

    model, extra_device_ids = put_main_model_on_the_device(model, cfg.use_gpu, args.gpu_num,
                                                           num_aux_models, args.split_models)

    if cfg.lr_finder.enable and not cfg.test.evaluate and not cfg.model.resume:
        aux_lr, model, optimizer, scheduler = run_lr_finder(cfg, datamanager, model,
                                                            optimizer, scheduler, args.classes,
                                                            rebuild_model=True,
                                                            gpu_num=args.gpu_num,
                                                            split_models=args.split_models)

    log_dir = cfg.data.tb_log_dir if cfg.data.tb_log_dir else cfg.data.save_dir
    run_training(cfg, datamanager, model, optimizer, scheduler, extra_device_ids,
                 aux_lr, tb_writer=SummaryWriter(log_dir=log_dir),
                 aux_config_opts=args.aux_config_opts,
                 should_freeze_aux_models=should_freeze_aux_models,
                 nncf_metainfo=nncf_metainfo,
                 compression_ctrl=compression_ctrl)


if __name__ == '__main__':
    main()
