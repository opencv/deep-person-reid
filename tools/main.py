import argparse
import os.path as osp
import sys
import time

from torch.utils.tensorboard import SummaryWriter
import torch
from scripts.default_config import (engine_run_kwargs, get_default_config,
                                    lr_finder_run_kwargs,
                                    lr_scheduler_kwargs, model_kwargs,
                                    optimizer_kwargs,
                                    merge_from_files_with_base)
from scripts.script_utils import (build_base_argparser, reset_config,
                                  check_classification_classes,
                                  build_datamanager, build_auxiliary_model,
                                  is_config_parameter_set_from_command_line,
                                  put_main_model_on_the_device)

import torchreid
from torchreid.engine import build_engine, get_initial_lr_from_checkpoint
from torchreid.utils import (Logger, check_isfile, collect_env_info,
                             compute_model_complexity, resume_from_checkpoint,
                             set_random_seed, load_pretrained_weights)
from torchreid.optim import LrFinder
from torchreid.integration.nncf.compression import is_checkpoint_nncf
from torchreid.integration.nncf.compression_script_utils import (get_nncf_changes_in_aux_training_config,
                                                                 make_nncf_changes_in_training,
                                                                 make_nncf_changes_in_main_training_config)


def main():
    parser = build_base_argparser()
    parser.add_argument('-e', '--auxiliary-models-cfg', type=str, nargs='*', default='',
                        help='path to extra config files')
    parser.add_argument('--split-models', action='store_true',
                        help='whether to split models on own gpu')

    parser.add_argument('--nncf', action='store_true',
                        help='If nncf compression should be used')
    parser.add_argument('--aux-config-opts', nargs='+', default=None,
                        help='Modify aux config options using the command-line')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available() and args.gpu_num > 0
    if args.config_file:
        merge_from_files_with_base(cfg, args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)

    is_nncf_used = args.nncf or cfg.nncf.enable or is_checkpoint_nncf(cfg.model.load_weights)
    if is_nncf_used:
        print(f'Using NNCF -- making NNCF changes in config')
        cfg = make_nncf_changes_in_main_training_config(cfg, args.opts)
        nncf_changes_in_aux_train_config = get_nncf_changes_in_aux_training_config(cfg)
    else:
        nncf_changes_in_aux_train_config = None

    set_random_seed(cfg.train.seed, cfg.train.deterministic)

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    num_aux_models = len(cfg.mutual_learning.aux_configs)
    datamanager = build_datamanager(cfg, args.classes)
    num_train_classes = datamanager.num_train_pids

    print('Building main model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(**model_kwargs(cfg, num_train_classes))
    num_params, flops = compute_model_complexity(model, (1, 3, cfg.data.height, cfg.data.width))
    print('Main model complexity: params={:,} flops={:,}'.format(num_params, flops))

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
        scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    # Loading model (and optimizer and scheduler in case of resuming training).
    # Note that if NNCF is used, loading is done inside NNCF part, so loading here is not required.
    if cfg.model.resume and check_isfile(cfg.model.resume) and not is_nncf_used:
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
            )
    elif cfg.model.load_weights and check_isfile(cfg.model.load_weights) and not is_nncf_used:
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.model.type == 'classification':
        check_classification_classes(model, datamanager, args.classes, test_only=cfg.test.evaluate)

    model, extra_device_ids = put_main_model_on_the_device(model, cfg.use_gpu, args.gpu_num, num_aux_models, args.split_models)

    if cfg.lr_finder.enable and not cfg.test.evaluate and not cfg.model.resume:
        if num_aux_models > 0:
            print("Mutual learning is enabled. Learning rate will be estimated for the main model only.")
        assert not is_nncf_used, "lr finder is incompatible with nncf"

        # build  engine for learning rate estimation
        engine = build_engine(cfg, datamanager, model, optimizer, scheduler, initial_lr=aux_lr)
        lr_finder = LrFinder(engine=engine, **lr_finder_run_kwargs(cfg))
        aux_lr = lr_finder.process()

        print(f"Estimated learning rate: {aux_lr}")
        if cfg.lr_finder.stop_after:
            print("Finding learning rate finished. Terminate the training process")
            exit()

        # reload all parts of the training
        # we do not check classification parameters
        # and do not get num_train_classes the second time
        # since it's done above and lr finder cannot change parameters of the datasets
        cfg.train.lr = aux_lr
        cfg.lr_finder.enable = False
        set_random_seed(cfg.train.seed, cfg.train.deterministic)
        datamanager = build_datamanager(cfg, args.classes)
        model = torchreid.models.build_model(**model_kwargs(cfg, num_train_classes))
        model, _ = put_main_model_on_the_device(model, cfg.use_gpu, args.gpu_num, num_aux_models, args.split_models)
        optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
        scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    if num_aux_models > 0:
        print('Enabled mutual learning between {} models.'.format(len(cfg.mutual_learning.aux_configs) + 1))

        models, optimizers, schedulers = [model], [optimizer], [scheduler]
        for config_file, device_ids in zip(cfg.mutual_learning.aux_configs, extra_device_ids):
            aux_model, aux_optimizer, aux_scheduler = build_auxiliary_model(
                config_file, num_train_classes, cfg.use_gpu, device_ids, lr=aux_lr,
                nncf_aux_config_file=nncf_changes_in_aux_train_config,
                aux_config_opts=args.aux_config_opts
            )

            models.append(aux_model)
            optimizers.append(aux_optimizer)
            schedulers.append(aux_scheduler)
    else:
        models, optimizers, schedulers = model, optimizer, scheduler
    print('Building {}-engine'.format(cfg.loss.name))
    engine = build_engine(cfg, datamanager, models, optimizers, schedulers,
                          should_freeze_aux_models=should_freeze_aux_models,
                          nncf_metainfo=nncf_metainfo,
                          initial_lr=aux_lr)

    log_dir = cfg.data.tb_log_dir if cfg.data.tb_log_dir else cfg.data.save_dir
    engine.run(**engine_run_kwargs(cfg), compression_ctrl=compression_ctrl,
               tb_writer=SummaryWriter(log_dir=log_dir))


if __name__ == '__main__':
    main()
