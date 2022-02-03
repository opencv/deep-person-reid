import os.path as osp
import os
import sys
import datetime
import time
import tempfile
from subprocess import run # nosec
import copy

import torch
import numpy as np
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from functools import partial
from ruamel.yaml import YAML
import json



from scripts.default_config import (get_default_config,
                                    lr_scheduler_kwargs, model_kwargs,
                                    optimizer_kwargs,
                                    merge_from_files_with_base)
from scripts.script_utils import (build_base_argparser, reset_config,
                                  check_classification_classes,
                                  build_datamanager, build_auxiliary_model,
                                  put_main_model_on_the_device)

import torchreid
from torchreid.engine import build_engine
from torchreid.utils import (Logger, AverageMeter, check_isfile, set_random_seed, load_pretrained_weights, mkdir_if_missing)


def read_json_cfg(cfg):
    with open(cfg) as f:
        config = json.load(f)
    return config

def make_change_in_cfg(main_cfg, field_name, value):
    keys = field_name.split(".")
    set_attr_dict(main_cfg, keys, value)
    return main_cfg

def set_attr_dict(dict_, keys, val, i=0):
    i = i if i else 0
    if not isinstance(dict_[keys[i]], dict):
        dict_[keys[i]] = val
    else:
        set_attr_dict(dict_[keys[i]], keys, val, i+1)

def read_yaml_config(yaml: YAML, config_path: str):
    yaml.default_flow_style = True
    with open(config_path, 'r') as f:
        cfg = yaml.load(f)
    return cfg

def finish_process(study):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return trial.params


def run_training(cfg, opt_cfg, args, trial):
    # define max epochs
    max_epochs = opt_cfg["epochs"] if opt_cfg else cfg['train']['max_epoch']

    if opt_cfg is not None:
        ### READING A JSON OPTIMIZATION CONFIG ####
        log_message = "\nnext trial with [ "
        if 'float' in opt_cfg:
            for param in opt_cfg['float']:
                field_name = param['name']
                step = param['step'] if param['step'] > 0 else None
                val = trial.suggest_float(field_name, *param['range'], step=step)
                cfg = make_change_in_cfg(cfg, field_name, val)
                log_message += f'{field_name} : {val}; '

        if 'int' in opt_cfg:
            for param in opt_cfg['int']:
                field_name = param['name']
                step = param['step'] if param['step'] > 0 else None
                val = trial.suggest_int(field_name, *param['range'], step=step)
                cfg = make_change_in_cfg(cfg, field_name, val)
                log_message += f'{field_name} : {val}; '

        if 'categorical' in opt_cfg:
            for param in opt_cfg['categorical']:
                field_name = param['name']
                val = trial.suggest_categorical(field_name, param['range'])
                cfg = make_change_in_cfg(cfg, field_name, val)
                log_message += f'{field_name} : {val}; '

        print(log_message + ']')

    # generate datamanager
    num_aux_models = len(cfg.mutual_learning.aux_configs)
    datamanager = build_datamanager(cfg, args.classes)

    # build the model
    num_train_classes = datamanager.num_train_ids
    print('Building main model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(**model_kwargs(cfg, num_train_classes))
    aux_lr = cfg.train.lr # placeholder, needed for aux models, may be filled by nncf part below
    compression_ctrl = None
    should_freeze_aux_models = False
    nncf_metainfo = None
    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(optimizer=optimizer,
                                                       num_iter=datamanager.num_iter,
                                                       **lr_scheduler_kwargs(cfg))
    # Loading model (and optimizer and scheduler in case of resuming training).
    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    check_classification_classes(model, datamanager, args.classes, test_only=cfg.test.evaluate)
    model, extra_device_ids = put_main_model_on_the_device(model, cfg.use_gpu, args.gpu_num, num_aux_models, args.split_models)
    num_aux_models = len(cfg.mutual_learning.aux_configs)
    if num_aux_models > 0:
        print(f'Enabled mutual learning between {len(cfg.mutual_learning.aux_configs) + 1} models.')

        models, optimizers, schedulers = [model], [optimizer], [scheduler]
        for config_file, device_ids in zip(cfg.mutual_learning.aux_configs, extra_device_ids):
            aux_model, aux_optimizer, aux_scheduler = build_auxiliary_model(
                config_file, num_train_classes, cfg.use_gpu, device_ids, num_iter=datamanager.num_iter,
                lr=aux_lr, aux_config_opts=args.aux_config_opts)

            models.append(aux_model)
            optimizers.append(aux_optimizer)
            schedulers.append(aux_scheduler)
    else:
        models, optimizers, schedulers = model, optimizer, scheduler
    print(f'Building {cfg.loss.name}-engine')
    engine = build_engine(cfg, datamanager, models, optimizers, schedulers,
                          should_freeze_aux_models=should_freeze_aux_models,
                          nncf_metainfo=nncf_metainfo,
                          compression_ctrl=compression_ctrl,
                          initial_lr=aux_lr)
    test_acc = AverageMeter()
    obj = 0
    engine.start_epoch = 0
    engine.max_epoch = max_epochs

    for engine.epoch in range(max_epochs):
        np.random.seed(cfg.train.seed + engine.epoch)
        avg_loss = engine.train(
            print_freq=20000,
            fixbase_epoch=0,
            open_layers=None,
            lr_finder=False,
            perf_monitor=None,
            stop_callback=None
        )

        top1, _ = engine.test(
                engine.epoch,
                lr_finder=False,
                )

        test_acc.update(top1)
        smooth_top1 = test_acc.avg
        target_metric = smooth_top1 if engine.target_metric == 'test_acc' else avg_loss

        obj = top1
        if not engine.per_batch_annealing:
            engine.update_lr(output_avg_metric = target_metric)

        if trial is not None:
            trial.report(obj, engine.epoch)

        # Handle pruning based on the intermediate value.
        if trial is not None and trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        should_exit, _ = engine.exit_on_plateau_and_choose_best(top1)
        should_exit = engine.early_stopping and should_exit
        if should_exit:
            break

    return obj

def main():
    # parse arguments
    parser = build_base_argparser()
    parser.add_argument('-e', '--auxiliary-models-cfg', type=str, nargs='*', default='',
                        help='path to extra config files')
    parser.add_argument('--split-models', action='store_true',
                        help='whether to split models on own gpu')
    parser.add_argument('--aux-config-opts', nargs='+', default=None,
                        help='Modify aux config options using the command-line')
    parser.add_argument('--epochs', default=10, type=int, help='amount of the epochs')
    parser.add_argument('-drt', '--disable_running_training', default=False,
                        action='store_true', help='disable full training after optimization')
    parser.add_argument('--opt-configs', nargs="+", default=['./opt_configs/example.json'],
                        help='path to optimization config')

    optimized_params = None
    args = parser.parse_args()
    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available() and args.gpu_num > 0
    if args.config_file:
        merge_from_files_with_base(cfg, args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    logger = Logger(None)
    sys.stdout = logger

    for i, optim_cfg in enumerate(args.opt_configs):
        if logger.file is not None:
            logger.file.close()
        set_random_seed(cfg.train.seed, cfg.train.deterministic)
        opt_cfg = read_json_cfg(optim_cfg)
        strftime = time.strftime('-%Y-%m-%d-%H-%M-%S')
        log_file = osp.join(cfg.data.save_dir, f'optuna_{i}{strftime}.log')
        mkdir_if_missing(osp.dirname(log_file))
        logger.file = open(log_file, 'w')
        if optimized_params:
            for name, value in optimized_params.items():
                cfg = make_change_in_cfg(cfg, name, value)

        print('Show configuration\n{}\n'.format(cfg))

        if cfg.use_gpu:
            torch.backends.cudnn.benchmark = True

        sampler = TPESampler(n_startup_trials=5, seed=True)
        study = optuna.create_study(study_name='classification task', direction="maximize", sampler=sampler)
        objective_partial = partial(run_training, cfg, opt_cfg, args)
        try:
            start_time = time.time()
            study.optimize(objective_partial, n_trials=opt_cfg['n_trials'], timeout=None)
            elapsed = round(time.time() - start_time)
            print(f"SUCCESS:: --- optimization is finished: {datetime.timedelta(seconds=elapsed)} ---")

        except KeyboardInterrupt:
            optimized_params = finish_process(study)

        except: # there is some general exception (some error in training)
            print("ERROR:: --- optimization is failed! ---")
            exit()

        else:
            optimized_params = finish_process(study)

    if not args.disable_running_training:
        assert optimized_params, "There is no optimized hyperparameter!"
        del study
        del objective_partial
        logger.file.close()
        for name, value in optimized_params.items():
            cfg = make_change_in_cfg(cfg, name, value)
        set_random_seed(cfg.train.seed, cfg.train.deterministic)
        strftime = time.strftime('-%Y-%m-%d-%H-%M-%S')
        log_file = osp.join(cfg.data.save_dir, f'train{strftime}.log')
        mkdir_if_missing(osp.dirname(log_file))
        logger.file = open(log_file, 'w')
        print('Show configuration\n{}\n'.format(cfg))
        run_training(cfg, opt_cfg=None, args=args, trial=None)

if __name__ == "__main__":
    main()
