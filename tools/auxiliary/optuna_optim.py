import os.path as osp
import os
import sys
import datetime
import time
import tempfile
from subprocess import run # nosec


import torch
import numpy as np
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from functools import partial
from ruamel.yaml import YAML



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
from torchreid.utils import (Logger, AverageMeter, check_isfile, set_random_seed, load_pretrained_weights)


def read_yaml_config(yaml: YAML, config_path: str):
    yaml.default_flow_style = True
    with open(config_path, 'r') as f:
        cfg = yaml.load(f)
    return cfg

def run_training(args, params):
    yaml = YAML()
    path_to_main = osp.relpath(path='/tools/main.py', start=args.root)
    cfg = read_yaml_config(yaml, args.config)
    fd, tmp_path_to_cfg = tempfile.mkstemp(suffix='.yml')
    try:
        with os.fdopen(fd, 'w') as tmp:
            # do stuff with temp file
            yaml.default_flow_style = True
            yaml.dump(cfg, tmp)
            tmp.close()

        # run training
        run(['python', f'{path_to_main}',
            '--config', f'{tmp_path_to_cfg}',
            '--gpu-num', f'{int(args.gpu_num)}'],
            shell=False)
    finally:
        os.remove(tmp_path_to_cfg)

def finish_process(study, args):
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

    if not args.disable_running_training:
        params = trial.params
        run_training(args, params)


def objective(cfg, args, trial):
    # Generate the trials.
    # g_ = trial.suggest_int("g_", 1, 7)
    # asl_pm = trial.suggest_float("asl_pm", 0, 0.5)
    # m = trial.suggest_float("m", 0.01, 0.7)
    # s = trial.suggest_int("s", 5, 60)
    lr = trial.suggest_float("train.lr", 0.001, 0.5)
    # t = trial.suggest_int("t", 1, 7)
    # cfg.loss.softmax.m = m
    # cfg.loss.softmax.s = s
    # cfg.loss.asl.p_m = asl_pm
    # cfg.loss.am_binary.amb_t = t
    cfg.train.lr = lr

    # geterate damanager
    num_aux_models = len(cfg.mutual_learning.aux_configs)
    datamanager = build_datamanager(cfg, args.classes)

    # build the model.
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
    engine.max_epoch = args.epochs
    print(f"\nnext trial with [lr: {lr}]")

    for engine.epoch in range(args.epochs):
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

        trial.report(obj, engine.epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        should_exit, _ = engine.exit_on_plateau_and_choose_best(top1, smooth_top1)
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

    args = parser.parse_args()
    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available() and args.gpu_num > 0
    if args.config_file:
        merge_from_files_with_base(cfg, args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)

    set_random_seed(cfg.train.seed, cfg.train.deterministic)

    log_name = 'optuna.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    sampler = TPESampler(n_startup_trials=5, seed=True)
    study = optuna.create_study(study_name='classification task', direction="maximize", sampler=sampler)
    objective_partial = partial(objective, cfg, args)
    try:
        start_time = time.time()
        study.optimize(objective_partial, n_trials=cfg.lr_finder.n_trials, timeout=None)
        elapsed = round(time.time() - start_time)
        print(f"--- optimization is finished: {datetime.timedelta(seconds=elapsed)} ---")

    except KeyboardInterrupt:
        finish_process(study, args)

    else:
        finish_process(study, args)

if __name__ == "__main__":
    main()
