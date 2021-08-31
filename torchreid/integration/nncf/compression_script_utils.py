from pprint import pformat

from torchreid.engine import get_initial_lr_from_checkpoint
from torchreid.utils import check_isfile, load_pretrained_weights, read_yaml

from scripts.default_config import merge_from_files_with_base
from scripts.script_utils import build_datamanager, is_config_parameter_set_from_command_line

from .compression import (is_checkpoint_nncf, wrap_nncf_model)

def get_coeff_decrease_lr_for_nncf(nncf_training_config):
    if nncf_training_config and nncf_training_config.get('coeff_decrease_lr_for_nncf'):
        return nncf_training_config.get('coeff_decrease_lr_for_nncf')
    raise RuntimeError('The default value for coeff_decrease_lr_for_nncf is not set')

def calculate_lr_for_nncf_training(lr_from_cfg, checkpoint_path,
                                   coeff_decrease_lr_for_nncf,
                                   is_initial_lr_set_from_opts):
    if is_initial_lr_set_from_opts:
        print(f'Since initial LR is set from command line arguments, do not calculate initial LR for NNCF, '
              f'taking lr from cfg, lr={lr_from_cfg}')
        return lr_from_cfg

    initial_lr_from_checkpoint = get_initial_lr_from_checkpoint(checkpoint_path)
    print(f'initial_lr_from_checkpoint = {initial_lr_from_checkpoint}')
    if initial_lr_from_checkpoint is None:
        print(f'The checkpoint does not contain initial lr -- will not calculate initial LR for NNCF, '
              f'taking lr from cfg, lr={lr_from_cfg}')
        return lr_from_cfg

    if is_checkpoint_nncf(checkpoint_path):
        print('The loaded checkpoint was received from NNCF training')
        print('WARNING: Loading NNCF checkpoint for fine-tuning -- since LR is NOT set from command line arguments '
              'the initial LR stored in checkpoint will be used again')
        print('Please, set LR from command line arguments to avoid this behavior')
        lr = initial_lr_from_checkpoint
        print(f'lr = {lr}')
        return lr

    print('Try to calculate initial LR for NNCF')
    print(f'coeff_decrease_lr_for_nncf = {coeff_decrease_lr_for_nncf}')
    lr = initial_lr_from_checkpoint * coeff_decrease_lr_for_nncf
    print(f'calculated lr = {lr}')
    return lr

def _sanity_check_nncf_changes_in_config(path_change_file):
    # sanity check -- to avoid complicated/recursive side effects
    if not path_change_file:
        return

    cfg_changes = read_yaml(path_change_file)

    if (cfg_changes.get('model', {}).get('load_weights') or
        cfg_changes.get('model', {}).get('resume') or
        cfg_changes.get('nncf') or
        cfg_changes.get('aux_configs')
        ):
        raise RuntimeError(f'Try to make too complicated changes for NNCF training, '
                           f'NNCF changes to main config =\n{pformat(cfg_changes)}')

def make_nncf_changes_in_main_training_config(cfg, command_line_cfg_opts):
    # Take the part of the main config, related to NNCF.
    # Please, note that at the moment the part should be a dict of the following form:
    # ```
    # nncf = {
    #     'nncf_config_path': '...' #this is the path to a json file with NNCF config dict itself
    #     'coeff_decrease_lr_for_nncf': <float value> # this is a coefficient to decrease the LR for NNCF training
    #     'changes_in_main_train_config': '...' # a path to a YAML file with changes of main cfg for NNCF training
    #     'changes_in_aux_train_config': '...'  # a path to a YAML file with changes of aux cfg for NNCF training
    # }
    # ```
    # -- in case if some section of this NNCF part is absent the default values will be used
    nncf_training_config = cfg.get('nncf', {})

    nncf_changes_in_main_train_config = nncf_training_config.get('changes_in_main_train_config')
    if nncf_changes_in_main_train_config:
        _sanity_check_nncf_changes_in_config(nncf_changes_in_main_train_config)

        print(f'applying changes to the main training config from the file {nncf_changes_in_main_train_config}')
        merge_from_files_with_base(cfg, nncf_changes_in_main_train_config)
        # then command line options should be applied again,
        # since the options set from the command line should have preference
        print(f'applying changes to the main training config from the command line options just after that. '
              f'The list of options = \n{pformat(command_line_cfg_opts)}')
        cfg.merge_from_list(command_line_cfg_opts)

    return cfg

def get_nncf_changes_in_aux_training_config(cfg):
    # See details on nncf_training_config in the comment in
    # the function make_nncf_changes_in_main_training_config
    nncf_training_config = cfg.get('nncf', {})
    nncf_changes_in_aux_train_config = nncf_training_config.get('changes_in_aux_train_config')
    _sanity_check_nncf_changes_in_config(nncf_changes_in_aux_train_config)
    return nncf_changes_in_aux_train_config

def make_nncf_changes_in_training(model, cfg, classes, command_line_cfg_opts):
    # See details on nncf_training_config in the comment in
    # the function make_nncf_changes_in_main_training_config
    nncf_training_config = cfg.get('nncf', {})
    nncf_config_path = nncf_training_config.get('nncf_config_path')
    print(f'NNCF config path = {nncf_config_path}')

    lr = None
    if cfg.model.resume:
        raise NotImplementedError('Resuming NNCF training is not implemented yet')
    if not cfg.model.load_weights:
        raise RuntimeError('NNCF training should be started from a non-NNCF or NNCF pre-trained model')
    checkpoint_path = cfg.model.load_weights
    if not check_isfile(checkpoint_path):
        raise RuntimeError(f'Cannot find checkpoint at {checkpoint_path}')

    is_curr_checkpoint_nncf = is_checkpoint_nncf(checkpoint_path)
    print(f'First stage of NNCF model wrapping -- loading weights from {checkpoint_path}')
    if is_curr_checkpoint_nncf:
        print('Note that it is an NNCF checkpoint, so warnings that some layers are discarded '
              'during loading due to unmatched keys will be printed -- it is normal for this case')

    load_pretrained_weights(model, checkpoint_path)

    if is_curr_checkpoint_nncf:
        print(f'Using NNCF checkpoint {checkpoint_path}')
        # just skipping loading special datamanager
        datamanager_for_nncf = None
        checkpoint_path_for_wrapping = checkpoint_path
    else:
        print('before building datamanager for nncf initializing')
        datamanager_for_nncf = build_datamanager(cfg, classes)
        print('after building datamanager for nncf initializing')
        checkpoint_path_for_wrapping = None

    compression_ctrl, model, nncf_metainfo = \
            wrap_nncf_model(model, cfg, datamanager_for_nncf,
                            checkpoint_path=checkpoint_path_for_wrapping,
                            nncf_config_path=nncf_config_path)
    print(f'Received from wrapping nncf_metainfo =\n{pformat(nncf_metainfo)}')
    if cfg.lr_finder.enable:
        print('Turn off LR finder -- it should not be used together with NNCF compression')
        cfg.lr_finder.enable = False

    # calculating initial LR for NNCF training
    is_initial_lr_set_from_opts = is_config_parameter_set_from_command_line(command_line_cfg_opts, 'train.lr')
    coeff_decrease_lr_for_nncf = get_coeff_decrease_lr_for_nncf(nncf_training_config)
    assert isinstance(coeff_decrease_lr_for_nncf, float)
    lr = calculate_lr_for_nncf_training(cfg.train.lr, checkpoint_path,
                                        coeff_decrease_lr_for_nncf,
                                        is_initial_lr_set_from_opts)
    assert lr is not None
    cfg.train.lr = lr
    return compression_ctrl, model, cfg, lr, nncf_metainfo

def make_nncf_changes_in_eval(model, cfg):
    # See details on nncf_training_config in the comment in
    # the function make_nncf_changes_in_main_training_config
    nncf_training_config = cfg.get('nncf', {})
    nncf_config_path = nncf_training_config.get('nncf_config_path')
    print(f'NNCF config path = {nncf_config_path}')
    checkpoint_path = cfg.model.load_weights
    datamanager_for_nncf = None
    compression_ctrl, model, _ = \
            wrap_nncf_model(model, cfg, datamanager_for_nncf,
                            checkpoint_path=checkpoint_path,
                            nncf_config_path=nncf_config_path)
    return model
