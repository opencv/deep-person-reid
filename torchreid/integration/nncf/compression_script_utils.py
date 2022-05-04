# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import logging
from pprint import pformat

from scripts.script_utils import build_datamanager, is_config_parameter_set_from_command_line
from torchreid.integration.nncf.compression import is_nncf_state, wrap_nncf_model
from torchreid.integration.nncf.config import compose_nncf_config, load_nncf_config
from torchreid.integration.nncf.config import merge_dicts_and_lists_b_into_a
from torchreid.utils import load_checkpoint, load_pretrained_weights

logger = logging.getLogger(__name__)


def get_coeff_decrease_lr_for_nncf(nncf_training_config):
    coef = nncf_training_config.get('coeff_decrease_lr_for_nncf')
    if isinstance(coef, float):
        return coef
    raise RuntimeError('The default value for coeff_decrease_lr_for_nncf is not set')


def calculate_lr_for_nncf_training(cfg, initial_lr_from_checkpoint, is_initial_lr_set_from_opts):
    lr_from_cfg = cfg.train.lr
    nncf_training_config = cfg.get('nncf', {})
    coeff_decrease_lr_for_nncf = get_coeff_decrease_lr_for_nncf(nncf_training_config)

    if is_initial_lr_set_from_opts:
        logger.info(f'Since initial LR is set from command line arguments, do not calculate initial LR for NNCF, '
              f'taking lr from cfg, lr={lr_from_cfg}')
        return lr_from_cfg

    logger.info(f'initial_lr_from_checkpoint = {initial_lr_from_checkpoint}')
    if initial_lr_from_checkpoint is None:
        logger.info(f'The checkpoint does not contain initial lr -- will not calculate initial LR for NNCF, '
              f'taking lr from cfg, lr={lr_from_cfg}')
        return lr_from_cfg

    logger.info('Try to calculate initial LR for NNCF')
    logger.info(f'coeff_decrease_lr_for_nncf = {coeff_decrease_lr_for_nncf}')
    lr = initial_lr_from_checkpoint * coeff_decrease_lr_for_nncf
    logger.info(f'calculated lr = {lr}')
    return lr


def get_nncf_preset_name(enable_quantization, enable_pruning):
    if not isinstance(enable_quantization, bool) or not isinstance(enable_pruning, bool):
        return None
    if enable_quantization and enable_pruning:
        return 'nncf_quantization_pruning'
    if enable_quantization and not enable_pruning:
        return'nncf_quantization'
    if not enable_quantization and enable_pruning:
        return 'nncf_pruning'
    return None


def make_nncf_changes_in_config(cfg,
                                enable_quantization=False,
                                enable_pruning=False,
                                command_line_cfg_opts=None):

    if not enable_quantization and not enable_pruning:
        raise ValueError('None of the optimization algorithms are enabled')

    # default changes
    if cfg.lr_finder.enable:
        logger.info('Turn off LR finder -- it should not be used together with NNCF compression')
        cfg.lr_finder.enable = False

    cfg.nncf.enable_quantization = enable_quantization
    cfg.nncf.enable_pruning = enable_pruning
    nncf_preset = get_nncf_preset_name(enable_quantization, enable_pruning)

    cfg = patch_config(cfg, nncf_preset)

    if command_line_cfg_opts is not None:
        logger.info(f'applying changes to the main training config from the command line options just after that. '
              f'The list of options = \n{pformat(command_line_cfg_opts)}')
        cfg.merge_from_list(command_line_cfg_opts)

    return cfg


def patch_config(cfg, nncf_preset, max_acc_drop=None):
    # TODO: use default config here
    nncf_config = load_nncf_config(cfg.nncf.nncf_config_path)

    optimization_config = compose_nncf_config(nncf_config, [nncf_preset])

    if "accuracy_aware_training" in optimization_config["nncf_config"]:
        # Update maximal_absolute_accuracy_degradation
        (optimization_config["nncf_config"]["accuracy_aware_training"]
        ["params"]["maximal_absolute_accuracy_degradation"]) = max_acc_drop
    else:
        logger.info("NNCF config has no accuracy_aware_training parameters")

    return merge_dicts_and_lists_b_into_a(cfg, optimization_config)


def make_nncf_changes_in_training(model, cfg, classes, command_line_cfg_opts):
    if cfg.lr_finder.enable:
        raise RuntimeError('LR finder could not be used together with NNCF compression')

    if cfg.model.resume:
        raise NotImplementedError('Resuming NNCF training is not implemented yet')
    if not cfg.model.load_weights:
        raise RuntimeError('NNCF training should be started from a pre-trained model')
    checkpoint_path = cfg.model.load_weights
    checkpoint_dict = load_checkpoint(checkpoint_path, map_location='cpu')
    if is_nncf_state(checkpoint_dict):
        raise RuntimeError(f'The checkpoint is NNCF checkpoint at {checkpoint_path}')

    logger.info(f'Loading weights from {checkpoint_path}')
    load_pretrained_weights(model, pretrained_dict=checkpoint_dict)
    datamanager_for_init = build_datamanager(cfg, classes)

    compression_ctrl, model, nncf_metainfo = \
        wrap_nncf_model(model, cfg, datamanager_for_init=datamanager_for_init)
    logger.info(f'Received from wrapping nncf_metainfo =\n{pformat(nncf_metainfo)}')

    # calculating initial LR for NNCF training
    lr = None
    initial_lr_from_checkpoint = checkpoint_dict.get('initial_lr')
    is_initial_lr_set_from_opts = is_config_parameter_set_from_command_line(command_line_cfg_opts, 'train.lr')
    lr = calculate_lr_for_nncf_training(cfg, initial_lr_from_checkpoint,
                                        is_initial_lr_set_from_opts)
    assert lr is not None
    cfg.train.lr = lr
    return compression_ctrl, model, cfg, lr, nncf_metainfo


def make_nncf_changes_in_eval(model, cfg):
    _, model, _ = wrap_nncf_model(model, cfg)
    return model
