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

# pylint: disable=R0915

import importlib
import os
from contextlib import contextmanager
from pprint import pformat

import logging
import torch

from torchreid.metrics import evaluate_multilabel_classification
from torchreid.utils import get_model_attr
from torchreid.utils.tools import random_image

logger = logging.getLogger(__name__)
_is_nncf_enabled = importlib.util.find_spec('nncf') is not None


@contextmanager
def nullcontext():
    """
    Context which does nothing
    """
    yield


def is_nncf_enabled():
    return _is_nncf_enabled


def check_nncf_is_enabled():
    if not is_nncf_enabled():
        raise RuntimeError('Tried to use NNCF, but NNCF is not installed')


def get_compression_parameter():
    try:
        from nncf.torch.layer_utils import CompressionParameter
        return CompressionParameter
    except ImportError:
        return None


def get_no_nncf_trace_context_manager():
    try:
        from nncf.torch.dynamic_graph.context import \
            no_nncf_trace as original_no_nncf_trace
        return original_no_nncf_trace
    except ImportError:
        return nullcontext


def safe_load_checkpoint(path, map_location=None):
    try:
        from torchreid.utils import load_checkpoint
        return load_checkpoint(path, map_location=map_location)
    except FileNotFoundError:
        return None


def _get_nncf_metainfo_from_state(state):
    if not isinstance(state, dict):
        return None
    return state.get('nncf_metainfo', None)


def create_nncf_metainfo(enable_quantization, enable_pruning, nncf_config):
    nncf_metainfo = {
        'hyperparams': {
            'enable_quantization': enable_quantization,
            'enable_pruning': enable_pruning,
        },
        'nncf_config': nncf_config,
    }
    return nncf_metainfo


def is_nncf_state(state):
    nncf_metainfo = _get_nncf_metainfo_from_state(state)
    if nncf_metainfo is None:
        return False
    hyperparams = nncf_metainfo.get('hyperparams', {})
    return hyperparams.get('enable_quantization', False) or hyperparams.get('enable_pruning', False)


def is_nncf_checkpoint(path):
    checkpoint = safe_load_checkpoint(path, map_location='cpu')
    return is_nncf_state(checkpoint)


def get_compression_hyperparams_from_state(state):
    hyperparams = {
        'enable_quantization': False,
        'enable_pruning': False,
    }

    nncf_metainfo = _get_nncf_metainfo_from_state(state)
    if nncf_metainfo is None:
        return hyperparams

    loaded_hyperparams = nncf_metainfo('hyperparams', {})
    for key in loaded_hyperparams:
        if key in hyperparams:
            hyperparams[key] = loaded_hyperparams[key]
    return hyperparams


def get_compression_hyperparams(path):
    checkpoint = safe_load_checkpoint(path, map_location='cpu')
    return get_compression_hyperparams_from_state(checkpoint)


def extract_model_and_compression_states(resuming_checkpoint):
    """
    The function return from checkpoint state_dict and compression_state.
    """
    if 'model' in resuming_checkpoint:
        model_state_dict = resuming_checkpoint['model']
    elif 'state_dict' in resuming_checkpoint:
        model_state_dict = resuming_checkpoint['state_dict']
    else:
        model_state_dict = resuming_checkpoint
    compression_state = resuming_checkpoint['compression_state']
    return model_state_dict, compression_state


def get_default_nncf_compression_config(h, w):
    """
    This function returns the default NNCF config for this repository.
    The config makes NNCF int8 quantization.
    """
    nncf_config_data = {
        'input_info': {
            'sample_size': [1, 3, h, w]
        },
        'compression': [
            {
                'algorithm': 'quantization',
                'initializer': {
                    'range': {
                        'num_init_samples': 8192,
                    },
                    'batchnorm_adaptation': {
                        'num_bn_adaptation_samples': 8192,
                    }
                }
            }
        ],
        'log_dir': '.'
    }
    return nncf_config_data


def register_custom_modules():
    # Register custom modules.
    # Users of nncf should manually check every custom
    # layer with weights which should be compressed and
    # in case such layers are not wrapping by nncf,
    # wrap such custom module by yourself.
    from nncf.torch import register_module
    from timm.models.layers.conv2d_same import Conv2dSame
    register_module(ignored_algorithms=[])(Conv2dSame)


def wrap_nncf_model(model, cfg,
                    checkpoint_dict=None,
                    datamanager_for_init=None):
    # Note that we require to import it here to avoid cyclic imports when import get_no_nncf_trace_context_manager
    # from mobilenetv3
    from torchreid.data.transforms import build_inference_transform

    from nncf import NNCFConfig
    from nncf.torch import create_compressed_model, load_state
    from nncf.torch.initialization import register_default_init_args
    from nncf.torch.dynamic_graph.io_handling import nncf_model_input
    from nncf.torch.dynamic_graph.trace_tensor import TracedTensor
    from nncf.torch.initialization import PTInitializingDataLoader

    if checkpoint_dict is None:
        checkpoint_path = cfg.model.load_weights
        resuming_checkpoint = safe_load_checkpoint(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint_path = 'pretrained_dict'
        resuming_checkpoint = checkpoint_dict

    if datamanager_for_init is None and not is_nncf_state(resuming_checkpoint):
        raise RuntimeError('Either datamanager_for_init or NNCF pre-trained '
                           'model checkpoint should be set')

    nncf_metainfo = None
    if is_nncf_state(resuming_checkpoint):
        nncf_metainfo = _get_nncf_metainfo_from_state(resuming_checkpoint)
        nncf_config_data = nncf_metainfo['nncf_config']
        datamanager_for_init = None
        logger.info(f'Read NNCF metainfo with NNCF config from the checkpoint:'
                    f'nncf_metainfo=\n{pformat(nncf_metainfo)}')
    else:
        resuming_checkpoint = None
        nncf_config_data = cfg.get('nncf_config')

        if nncf_config_data is None:
            logger.info('Cannot read nncf_config from config file')
        else:
            logger.info(f' nncf_config=\n{pformat(nncf_config_data)}')

    h, w = cfg.data.height, cfg.data.width
    if not nncf_config_data:
        logger.info('Using the default NNCF int8 quantization config')
        nncf_config_data = get_default_nncf_compression_config(h, w)

    # do it even if nncf_config_data is loaded from a checkpoint -- for the rare case when
    # the width and height of the model's input was changed in the config
    # and then finetuning of NNCF model is run
    nncf_config_data.setdefault('input_info', {})
    nncf_config_data['input_info']['sample_size'] = [1, 3, h, w]

    nncf_config = NNCFConfig(nncf_config_data)
    logger.info(f'nncf_config =\n{pformat(nncf_config)}')
    if not nncf_metainfo:
        nncf_metainfo = create_nncf_metainfo(enable_quantization=True,
                                             enable_pruning=False,
                                             nncf_config=nncf_config_data)
    else:
        # update it just to be on the safe side
        nncf_metainfo['nncf_config'] = nncf_config_data

    class ReidInitializeDataLoader(PTInitializingDataLoader):
        def get_inputs(self, dataloader_output):
            # define own InitializingDataLoader class using approach like
            # parse_data_for_train and parse_data_for_eval in the class Engine
            # dataloader_output[0] should be image here
            args = (dataloader_output[0],)
            return args, {}

    @torch.no_grad()
    def model_eval_fn(model):
        """
        Runs evaluation of the model on the validation set and
        returns the target metric value.
        Used to evaluate the original model before compression
        if NNCF-based accuracy-aware training is used.
        """
        from torchreid.metrics.classification import evaluate_classification

        if test_loader is None:
            raise RuntimeError('Cannot perform a model evaluation on the validation '
                               'dataset since the validation data loader was not passed '
                               'to wrap_nncf_model')

        model_type = get_model_attr(model, 'type')
        use_gpu = cur_device.type == 'cuda'
        print('##### Evaluating test dataset #####')
        if model_type == 'classification':
            cmc, _, _ = evaluate_classification(
                test_loader,
                model,
                use_gpu=use_gpu
            )
            accuracy = cmc[0]
        elif model_type == 'multilabel':
            mAP, _, _, _, _, _, _ = evaluate_multilabel_classification(
                test_loader,
                model,
                use_gpu=use_gpu
            )
            accuracy = mAP
        else:
            raise ValueError(f'Cannot perform a model evaluation on the validation dataset'
                                f'since the model has unsupported model_type {model_type or "None"}')

        return accuracy

    cur_device = next(model.parameters()).device
    logger.info(f'NNCF: cur_device = {cur_device}')

    if resuming_checkpoint is None:
        logger.info('No NNCF checkpoint is provided -- register initialize data loader')
        train_loader = datamanager_for_init.train_loader
        test_loader = datamanager_for_init.test_loader
        wrapped_loader = ReidInitializeDataLoader(train_loader)
        nncf_config = register_default_init_args(nncf_config, wrapped_loader,
                                                 model_eval_fn=model_eval_fn, device=cur_device)
        model_state_dict = None
        compression_state = None
    else:
        model_state_dict, compression_state = extract_model_and_compression_states(resuming_checkpoint)

    transform = build_inference_transform(
        cfg.data.height,
        cfg.data.width,
        norm_mean=cfg.data.norm_mean,
        norm_std=cfg.data.norm_std,
    )

    def dummy_forward(model):
        prev_training_state = model.training
        model.eval()
        input_img = random_image(cfg.data.height, cfg.data.width)
        input_blob = transform(input_img).unsqueeze(0)
        assert len(input_blob.size()) == 4
        input_blob = input_blob.to(device=cur_device)
        input_blob = nncf_model_input(input_blob)
        model(input_blob)
        model.train(prev_training_state)

    def wrap_inputs(args, kwargs):
        assert len(args) == 1
        if isinstance(args[0], TracedTensor):
            logger.info('wrap_inputs: do not wrap input TracedTensor')
            return args, {}
        return (nncf_model_input(args[0]),), kwargs

    model.dummy_forward_fn = dummy_forward
    if 'log_dir' in nncf_config:
        os.makedirs(nncf_config['log_dir'], exist_ok=True)
    logger.info(f'nncf_config["log_dir"] = {nncf_config["log_dir"]}')

    register_custom_modules()

    compression_ctrl, model = create_compressed_model(model,
                                                      nncf_config,
                                                      dummy_forward_fn=dummy_forward,
                                                      wrap_inputs_fn=wrap_inputs,
                                                      compression_state=compression_state)

    if model_state_dict:
        logger.info(f'Loading NNCF model from {checkpoint_path}')
        load_state(model, model_state_dict, is_resume=True)

    return compression_ctrl, model, nncf_metainfo


def get_nncf_complession_stage():
    try:
        from nncf.api.compression import CompressionStage
        return CompressionStage
    except ImportError:
        return lambda _: None


def get_nncf_prepare_for_tensorboard():
    try:
        from nncf.common.utils.tensorboard import prepare_for_tensorboard
        return prepare_for_tensorboard
    except ImportError:
        return lambda _: None


def is_accuracy_aware_training_set(nncf_config):
    if not is_nncf_enabled():
        return False
    from nncf.config.utils import is_accuracy_aware_training
    is_acc_aware_training_set = is_accuracy_aware_training(nncf_config)
    return is_acc_aware_training_set
