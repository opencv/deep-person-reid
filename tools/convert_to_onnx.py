"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import os
import warnings

import torch

from scripts.default_config import get_default_config, model_kwargs, merge_from_files_with_base
from scripts.script_utils import patch_InplaceAbn_forward
from torchreid.apis.export import export_onnx, export_ir
from torchreid.integration.nncf.compression import get_compression_hyperparams
from torchreid.models import build_model
from torchreid.utils import load_checkpoint, load_pretrained_weights
from torchreid.integration.nncf.compression_script_utils import (make_nncf_changes_in_eval,
                                                                 make_nncf_changes_in_config)


def parse_num_classes(source_datasets, classification=False, num_classes=None, snap_path=None):
    if classification:
        if snap_path:
            chkpt = load_checkpoint(snap_path)
            num_classes_from_snap = chkpt['num_classes'] if 'num_classes' in chkpt else None

            if isinstance(num_classes_from_snap, int):
                num_classes_from_snap = [num_classes_from_snap]

            if num_classes is None:
                num_classes = num_classes_from_snap
            else:
                print('Warning: number of classes in model was overriden via command line')
        assert num_classes is not None and len(num_classes) > 0

    if num_classes is not None and len(num_classes) > 0:
        return num_classes

    num_clustered = 0
    num_rest = 0
    for src in source_datasets:
        if isinstance(src, (tuple, list)):
            num_clustered += 1
        else:
            num_rest += 1

    total_num_sources = num_clustered + int(num_rest > 0)
    assert total_num_sources > 0

    return [0] * total_num_sources  # dummy number of classes


def reset_config(cfg):
    cfg.model.download_weights = False


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default='', required=True,
                        help='Path to config file')
    parser.add_argument('--output-name', type=str, default='model',
                        help='Path to save ONNX model')
    parser.add_argument('--num-classes', type=int, nargs='+', default=None)
    parser.add_argument('--opset', type=int, default=11)
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose mode for onnx.export')
    parser.add_argument('--disable-dyn-axes', default=False, action='store_true')
    parser.add_argument('--export_ir', action='store_true')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        merge_from_files_with_base(cfg, args.config_file)
    reset_config(cfg)
    cfg.merge_from_list(args.opts)

    compression_hyperparams = get_compression_hyperparams(cfg.model.load_weights)
    is_nncf_used = compression_hyperparams['enable_quantization'] or compression_hyperparams['enable_pruning']
    if is_nncf_used:
        print(f'Using NNCF -- making NNCF changes in config')
        cfg = make_nncf_changes_in_config(cfg,
                                          compression_hyperparams['enable_quantization'],
                                          compression_hyperparams['enable_pruning'],
                                          args.opts)
    cfg.train.mix_precision = False
    cfg.freeze()
    num_classes = parse_num_classes(source_datasets=cfg.data.sources,
                                    classification=cfg.model.type == 'classification' or cfg.model.type == 'multilabel',
                                    num_classes=args.num_classes,
                                    snap_path=cfg.model.load_weights)
    model = build_model(**model_kwargs(cfg, num_classes))
    if cfg.model.load_weights:
        load_pretrained_weights(model, cfg.model.load_weights)
    else:
        warnings.warn("No weights are passed through 'load_weights' parameter! "
              "The model will be converted with random or pretrained weights", category=RuntimeWarning)
    if 'tresnet' in cfg.model.name:
        patch_InplaceAbn_forward()
    if is_nncf_used:
        print('Begin making NNCF changes in model')
        model = make_nncf_changes_in_eval(model, cfg)
        print('End making NNCF changes in model')
    onnx_file_path = export_onnx(model=model.eval(),
                                 cfg=cfg,
                                 output_file_path=args.output_name,
                                 disable_dyn_axes=args.disable_dyn_axes,
                                 verbose=args.verbose,
                                 opset=args.opset,
                                 extra_check=True)
    if args.export_ir:
        input_shape = [1, 3, cfg.data.height, cfg.data.width]
        export_ir(onnx_model_path=onnx_file_path,
                  norm_mean=cfg.data.norm_mean,
                  norm_std=cfg.data.norm_std,
                  input_shape=input_shape,
                  optimized_model_dir=os.path.dirname(os.path.abspath(onnx_file_path)),
                  data_type='FP32')


if __name__ == '__main__':
    main()
