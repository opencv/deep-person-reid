"""
 Copyright (c) 2021 Intel Corporation

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

from subprocess import run, DEVNULL, CalledProcessError

import onnx
import torch

from torch.onnx.symbolic_registry import register_op

from scripts.script_utils import (group_norm_symbolic, hardsigmoid_symbolic,
                                  relu6_symbolic)
from torchreid.utils import random_image
from torchreid.data.transforms import build_inference_transform


def export_onnx(model, cfg, output_file_path='model', disable_dyn_axes=True,
                verbose=False, opset=9, extra_check=False):
    transform = build_inference_transform(
        cfg.data.height,
        cfg.data.width,
        norm_mean=cfg.data.norm_mean,
        norm_std=cfg.data.norm_std,
    )

    input_img = random_image(cfg.data.height, cfg.data.width)
    device = next(model.parameters()).device
    input_blob = transform(input_img).unsqueeze(0).to(device)

    input_names = ['data']
    output_names = ['output']
    if not disable_dyn_axes:
        dynamic_axes = {'data': {0: 'batch_size', 1: 'channels', 2: 'height', 3: 'width'}}
    else:
        dynamic_axes = {}

    if not output_file_path.endswith('.onnx'):
        output_file_path += '.onnx'

    register_op("group_norm", group_norm_symbolic, "", opset)
    register_op("relu6", relu6_symbolic, "", opset)
    register_op("hardsigmoid", hardsigmoid_symbolic, "", opset)

    with torch.no_grad():
        torch.onnx.export(
            model,
            input_blob,
            output_file_path,
            verbose=verbose,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )

    if extra_check:
        net_from_onnx = onnx.load(output_file_path)
        try:
            onnx.checker.check_model(net_from_onnx)
            print('ONNX check passed.')
        except onnx.onnx_cpp2py_export.checker.ValidationError as ex:
            print(f'ONNX check failed: {ex}.')

    return output_file_path


def export_ir(onnx_model_path, norm_mean=[0,0,0], norm_std=[1,1,1], input_shape=None,
                optimized_model_dir='./ir_model', data_type='FP32'):
    def get_mo_cmd():
        for mo_cmd in ('mo', 'mo.py'):
            try:
                run([mo_cmd, '-h'], stdout=DEVNULL, stderr=DEVNULL, shell=False, check=True)
                return mo_cmd
            except CalledProcessError:
                pass
        raise RuntimeError('OpenVINO Model Optimizer is not found or configured improperly')

    mean_values = str([s*255 for s in norm_mean])
    scale_values = str([s*255 for s in norm_std])

    mo_cmd = get_mo_cmd()

    command_line = [mo_cmd, f'--input_model={onnx_model_path}',
                    f'--mean_values={mean_values}',
                    f'--scale_values={scale_values}',
                    f'--output_dir={optimized_model_dir}',
                    '--data_type', f'{data_type}',
                    '--reverse_input_channels']

    if input_shape:
        command_line.append('--input_shape', f"{input_shape}")

    run(command_line, shell=False, check=True)
