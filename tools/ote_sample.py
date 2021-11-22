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

import argparse
import logging
import os.path as osp
import sys

from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.configuration.helper import create
from ote_sdk.entities.datasets import Subset
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.model import ModelEntity, ModelStatus
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.task_environment import TaskEnvironment

from torchreid.integration.sc.utils import (ClassificationDatasetAdapter,
                                            generate_label_schema,
                                            get_task_class)


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Sample showcasing the new API')
    parser.add_argument('template_file_path', help='path to template file')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--debug-dump-path', default='')
    args = parser.parse_args()
    return args


def main(args):
    if args.debug_dump_path:
        from torchreid.utils import Logger
        sys.stdout = Logger(args.debug_dump_path)
    logger.info('Initialize dataset')
    dataset = ClassificationDatasetAdapter(
        train_data_root=osp.join(args.data_dir, 'train'),
        train_ann_file=osp.join(args.data_dir, 'train.json'),
        val_data_root=osp.join(args.data_dir, 'val'),
        val_ann_file=osp.join(args.data_dir, 'val.json'),
        test_data_root=osp.join(args.data_dir, 'val'),
        test_ann_file=osp.join(args.data_dir, 'val.json'))

    labels_schema = generate_label_schema(dataset.get_labels(), dataset.is_multilabel())
    logger.info(f'Train dataset: {len(dataset.get_subset(Subset.TRAINING))} items')
    logger.info(f'Validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items')

    logger.info('Load model template')
    model_template = parse_model_template(args.template_file_path)

    logger.info('Set hyperparameters')

    params = create(model_template.hyper_parameters.data)
    logger.info('Setup environment')
    environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=labels_schema, model_template=model_template)

    logger.info('Create base Task')
    task_impl_path = model_template.entrypoints.base
    task_cls = get_task_class(task_impl_path)
    task = task_cls(task_environment=environment)

    logger.info('Train model')
    output_model = ModelEntity(
        dataset,
        environment.get_model_configuration(),
        model_status=ModelStatus.NOT_READY)
    task.train(dataset, output_model)

    logger.info('Get predictions on the validation set')
    validation_dataset = dataset.get_subset(Subset.VALIDATION)
    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=True))
    resultset = ResultSetEntity(
        model=output_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    logger.info('Estimate quality on validation set')
    task.evaluate(resultset)
    logger.info(str(resultset.performance))

    if args.export:
        logger.info('Export model')
        exported_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)
        task.export(ExportType.OPENVINO, exported_model)

        logger.info('Create OpenVINO Task')
        environment.model = exported_model
        openvino_task_impl_path = model_template.entrypoints.openvino
        openvino_task_cls = get_task_class(openvino_task_impl_path)
        openvino_task = openvino_task_cls(environment)

        logger.info('Get predictions on the validation set')
        predicted_validation_dataset = openvino_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        resultset = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        logger.info('Estimate quality on validation set')
        openvino_task.evaluate(resultset)
        logger.info(str(resultset.performance))

        logger.info('Run POT optimization')
        optimized_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)
        openvino_task.optimize(
            OptimizationType.POT,
            dataset.get_subset(Subset.TRAINING),
            optimized_model,
            OptimizationParameters())

        logger.info('Get predictions on the validation set')
        predicted_validation_dataset = openvino_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        resultset = ResultSetEntity(
            model=optimized_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        logger.info('Performance of optimized model:')
        openvino_task.evaluate(resultset)
        logger.info(str(resultset.performance))


if __name__ == '__main__':
    args = parse_args()
    print(args)
    sys.exit(main(args) or 0)
