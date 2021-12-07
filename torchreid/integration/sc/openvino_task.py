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
import os
import tempfile

from addict import Dict as ADDict
from typing import Any, Dict, Tuple, Optional, Union

import cv2 as cv
import numpy as np

from ote_sdk.entities.annotation import Annotation, AnnotationSceneEntity, AnnotationSceneKind
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters, default_progress_callback
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import (
    ModelStatus,
    ModelEntity,
    ModelFormat,
    OptimizationMethod,
    ModelPrecision,
)
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.serialization.label_mapper import label_schema_to_bytes
from ote_sdk.usecases.exportable_code.inference import BaseOpenVINOInferencer
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)

from compression.api import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights, get_nodes_by_type
from compression.pipeline.initializer import create_pipeline

from torchreid.integration.sc.parameters import OTEClassificationParameters
from torchreid.integration.sc.utils import (get_ancestors_by_prediction, get_multiclass_predictions,
                                            get_multilabel_predictions,
                                            get_empty_label, get_leaf_labels)

logger = logging.getLogger(__name__)


def get_output(net, outputs, name):
    try:
        key = net.get_ov_name_for_tensor(name)
        assert key in outputs, f'"{key}" is not a valid output identifier'
    except KeyError as err:
        if name not in outputs:
            raise KeyError(f'Failed to identify output "{name}"') from err
        key = name
    return outputs[key]


class OpenVINOClassificationInferencer(BaseOpenVINOInferencer):
    def __init__(
        self,
        hparams: OTEClassificationParameters,
        label_schema: LabelSchemaEntity,
        model_file: Union[str, bytes],
        weight_file: Union[str, bytes, None] = None,
        device: str = "CPU",
        num_requests: int = 1,
    ):
        """
        Inferencer implementation for OTEDetection using OpenVINO backend.
        :param model: Path to model to load, `.xml`, `.bin` or `.onnx` file.
        :param hparams: Hyper parameters that the model should use.
        :param num_requests: Maximum number of requests that the inferencer can make.
            Good value is the number of available cores. Defaults to 1.
        :param device: Device to run inference on, such as CPU, GPU or MYRIAD. Defaults to "CPU".
        """
        super().__init__(model_file, weight_file, device, num_requests)

        if len(label_schema.get_labels(False)) == 1:
            self.labels = label_schema.get_labels(include_empty=True)
        else:
            self.labels = label_schema.get_labels(include_empty=False)
        self.empty_label = get_empty_label(label_schema)
        self.multilabel = len(label_schema.get_groups(False)) > 1 and \
                len(label_schema.get_groups(False)) == len(label_schema.get_labels(include_empty=False))

        self.hierarchical = False
        if not self.multilabel and len(label_schema.get_groups(False)) > 1:
            self.labels = get_leaf_labels(label_schema)
            self.hierarchical = True

        self.label_schema = label_schema
        self.input_blob_name = 'data'
        self.n, self.c, self.h, self.w = self.net.input_info[self.input_blob_name].tensor_desc.dims
        self.keep_aspect_ratio_resize = False
        self.pad_value = 0

    @staticmethod
    def resize_image(image: np.ndarray, size: Tuple[int], keep_aspect_ratio: bool = False) -> np.ndarray:
        if not keep_aspect_ratio:
            resized_frame = cv.resize(image, dsize=size)
        else:
            h, w = image.shape[:2]
            scale = min(size[1] / h, size[0] / w)
            resized_frame = cv.resize(image, None, fx=scale, fy=scale)
        return resized_frame

    def pre_process(self, image: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        resized_image = self.resize_image(image, (self.w, self.h), self.keep_aspect_ratio_resize)
        resized_image = cv.cvtColor(resized_image, cv.COLOR_RGB2BGR)
        meta = {'original_shape': image.shape,
                'resized_shape': resized_image.shape}

        h, w = resized_image.shape[:2]
        if h != self.h or w != self.w:
            resized_image = np.pad(resized_image, ((0, self.h - h), (0, self.w - w), (0, 0)),
                                   mode='constant', constant_values=self.pad_value)
        resized_image = resized_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        resized_image = resized_image.reshape((self.n, self.c, self.h, self.w))
        dict_inputs = {self.input_blob_name: resized_image}
        return dict_inputs, meta

    def post_process(self, prediction: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> AnnotationSceneEntity:
        raw_output = get_output(self.net, prediction, 'output').reshape(-1)
        if self.multilabel:
            item_labels = get_multilabel_predictions(raw_output, self.labels)
            if not item_labels:
                item_labels = [ScoredLabel(self.empty_label, probability=1.)]
        else:
            item_labels = get_multiclass_predictions(raw_output, self.labels)
            if self.hierarchical:
                item_labels.extend(get_ancestors_by_prediction(self.label_schema, item_labels[0]))
        anno = [Annotation(Rectangle.generate_full_box(), labels=item_labels)]

        return AnnotationSceneEntity(kind=AnnotationSceneKind.PREDICTION, annotations=anno)

    def forward(self, image: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.model.infer(image)


class OTEOpenVinoDataLoader(DataLoader):
    def __init__(self, dataset: DatasetEntity, inferencer: BaseOpenVINOInferencer):
        super().__init__(config=None)
        self.dataset = dataset
        self.inferencer = inferencer

    def __getitem__(self, index):
        image = self.dataset[index].numpy
        annotation = self.dataset[index].annotation_scene
        inputs, metadata = self.inferencer.pre_process(image)

        return (index, annotation), inputs, metadata

    def __len__(self):
        return len(self.dataset)


class OpenVINOClassificationTask(IInferenceTask, IEvaluationTask, IOptimizationTask):
    def __init__(self, task_environment: TaskEnvironment):
        self.task_environment = task_environment
        self.hparams = self.task_environment.get_hyper_parameters(OTEClassificationParameters)
        self.model = self.task_environment.model
        self.inferencer = self.load_inferencer()

    def load_inferencer(self) -> OpenVINOClassificationInferencer:
        return OpenVINOClassificationInferencer(self.hparams,
                                                self.task_environment.label_schema,
                                                self.model.get_data("openvino.xml"),
                                                self.model.get_data("openvino.bin"))

    def infer(self, dataset: DatasetEntity,
              inference_parameters: Optional[InferenceParameters] = None) -> DatasetEntity:
        update_progress_callback = default_progress_callback
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
        dataset_size = len(dataset)
        for i, dataset_item in enumerate(dataset, 1):
            predicted_scene = self.inferencer.predict(dataset_item.numpy)
            dataset_item.append_labels(predicted_scene.annotations[0].get_labels())
            update_progress_callback(int(i / dataset_size * 100))
        return dataset

    def evaluate(self,
                 output_result_set: ResultSetEntity,
                 evaluation_metric: Optional[str] = None):
        if evaluation_metric is not None:
            logger.warning(f'Requested to use {evaluation_metric} metric,'
                            'but parameter is ignored. Use accuracy instead.')
        output_result_set.performance = MetricsHelper.compute_accuracy(output_result_set).get_performance()

    def optimize(self,
                 optimization_type: OptimizationType,
                 dataset: DatasetEntity,
                 output_model: ModelEntity,
                 optimization_parameters: Optional[OptimizationParameters]):

        if optimization_type is not OptimizationType.POT:
            raise ValueError("POT is the only supported optimization type for OpenVino models")

        data_loader = OTEOpenVinoDataLoader(dataset, self.inferencer)

        with tempfile.TemporaryDirectory() as tempdir:
            xml_path = os.path.join(tempdir, "model.xml")
            bin_path = os.path.join(tempdir, "model.bin")
            with open(xml_path, "wb") as f:
                f.write(self.model.get_data("openvino.xml"))
            with open(bin_path, "wb") as f:
                f.write(self.model.get_data("openvino.bin"))

            model_config = ADDict({
                'model_name': 'openvino_model',
                'model': xml_path,
                'weights': bin_path
            })

            model = load_model(model_config)

            if get_nodes_by_type(model, ['FakeQuantize']):
                logger.warning("Model is already optimized by POT")
                output_model.model_status = ModelStatus.FAILED
                return

        engine_config = ADDict({
            'device': 'CPU'
        })

        stat_subset_size = self.hparams.pot_parameters.stat_subset_size
        preset = self.hparams.pot_parameters.preset.name.lower()

        algorithms = [
            {
                'name': 'DefaultQuantization',
                'params': {
                    'target_device': 'ANY',
                    'preset': preset,
                    'stat_subset_size': min(stat_subset_size, len(data_loader)),
                    'shuffle_data': True
                }
            }
        ]

        engine = IEEngine(config=engine_config, data_loader=data_loader, metric=None)

        pipeline = create_pipeline(algorithms, engine)

        compressed_model = pipeline.run(model)

        compress_model_weights(compressed_model)

        with tempfile.TemporaryDirectory() as tempdir:
            save_model(compressed_model, tempdir, model_name="model")
            with open(os.path.join(tempdir, "model.xml"), "rb") as f:
                output_model.set_data("openvino.xml", f.read())
            with open(os.path.join(tempdir, "model.bin"), "rb") as f:
                output_model.set_data("openvino.bin", f.read())

        output_model.set_data("label_schema.json", label_schema_to_bytes(self.task_environment.label_schema))

        # set model attributes for quantized model
        output_model.model_status = ModelStatus.SUCCESS
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = OptimizationType.POT
        output_model.optimization_methods = [OptimizationMethod.QUANTIZATION]
        output_model.precision = [ModelPrecision.INT8]

        self.model = output_model
        self.inferencer = self.load_inferencer()
