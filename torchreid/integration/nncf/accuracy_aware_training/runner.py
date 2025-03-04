# Copyright (C) 2022 Intel Corporation
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

import io
import os.path as osp
from shutil import copyfile
from typing import Dict

import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from nncf.api.compression import CompressionStage, CompressionAlgorithmController

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import PIL.Image
    from torchvision.transforms import ToTensor

    IMG_PACKAGES_AVAILABLE = True
except ImportError:
    IMG_PACKAGES_AVAILABLE = False

from nncf.torch.checkpoint_loading import load_state
from nncf.torch.accuracy_aware_training.utils import is_main_process
from nncf.common.accuracy_aware_training.runner import TrainingRunner
from nncf.common.accuracy_aware_training.runner_factory import TrainingRunnerCreator
from nncf.common.utils.backend import infer_backend_from_compression_controller, BackendType
from nncf.common.utils.helpers import configure_accuracy_aware_paths
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.utils.tensorboard import prepare_for_tensorboard

from torchreid.integration.nncf.accuracy_aware_training.base_runner import (
    BaseAccuracyAwareTrainingRunner,
    BaseAdaptiveCompressionLevelTrainingRunner
)


class EarlyExitTrainingRunnerCreator(TrainingRunnerCreator):
    """
    Class creates an Early Exit Training Runner depending on an used backend.
    """

    def __init__(self, accuracy_aware_training_params: Dict[str, object],
                 compression_controller: CompressionAlgorithmController,
                 lr_updates_needed: bool, verbose: bool, dump_checkpoints: bool):
        self.accuracy_aware_training_params = accuracy_aware_training_params
        self.compression_controller = compression_controller
        self.lr_updates_needed = lr_updates_needed
        self.verbose = verbose
        self.dump_checkpoints = dump_checkpoints

    def create_training_loop(self) -> TrainingRunner:
        """
        Creates an object of AccuracyAwareTrainingRunner depending on user backend

        :return: AccuracyAwareTrainingRunner object
        """
        nncf_backend = infer_backend_from_compression_controller(self.compression_controller)
        if nncf_backend is BackendType.TORCH:
            return PTAccuracyAwareTrainingRunner(self.accuracy_aware_training_params, self.lr_updates_needed,
                                                 self.verbose, self.dump_checkpoints)
        raise RuntimeError('Got an unsupported value of nncf_backend')


class AdaptiveCompressionLevelTrainingRunnerCreator(TrainingRunnerCreator):
    """
    Class creates an Adaptive Compression Level Training Runner depending on an used backend.
    """

    def __init__(self, accuracy_aware_training_params: Dict[str, object],
                 compression_controller: CompressionAlgorithmController,
                 lr_updates_needed: bool, verbose: bool, minimal_compression_rate: float,
                 maximal_compression_rate: float, dump_checkpoints: bool):
        self.accuracy_aware_training_params = accuracy_aware_training_params
        self.compression_controller = compression_controller
        self.lr_updates_needed = lr_updates_needed
        self.verbose = verbose
        self.minimal_compression_rate = minimal_compression_rate
        self.maximal_compression_rate = maximal_compression_rate
        self.dump_checkpoints = dump_checkpoints

    def create_training_loop(self) -> TrainingRunner:
        """
        Creates an object of AdaptiveCompressionLevelTrainingRunner depending on user backend

        :return: AdaptiveCompressionLevelTrainingRunner object
        """
        nncf_backend = infer_backend_from_compression_controller(self.compression_controller)

        if nncf_backend is BackendType.TORCH:
            return PTAdaptiveCompressionLevelTrainingRunner(self.accuracy_aware_training_params,
                                                            self.lr_updates_needed, self.verbose,
                                                            self.minimal_compression_rate,
                                                            self.maximal_compression_rate,
                                                            self.dump_checkpoints)
        raise RuntimeError('Got an unsupported value of nncf_backend')


class PTAccuracyAwareTrainingRunner(BaseAccuracyAwareTrainingRunner):
    """
    BaseAccuracyAwareTrainingRunner
    The Training Runner implementation for PyTorch training code.
    """

    def __init__(self, accuracy_aware_training_params,
                 lr_updates_needed=True, verbose=True,
                 dump_checkpoints=True):
        super().__init__(accuracy_aware_training_params, verbose, dump_checkpoints)

        self.base_lr_reduction_factor_during_search = 1.0
        self.lr_updates_needed = lr_updates_needed
        self.load_checkpoint_fn = None
        self.early_stopping_fn = None
        self.update_learning_rate_fn = None

    def initialize_training_loop_fns(self, train_epoch_fn, validate_fn, configure_optimizers_fn,
                                     dump_checkpoint_fn, tensorboard_writer=None, log_dir=None):
        super().initialize_training_loop_fns(train_epoch_fn, validate_fn, configure_optimizers_fn, dump_checkpoint_fn,
                                             tensorboard_writer=tensorboard_writer, log_dir=log_dir)
        self._log_dir = self._log_dir if self._log_dir is not None \
            else os.path.join(os.getcwd(), 'runs')
        self._log_dir = configure_accuracy_aware_paths(self._log_dir)
        self._checkpoint_save_dir = self._log_dir
        if self._tensorboard_writer is None and TENSORBOARD_AVAILABLE:
            self._tensorboard_writer = SummaryWriter(self._log_dir)

    def retrieve_uncompressed_model_accuracy(self, model):
        if hasattr(model, 'original_model_accuracy') or hasattr(model.module, 'original_model_accuracy'):
            if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                self.uncompressed_model_accuracy = model.module.original_model_accuracy
            else:
                self.uncompressed_model_accuracy = model.original_model_accuracy
        else:
            raise RuntimeError('Original model does not contain the pre-calculated reference metric value')

    def train_epoch(self, model, compression_controller):
        compression_controller.scheduler.epoch_step()
        # assuming that epoch number is only used for logging in train_fn:
        self.loss = self._train_epoch_fn(compression_controller,
                                         model,
                                         epoch=self.cumulative_epoch_count,
                                         optimizer=self.optimizer,
                                         lr_scheduler=self.lr_scheduler)
        self.training_epoch_count += 1
        self.cumulative_epoch_count += 1

    def validate(self, model):
        with torch.no_grad():
            self.current_val_metric_value = self._validate_fn(model, epoch=self.cumulative_epoch_count)
        is_better_by_accuracy = (not self.is_higher_metric_better) != (
                self.current_val_metric_value > self.best_val_metric_value)
        if is_better_by_accuracy:
            self.best_val_metric_value = self.current_val_metric_value

        if is_main_process():
            self.add_tensorboard_scalar('val/accuracy_aware/metric_value',
                                        self.current_val_metric_value, self.cumulative_epoch_count)

        return self.current_val_metric_value

    def update_learning_rate(self):
        if self.update_learning_rate_fn is not None:
            self.update_learning_rate_fn(self.lr_scheduler,
                                        self.training_epoch_count,
                                        self.current_val_metric_value,
                                        self.loss)
        else:
            if self.lr_scheduler is not None and self.lr_updates_needed:
                self.lr_scheduler.step(
                    self.training_epoch_count if not isinstance(self.lr_scheduler, ReduceLROnPlateau)
                    else self.best_val_metric_value)

    def configure_optimizers(self):
        self.optimizer, self.lr_scheduler = self._configure_optimizers_fn()

    def reset_training(self):
        self.configure_optimizers()

        optimizers = self.optimizer if isinstance(self.optimizer, (tuple, list)) else [self.optimizer]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.base_lr_reduction_factor_during_search

        lr_schedulers = self.lr_scheduler if isinstance(self.lr_scheduler, (tuple, list)) else [self.lr_scheduler]
        for lr_scheduler in lr_schedulers:
            if lr_scheduler is not None:
                for attr_name in ['base_lrs', 'init_lr']:
                    if hasattr(lr_scheduler, attr_name):
                        setattr(
                            lr_scheduler,
                            attr_name,
                            [base_lr * self.base_lr_reduction_factor_during_search
                            for base_lr in getattr(lr_scheduler, attr_name)]
                        )

        self.training_epoch_count = 0
        self.best_val_metric_value = 0
        self.current_val_metric_value = 0
        self.loss = None

    def dump_statistics(self, model, compression_controller):
        if not is_main_process():
            return

        statistics = compression_controller.statistics()

        if self.verbose:
            nncf_logger.info(statistics.to_str())

        for key, value in prepare_for_tensorboard(statistics).items():
            if isinstance(value, (int, float)):
                self.add_tensorboard_scalar(f'compression/statistics/{key}',
                                            value, self.cumulative_epoch_count)

        self.dump_checkpoint(model, compression_controller)

    def _save_best_checkpoint(self, checkpoint_path, ):
        best_checkpoint_filename = 'acc_aware_checkpoint_best.pth'
        best_path = osp.join(self._checkpoint_save_dir, best_checkpoint_filename)
        self._best_checkpoint = best_path
        copyfile(checkpoint_path, best_path)
        nncf_logger.info(f'Copy best checkpoint {checkpoint_path} -> {best_path}')

    def dump_checkpoint(self, model, compression_controller):
        if not is_main_process():
            return

        is_best_checkpoint = (self.best_val_metric_value == self.current_val_metric_value and
                              compression_controller.compression_stage() == CompressionStage.FULLY_COMPRESSED)
        if not self.dump_checkpoints and not is_best_checkpoint:
            return

        if self._dump_checkpoint_fn is not None:
            checkpoint_path = self._dump_checkpoint_fn(model, compression_controller, self, self._checkpoint_save_dir)
        else:
            checkpoint = {
                'epoch': self.cumulative_epoch_count + 1,
                'state_dict': model.state_dict(),
                'compression_state': compression_controller.get_compression_state(),
                'best_metric_val': self.best_val_metric_value,
                'current_val_metric_value': self.current_val_metric_value,
                'optimizer': self.optimizer.state_dict(),
            }
            checkpoint_path = osp.join(self._checkpoint_save_dir, 'acc_aware_checkpoint_last.pth')
            torch.save(checkpoint, checkpoint_path)
        nncf_logger.info(f'The checkpoint is saved in {checkpoint_path}')
        if is_best_checkpoint:
            self._save_best_checkpoint(checkpoint_path)

    def add_tensorboard_scalar(self, key, data, step):
        if self.verbose and self._tensorboard_writer is not None:
            self._tensorboard_writer.add_scalar(key, data, step)

    def add_tensorboard_image(self, key, data, step):
        if self.verbose and self._tensorboard_writer is not None:
            self._tensorboard_writer.add_image(key, data, step)

    def update_training_history(self, compression_rate, best_metric_value):
        best_accuracy_budget = best_metric_value - self.minimal_tolerable_accuracy
        self._compressed_training_history.append((compression_rate, best_accuracy_budget))

        if IMG_PACKAGES_AVAILABLE:
            plt.figure()
            plt.plot(self.compressed_training_history.keys(),
                     self.compressed_training_history.values())
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = ToTensor()(image)
            self.add_tensorboard_image('compression/accuracy_aware/acc_budget_vs_comp_rate',
                                       image,
                                       len(self.compressed_training_history))

    @property
    def compressed_training_history(self):
        return dict(self._compressed_training_history)

    def load_best_checkpoint(self, model):
        resuming_checkpoint_path = self._best_checkpoint
        nncf_logger.info(f'Loading the best checkpoint found during training '
                         f'{resuming_checkpoint_path}...')
        if self.load_checkpoint_fn is not None:
            self.load_checkpoint_fn(model, resuming_checkpoint_path)
        else:
            resuming_checkpoint = torch.load(resuming_checkpoint_path, map_location='cpu')
            resuming_model_state_dict = resuming_checkpoint.get('state_dict', resuming_checkpoint)
            load_state(model, resuming_model_state_dict, is_resume=True)

    def stop_training(self, compression_controller):
        if compression_controller.compression_stage() == CompressionStage.FULLY_COMPRESSED \
                and self.early_stopping_fn is not None:
            return self.early_stopping_fn(self.current_val_metric_value)
        return False


class PTAdaptiveCompressionLevelTrainingRunner(PTAccuracyAwareTrainingRunner,
                                               BaseAdaptiveCompressionLevelTrainingRunner):
    def __init__(self, accuracy_aware_training_params,
                 lr_updates_needed=True, verbose=True,
                 minimal_compression_rate=0.05,
                 maximal_compression_rate=0.95,
                 dump_checkpoints=True):

        PTAccuracyAwareTrainingRunner.__init__(self, accuracy_aware_training_params,
                                               lr_updates_needed, verbose,
                                               dump_checkpoints)

        BaseAdaptiveCompressionLevelTrainingRunner.__init__(self, accuracy_aware_training_params,
                                                            verbose,
                                                            minimal_compression_rate=minimal_compression_rate,
                                                            maximal_compression_rate=maximal_compression_rate,
                                                            dump_checkpoints=dump_checkpoints)

    def dump_statistics(self, model, compression_controller):
        if not is_main_process():
            return

        self.update_training_history(self.compression_rate_target, self.current_val_metric_value)
        super().dump_statistics(model, compression_controller)

    def _save_best_checkpoint(self, checkpoint_path):
        best_checkpoint_filename = f'acc_aware_checkpoint_best_compression_rate_' \
                                   f'{self.compression_rate_target:.3f}.pth'
        best_path = osp.join(self._checkpoint_save_dir, best_checkpoint_filename)
        self._best_checkpoints[self.compression_rate_target] = best_path
        copyfile(checkpoint_path, best_path)
        nncf_logger.info(f'Copy best checkpoint {checkpoint_path} -> {best_path}')

    def load_best_checkpoint(self, model):
        # load checkpoint with highest compression rate and positive acc budget
        possible_checkpoint_rates = self.get_compression_rates_with_positive_acc_budget()
        if not possible_checkpoint_rates:
            nncf_logger.warning('Could not produce a compressed model satisfying the set accuracy '
                                'degradation criterion during training. Increasing the number of training '
                                'epochs')
            possible_checkpoint_rates = self.get_compression_rates()
        best_checkpoint_compression_rate = sorted(possible_checkpoint_rates)[-1]
        resuming_checkpoint_path = self._best_checkpoints[best_checkpoint_compression_rate]
        nncf_logger.info(f'Loading the best checkpoint found during training '
                         f'{resuming_checkpoint_path}...')
        if self.load_checkpoint_fn is not None:
            self.load_checkpoint_fn(model, resuming_checkpoint_path)
        else:
            resuming_checkpoint = torch.load(resuming_checkpoint_path, map_location='cpu')
            resuming_model_state_dict = resuming_checkpoint.get('state_dict', resuming_checkpoint)
            load_state(model, resuming_model_state_dict, is_resume=True)
