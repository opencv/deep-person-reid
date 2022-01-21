# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import, division, print_function

import numpy as np
import torch

from torchreid.data.datasets import init_image_dataset
from torchreid.data.sampler import build_train_sampler
from torchreid.data.transforms import build_transforms
from torchreid.utils import worker_init_fn


class ImageDataManager():
    r"""Image data manager.

    Args:
        root (str): root path to datasets.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
        batch_size_train (int, optional): number of images in a training batch. Default is 32.
        batch_size_test (int, optional): number of images in a test batch. Default is 32.
        workers (int, optional): number of workers. Default is 4.
        train_sampler (str, optional): sampler. Default is RandomSampler.
        correct_batch_size (bool, optional): this heuristic improves multilabel training on small datasets
    """

    def __init__(
        self,
        root='',
        height=224,
        width=224,
        transforms='random_flip',
        norm_mean=None,
        norm_std=None,
        use_gpu=True,
        batch_size_train=32,
        batch_size_test=32,
        correct_batch_size = False,
        workers=4,
        train_sampler='RandomSampler',
        custom_dataset_roots=[''],
        custom_dataset_types=[''],
        filter_classes=None,
    ):
        self.height = height
        self.width = width

        self.transform_train, self.transform_test = build_transforms(
            self.height,
            self.width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.use_gpu = (torch.cuda.is_available() and use_gpu)

        print('=> Loading train dataset')
        train_dataset = init_image_dataset(
            transform=self.transform_train,
            mode='train',
            root=root,
            custom_dataset_roots=custom_dataset_roots,
            custom_dataset_types=custom_dataset_types,
            filter_classes=filter_classes,
        )
        self.train_dataset = train_dataset

        self._data_counts = train_dataset.data_counts
        self._num_train_ids = train_dataset.num_train_ids
        if correct_batch_size:
            batch_size_train = self.calculate_batch(batch_size_train, len(train_dataset))
        batch_size_train = max(1, min(batch_size_train, len(train_dataset)))
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=build_train_sampler(
                train_dataset.train,
                train_sampler,
            ),
            batch_size=batch_size_train,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )
        self.num_iter = len(self.train_loader)
        print('=> Loading test dataset')

        # build test loader
        test_dataset = init_image_dataset(
            transform=self.transform_test,
            mode='test',
            root=root,
            custom_dataset_roots=custom_dataset_roots,
            custom_dataset_types=custom_dataset_types,
            filter_classes=filter_classes
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=max(min(batch_size_test, len(test_dataset)), 1),
            shuffle=False,
            num_workers=workers,
            worker_init_fn=worker_init_fn,
            pin_memory=self.use_gpu,
            drop_last=False
        )

        print('\n')
        print('  **************** Summary ****************')
        print(f'  # categories      : {self._num_train_ids}')
        print(f'  # train images    : {len(train_dataset)}')
        print(f'  # test images     : {len(test_dataset)}')
        print('  *****************************************')
        print('\n')

    @staticmethod
    def calculate_batch(cur_batch, data_len):
        ''' This heuristic improves multilabel training on small datasets '''
        if data_len <= 2500:
            return max(int(np.ceil(np.sqrt(data_len) / 2.5)), 6)
        return cur_batch

    @property
    def num_train_ids(self):
        """Returns the number of training categories."""
        return self._num_train_ids

    @property
    def data_counts(self):
        """Returns the number of samples for each ID."""
        return self._data_counts
