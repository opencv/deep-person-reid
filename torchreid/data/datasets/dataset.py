# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import, division, print_function
import os.path as osp

import torch

from torchreid.utils import read_image


class ImageDataset:
    """A base class representing ImageDataset.

    All other image datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``img``, ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self,
                 train,
                 test,
                 transform=None,
                 mode='train',
                 verbose=True,
                 **kwargs):

        self.classes = {}
        self.train = train
        self.test = test
        self.transform = transform
        self.mode = mode
        self.verbose = verbose

        self.num_train_ids = self.get_num_ids(self.train)
        if self.num_train_ids == 0: # workaround: test is a validation set
            self.num_train_ids = self.get_num_ids(self.test)

        self.data_counts = self.get_data_counts(self.train)

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'test':
            self.data = self.test
        else:
            raise ValueError(f'Invalid mode. Got {self.mode}, but expected to be '
                             'one of [train | test]')

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        input_record = self.data[index]

        image = read_image(input_record[0], grayscale=False)
        obj_id = input_record[1]

        if isinstance(obj_id, (tuple, list)): # when multi-label classification is available
            if len(obj_id) and isinstance(obj_id[0], (tuple, list)):
                num_cls_heads = self.mixed_cls_heads_info['num_multiclass_heads']
                targets = torch.zeros(self.mixed_cls_heads_info['num_multiclass_heads'] + \
                                        self.mixed_cls_heads_info['num_multilabel_classes'])
                targets[0 : num_cls_heads] = -1
                for group_idx, in_group_idx in obj_id:
                    if group_idx < num_cls_heads:
                        targets[group_idx] = in_group_idx
                    else:
                        targets[num_cls_heads + in_group_idx] = 1
            else:
                targets = torch.zeros(self.num_train_ids)
                for obj in obj_id:
                    targets[obj] = 1
            obj_id = targets

        if self.transform is not None:
            transformed_image = self.transform(image)
        else:
            transformed_image = image

        output_record = (transformed_image, obj_id)

        return output_record

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_data_counts(data):
        counts = {}
        for record in data:
            obj_id = record[1]
            if obj_id not in counts:
                counts[obj_id] = 1
            else:
                counts[obj_id] += 1

        return counts

    @staticmethod
    def parse_data(data):
        """Parses data list and returns the number of categories.
        """
        if not data:
            return 0
        ids = set()
        for record in data:
            label = record[1]
            if isinstance(label, (list, tuple)):
                if len(label) and isinstance(label[0], (tuple, list)):
                    ids.update([1])
                else:
                    ids.update(set(label))
            else:
                ids.add(label)

        if len(ids) != max(ids) + 1:
            print("WARNING:: There are some categories are missing in this split for this dataset.")
        num_cats = max(ids) + 1
        return num_cats

    def get_num_ids(self, data):
        """Returns the number of training categories."""
        return self.parse_data(data)

    @staticmethod
    def check_before_run(required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError(f'"{fpath}" is not found')

    def __repr__(self):
        num_train_ids = self.parse_data(self.train)
        num_test_ids = self.parse_data(self.test)

        msg = '  ------------------------------\n' \
              '  subset   | # ids | # items\n' \
              '  ------------------------------\n' \
              f'  train    | {num_train_ids:5d} | {len(self.train):7d}\n' \
              f'  test     | {num_test_ids:5d} | {len(self.test):7d}\n' \
              '  -------------------------------\n'

        return msg

    def show_summary(self):
        print(self)
