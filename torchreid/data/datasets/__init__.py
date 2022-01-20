# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import, print_function
from copy import copy

from .image import (Classification, ClassificationImageFolder, ExternalDatasetWrapper,
                    MultiLabelClassification, MultiheadClassification)

__image_datasets = {
    'classification': Classification,
    'classification_image_folder' : ClassificationImageFolder,
    'external_classification_wrapper' : ExternalDatasetWrapper,
    'multilabel_classification': MultiLabelClassification,
    'multihead_classification': MultiheadClassification,
}


def init_image_dataset(mode,
                       custom_dataset_roots=[''],
                       custom_dataset_types=[''], **kwargs):
    """Initializes an image dataset."""

    # handle also custom datasets
    avai_datasets = list(__image_datasets.keys())
    for data_type in custom_dataset_types:
        assert data_type in avai_datasets
    new_kwargs = copy(kwargs)
    i = 0 if mode == 'train' else 1
    if custom_dataset_types[i] == 'external_classification_wrapper':
        new_kwargs['data_provider'] = custom_dataset_roots[i]
    else:
        new_kwargs['root'] = custom_dataset_roots[i]
    return __image_datasets[custom_dataset_types[i]](**new_kwargs)
