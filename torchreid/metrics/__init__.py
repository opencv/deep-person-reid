# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import

from .accuracy import accuracy, accuracy_multilabel
from .classification import (evaluate_classification, show_confusion_matrix, evaluate_multilabel_classification,
                             evaluate_multihead_classification)
