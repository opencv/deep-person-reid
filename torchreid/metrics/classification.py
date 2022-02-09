# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from terminaltables import AsciiTable

from torchreid.utils import get_model_attr


__FEATURE_DUMP_MODES = ['none', 'all', 'vecs']

def score_extraction(data_loader, model, use_gpu, labelmap=[], head_id=0,
                        perf_monitor=None, feature_dump_mode='none', apply_scale=False):

    assert feature_dump_mode in __FEATURE_DUMP_MODES
    return_featuremaps = feature_dump_mode != __FEATURE_DUMP_MODES[0]

    with torch.no_grad():
        out_scores, gt_labels, all_feature_maps, all_feature_vecs = [], [], [], []
        for batch_idx, data in enumerate(data_loader):
            batch_images, batch_labels = data[0], data[1]
            if perf_monitor: perf_monitor.on_test_batch_begin(batch_idx, None)
            if use_gpu:
                batch_images = batch_images.cuda()

            if labelmap:
                for i, label in enumerate(labelmap):
                    batch_labels[torch.where(batch_labels==i)] = label

            if perf_monitor: perf_monitor.on_test_batch_end(batch_idx, None)

            if return_featuremaps:
                logits, features, global_features = model.forward(batch_images,
                                                                  return_all=return_featuremaps)[head_id]
                if feature_dump_mode == __FEATURE_DUMP_MODES[1]:
                    all_feature_maps.append(features)
                all_feature_vecs.append(global_features)
            else:
                logits = model.forward(batch_images)[head_id]
            out_scores.append(logits * get_model_attr(model, 'scale'))
            gt_labels.append(batch_labels)

        out_scores = torch.cat(out_scores, 0).data.cpu().numpy()
        gt_labels = torch.cat(gt_labels, 0).data.cpu().numpy()
        if apply_scale:
            s = get_model_attr(model, 'scale')
            if s != 1.:
                out_scores *= s

        if all_feature_vecs:
            all_feature_vecs = torch.cat(all_feature_vecs, 0).data.cpu().numpy()
            all_feature_vecs = all_feature_vecs.reshape(all_feature_vecs.shape[0], -1)
            if feature_dump_mode == __FEATURE_DUMP_MODES[2]:
                return (out_scores, all_feature_vecs), gt_labels

        if all_feature_maps:
            all_feature_maps = torch.cat(all_feature_maps, 0).data.cpu().numpy()
            return (out_scores, all_feature_maps, all_feature_vecs), gt_labels

    return out_scores, gt_labels


def score_extraction_from_ir(data_loader, model, labelmap=[], apply_scale=False):
    out_scores, gt_labels = [], []
    for data in data_loader.dataset:
        image, label = np.asarray(data[0]), data[1]
        if labelmap:
            label = labelmap[label]
        scores = model.forward([image])[0]
        out_scores.append(scores *  model.scale)
        gt_labels.append(label)

    out_scores = np.concatenate(out_scores, 0)
    if apply_scale:
        s = get_model_attr(model, 'scale')
        if s != 1.:
            out_scores *= s
    if model.type == 'multilabel':
        gt_labels = np.concatenate(gt_labels, 0)
    else:
        gt_labels = np.array(gt_labels)
    gt_labels = gt_labels.reshape(out_scores.shape[0], -1)

    return out_scores, gt_labels


def mean_top_k_accuracy(scores, labels, k=1):
    idx = np.argsort(-scores, axis=-1)[:, :k]
    labels = np.array(labels)
    matches = np.any(idx == labels.reshape([-1, 1]), axis=-1)

    classes = np.unique(labels)

    accuracy_values = []
    for class_id in classes:
        mask = labels == class_id
        num_valid = np.sum(mask)
        if num_valid == 0:
            continue

        accuracy_values.append(np.sum(matches[mask]) / float(num_valid))

    return np.mean(accuracy_values) if len(accuracy_values) > 0 else 1.0


def mean_average_precision(scores, labels):
    def _ap(in_recall, in_precision):
        mrec = np.concatenate((np.zeros([1, in_recall.shape[1]], dtype=np.float32),
                               in_recall,
                               np.ones([1, in_recall.shape[1]], dtype=np.float32)))
        mpre = np.concatenate((np.zeros([1, in_precision.shape[1]], dtype=np.float32),
                               in_precision,
                               np.zeros([1, in_precision.shape[1]], dtype=np.float32)))

        for i in range(mpre.shape[0] - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        all_ap = []
        cond = mrec[1:] != mrec[:-1]
        for k in range(cond.shape[1]):
            i = np.where(cond[:, k])[0]
            all_ap.append(np.sum((mrec[i + 1, k] - mrec[i, k]) * mpre[i + 1, k]))

        return np.array(all_ap, dtype=np.float32)

    one_hot_labels = np.zeros_like(scores, dtype=np.int32)
    one_hot_labels[np.arange(len(labels)), labels] = 1

    idx = np.argsort(-scores, axis=0)
    sorted_labels = np.take_along_axis(one_hot_labels, idx, axis=0)

    matched = sorted_labels == 1

    tp = np.cumsum(matched, axis=0).astype(np.float32)
    fp = np.cumsum(~matched, axis=0).astype(np.float32)

    num_pos = np.sum(one_hot_labels, axis=0)
    valid_mask = num_pos > 0
    num_pos[~valid_mask] = 1
    num_pos = num_pos.astype(np.float32)

    recall = tp / num_pos.reshape([1, -1])
    precision = tp / (tp + fp)

    ap = _ap(recall, precision)
    valid_ap = ap[valid_mask]
    mean_ap = np.mean(ap) if len(valid_ap) > 0 else 1.0

    return mean_ap


def norm_confusion_matrix(scores, labels):
    pred = np.argmax(scores, axis=1)
    cf = confusion_matrix(labels, pred).astype(float)

    cls_cnt = np.sum(cf, axis=1, keepdims=True)
    norm_cm = cf / cls_cnt

    return norm_cm


def show_confusion_matrix(norm_cm):
    header = ['class {}'.format(i) for i in range(norm_cm.shape[0])]
    data_info = []
    for line in norm_cm:
        data_info.append(['{:.2f}'.format(1e2 * v) for v in line])
    table_data = [header] + data_info
    table = AsciiTable(table_data)
    print('Confusion matrix:\n' + table.table)


def get_invalid(scores, gt_labels, data_info):
    pred_labels = np.argmax(scores, axis=1)
    matches = pred_labels != gt_labels

    unmatched = defaultdict(list)
    for i in range(len(matches)):
        if matches[i]:
            unmatched[gt_labels[i]].append((data_info[i], pred_labels[i]))

    return unmatched


def evaluate_classification(dataloader, model, use_gpu, topk=(1,), labelmap=[]):
    if get_model_attr(model, 'is_ie_model'):
        scores, labels = score_extraction_from_ir(dataloader, model, labelmap)
    else:
        scores, labels = score_extraction(dataloader, model, use_gpu, labelmap)

    m_ap = mean_average_precision(scores, labels)

    cmc = []
    for k in topk:
        cmc.append(mean_top_k_accuracy(scores, labels, k=k))

    norm_cm = norm_confusion_matrix(scores, labels)

    return cmc, m_ap, norm_cm


def evaluate_multilabel_classification(dataloader, model, use_gpu):

    def average_precision(output, target):
        epsilon = 1e-8

        # sort examples
        indices = output.argsort()[::-1]
        # Computes prec@i
        total_count_ = np.cumsum(np.ones((len(output), 1)))

        target_ = target[indices]
        ind = target_ == 1
        pos_count_ = np.cumsum(ind)
        total = pos_count_[-1]
        pos_count_[np.logical_not(ind)] = 0
        pp = pos_count_ / total_count_
        precision_at_i_ = np.sum(pp)
        precision_at_i = precision_at_i_ / (total + epsilon)

        return precision_at_i

    def mAP(targs, preds, pos_thr=0.5):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """
        if np.size(preds) == 0:
            return 0
        ap = np.zeros((preds.shape[1]))
        # compute average precision for each class
        for k in range(preds.shape[1]):
            scores = preds[:, k]
            targets = targs[:, k]
            ap[k] = average_precision(scores, targets)
        tp, fp, fn, tn = [], [], [], []
        for k in range(preds.shape[0]):
            scores = preds[k,:]
            targets = targs[k,:]
            pred = (scores > pos_thr).astype(np.int32)
            tp.append(((pred + targets) == 2).sum())
            fp.append(((pred - targets) == 1).sum())
            fn.append(((pred - targets) == -1).sum())
            tn.append(((pred + targets) == 0).sum())

        p_c = [tp[i] / (tp[i] + fp[i]) if tp[i] > 0 else 0.0 for i in range(len(tp))]
        r_c = [tp[i] / (tp[i] + fn[i]) if tp[i] > 0 else 0.0
                    for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0
                    for i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = sum(tp) / (np.array(tp) + np.array(fp)).sum()
        r_o = sum(tp) / (np.array(tp) + np.array(fn)).sum()
        f_o = 2 * p_o * r_o / (p_o + r_o)

        return ap.mean(), mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o


    if get_model_attr(model, 'is_ie_model'):
        scores, labels = score_extraction_from_ir(dataloader, model)
    else:
        scores, labels = score_extraction(dataloader, model, use_gpu)

    scores = 1. / (1 + np.exp(-scores))
    mAP_score = mAP(labels, scores)

    return mAP_score
