from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from terminaltables import AsciiTable


def score_extraction(data_loader, model, use_gpu, labelmap=[], head_id=0):
    with torch.no_grad():
        out_scores, gt_labels = [], []
        for batch_idx, data in enumerate(data_loader):
            batch_images, batch_labels = data[0], data[1]
            if use_gpu:
                batch_images = batch_images.cuda()

            if labelmap:
                for i, label in enumerate(labelmap):
                    batch_labels[torch.where(batch_labels==i)] = label

            out_scores.append(model(batch_images)[head_id])
            gt_labels.extend(batch_labels)

        out_scores = torch.cat(out_scores, 0).data.cpu().numpy()
        gt_labels = np.asarray(gt_labels)

    return out_scores, gt_labels


def score_extraction_from_ir(data_loader, model, use_gpu, labelmap=[]):
    out_scores, gt_labels = [], []
    for data in data_loader.dataset:
        image, label = np.asarray(data[0]), data[1]
        if labelmap:
            label = labelmap[label]
        scores = model.forward([image])[0]
        out_scores.append(scores)
        gt_labels.append(label)

    out_scores = np.concatenate(out_scores, 0)
    gt_labels = np.asarray(gt_labels)

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
    is_ie_model = model.module.is_ie_model if isinstance(model, torch.nn.parallel.DataParallel) \
        else model.is_ie_model

    if is_ie_model:
        scores, labels = score_extraction_from_ir(dataloader, model, labelmap)
    else:
        scores, labels = score_extraction(dataloader, model, use_gpu, labelmap)

    m_ap = mean_average_precision(scores, labels)

    cmc = []
    for k in topk:
        cmc.append(mean_top_k_accuracy(scores, labels, k=k))

    norm_cm = norm_confusion_matrix(scores, labels)

    return cmc, m_ap, norm_cm
