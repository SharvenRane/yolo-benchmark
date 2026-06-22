"""Mean average precision for object detection.

The implementation follows the standard greedy matching protocol. For each
class, predictions are sorted by descending confidence. Each prediction is
matched to the highest IoU ground truth box that is still available and clears
the IoU threshold. A matched prediction counts as a true positive, an unmatched
one as a false positive, and any ground truth box left over is a false negative.
The precision and recall curve is then integrated to give average precision.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .iou import box_iou


def _ap_from_pr(recall, precision):
    """Integrate precision over recall using the all points interpolation.

    This is the area under the precision recall curve after making precision
    monotonically non increasing from right to left, which matches the COCO and
    Pascal VOC 2010 onward convention.
    """
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    change = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[change + 1] - recall[change]) * precision[change + 1])
    return float(ap)


def average_precision(predictions, ground_truths, iou_threshold=0.5):
    """Average precision for a single class.

    ``predictions`` is an iterable of ``(image_id, score, box)`` tuples and
    ``ground_truths`` is an iterable of ``(image_id, box)`` tuples, where each
    box is ``[x1, y1, x2, y2]``. The returned float is the average precision at
    the given IoU threshold.

    When there are no ground truth boxes the AP is defined as ``1.0`` if there
    are also no predictions and ``0.0`` otherwise.
    """
    predictions = list(predictions)
    ground_truths = list(ground_truths)

    total_gt = len(ground_truths)
    if total_gt == 0:
        return 1.0 if len(predictions) == 0 else 0.0

    if len(predictions) == 0:
        return 0.0

    gt_by_image = defaultdict(list)
    for image_id, box in ground_truths:
        gt_by_image[image_id].append(np.asarray(box, dtype=np.float64))

    matched = {image_id: [False] * len(boxes) for image_id, boxes in gt_by_image.items()}

    predictions.sort(key=lambda item: item[1], reverse=True)

    tp = np.zeros(len(predictions), dtype=np.float64)
    fp = np.zeros(len(predictions), dtype=np.float64)

    for rank, (image_id, _score, box) in enumerate(predictions):
        gt_boxes = gt_by_image.get(image_id, [])
        if not gt_boxes:
            fp[rank] = 1.0
            continue

        ious = box_iou(np.asarray(box, dtype=np.float64)[None, :], np.asarray(gt_boxes))[0]
        best = int(np.argmax(ious))
        best_iou = float(ious[best])

        if best_iou >= iou_threshold and not matched[image_id][best]:
            tp[rank] = 1.0
            matched[image_id][best] = True
        else:
            fp[rank] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)

    return _ap_from_pr(recall, precision)


def mean_average_precision(predictions, ground_truths, iou_threshold=0.5):
    """Mean average precision across all classes.

    ``predictions`` is an iterable of ``(image_id, class_id, score, box)`` and
    ``ground_truths`` is an iterable of ``(image_id, class_id, box)``. The mean
    is taken over the set of classes that appear in the ground truth. The return
    value is a tuple ``(mAP, per_class)`` where ``per_class`` maps each class id
    to its average precision.
    """
    predictions = list(predictions)
    ground_truths = list(ground_truths)

    preds_by_class = defaultdict(list)
    for image_id, class_id, score, box in predictions:
        preds_by_class[class_id].append((image_id, score, box))

    gts_by_class = defaultdict(list)
    for image_id, class_id, box in ground_truths:
        gts_by_class[class_id].append((image_id, box))

    classes = sorted(gts_by_class.keys())
    if not classes:
        return 0.0, {}

    per_class = {}
    for class_id in classes:
        per_class[class_id] = average_precision(
            preds_by_class.get(class_id, []),
            gts_by_class[class_id],
            iou_threshold=iou_threshold,
        )

    map_value = float(np.mean([per_class[c] for c in classes]))
    return map_value, per_class
