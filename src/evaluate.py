"""A small evaluation loop that ties IoU, NMS, and mAP together.

The loop takes raw detections per image, applies per class NMS to suppress
overlapping boxes, then scores the surviving detections against the ground
truth with mean average precision.
"""

from __future__ import annotations

from collections import defaultdict

from .nms import nms
from .map import mean_average_precision


def apply_nms_per_image(detections, iou_threshold=0.5):
    """Apply class aware NMS to a list of raw detections.

    ``detections`` is an iterable of ``(image_id, class_id, score, box)``. NMS is
    run independently for each ``(image_id, class_id)`` group so that boxes of
    different classes never suppress each other. The return value is a filtered
    list in the same tuple format.
    """
    grouped = defaultdict(list)
    for image_id, class_id, score, box in detections:
        grouped[(image_id, class_id)].append((score, box))

    kept = []
    for (image_id, class_id), items in grouped.items():
        boxes = [box for _score, box in items]
        scores = [score for score, _box in items]
        keep_idx = nms(boxes, scores, iou_threshold=iou_threshold)
        for i in keep_idx:
            kept.append((image_id, class_id, scores[i], boxes[i]))
    return kept


def evaluate(detections, ground_truths, nms_iou_threshold=0.5, map_iou_threshold=0.5):
    """Run the full evaluation loop.

    ``detections`` is an iterable of ``(image_id, class_id, score, box)`` and
    ``ground_truths`` is an iterable of ``(image_id, class_id, box)``. The
    function suppresses overlapping detections, then computes mean average
    precision.

    The return value is a dict with keys ``map``, ``per_class``, and
    ``num_detections`` (the count of detections that survived NMS).
    """
    filtered = apply_nms_per_image(detections, iou_threshold=nms_iou_threshold)
    map_value, per_class = mean_average_precision(
        filtered, ground_truths, iou_threshold=map_iou_threshold
    )
    return {
        "map": map_value,
        "per_class": per_class,
        "num_detections": len(filtered),
    }
