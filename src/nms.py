"""Greedy non maximum suppression."""

from __future__ import annotations

import numpy as np

from .iou import box_iou


def nms(boxes, scores, iou_threshold=0.5):
    """Run greedy non maximum suppression.

    ``boxes`` has shape ``(N, 4)`` in corner format and ``scores`` has shape
    ``(N,)``. The function repeatedly keeps the highest scoring box and discards
    every remaining box whose IoU with it exceeds ``iou_threshold``.

    The return value is a list of integer indices into the original arrays,
    ordered by descending score.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    if boxes.shape[0] != scores.shape[0]:
        raise ValueError("boxes and scores must have matching length")

    if boxes.shape[0] == 0:
        return []

    order = list(np.argsort(-scores, kind="stable"))
    keep = []

    while order:
        current = order.pop(0)
        keep.append(int(current))
        if not order:
            break

        ious = box_iou(boxes[current][None, :], boxes[order])[0]
        survivors = [idx for idx, iou in zip(order, ious) if iou <= iou_threshold]
        order = survivors

    return keep
