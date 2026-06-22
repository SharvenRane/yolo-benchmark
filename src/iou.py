"""Intersection over union for axis aligned bounding boxes.

Boxes use the corner format ``[x1, y1, x2, y2]`` where ``(x1, y1)`` is the top
left corner and ``(x2, y2)`` is the bottom right corner.
"""

from __future__ import annotations

import numpy as np


def iou_pair(box_a, box_b):
    """Return the IoU of two single boxes.

    Each box is a sequence of four numbers ``[x1, y1, x2, y2]``. The result is a
    float in the closed interval ``[0, 1]``.
    """
    a = np.asarray(box_a, dtype=np.float64)
    b = np.asarray(box_b, dtype=np.float64)

    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0
    return float(inter / union)


def box_iou(boxes_a, boxes_b):
    """Compute the pairwise IoU matrix between two sets of boxes.

    ``boxes_a`` has shape ``(N, 4)`` and ``boxes_b`` has shape ``(M, 4)``. The
    return value has shape ``(N, M)`` where entry ``[i, j]`` is the IoU of
    ``boxes_a[i]`` with ``boxes_b[j]``.
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)

    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    area_a = (a[:, 2] - a[:, 0]).clip(min=0) * (a[:, 3] - a[:, 1]).clip(min=0)
    area_b = (b[:, 2] - b[:, 0]).clip(min=0) * (b[:, 3] - b[:, 1]).clip(min=0)

    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clip(min=0)
    inter_h = (inter_y2 - inter_y1).clip(min=0)
    inter = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter

    iou = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
    return iou.astype(np.float64)
