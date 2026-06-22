"""Detection benchmarking harness: IoU, NMS, and mAP from scratch."""

from .iou import box_iou, iou_pair
from .nms import nms
from .map import average_precision, mean_average_precision
from .evaluate import apply_nms_per_image, evaluate

__all__ = [
    "box_iou",
    "iou_pair",
    "nms",
    "average_precision",
    "mean_average_precision",
    "apply_nms_per_image",
    "evaluate",
]
