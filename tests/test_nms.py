import numpy as np
import pytest

from src.iou import box_iou
from src.nms import nms


def test_nms_removes_overlapping_duplicate():
    # Two nearly identical boxes plus one far away. NMS should keep the higher
    # scoring of the duplicate pair and the distant box.
    boxes = [
        [0.0, 0.0, 10.0, 10.0],
        [0.5, 0.5, 10.5, 10.5],
        [100.0, 100.0, 110.0, 110.0],
    ]
    scores = [0.9, 0.8, 0.95]
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert set(keep) == {0, 2}
    # highest score overall comes first
    assert keep[0] == 2


def test_nms_keeps_all_when_no_overlap():
    boxes = [
        [0.0, 0.0, 5.0, 5.0],
        [10.0, 10.0, 15.0, 15.0],
        [20.0, 20.0, 25.0, 25.0],
    ]
    scores = [0.5, 0.6, 0.7]
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert set(keep) == {0, 1, 2}


def test_nms_keeps_overlap_below_threshold():
    # Two boxes overlapping at IoU 1/7 which is below a 0.5 threshold.
    boxes = [
        [0.0, 0.0, 2.0, 2.0],
        [1.0, 1.0, 3.0, 3.0],
    ]
    scores = [0.9, 0.8]
    assert box_iou(np.array(boxes[:1]), np.array(boxes[1:]))[0, 0] == pytest.approx(1.0 / 7.0)
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert set(keep) == {0, 1}


def test_nms_threshold_controls_suppression():
    boxes = [
        [0.0, 0.0, 2.0, 2.0],
        [1.0, 1.0, 3.0, 3.0],
    ]
    scores = [0.9, 0.8]
    # With a threshold below the actual IoU (1/7 ~= 0.143) the lower box is removed.
    keep_strict = nms(boxes, scores, iou_threshold=0.1)
    assert keep_strict == [0]


def test_nms_output_sorted_by_score():
    boxes = [
        [0.0, 0.0, 5.0, 5.0],
        [50.0, 50.0, 55.0, 55.0],
        [100.0, 100.0, 105.0, 105.0],
    ]
    scores = [0.3, 0.9, 0.6]
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep == [1, 2, 0]


def test_nms_cluster_collapses_to_one():
    # A tight cluster of five overlapping boxes should collapse to the top one.
    rng = np.random.default_rng(3)
    base = np.array([10.0, 10.0, 30.0, 30.0])
    boxes = [base + rng.uniform(-1.0, 1.0, size=4) for _ in range(5)]
    boxes.append(np.array([200.0, 200.0, 220.0, 220.0]))
    scores = [0.5, 0.55, 0.6, 0.65, 0.7, 0.4]
    keep = nms(boxes, scores, iou_threshold=0.5)
    # one survivor from the cluster (index 4, highest score) and the far box
    assert 4 in keep
    assert 5 in keep
    assert len(keep) == 2


def test_nms_empty_input():
    assert nms([], [], iou_threshold=0.5) == []


def test_nms_length_mismatch_raises():
    with pytest.raises(ValueError):
        nms([[0, 0, 1, 1]], [0.5, 0.6])
