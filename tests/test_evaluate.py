import pytest

from src.evaluate import apply_nms_per_image, evaluate


def test_nms_per_image_is_class_aware():
    # Two overlapping boxes of different classes must both survive.
    dets = [
        (0, 0, 0.9, [0, 0, 10, 10]),
        (0, 1, 0.8, [0, 0, 10, 10]),
    ]
    kept = apply_nms_per_image(dets, iou_threshold=0.5)
    assert len(kept) == 2
    classes = sorted(c for _img, c, _s, _b in kept)
    assert classes == [0, 1]


def test_nms_per_image_suppresses_same_class_duplicate():
    dets = [
        (0, 0, 0.9, [0, 0, 10, 10]),
        (0, 0, 0.8, [0, 0, 10, 10]),
    ]
    kept = apply_nms_per_image(dets, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0][2] == pytest.approx(0.9)


def test_evaluate_perfect_after_nms_is_one():
    gts = [
        (0, 0, [0, 0, 10, 10]),
        (0, 0, [50, 50, 60, 60]),
        (1, 1, [5, 5, 15, 15]),
    ]
    # Duplicate detections that NMS should collapse, leaving perfect coverage.
    dets = [
        (0, 0, 0.95, [0, 0, 10, 10]),
        (0, 0, 0.90, [0.5, 0.5, 10.5, 10.5]),
        (0, 0, 0.92, [50, 50, 60, 60]),
        (1, 1, 0.99, [5, 5, 15, 15]),
    ]
    result = evaluate(dets, gts, nms_iou_threshold=0.5, map_iou_threshold=0.5)
    assert result["map"] == pytest.approx(1.0)
    assert result["num_detections"] == 3


def test_evaluate_degrades_with_extra_noise():
    gts = [(0, 0, [0, 0, 10, 10]), (1, 0, [0, 0, 10, 10])]
    clean = [(0, 0, 0.9, [0, 0, 10, 10]), (1, 0, 0.9, [0, 0, 10, 10])]
    noisy = clean + [(0, 0, 0.99, [500, 500, 510, 510])]

    clean_result = evaluate(clean, gts)
    noisy_result = evaluate(noisy, gts)

    assert clean_result["map"] == pytest.approx(1.0)
    assert noisy_result["map"] < clean_result["map"]
