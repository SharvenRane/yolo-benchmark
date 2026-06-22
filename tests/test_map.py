import numpy as np
import pytest

from src.map import average_precision, mean_average_precision


def _grid_ground_truths(n, class_id=0):
    """Build n well separated ground truth boxes across several images."""
    gts = []
    for i in range(n):
        x = (i % 4) * 100.0
        y = (i // 4) * 100.0
        image_id = i % 3
        gts.append((image_id, class_id, [x, y, x + 20.0, y + 20.0]))
    return gts


def test_ap_perfect_predictions_is_one():
    gts = [(0, [0, 0, 10, 10]), (0, [50, 50, 60, 60]), (1, [5, 5, 15, 15])]
    preds = [(img, 0.9, box) for img, box in gts]
    assert average_precision(preds, gts, iou_threshold=0.5) == pytest.approx(1.0)


def test_ap_no_predictions_is_zero():
    gts = [(0, [0, 0, 10, 10])]
    assert average_precision([], gts) == pytest.approx(0.0)


def test_ap_no_gt_no_pred_is_one():
    assert average_precision([], []) == pytest.approx(1.0)


def test_ap_no_gt_with_pred_is_zero():
    preds = [(0, 0.9, [0, 0, 10, 10])]
    assert average_precision(preds, []) == pytest.approx(0.0)


def test_map_perfect_predictions_is_one():
    gts = [(g[0], 0, g[1]) for g in _grid_ground_truths(12)[:0]]  # placeholder
    gts = _grid_ground_truths(12, class_id=0) + _grid_ground_truths(8, class_id=1)
    preds = [(img, cid, 0.99, box) for img, cid, box in gts]
    map_value, per_class = mean_average_precision(preds, gts, iou_threshold=0.5)
    assert map_value == pytest.approx(1.0)
    assert per_class[0] == pytest.approx(1.0)
    assert per_class[1] == pytest.approx(1.0)


def test_map_degrades_with_missed_detections():
    gts = _grid_ground_truths(12, class_id=0)
    perfect = [(img, cid, 0.99, box) for img, cid, box in gts]
    map_perfect, _ = mean_average_precision(perfect, gts)

    # Drop half of the predictions: recall is capped, so AP must fall.
    degraded = perfect[: len(perfect) // 2]
    map_degraded, _ = mean_average_precision(degraded, gts)

    assert map_perfect == pytest.approx(1.0)
    assert map_degraded < map_perfect
    assert map_degraded < 0.75


def test_map_degrades_with_false_positives():
    gts = _grid_ground_truths(8, class_id=0)
    perfect = [(img, cid, 0.99, box) for img, cid, box in gts]
    map_perfect, _ = mean_average_precision(perfect, gts)

    # Add high confidence boxes that match nothing.
    junk = [(img, 0, 0.999, [900.0 + img, 900.0, 905.0 + img, 905.0]) for img in range(3)]
    map_noisy, _ = mean_average_precision(perfect + junk, gts)

    assert map_perfect == pytest.approx(1.0)
    assert map_noisy < map_perfect


def test_map_degrades_with_loose_boxes():
    # Predictions shifted so far that IoU falls below threshold count as misses.
    gts = _grid_ground_truths(8, class_id=0)
    shifted = [(img, cid, 0.9, [b[0] + 18.0, b[1] + 18.0, b[2] + 18.0, b[3] + 18.0])
               for img, cid, b in gts]
    map_value, _ = mean_average_precision(shifted, gts, iou_threshold=0.5)
    assert map_value < 1.0


def test_ap_ranking_matters():
    # Same set of one TP and one FP. Scoring the TP higher gives a better AP
    # than scoring the FP higher.
    gts = [(0, [0, 0, 10, 10])]
    tp_box = [0, 0, 10, 10]
    fp_box = [500, 500, 510, 510]

    good_order = [(0, 0.9, tp_box), (0, 0.1, fp_box)]
    bad_order = [(0, 0.9, fp_box), (0, 0.1, tp_box)]

    ap_good = average_precision(good_order, gts)
    ap_bad = average_precision(bad_order, gts)
    assert ap_good == pytest.approx(1.0)
    assert ap_bad < ap_good


def test_double_detection_counts_one_tp():
    # Two predictions on the same single ground truth: the first is a TP, the
    # second a duplicate FP. The duplicate does not lower AP here because all
    # points interpolation keeps the precision at full recall, but it must show
    # up as a false positive rather than a second true positive. We verify that
    # by adding a second ground truth that the duplicate cannot rescue.
    gts = [(0, [0, 0, 10, 10]), (1, [0, 0, 10, 10])]
    preds = [
        (0, 0.9, [0, 0, 10, 10]),   # TP on image 0
        (0, 0.8, [0, 0, 10, 10]),   # duplicate on image 0, must be an FP
    ]
    ap = average_precision(preds, gts)
    # Only one of two ground truths is ever recalled, so AP is capped at 0.5.
    assert ap == pytest.approx(0.5)


def test_duplicate_does_not_double_count_recall():
    # A single ground truth with two identical predictions reaches recall 1.0
    # exactly once. AP stays at 1.0 under all points interpolation.
    gts = [(0, [0, 0, 10, 10])]
    preds = [(0, 0.9, [0, 0, 10, 10]), (0, 0.8, [0, 0, 10, 10])]
    ap = average_precision(preds, gts)
    assert ap == pytest.approx(1.0)


def test_map_empty_gt_returns_zero():
    map_value, per_class = mean_average_precision([], [])
    assert map_value == pytest.approx(0.0)
    assert per_class == {}
