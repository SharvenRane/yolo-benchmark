import numpy as np
import pytest

from src.iou import box_iou, iou_pair


def test_identical_boxes_iou_is_one():
    box = [0.0, 0.0, 10.0, 10.0]
    assert iou_pair(box, box) == pytest.approx(1.0)


def test_disjoint_boxes_iou_is_zero():
    a = [0.0, 0.0, 10.0, 10.0]
    b = [20.0, 20.0, 30.0, 30.0]
    assert iou_pair(a, b) == pytest.approx(0.0)


def test_touching_boxes_iou_is_zero():
    # Boxes share an edge but have no overlapping area.
    a = [0.0, 0.0, 10.0, 10.0]
    b = [10.0, 0.0, 20.0, 10.0]
    assert iou_pair(a, b) == pytest.approx(0.0)


def test_half_overlap_hand_computed():
    # Two unit-area-scaled boxes overlapping on exactly half their width.
    a = [0.0, 0.0, 2.0, 2.0]   # area 4
    b = [1.0, 0.0, 3.0, 2.0]   # area 4
    # intersection is the 1x2 strip = 2, union = 4 + 4 - 2 = 6
    assert iou_pair(a, b) == pytest.approx(2.0 / 6.0)


def test_quarter_overlap_hand_computed():
    a = [0.0, 0.0, 2.0, 2.0]   # area 4
    b = [1.0, 1.0, 3.0, 3.0]   # area 4
    # intersection is the 1x1 corner = 1, union = 4 + 4 - 1 = 7
    assert iou_pair(a, b) == pytest.approx(1.0 / 7.0)


def test_contained_box_hand_computed():
    outer = [0.0, 0.0, 4.0, 4.0]   # area 16
    inner = [1.0, 1.0, 3.0, 3.0]   # area 4, fully inside
    # intersection = 4, union = 16
    assert iou_pair(outer, inner) == pytest.approx(4.0 / 16.0)


def test_iou_is_symmetric():
    a = [0.0, 0.0, 5.0, 7.0]
    b = [2.0, 3.0, 9.0, 8.0]
    assert iou_pair(a, b) == pytest.approx(iou_pair(b, a))


def test_iou_bounded_random():
    rng = np.random.default_rng(0)
    for _ in range(200):
        xs = np.sort(rng.uniform(0, 50, size=2))
        ys = np.sort(rng.uniform(0, 50, size=2))
        a = [xs[0], ys[0], xs[1], ys[1]]
        xs = np.sort(rng.uniform(0, 50, size=2))
        ys = np.sort(rng.uniform(0, 50, size=2))
        b = [xs[0], ys[0], xs[1], ys[1]]
        v = iou_pair(a, b)
        assert -1e-9 <= v <= 1.0 + 1e-9


def test_box_iou_matrix_shape_and_values():
    a = np.array([[0, 0, 2, 2], [0, 0, 2, 2]], dtype=float)
    b = np.array([[0, 0, 2, 2], [1, 0, 3, 2], [10, 10, 12, 12]], dtype=float)
    m = box_iou(a, b)
    assert m.shape == (2, 3)
    assert m[0, 0] == pytest.approx(1.0)
    assert m[0, 1] == pytest.approx(2.0 / 6.0)
    assert m[0, 2] == pytest.approx(0.0)
    # both rows of a are identical, so the matrix rows must match
    np.testing.assert_allclose(m[0], m[1])


def test_box_iou_matches_pairwise():
    rng = np.random.default_rng(7)
    a = []
    for _ in range(5):
        xs = np.sort(rng.uniform(0, 20, size=2))
        ys = np.sort(rng.uniform(0, 20, size=2))
        a.append([xs[0], ys[0], xs[1], ys[1]])
    b = []
    for _ in range(4):
        xs = np.sort(rng.uniform(0, 20, size=2))
        ys = np.sort(rng.uniform(0, 20, size=2))
        b.append([xs[0], ys[0], xs[1], ys[1]])
    m = box_iou(a, b)
    for i in range(len(a)):
        for j in range(len(b)):
            assert m[i, j] == pytest.approx(iou_pair(a[i], b[j]))


def test_box_iou_empty_inputs():
    empty = np.zeros((0, 4))
    full = np.array([[0, 0, 1, 1]], dtype=float)
    assert box_iou(empty, full).shape == (0, 1)
    assert box_iou(full, empty).shape == (1, 0)
    assert box_iou(empty, empty).shape == (0, 0)
