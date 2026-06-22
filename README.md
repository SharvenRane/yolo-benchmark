# yolo-benchmark

A small detection benchmarking harness written from scratch in NumPy. It
implements the three pieces you need to score an object detector and nothing
else: intersection over union, non maximum suppression, and mean average
precision. There is also a thin evaluation loop that wires them together so you
can hand it raw detections and get a number back.

I wrote this because I wanted the metric code to be readable rather than hidden
behind a framework. Every function here is a few dozen lines of plain array
math, and the test suite checks the behavior against values you can work out by
hand on a napkin.

## What is inside

```
src/
  iou.py        intersection over union, single pair and full pairwise matrix
  nms.py        greedy non maximum suppression
  map.py        average precision per class and mean average precision
  evaluate.py   a loop that runs class aware NMS then scores with mAP
tests/
  test_iou.py
  test_nms.py
  test_map.py
  test_evaluate.py
```

Boxes everywhere use the corner format `[x1, y1, x2, y2]`, with `(x1, y1)` the
top left corner and `(x2, y2)` the bottom right.

## The pieces

### Intersection over union

`iou_pair(box_a, box_b)` returns a single float in `[0, 1]`. `box_iou(a, b)`
takes two stacks of boxes shaped `(N, 4)` and `(M, 4)` and returns the full
`(N, M)` matrix of overlaps in one vectorized pass. Boxes that only touch along
an edge have zero overlapping area and therefore an IoU of zero, which the tests
pin down.

### Non maximum suppression

`nms(boxes, scores, iou_threshold)` runs the usual greedy loop. It keeps the
highest scoring box, throws away every remaining box that overlaps it by more
than the threshold, and repeats. The result is the list of surviving indices in
descending score order. A tight cluster of near duplicate boxes collapses to one
survivor, and boxes that overlap below the threshold are left alone.

### Mean average precision

`average_precision(predictions, ground_truths, iou_threshold)` scores one class.
Predictions are sorted by confidence, each one is greedily matched to the best
still available ground truth box above the IoU threshold, and the resulting
precision and recall curve is integrated. The integration uses all points
interpolation, the convention Pascal VOC adopted from 2010 onward and the one
COCO uses, where precision is made monotonically non increasing from the right
before taking the area.

`mean_average_precision(predictions, ground_truths, iou_threshold)` runs that
per class and averages over the classes present in the ground truth. It returns
both the mean and a per class breakdown.

A couple of edge cases worth calling out, since they trip people up. When a
class has no ground truth and no predictions, its AP is defined as 1.0. When it
has no ground truth but does have predictions, the AP is 0.0.

### Evaluation loop

`evaluate(detections, ground_truths, ...)` is the convenience wrapper. It runs
NMS independently for each image and class so boxes of different classes never
suppress one another, then scores whatever survives with mean average precision.
It hands back a dict with the mAP, the per class scores, and how many detections
made it through suppression.

```python
from src.evaluate import evaluate

detections = [
    # (image_id, class_id, score, [x1, y1, x2, y2])
    (0, 0, 0.95, [0, 0, 10, 10]),
    (0, 0, 0.90, [0.5, 0.5, 10.5, 10.5]),   # near duplicate, NMS drops it
    (1, 1, 0.99, [5, 5, 15, 15]),
]
ground_truths = [
    # (image_id, class_id, [x1, y1, x2, y2])
    (0, 0, [0, 0, 10, 10]),
    (1, 1, [5, 5, 15, 15]),
]

result = evaluate(detections, ground_truths)
print(result["map"], result["num_detections"])
```

## Running the tests

The suite is pure NumPy and runs in a fraction of a second on CPU with no
downloads. Point it at whichever interpreter has numpy and pytest installed:

```
pip install -r requirements.txt
python -m pytest tests/ -q
```

On my machine the full run reports `35 passed`. The tests check IoU against hand
computed fractions, confirm NMS removes overlapping boxes and respects the
threshold, and confirm mAP is exactly 1.0 for perfect predictions and drops once
you introduce missed detections, false positives, or loosely placed boxes.

## Notes

This is a metrics harness, not a detector. It assumes you already have boxes and
scores from somewhere, whether that is a real model or synthetic data, and its
job is to turn those into an honest number.
