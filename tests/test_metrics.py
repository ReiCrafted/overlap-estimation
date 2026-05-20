import pytest
import numpy as np
from overlap_detection.metrics import per_corner_errors, overlap_rms_error, overlap_iou

def test_per_corner_errors():
    p1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    p2 = np.array([[1, 0], [11, 0], [11, 10], [1, 10]])
    errors = per_corner_errors(p1, p2)
    np.testing.assert_allclose(errors, [1.0, 1.0, 1.0, 1.0])

def test_overlap_rms_error():
    errors = np.array([3.0, 4.0]) # RMS should be sqrt((9+16)/2) = sqrt(12.5) ~ 3.5355
    rms = overlap_rms_error(errors)
    assert np.isclose(rms, np.sqrt(12.5))

def test_overlap_iou():
    p1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    p2 = np.array([[5, 0], [15, 0], [15, 10], [5, 10]])
    # Intersection is [5,10] x [0,10] = 50 area
    # Union is [0,15] x [0,10] = 150 area
    iou = overlap_iou(p1, p2)
    assert np.isclose(iou, 50.0 / 150.0)
