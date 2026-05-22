import pytest
import numpy as np
from overlap_detection.metrics import (
    per_corner_errors, mean_corner_error, overlap_iou, categorize_result,
)

def test_per_corner_errors():
    p1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    p2 = np.array([[1, 0], [11, 0], [11, 10], [1, 10]])
    errors = per_corner_errors(p1, p2)
    np.testing.assert_allclose(errors, [1.0, 1.0, 1.0, 1.0])

def test_mean_corner_error():
    errors = np.array([2.0, 4.0, 6.0, 8.0])
    assert np.isclose(mean_corner_error(errors), 5.0)

def test_categorize_result_tiers():
    tiers = (3.0, 5.0, 10.0)
    assert categorize_result(True, 1.4, tiers) == "acc_at_3"
    assert categorize_result(True, 3.0, tiers) == "acc_at_3"   # inclusive
    assert categorize_result(True, 4.2, tiers) == "acc_at_5"
    assert categorize_result(True, 8.9, tiers) == "acc_at_10"
    assert categorize_result(True, 14.0, tiers) == "false_match"
    assert categorize_result(False, None, tiers) == "no_match"
    assert categorize_result(True, None, tiers) == "no_match"  # GT missing

def test_overlap_iou():
    p1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    p2 = np.array([[5, 0], [15, 0], [15, 10], [5, 10]])
    # Intersection is [5,10] x [0,10] = 50 area
    # Union is [0,15] x [0,10] = 150 area
    iou = overlap_iou(p1, p2)
    assert np.isclose(iou, 50.0 / 150.0)
