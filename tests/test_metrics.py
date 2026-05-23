import pytest
import numpy as np
from overlap_detection.metrics import (
    per_corner_errors, corner_errors_hpatches,
    mean_corner_error, overlap_iou, categorize_result,
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


# ---------------------------------------------------------------------------
# HPatches-convention corner error
# ---------------------------------------------------------------------------


def _translation(dx: float, dy: float) -> np.ndarray:
    """2x3 affine: identity rotation, (dx, dy) translation."""
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def test_corner_errors_hpatches_identical_affines_is_zero():
    """Same affine on both sides → every corner error is exactly 0."""
    M = _translation(7.0, -4.0)
    errors = corner_errors_hpatches(M, M, image_a_shape=(480, 640, 3))
    assert errors.shape == (4,)
    np.testing.assert_allclose(errors, 0.0)


def test_corner_errors_hpatches_pure_translation_offset():
    """Estimated affine offset by (3, 4) from GT → every corner error is 5 px
    (3-4-5 triangle), regardless of image size, regardless of corner."""
    gt = _translation(0.0, 0.0)
    est = _translation(3.0, 4.0)
    errors = corner_errors_hpatches(est, gt, image_a_shape=(200, 300, 3))
    assert errors.shape == (4,)
    np.testing.assert_allclose(errors, 5.0)


def test_corner_errors_hpatches_corner_order_is_tl_tr_br_bl():
    """A pure 1-px x-shear (rows scale linearly with y) hits each corner
    differently, so the per-corner errors expose the constructor order.
    Affine row 0: [1, 0, 0]; row 1: [k, 1, 0] shifts y by k*x. With k = 1e-3
    and W=1000, H=500: corners (0,0), (W,0), (W,H), (0,H) shift in y by
    0, k*W=1, k*W=1, 0 respectively.  So errors are [0, 1, 1, 0]."""
    gt = _translation(0.0, 0.0)
    est = np.array([[1.0, 0.0, 0.0], [1e-3, 1.0, 0.0]], dtype=np.float64)
    errors = corner_errors_hpatches(est, gt, image_a_shape=(500, 1000, 3))
    # Order TL, TR, BR, BL — only the right-side corners (x=W) shift by 1.0
    np.testing.assert_allclose(errors, [0.0, 1.0, 1.0, 0.0], atol=1e-9)


def test_corner_errors_hpatches_allows_corners_outside_b():
    """Sanity: even when one affine pushes corners well outside any
    plausible B image rectangle, the function still returns finite errors —
    no clipping happens."""
    gt = _translation(0.0, 0.0)
    est = _translation(10_000.0, 10_000.0)
    errors = corner_errors_hpatches(est, gt, image_a_shape=(100, 100, 3))
    assert np.all(np.isfinite(errors))
    # 10_000 in both directions = ||(10_000, 10_000)|| ≈ 14142.13 for each corner
    np.testing.assert_allclose(errors, np.hypot(10_000, 10_000))
