import pytest
import numpy as np
from overlap_detection.geometry import compute_overlap_polygon, invert_affine, apply_affine

def test_apply_invert_affine():
    M = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]])
    pts = np.array([[0, 0], [10, 10]])
    res = apply_affine(pts, M)
    np.testing.assert_allclose(res, [[10, 20], [20, 30]])
    
    inv_M = invert_affine(M)
    pts_back = apply_affine(res, inv_M)
    np.testing.assert_allclose(pts, pts_back)

def test_compute_overlap_polygon():
    M = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0]])
    A_shape = (100, 100, 3)
    B_shape = (100, 100, 3)
    
    poly_A, poly_B = compute_overlap_polygon(M, A_shape, B_shape)
    assert len(poly_A) == 4
    assert len(poly_B) == 4
    # Image A translated by +10 in X. So A's overlap is X in [0, 90], B's overlap is X in [10, 100]
