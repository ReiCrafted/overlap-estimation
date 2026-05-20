import pytest
import numpy as np
from overlap_detection.types import Keypoint
from overlap_detection.verification import verify_affine


def test_verify_affine():
    """Known pure-translation: points shifted by (5, 5) should recover that."""
    # Need enough non-collinear points for affine estimation
    np.random.seed(42)
    n = 20
    pts = np.random.uniform(10, 90, size=(n, 2))

    kps_A = [Keypoint(x=float(p[0]), y=float(p[1]), response=1.0) for p in pts]
    kps_B = [Keypoint(x=float(p[0]+5), y=float(p[1]+5), response=1.0) for p in pts]

    matches = np.array([[i, i, 0.1] for i in range(n)], dtype=np.float32)

    M, inliers = verify_affine(matches, kps_A, kps_B, "PROSAC", 5.0, 1000, 0.99)
    assert M is not None, "Affine estimation returned None"
    assert M.shape == (2, 3)
    assert np.all(inliers)

    # Check translation component is ~(5, 5)
    np.testing.assert_allclose(M[:, 2], [5.0, 5.0], atol=0.5)
