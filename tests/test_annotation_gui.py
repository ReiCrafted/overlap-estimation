from overlap_detection.annotation_gui import decompose_affine, build_affine, compose_affine
from overlap_detection.geometry import invert_affine

def test_affine_decomposition_roundtrip():
    import numpy as np
    tx, ty = 15.0, -30.0
    angle = 45.0
    sx, sy = 1.2, 0.8
    
    M = build_affine(tx, ty, angle, sx, sy)
    
    dtx, dty, dang, dsx, dsy = decompose_affine(M)
    
    np.testing.assert_allclose(tx, dtx)
    np.testing.assert_allclose(ty, dty)
    np.testing.assert_allclose(angle, dang)
    np.testing.assert_allclose(sx, dsx)
    np.testing.assert_allclose(sy, dsy)
