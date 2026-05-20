import pytest
import numpy as np
import cv2
from overlap_detection.types import Keypoint
from overlap_detection.description import describe, is_binary_descriptor

def test_is_binary_descriptor():
    assert is_binary_descriptor("BRIEF")
    assert not is_binary_descriptor("SIFT")

def test_describe_sift():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = 255
    kps = [Keypoint(x=50.0, y=50.0, response=1.0)]
    
    out_kps, desc = describe(img, kps, "SIFT", {}, 6.0)
    assert len(out_kps) == 1
    assert desc.shape == (1, 128)
    assert desc.dtype == np.float32

def test_describe_rootsift():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = 255
    kps = [Keypoint(x=50.0, y=50.0, response=1.0)]
    
    out_kps, desc = describe(img, kps, "RootSIFT", {}, 6.0)
    assert len(out_kps) == 1
    assert desc.shape == (1, 128)
    # Check L2 norm is approx 1
    norms = np.linalg.norm(desc, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
