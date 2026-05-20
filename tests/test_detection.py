import pytest
import numpy as np
import cv2
from overlap_detection.detection import detect


def _make_checkerboard(size: int = 200, sq: int = 25) -> np.ndarray:
    """Create a checkerboard with strong corners that all detectors can find.
    Apply slight blur so gradient-based detectors (FAST/AGAST) see
    smooth intensity transitions rather than hard binary edges."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(0, size, sq):
        for x in range(0, size, sq):
            if ((x // sq) + (y // sq)) % 2 == 0:
                img[y:y+sq, x:x+sq] = 255
    # Slight blur creates gradients for FAST/AGAST circle test
    import cv2
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=1.0)
    return img


@pytest.mark.parametrize("detector_name", [
    "Harris", "GFTT", "FAST", "AGAST", "BRISK", "SIFT", "KAZE", "AKAZE", "MSER"
])
def test_detect_synthetic(detector_name):
    """Each detector should find at least one keypoint on a checkerboard."""
    img = _make_checkerboard()
    mask = np.ones((200, 200), dtype=np.uint8) * 255
    kps = detect(img, mask, detector_name, {}, max_keypoints=500)
    assert len(kps) > 0, f"{detector_name} found 0 keypoints on checkerboard"


def test_detect_masked():
    """Zero-mask should yield zero keypoints."""
    img = _make_checkerboard()
    mask = np.zeros((200, 200), dtype=np.uint8)
    kps = detect(img, mask, "FAST", {}, max_keypoints=100)
    assert len(kps) == 0
