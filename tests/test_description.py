import pytest
import numpy as np
import cv2
from overlap_detection.types import Keypoint
from overlap_detection.description import describe, is_binary_descriptor
from overlap_detection.detection import detect


def _textured_image(size: int = 200) -> np.ndarray:
    """Solid white square on a dark, slightly-noisy background.
    Provides enough texture for every descriptor's gradient/intensity
    sampling to produce a meaningful (non-zero) feature vector."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[size // 5 : 4 * size // 5, size // 5 : 4 * size // 5] = 255
    rng = np.random.default_rng(0)
    img += (rng.random((size, size, 3)) * 20).astype(np.uint8)
    return img


def test_is_binary_descriptor():
    assert is_binary_descriptor("BRIEF")
    assert not is_binary_descriptor("SIFT")


# Canonical (dim, dtype) per descriptor.  Probed once from the running
# OpenCV / custom-impl combo; if any of these change the tests will catch it.
_DESCRIPTOR_SPECS: dict[str, tuple[int, type]] = {
    "SIFT":     (128, np.float32),
    "RootSIFT": (128, np.float32),
    "USURF":    (64,  np.float32),
    "DAISY":    (200, np.float32),
    "BRIEF":    (32,  np.uint8),     # 256 bits / 8
    "BRISK":    (64,  np.uint8),     # 512 bits / 8
    "SUFREAK":  (64,  np.uint8),     # 512 bits / 8
    "MLDB":     (61,  np.uint8),     # 486 bits, packed (2 padding bits)
    "LIOP":     (144, np.float32),   # 6 bins × 24 codes
}


@pytest.mark.parametrize("descriptor_name", list(_DESCRIPTOR_SPECS))
def test_describe_shape_and_dtype(descriptor_name):
    """Every descriptor produces an (N, D) matrix of the expected dim & dtype
    for a single keypoint near the centre of a textured image."""
    expected_dim, expected_dtype = _DESCRIPTOR_SPECS[descriptor_name]
    img = _textured_image()
    kps = [Keypoint(x=100.0, y=100.0, response=1.0, sigma=6.0)]

    out_kps, desc = describe(img, kps, descriptor_name, {}, default_sigma=6.0)

    assert len(out_kps) == 1, f"{descriptor_name} unexpectedly filtered the keypoint"
    assert desc.shape == (1, expected_dim), (
        f"{descriptor_name}: expected shape (1, {expected_dim}), got {desc.shape}"
    )
    assert desc.dtype == expected_dtype, (
        f"{descriptor_name}: expected dtype {expected_dtype}, got {desc.dtype}"
    )
    # Sanity: descriptor isn't all-zero on a textured input
    assert np.any(desc != 0), f"{descriptor_name} produced an all-zero descriptor"


def test_describe_rootsift_is_l2_normalised():
    """RootSIFT's post-processing must leave each row at unit L2 norm."""
    img = _textured_image()
    kps = [Keypoint(x=100.0, y=100.0, response=1.0, sigma=6.0)]
    _, desc = describe(img, kps, "RootSIFT", {}, default_sigma=6.0)
    norms = np.linalg.norm(desc, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_describe_liop_per_bin_l2_norm():
    """LIOP normalises each of its 6 ordinal-bin histograms independently
    to unit L2 norm before concatenation.  The full vector therefore has
    L2 norm equal to sqrt(number_of_nonzero_bins) ≤ √6."""
    img = _textured_image()
    kps = [Keypoint(x=100.0, y=100.0, response=1.0, sigma=6.0)]
    _, desc = describe(img, kps, "LIOP", {}, default_sigma=6.0)
    assert desc.shape == (1, 144)
    # 6 sub-blocks of 24 entries each, each with L2 norm in {0, 1}.
    for b in range(6):
        block_norm = float(np.linalg.norm(desc[0, b * 24 : (b + 1) * 24]))
        assert block_norm == pytest.approx(0.0) or block_norm == pytest.approx(1.0, abs=1e-5), (
            f"LIOP bin {b}: per-bin norm should be 0 or 1, got {block_norm}"
        )


def test_describe_mldb_via_akaze_native_path():
    """MLDB has two paths: the default custom NumPy implementation and the
    native OpenCV AKAZE compute() route (gated on AKAZE-detected keypoints
    carrying a valid class_id).  Exercise the native path explicitly."""
    img = _textured_image()
    mask = np.full(img.shape[:2], 255, dtype=np.uint8)
    kps = detect(img, mask, "AKAZE", {}, max_keypoints=20)
    if not kps:
        pytest.skip("AKAZE found no keypoints on the test image")
    assert kps[0].class_id is not None, "AKAZE keypoints should carry class_id"

    out_kps, desc = describe(img, kps, "MLDB", {}, default_sigma=6.0,
                             detector_name="AKAZE")
    assert len(out_kps) > 0
    assert desc.ndim == 2 and desc.shape[0] == len(out_kps)
    assert desc.dtype == np.uint8
