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


def test_describe_brief_descriptor_params_change_byte_count():
    """Sanity that ``descriptor_params`` keeps reaching OpenCV factories too.
    BRIEF supports ``bytes ∈ {16, 32, 64}``; pick a non-default and confirm
    the descriptor width follows."""
    img = _textured_image()
    kps = [Keypoint(x=100.0, y=100.0, response=1.0, sigma=6.0)]
    _, d_default = describe(img, kps, "BRIEF", {}, default_sigma=6.0)
    _, d_wide = describe(img, kps, "BRIEF", {"bytes": 64}, default_sigma=6.0)
    assert d_default.shape == (1, 32)
    assert d_wide.shape == (1, 64)


def test_describe_sift_descriptor_params_pass_through():
    """SIFT's ``contrastThreshold`` is a description-time param that doesn't
    change shape; just verify the kwarg is accepted (no TypeError)."""
    img = _textured_image()
    kps = [Keypoint(x=100.0, y=100.0, response=1.0, sigma=6.0)]
    _, desc = describe(img, kps, "SIFT",
                       {"contrastThreshold": 0.02}, default_sigma=6.0)
    assert desc.shape == (1, 128)


def test_describe_liop_descriptor_params_change_dim():
    """``descriptor_params`` flows into the custom LIOP path.  Halving the
    number of bins halves the descriptor dimension (`n_bins * n_neighbors!`)."""
    img = _textured_image()
    kps = [Keypoint(x=100.0, y=100.0, response=1.0, sigma=6.0)]
    _, desc_default = describe(img, kps, "LIOP", {}, default_sigma=6.0)
    _, desc_tuned = describe(img, kps, "LIOP",
                             {"n_bins": 3}, default_sigma=6.0)
    assert desc_default.shape == (1, 6 * 24)   # 144
    assert desc_tuned.shape == (1, 3 * 24)     # 72


def test_describe_liop_descriptor_params_n_neighbors_changes_dim():
    """K = 3 ⇒ 3! = 6 codes per bin → 6 * 6 = 36 values."""
    img = _textured_image()
    kps = [Keypoint(x=100.0, y=100.0, response=1.0, sigma=6.0)]
    _, desc = describe(img, kps, "LIOP",
                       {"n_neighbors": 3}, default_sigma=6.0)
    assert desc.shape == (1, 6 * 6)


def test_describe_liop_rejects_invalid_patch_size():
    """patch_size must be odd and >= 3 — invalid values surface as ValueError
    from the underlying geometry validation, not silently."""
    img = _textured_image()
    kps = [Keypoint(x=100.0, y=100.0, response=1.0, sigma=6.0)]
    with pytest.raises(ValueError, match="patch_size"):
        describe(img, kps, "LIOP", {"patch_size": 40}, default_sigma=6.0)


def test_describe_mldb_descriptor_params_change_bit_count():
    """``descriptor_params`` flows into the custom MLDB path.  A single 2×2
    grid produces C(4, 2) = 6 pairs × 3 channels = 18 bits → 3 bytes."""
    img = _textured_image()
    kps = [Keypoint(x=100.0, y=100.0, response=1.0, sigma=6.0)]
    _, desc_default = describe(img, kps, "MLDB", {}, default_sigma=6.0)
    _, desc_tuned = describe(img, kps, "MLDB",
                             {"grids": ((2, 2),)}, default_sigma=6.0)
    assert desc_default.shape == (1, 61)   # 486 bits / 8 rounded up
    assert desc_tuned.shape == (1, 3)      # 18 bits / 8 rounded up


def test_describe_mldb_rejects_indivisible_patch_size():
    """patch_size must be divisible by every grid dim."""
    img = _textured_image()
    kps = [Keypoint(x=100.0, y=100.0, response=1.0, sigma=6.0)]
    with pytest.raises(ValueError, match="divisible"):
        # 30 is divisible by 2 and 3 but not by 4 → grid (4, 4) fails
        describe(img, kps, "MLDB", {"patch_size": 30}, default_sigma=6.0)


def test_describe_mldb_custom_param_changes_descriptor_value():
    """The smoothing sigma changes the descriptor bits — verifies the param
    actually reaches the algorithm rather than just being passed and dropped."""
    img = _textured_image()
    kps = [Keypoint(x=100.0, y=100.0, response=1.0, sigma=6.0)]
    _, d_low = describe(img, kps, "MLDB",
                        {"smooth_sigma": 0.5}, default_sigma=6.0)
    _, d_high = describe(img, kps, "MLDB",
                         {"smooth_sigma": 5.0}, default_sigma=6.0)
    assert d_low.shape == d_high.shape == (1, 61)
    assert not np.array_equal(d_low, d_high), (
        "smooth_sigma should affect MLDB descriptor bits — got identical output"
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
