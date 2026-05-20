import numpy as np
from overlap_detection.preprocessing import (
    make_overlap_band_mask,
    make_grayness_mask,
    combine_masks,
    apply_mask_mode,
)

def test_make_overlap_band_mask():
    shape = (100, 200, 3)
    
    # Test "both" (default)
    mask_both = make_overlap_band_mask(shape, 0.2)
    assert mask_both.shape == (100, 200)
    assert np.all(mask_both[:, :40] == 255)
    assert np.all(mask_both[:, -40:] == 255)
    assert np.all(mask_both[:, 40:-40] == 0)

    # Test "left"
    mask_left = make_overlap_band_mask(shape, 0.2, side="left")
    assert np.all(mask_left[:, :40] == 255)
    assert np.all(mask_left[:, 40:] == 0)

    # Test "right"
    mask_right = make_overlap_band_mask(shape, 0.2, side="right")
    assert np.all(mask_right[:, -40:] == 255)
    assert np.all(mask_right[:, :-40] == 0)

def test_make_grayness_mask():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:5, :, :] = [100, 100, 105]  # Gray, ptp=5 <= 15
    image[5:, :, :] = [200, 50, 50]    # Red, ptp=150 > 15
    mask = make_grayness_mask(image, 15)
    assert np.all(mask[:5, :] == 0)
    assert np.all(mask[5:, :] == 255)

def test_combine_masks():
    m1 = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    m2 = np.array([[255, 0], [255, 0]], dtype=np.uint8)
    combined = combine_masks(m1, m2)
    expected = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(combined, expected)

def test_apply_mask_mode():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # All colorful

    mask_no_mask = apply_mask_mode(image, "no_mask", 0.2, 15)
    assert np.sum(mask_no_mask == 255) == 100 * 80  # 40 left + 40 right

    mask_left = apply_mask_mode(image, "no_mask", 0.2, 15, side="left")
    assert np.sum(mask_left == 255) == 100 * 40

    mask_mask = apply_mask_mode(image, "mask", 0.2, 15)
    assert np.sum(mask_mask == 255) == 100 * 80  # All colorful, so same as band mask

    image[:50, :40, :] = 100 # Make top-left gray
    mask_mask_gray = apply_mask_mode(image, "mask", 0.2, 15)
    assert np.sum(mask_mask_gray == 255) == 100 * 80 - 50 * 40
