import numpy as np
from overlap_detection.preprocessing import (
    make_overlap_band_mask,
    make_saturation_brightness_mask,
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


def test_make_saturation_brightness_mask_low_sat_mid_brightness_excluded():
    # Frame-like pixel: sat ≈ 0.04, brightness = 105 → inside frame band → excluded (0).
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :, :] = [100, 102, 105]
    mask = make_saturation_brightness_mask(image)
    assert np.all(mask == 0)


def test_make_saturation_brightness_mask_high_sat_kept():
    # Saturated pixel (red): sat = (200-50)/200 = 0.75 → kept (255).
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :, :] = [200, 50, 50]
    mask = make_saturation_brightness_mask(image)
    assert np.all(mask == 255)


def test_make_saturation_brightness_mask_dark_pixel_kept():
    # Low-sat but below brightness_lo (default 15) → kept (content can be very dark).
    image = np.full((4, 4, 3), 10, dtype=np.uint8)
    mask = make_saturation_brightness_mask(image)
    assert np.all(mask == 255)


def test_make_saturation_brightness_mask_bright_pixel_kept():
    # Low-sat but above brightness_hi (default 180) → kept (bright highlights).
    image = np.full((4, 4, 3), 220, dtype=np.uint8)
    mask = make_saturation_brightness_mask(image)
    assert np.all(mask == 255)


def test_make_saturation_brightness_mask_mixed_image():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:5, :, :] = [100, 102, 105]   # frame-like
    image[5:, :, :] = [200, 50, 50]     # saturated content
    mask = make_saturation_brightness_mask(image)
    assert np.all(mask[:5, :] == 0)
    assert np.all(mask[5:, :] == 255)


def test_make_saturation_brightness_mask_zero_safe():
    # All-black pixels: ch_max = 0, divide guard must return sat = 0.
    # ch_max = 0 < brightness_lo = 15, so they are kept (255).
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = make_saturation_brightness_mask(image)
    assert np.all(mask == 255)


def test_combine_masks():
    m1 = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    m2 = np.array([[255, 0], [255, 0]], dtype=np.uint8)
    combined = combine_masks(m1, m2)
    expected = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(combined, expected)


def test_apply_mask_mode():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # saturated red everywhere → kept by frame mask

    mask_no_mask = apply_mask_mode(image, "no_mask", 0.2)
    assert np.sum(mask_no_mask == 255) == 100 * 80  # 40 left + 40 right

    mask_left = apply_mask_mode(image, "no_mask", 0.2, side="left")
    assert np.sum(mask_left == 255) == 100 * 40

    mask_mask = apply_mask_mode(image, "mask", 0.2)
    # All pixels saturated → frame mask keeps all → result matches band mask.
    assert np.sum(mask_mask == 255) == 100 * 80

    # Paint top-left frame-like (low sat, mid brightness) → those pixels excluded.
    image[:50, :40, :] = [100, 102, 105]
    mask_mask_frame = apply_mask_mode(image, "mask", 0.2)
    assert np.sum(mask_mask_frame == 255) == 100 * 80 - 50 * 40
