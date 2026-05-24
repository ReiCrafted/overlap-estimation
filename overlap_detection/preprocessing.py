import cv2
import numpy as np

def make_overlap_band_mask(
    image_shape: tuple[int, int, int],
    band_fraction: float,
    side: str = "both",
) -> np.ndarray:
    """Returns mask of shape (H, W) with 255 in the left and/or right edge
    bands (each `band_fraction * W` wide), 0 elsewhere. The top and
    bottom edges are NOT masked — overlap is expected along the horizontal
    motion axis of the gantry."""
    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    band_width = int(W * band_fraction)
    if band_width > 0:
        if side in ("left", "both"):
            mask[:, :band_width] = 255
        if side in ("right", "both"):
            mask[:, -band_width:] = 255
    return mask


def make_saturation_brightness_mask(
    image: np.ndarray,
    sat_threshold: float = 0.12,
    brightness_lo: int = 15,
    brightness_hi: int = 180,
) -> np.ndarray:
    """Returns mask with 0 where the pixel matches the cassette-frame
    signature (low saturation AND brightness in the plastic-frame band),
    255 elsewhere.

    Operates on uint8 RGB. Empirically derived from per-pixel CSV samples
    of frame vs. non-frame regions (see project_overview.md §Stage 1).
    The frame pixels cluster tightly at low saturation = (max-min)/max
    and brightness = max(R,G,B) in a mid-range band; non-frame (plant /
    soil) pixels are either much darker, much brighter, or significantly
    chromatic.
    """
    img = image.astype(np.float32)
    ch_max = img.max(axis=2)
    ch_min = img.min(axis=2)
    sat = np.divide(ch_max - ch_min, ch_max,
                    out=np.zeros_like(ch_max), where=ch_max > 0)
    is_frame = (sat < sat_threshold) & (ch_max >= brightness_lo) & (ch_max <= brightness_hi)
    return np.where(is_frame, 0, 255).astype(np.uint8)


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    """Bitwise AND of multiple masks."""
    if not masks:
        raise ValueError("At least one mask is required.")
    result = masks[0]
    for mask in masks[1:]:
        result = cv2.bitwise_and(result, mask)
    return result


def apply_mask_mode(
    image: np.ndarray,
    mode: str,
    band_fraction: float,
    side: str = "both",
    sat_threshold: float = 0.12,
    brightness_lo: int = 15,
    brightness_hi: int = 180,
) -> np.ndarray:
    """Returns the final detection mask per mode:
    - "no_mask":  overlap band mask only
    - "mask":     overlap band AND saturation/brightness frame mask
    - "fallback": same as "no_mask" (legacy alias); fallback re-run logic
                  that swaps in the stricter mask lives in the orchestrator
    """
    band_mask = make_overlap_band_mask(image.shape, band_fraction, side=side)
    if mode == "mask":
        sat_mask = make_saturation_brightness_mask(
            image, sat_threshold, brightness_lo, brightness_hi,
        )
        return combine_masks(band_mask, sat_mask)
    elif mode in ("no_mask", "fallback"):
        return band_mask
    else:
        raise ValueError(f"Unknown mask mode: {mode}")
