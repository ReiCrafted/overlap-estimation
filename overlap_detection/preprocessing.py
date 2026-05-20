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

def make_grayness_mask(
    image: np.ndarray,
    gray_threshold: int,
) -> np.ndarray:
    """Returns mask of shape (H, W) with 0 where pixel is 'gray-ish'
    (max channel - min channel <= gray_threshold) and 255 elsewhere.
    Operates on RGB images (uint8). Excludes gray plastic cassette
    frames per thesis §5.2.1."""
    ptp = np.ptp(image, axis=2)
    mask = np.where(ptp <= gray_threshold, 0, 255).astype(np.uint8)
    return mask

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
    gray_threshold: int,
    side: str = "both",
) -> np.ndarray:
    """Returns final detection mask per mode:
    - "no_mask": only the overlap band mask
    - "mask": overlap band AND grayness mask combined
    - "fallback": returns the no_mask mask; fallback logic that re-runs
      with the stricter mask is handled by the orchestrator, not here
    """
    band_mask = make_overlap_band_mask(image.shape, band_fraction, side=side)
    if mode == "mask":
        gray_mask = make_grayness_mask(image, gray_threshold)
        return combine_masks(band_mask, gray_mask)
    elif mode in ("no_mask", "fallback"):
        return band_mask
    else:
        raise ValueError(f"Unknown mask mode: {mode}")
