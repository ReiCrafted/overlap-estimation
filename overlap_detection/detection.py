"""detection.py — OpenCV feature detector wrappers.

Each detector is built from a dispatch table mapping name → factory function.
User-provided ``detector_params`` override sensible per-detector defaults.
"""

import cv2
import numpy as np
from overlap_detection.types import Keypoint

# ---------------------------------------------------------------------------
# Per-detector default parameters
# ---------------------------------------------------------------------------

_DETECTOR_DEFAULTS = {
    "Harris": {"maxCorners": 5000, "qualityLevel": 0.01, "minDistance": 10, "k": 0.04},
    "GFTT":   {"maxCorners": 5000, "qualityLevel": 0.01, "minDistance": 10},
    "FAST":   {"threshold": 8, "nonmaxSuppression": True, "type": cv2.FAST_FEATURE_DETECTOR_TYPE_9_16},
    "AGAST":  {"threshold": 8, "nonmaxSuppression": True},
    "BRISK":  {"thresh": 30, "octaves": 4},
    "SIFT":   {"nfeatures": 0, "contrastThreshold": 0.04, "edgeThreshold": 10},
    "USURF":  {"hessianThreshold": 100, "upright": True},
    "STAR":   {"maxSize": 15, "responseThreshold": 20},
    "KAZE":   {"threshold": 0.001},
    "AKAZE":  {"threshold": 0.001},
    "MSER":   {"min_area": 20, "max_area": 8100, "max_variation": 0.25},
}

# Detectors that return x,y only (no scale or orientation)
_SCALELESS_DETECTORS = {"Harris", "GFTT", "FAST", "AGAST", "MSER"}
_ORIENTATIONLESS_DETECTORS = {"Harris", "GFTT", "FAST", "AGAST", "MSER", "USURF"}


def _merged_params(detector_name: str, user_params: dict) -> dict:
    """Merge user overrides on top of detector defaults."""
    params = _DETECTOR_DEFAULTS.get(detector_name, {}).copy()
    params.update(user_params)
    return params


def _detect_corner_based(gray: np.ndarray, mask: np.ndarray,
                         params: dict, use_harris: bool) -> list[Keypoint]:
    """Shared implementation for Harris and GFTT (both use goodFeaturesToTrack)."""
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=params.get("maxCorners", 5000),
        qualityLevel=params.get("qualityLevel", 0.01),
        minDistance=params.get("minDistance", 10),
        mask=mask,
        useHarrisDetector=use_harris,
        k=params.get("k", 0.04),
    )
    if corners is None:
        return []
    return [Keypoint(x=float(pt[0][0]), y=float(pt[0][1]), response=1.0)
            for pt in corners]


def _detect_mser(gray: np.ndarray, mask: np.ndarray,
                 params: dict) -> list[Keypoint]:
    """MSER: convert detected regions to centroid keypoints."""
    detector = cv2.MSER_create(**params)
    regions, _ = detector.detectRegions(gray)
    H, W = gray.shape[:2]
    keypoints = []
    for region in regions:
        if len(region) == 0:
            continue
        mean_x = float(np.mean(region[:, 0]))
        mean_y = float(np.mean(region[:, 1]))
        # Bounds-check before mask lookup
        ix, iy = int(round(mean_x)), int(round(mean_y))
        ix = max(0, min(ix, W - 1))
        iy = max(0, min(iy, H - 1))
        if mask is not None and mask[iy, ix] == 0:
            continue
        keypoints.append(Keypoint(x=mean_x, y=mean_y,
                                  response=float(len(region))))
    return keypoints


# Dispatch table: name → factory that builds an OpenCV Feature2D detector
_DETECTOR_BUILDERS = {
    "FAST":  lambda p: cv2.FastFeatureDetector_create(**p),
    "AGAST": lambda p: cv2.AgastFeatureDetector_create(**p),
    "BRISK": lambda p: cv2.BRISK_create(**p),
    "SIFT":  lambda p: cv2.SIFT_create(**p),
    "USURF": lambda p: cv2.xfeatures2d.SURF_create(**p),
    "STAR":  lambda p: cv2.xfeatures2d.StarDetector_create(**p),
    "KAZE":  lambda p: cv2.KAZE_create(**p),
    "AKAZE": lambda p: cv2.AKAZE_create(**p),
}


def _cv_kp_to_keypoint(kp: cv2.KeyPoint, detector_name: str) -> Keypoint:
    """Convert an OpenCV KeyPoint to our Keypoint dataclass."""
    x, y = kp.pt
    sigma = (kp.size / 2) if kp.size > 0 else None
    theta = np.radians(kp.angle) if kp.angle != -1 else None

    # Override for detectors that don't truly provide scale/orientation
    if detector_name in _SCALELESS_DETECTORS:
        sigma = None
    if detector_name in _ORIENTATIONLESS_DETECTORS:
        theta = None

    # Preserve class_id for AKAZE/KAZE: it encodes the scale-space evolution
    # layer and is required by OpenCV's native MLDB descriptor compute() path.
    # All other detectors leave class_id at the cv2.KeyPoint default of -1.
    class_id = kp.class_id if kp.class_id >= 0 else None

    return Keypoint(
        x=float(x), y=float(y), response=float(kp.response),
        sigma=sigma, theta=theta, octave=kp.octave, class_id=class_id,
    )


def detect(
    image: np.ndarray,
    mask: np.ndarray,
    detector_name: str,
    detector_params: dict,
    max_keypoints: int,
) -> list[Keypoint]:
    """Run detector on image with optional mask. Returns list of Keypoint
    dataclass instances. Keypoints outside mask are excluded. If more
    keypoints than max_keypoints are found, keep the strongest by response.
    The Keypoint.sigma and Keypoint.theta fields may be None when the
    detector doesn't compute them."""

    if detector_name not in _DETECTOR_DEFAULTS:
        raise ValueError(f"Unknown detector: {detector_name}")

    params = _merged_params(detector_name, detector_params)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

    # --- Corner-based detectors (Harris, GFTT) ---
    if detector_name == "Harris":
        keypoints = _detect_corner_based(gray, mask, params, use_harris=True)
    elif detector_name == "GFTT":
        keypoints = _detect_corner_based(gray, mask, params, use_harris=False)
    # --- Region-based detector (MSER) ---
    elif detector_name == "MSER":
        keypoints = _detect_mser(gray, mask, params)
    # --- Standard Feature2D detectors ---
    else:
        builder = _DETECTOR_BUILDERS[detector_name]
        detector = builder(params)
        cv_kps = detector.detect(gray, mask)
        keypoints = [_cv_kp_to_keypoint(kp, detector_name) for kp in cv_kps]

    # Sort by response (descending) and truncate
    keypoints.sort(key=lambda k: k.response, reverse=True)
    if len(keypoints) > max_keypoints:
        keypoints = keypoints[:max_keypoints]

    return keypoints
