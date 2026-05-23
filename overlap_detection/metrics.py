import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from overlap_detection.types import PairResult, GroundTruth
from overlap_detection.geometry import compute_overlap_polygon, apply_affine


def _format_tier(t: float) -> str:
    """Format a tier threshold for a ``"acc_at_..."`` label without trailing
    zeros (``3.0 → "3"``, ``2.5 → "2.5"``)."""
    return f"{t:g}"


def categorize_result(
    has_transform: bool,
    mean_corner_error: float | None,
    accuracy_tiers_px: tuple[float, ...] | list[float],
) -> str:
    """Map an attempt's outcome to its ordinal accuracy label.

    Parameters
    ----------
    has_transform
        Whether the pipeline produced an accepted affine matrix (i.e. it
        survived the affine sanity check and the min-inliers gate).
    mean_corner_error
        Mean corner reprojection error vs. ground truth, in pixels.  Must be
        supplied when ``has_transform`` is ``True``; ignored otherwise.
    accuracy_tiers_px
        Tier thresholds; will be sorted ascending.

    Returns
    -------
    label : str
        One of ``"no_match"``, ``"false_match"``, or ``"acc_at_<T>"``.
    """
    if not has_transform:
        return "no_match"
    if mean_corner_error is None or not np.isfinite(mean_corner_error):
        # Pipeline produced a transform but we have no GT-derived error to
        # grade it against — treat as no_match to keep counts honest.  Real
        # experiments are always run with ground truth (see project_overview).
        return "no_match"
    for t in sorted(accuracy_tiers_px):
        if mean_corner_error <= t:
            return f"acc_at_{_format_tier(t)}"
    return "false_match"


def per_corner_errors(
    predicted_corners: np.ndarray,   # Nx2
    ground_truth_corners: np.ndarray, # Nx2
) -> np.ndarray:
    """Generic pairwise Euclidean distance between two ordered point sets.

    Returns an ``N``-element array of distances.  If the input lengths
    disagree, returns ``np.nan`` for every entry (callers downstream treat
    NaN errors as ungradable).
    """
    if len(predicted_corners) != len(ground_truth_corners):
        return np.full((len(ground_truth_corners),), np.nan)

    return np.linalg.norm(predicted_corners - ground_truth_corners, axis=1)


def corner_errors_hpatches(
    estimated_affine: np.ndarray,    # 2x3, A→B
    gt_affine: np.ndarray,           # 2x3, A→B
    image_a_shape: tuple,            # (H, W, …)
) -> np.ndarray:
    """HPatches / SuperGlue convention corner error.

    Warps image-A's four image-rectangle corners through both the estimated
    and the ground-truth affine; returns the four Euclidean distances (in
    **B-pixels**) between the two sets of projected corners.

    Returned array is always shape ``(4,)`` in the order
    ``[top-left, top-right, bottom-right, bottom-left]`` — the same order
    in which A's corners are constructed below.  No clipping is performed;
    projected corners may legitimately land outside B's image rectangle.
    """
    H_A, W_A = image_a_shape[:2]
    A_corners = np.array(
        [[0, 0], [W_A, 0], [W_A, H_A], [0, H_A]], dtype=np.float64,
    )
    pred = apply_affine(A_corners, estimated_affine)
    gt = apply_affine(A_corners, gt_affine)
    return np.linalg.norm(pred - gt, axis=1)


def mean_corner_error(corner_errors: np.ndarray) -> float:
    """Mean of per-corner reprojection errors, in pixels.

    Returns ``nan`` if any input is ``nan``.  The canonical input is the
    output of :func:`corner_errors_hpatches`, matching the SuperGlue /
    glue-factory / LoFTR convention so the numbers are directly comparable
    to published tables.
    """
    if len(corner_errors) == 0 or np.any(np.isnan(corner_errors)):
        return float('nan')
    return float(np.mean(corner_errors))


def overlap_iou(
    predicted_polygon: np.ndarray,
    ground_truth_polygon: np.ndarray,
) -> float:
    """Intersection over Union of two polygons. Returns float in [0, 1].
    Use shapely."""
    if len(predicted_polygon) < 3 or len(ground_truth_polygon) < 3:
        return 0.0

    p1 = Polygon(predicted_polygon)
    p2 = Polygon(ground_truth_polygon)

    if not p1.is_valid or not p2.is_valid:
        return 0.0

    intersection = p1.intersection(p2).area
    union = p1.union(p2).area

    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_pair_metrics(
    result: PairResult,
    ground_truth: GroundTruth | None,
    accuracy_tiers_px: tuple[float, ...] | list[float] = (3.0, 5.0, 10.0),
) -> dict:
    """Compute all metrics for a single pair result.  Returns a flat dict
    suitable for CSV writing.  Also assigns ``result.result_label`` based on
    the configured ``accuracy_tiers_px`` and the measured mean corner error.

    Two related geometric measurements are taken, each from a different
    polygon:

    * **Corner error** uses :func:`corner_errors_hpatches` — image-A's four
      image-rectangle corners projected through both the estimated and
      ground-truth affines, measured in B-pixels with no clipping.  Always
      4 corners; reported as ``corner_error_{0..3}`` in the order
      ``TL, TR, BR, BL`` and aggregated as ``mean_corner_error``.  This
      matches the HPatches / SuperGlue / LoFTR convention.
    * **IoU** uses the clipped overlap polygon (``result.overlap_polygon_a``
      vs. a freshly-computed GT overlap polygon), since IoU is fundamentally
      an area metric and needs the actual overlap region.
    """
    inlier_ratio = result.n_inliers / result.n_raw_matches if result.n_raw_matches > 0 else 0.0

    metrics = {
        "num_keypoints_A": result.n_kp_a,
        "num_keypoints_B": result.n_kp_b,
        "num_tentative_matches": result.n_raw_matches,
        "num_inliers": result.n_inliers,
        "inlier_ratio": float(inlier_ratio),
        "detection_ms": result.time_detection_s * 1000,
        "description_ms": result.time_description_s * 1000,
        "matching_ms": result.time_matching_s * 1000,
        "verification_ms": result.time_verification_s * 1000,
        "geometry_ms": result.time_geometry_s * 1000,
        "total_ms": result.time_total_s * 1000,
        "corner_error_0": None,   # TL
        "corner_error_1": None,   # TR
        "corner_error_2": None,   # BR
        "corner_error_3": None,   # BL
        "mean_corner_error": None,
        "iou": None,
        "result_label": "no_match",
    }

    has_gt = ground_truth is not None
    has_affine = result.affine_matrix is not None

    # Corner error (HPatches convention) — needs both affines, no polygons.
    if has_gt and has_affine:
        errors = corner_errors_hpatches(
            result.affine_matrix,
            ground_truth.affine_matrix_A_to_B,
            ground_truth.image_a_shape,
        )
        for i, e in enumerate(errors):
            metrics[f"corner_error_{i}"] = float(e)
        metrics["mean_corner_error"] = mean_corner_error(errors)

    # IoU — uses the clipped overlap polygons in A's frame.
    if has_gt and result.overlap_polygon_a is not None and len(result.overlap_polygon_a) > 0:
        gt_poly_A, _ = compute_overlap_polygon(
            ground_truth.affine_matrix_A_to_B,
            ground_truth.image_a_shape,
            ground_truth.image_b_shape,
        )
        if len(gt_poly_A) > 0:
            metrics["iou"] = overlap_iou(result.overlap_polygon_a, gt_poly_A)

    label = categorize_result(
        has_transform=has_affine,
        mean_corner_error=metrics["mean_corner_error"],
        accuracy_tiers_px=accuracy_tiers_px,
    )
    metrics["result_label"] = label
    result.result_label = label
    return metrics
