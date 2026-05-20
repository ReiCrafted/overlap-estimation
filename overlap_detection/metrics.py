import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from overlap_detection.types import PairResult, GroundTruth
from overlap_detection.geometry import compute_overlap_polygon


def per_corner_errors(
    predicted_corners: np.ndarray,   # Nx2 from compute_overlap_polygon
    ground_truth_corners: np.ndarray, # Nx2 from manual annotation
) -> np.ndarray:
    """Returns N-element array of Euclidean distances per corner, in pixels.
    Assumes corners are in matching order (use polygon vertex matching
    if order is ambiguous — for axis-aligned rectangles, top-left
    should be first in both)."""
    if len(predicted_corners) != len(ground_truth_corners):
        return np.full((len(ground_truth_corners),), np.nan)

    return np.linalg.norm(predicted_corners - ground_truth_corners, axis=1)


def overlap_rms_error(corner_errors: np.ndarray) -> float:
    """RMS of per-corner errors. Single scalar in pixels."""
    if np.any(np.isnan(corner_errors)):
        return float('nan')
    return float(np.sqrt(np.mean(corner_errors**2)))


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
) -> dict:
    """Compute all metrics for a single pair result. Returns a flat dict
    suitable for CSV writing."""
    inlier_ratio = result.n_inliers / result.n_raw_matches if result.n_raw_matches > 0 else 0.0

    metrics = {
        "num_keypoints_A": result.n_kp_a,
        "num_keypoints_B": result.n_kp_b,
        "num_tentative_matches": result.n_raw_matches,
        "num_inliers": result.n_inliers,
        "inlier_ratio": float(inlier_ratio),
        "estimation_succeeded": result.success,
        "detection_ms": result.time_detection_s * 1000,
        "description_ms": result.time_description_s * 1000,
        "matching_ms": result.time_matching_s * 1000,
        "verification_ms": result.time_verification_s * 1000,
        "geometry_ms": result.time_geometry_s * 1000,
        "total_ms": result.time_total_s * 1000,
        "corner_error_0": None,
        "corner_error_1": None,
        "corner_error_2": None,
        "corner_error_3": None,
        "rms_corner_error": None,
        "iou": None,
    }

    if ground_truth is not None and result.overlap_polygon_a is not None and len(result.overlap_polygon_a) > 0:
        gt_poly_A, _ = compute_overlap_polygon(
            ground_truth.affine_matrix_A_to_B,
            ground_truth.image_a_shape,
            ground_truth.image_b_shape,
        )
        if len(gt_poly_A) > 0:
            errors = per_corner_errors(result.overlap_polygon_a, gt_poly_A)
            if not np.any(np.isnan(errors)):
                for i, e in enumerate(errors):
                    metrics[f"corner_error_{i}"] = float(e)
                metrics["rms_corner_error"] = overlap_rms_error(errors)
            metrics["iou"] = overlap_iou(result.overlap_polygon_a, gt_poly_A)

    return metrics
