import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from shapely import contains_xy as _shp_contains_xy
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
        survived the min-inliers gate).
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


def corner_errors_overlap_polygon(
    estimated_affine: np.ndarray,    # 2x3, A→B
    gt_affine: np.ndarray,           # 2x3, A→B
    image_a_shape: tuple,            # (H, W, …)
    image_b_shape: tuple,            # (H, W, …)
) -> np.ndarray:
    """Per-vertex error on the **clipped overlap polygon**, in B-pixels.

    Computes the overlap polygon in B's frame for both the estimated and
    ground-truth affines (each is A's warped rectangle ∩ B's rectangle),
    then returns the Euclidean distance between corresponding vertices in
    the canonical clockwise / top-left-first order applied by
    :func:`compute_overlap_polygon`.

    Vertex count is **3–8** depending on how A's warped rectangle clips
    against B — in practice usually 3–5 for the camera-rig overlap geometry
    in this project.  This metric is operationally meaningful: it grades
    the affine over the region where features can actually be matched,
    rather than extrapolating to image corners that may sit far outside the
    overlap (where small rotational errors get amplified into huge pixel
    deviations).

    Returns a single ``nan`` (and only one element) when the metric is
    ungradable:

    * either polygon is empty (degenerate affine, or A's warp does not
      intersect B at all), or
    * the two polygons have different vertex counts because they clip
      against different sides of B — the per-vertex correspondence breaks
      down and the pair is downgraded to ``no_match`` downstream.
    """
    _, est_poly_B = compute_overlap_polygon(
        estimated_affine, image_a_shape, image_b_shape,
    )
    _, gt_poly_B = compute_overlap_polygon(
        gt_affine, image_a_shape, image_b_shape,
    )

    if len(est_poly_B) == 0 or len(gt_poly_B) == 0:
        return np.array([np.nan])
    if len(est_poly_B) != len(gt_poly_B):
        return np.array([np.nan])

    return np.linalg.norm(est_poly_B - gt_poly_B, axis=1)


def mean_corner_error(corner_errors: np.ndarray) -> float:
    """Mean of per-vertex reprojection errors, in pixels.

    Returns ``nan`` if any input is ``nan``.  The canonical input is the
    output of :func:`corner_errors_overlap_polygon` — the per-vertex
    distances on the clipped overlap polygon.
    """
    if len(corner_errors) == 0 or np.any(np.isnan(corner_errors)):
        return float('nan')
    return float(np.mean(corner_errors))



def pixel_correspondence_rate(
    estimated_affine: np.ndarray,    # 2x3, A→B
    gt_affine: np.ndarray,           # 2x3, A→B
    image_a_shape: tuple,
    image_b_shape: tuple,
    tolerance_px: float = 1.0,
    max_samples: int = 100_000,
) -> float:
    """Pixel-level correspondence between two affine transforms.

    For every pixel position ``p`` inside the **ground-truth overlap
    polygon** (in A's frame), compute the per-pixel disagreement
    ``e(p) = ||M_est @ p − M_gt @ p||`` (in B-pixels), then return the
    fraction of pixels with ``e(p) ≤ tolerance_px``.

    Why this isn't redundant with the corner-error metric.  ``mean_corner_error``
    grades the polygon **vertices** — a small set of points on the boundary
    of the overlap region.  PCR grades every interior pixel.  These are
    different geometric questions:

    * a transform that exactly maps the polygon onto itself but rotates the
      *interior* (e.g. a 180° rotation about the polygon centroid) yields
      ``mean_corner_error ≈ 0`` (vertices land on themselves) but
      ``PCR ≈ 0`` (every interior pixel is moved);
    * a transform that shears the overlap so vertices drift but the rest
      stays correct will have non-zero ``mean_corner_error`` and
      proportionally smaller PCR penalty.

    Returns NaN when the metric is ungradable — empty GT overlap polygon,
    invalid polygon geometry, or no integer pixel falls inside the polygon
    (vanishingly small overlap).

    Parameters
    ----------
    tolerance_px
        Per-pixel error budget (B-pixels).  A pixel is counted as
        correctly placed when its reprojection error is at or under this
        value.  Default 1.0 — sub-pixel alignment.
    max_samples
        Cap on the number of pixels actually evaluated.  Pixels are
        sampled on a regular stride chosen so the bounding-box grid has
        ≤ ``max_samples`` points; the in-polygon mask is then applied.
        Default 100 000 — for a typical 2464×2056 cassette image with
        ~500 k overlap pixels this corresponds to stride ≈ 2.
    """
    gt_poly_A, _ = compute_overlap_polygon(
        gt_affine, image_a_shape, image_b_shape,
    )
    if len(gt_poly_A) < 3:
        return float('nan')

    poly = Polygon(gt_poly_A)
    if poly.is_empty or not poly.is_valid:
        return float('nan')

    # Bounding-box pixel grid, strided to cap sample count.
    minx, miny, maxx, maxy = poly.bounds
    minx_i, miny_i = int(np.floor(minx)), int(np.floor(miny))
    maxx_i, maxy_i = int(np.ceil(maxx)),  int(np.ceil(maxy))
    bbox_size = max(1, (maxx_i - minx_i + 1) * (maxy_i - miny_i + 1))
    stride = max(1, int(np.sqrt(bbox_size / max_samples)))

    xs = np.arange(minx_i, maxx_i + 1, stride, dtype=np.float64)
    ys = np.arange(miny_i, maxy_i + 1, stride, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    pts_x = X.ravel()
    pts_y = Y.ravel()

    inside = _shp_contains_xy(poly, pts_x, pts_y)
    if not np.any(inside):
        return float('nan')

    xs_in = pts_x[inside]
    ys_in = pts_y[inside]

    # e(p) = || (M_est − M_gt) · [x, y, 1] ||
    dM = estimated_affine - gt_affine    # (2, 3)
    dx = dM[0, 0] * xs_in + dM[0, 1] * ys_in + dM[0, 2]
    dy = dM[1, 0] * xs_in + dM[1, 1] * ys_in + dM[1, 2]
    errors = np.hypot(dx, dy)

    return float(np.mean(errors <= tolerance_px))


def compute_pair_metrics(
    result: PairResult,
    ground_truth: GroundTruth | None,
    accuracy_tiers_px: tuple[float, ...] | list[float] = (3.0, 5.0, 10.0),
    pixel_correspondence_tolerance_px: float = 1.0,
) -> dict:
    """Compute all metrics for a single pair result.  Returns a flat dict
    suitable for CSV writing.  Also assigns ``result.result_label`` based on
    the configured ``accuracy_tiers_px`` and the measured mean corner error.

    Two complementary geometric measurements are taken, **both** rooted in
    the clipped overlap polygon but probing different aspects:

    * **Corner error** uses :func:`corner_errors_overlap_polygon` — the
      per-vertex Euclidean distance between the estimated and ground-truth
      overlap polygons (each is A's warped rectangle ∩ B's rectangle).
      Vertex count is variable (3–8, usually 3–5 in practice).  Reported
      as the list ``corner_errors`` plus ``n_corners`` for diagnostics,
      and aggregated as ``mean_corner_error`` for grading the affine via
      the configured ``accuracy_tiers_px``.  Polygons with mismatched
      vertex counts (estimated vs. GT clip against different sides of B)
      produce a single NaN error → ``no_match`` downstream.
    * **Pixel correspondence rate** uses
      :func:`pixel_correspondence_rate` — for every interior pixel of the
      GT overlap region, what fraction lands within
      ``pixel_correspondence_tolerance_px`` of where the GT affine places
      it?  This is the orthogonal signal corner-error can miss: a
      transform that fakes the polygon footprint but scrambles the
      interior content (e.g. a 180° rotation about the polygon centroid)
      shows up here as PCR ≈ 0 even though corner_error ≈ 0.
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
        "corner_errors": None,
        "n_corners": None,
        "mean_corner_error": None,
        "pixel_correspondence_rate": None,
        "result_label": "no_match",
    }

    has_gt = ground_truth is not None
    has_affine = result.affine_matrix is not None

    # Corner error on the clipped overlap polygon (variable 3-8 vertices).
    if has_gt and has_affine:
        errors = corner_errors_overlap_polygon(
            result.affine_matrix,
            ground_truth.affine_matrix_A_to_B,
            ground_truth.image_a_shape,
            ground_truth.image_b_shape,
        )
        # A single-NaN array signals "ungradable" (mismatched vertex count
        # or empty polygon); record that as n_corners = 0 to distinguish
        # from a real but bad gradable result.
        if len(errors) == 1 and np.isnan(errors[0]):
            metrics["corner_errors"] = None
            metrics["n_corners"] = 0
        else:
            metrics["corner_errors"] = [float(e) for e in errors]
            metrics["n_corners"] = int(len(errors))
        metrics["mean_corner_error"] = mean_corner_error(errors)

    # Pixel correspondence rate — per-pixel transform agreement on the GT
    # overlap region, at the configured tolerance.
    if has_gt and has_affine:
        pcr = pixel_correspondence_rate(
            result.affine_matrix,
            ground_truth.affine_matrix_A_to_B,
            ground_truth.image_a_shape,
            ground_truth.image_b_shape,
            tolerance_px=pixel_correspondence_tolerance_px,
        )
        metrics["pixel_correspondence_rate"] = (
            None if (pcr is None or (isinstance(pcr, float) and np.isnan(pcr)))
            else float(pcr)
        )

    label = categorize_result(
        has_transform=has_affine,
        mean_corner_error=metrics["mean_corner_error"],
        accuracy_tiers_px=accuracy_tiers_px,
    )
    metrics["result_label"] = label
    result.result_label = label
    return metrics
