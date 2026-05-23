"""
types.py — Shared data contracts for the overlap-detection pipeline.

This module defines:

* **Type aliases** for the three fundamental array types that flow between
  pipeline stages (``Image``, ``Mask``, ``DescriptorMatrix``).
* :class:`Keypoint` — a dataclass representation of a single detected feature
  point.  It is the canonical structured form used by detection → description
  → verification stages.
* :class:`PairResult` — a dataclass representing the complete output record
  for one image-pair run, covering configuration snapshot, timing, matching
  statistics, geometric verification outcome, and accuracy metrics.

Notes
-----
* ``Image``, ``Mask``, and ``DescriptorMatrix`` are **type aliases** for
  ``numpy.ndarray``.  They carry no runtime enforcement; they exist solely to
  make function signatures self-documenting.
* :class:`Keypoint` mirrors the keypoint *dict* contract in
  ``project_overview.md`` §3 but as a proper dataclass so that field access
  is IDE-friendly and static-analysis tools can check it.
* :class:`PairResult` corresponds to the pair-result *dict* contract in
  ``project_overview.md`` §8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Image = np.ndarray
"""RGB image array.

Shape  : ``(H, W, 3)``
Dtype  : ``uint8``
Channel order : RGB
"""

Mask = np.ndarray
"""Binary region-of-interest mask.

Shape  : ``(H, W)``
Dtype  : ``uint8``
Values : ``0`` (ignore) or ``255`` (keep)
"""

DescriptorMatrix = np.ndarray
"""Dense descriptor matrix for a set of keypoints.

Shape  : ``(N, D)`` where *N* = number of keypoints, *D* = descriptor dim.
Dtype  : ``float32`` for float descriptors (SIFT, KAZE …)
         ``uint8``   for packed binary descriptors (BRISK, BRIEF …)
"""

# ---------------------------------------------------------------------------
# Keypoint
# ---------------------------------------------------------------------------


@dataclass
class Keypoint:
    """A single feature keypoint extracted by a detector.

    All coordinates are in *image pixel space* (origin at top-left corner).
    Fields that a detector cannot provide are stored as ``None``; the
    description stage fills them in from :attr:`RunConfig.descriptor_default_sigma`
    before passing keypoints to the descriptor.

    Attributes
    ----------
    x : float
        Sub-pixel horizontal coordinate.
    y : float
        Sub-pixel vertical coordinate.
    sigma : float or None
        Characteristic scale in pixels (½ the kernel diameter at detection).
        ``None`` for scale-less detectors (Harris, FAST, AGAST …).
    theta : float or None
        Dominant orientation in **radians**, measured counter-clockwise from
        the positive x-axis.  ``None`` for orientation-less detectors.
    response : float
        Detector response / cornerness score used for keypoint ranking.
    octave : int or None
        Pyramidal scale-space octave at which the keypoint was detected.
        ``None`` for detectors that do not use a Gaussian pyramid.
    class_id : int or None
        Scale-space evolution layer index set by AKAZE (and KAZE) during
        detection.  OpenCV's native MLDB descriptor reads derivative images
        from this specific layer in AKAZE's internal diffusion pyramid; the
        field must be round-tripped faithfully for that path to work.
        ``None`` for all other detectors (``cv2.KeyPoint`` default is -1).
    """

    x: float
    y: float
    response: float
    sigma: Optional[float] = None
    theta: Optional[float] = None
    octave: Optional[int] = None
    class_id: Optional[int] = None

    @property
    def pt(self) -> tuple[float, float]:
        """Convenience accessor returning ``(x, y)`` as a tuple."""
        return (self.x, self.y)


# ---------------------------------------------------------------------------
# GroundTruth
# ---------------------------------------------------------------------------


@dataclass
class GroundTruth:
    """Manual annotation of the overlap region for one image pair.

    Produced by the annotation GUI and consumed by the metrics stage
    to compute accuracy metrics (corner error, IoU).

    The source of truth is ``affine_matrix_A_to_B`` (2×3, maps image-A
    pixel coordinates to image-B pixel coordinates) together with the two
    image shapes.  Overlap polygons are derived on demand from these three
    values via ``compute_overlap_polygon`` rather than being stored.
    """

    image_A_path: Path
    image_B_path: Path
    affine_matrix_A_to_B: np.ndarray   # 2x3, A→B
    image_a_shape: tuple               # (H, W, 3)
    image_b_shape: tuple               # (H, W, 3)
    annotator: str
    annotation_date: str


# ---------------------------------------------------------------------------
# PairResult
# ---------------------------------------------------------------------------


@dataclass
class PairResult:
    """Complete result record for a single image-pair pipeline run.

    One :class:`PairResult` is produced per ``(image_A, image_B, RunConfig)``
    triple.  It is the unit of output written by the reporting stage and
    consumed by the metrics aggregation stage.

    Attributes
    ----------
    image_a_path : Path
        Filesystem path to image A.
    image_b_path : Path
        Filesystem path to image B.

    detector : str
        Name of the detector used (mirrors ``RunConfig.detector``).
    descriptor : str
        Name of the descriptor used (mirrors ``RunConfig.descriptor``).
    mask_mode : str
        Masking mode actually applied (may differ from config if fallback
        was triggered).

    n_kp_a : int
        Number of keypoints detected in image A (after truncation).
    n_kp_b : int
        Number of keypoints detected in image B (after truncation).
    n_raw_matches : int
        Number of descriptor matches before geometric filtering.
    n_inliers : int
        Number of RANSAC / USAC inliers supporting the estimated affine.

    affine_matrix : numpy.ndarray or None
        Estimated ``(2, 3)`` affine transformation matrix (``float64``)
        mapping image-A coordinates to image-B coordinates, or ``None`` if
        verification failed (RANSAC returned no transform, or the inlier
        count fell below ``RunConfig.min_inliers``).
    inlier_mask : numpy.ndarray or None
        Boolean array of shape ``(n_raw_matches,)`` indicating inliers, or
        ``None`` if verification failed.

    overlap_polygon_a : numpy.ndarray or None
        Overlap region corners in image-A coordinates, shape ``(K, 2)``.
    overlap_polygon_b : numpy.ndarray or None
        Overlap region corners in image-B coordinates, shape ``(K, 2)``.

    time_detection_s : float
        Wall-clock time for the detection stage (seconds).
    time_description_s : float
        Wall-clock time for the description stage (seconds).
    time_matching_s : float
        Wall-clock time for the matching stage (seconds).
    time_verification_s : float
        Wall-clock time for the verification stage (seconds).
    time_geometry_s : float
        Wall-clock time for the overlap-geometry stage (seconds).
    time_total_s : float
        Total wall-clock time for the full pipeline run (seconds).

    error_message : str or None
        Human-readable description of any exception or failure condition.
        ``None`` when the pipeline ran without hitting any rejection gate.

    result_label : str
        Ordinal accuracy categorisation; see the field docstring below.

    extra : dict
        Arbitrary key-value store for experimental metadata not captured by
        the fields above (e.g. custom timing breakdowns, debug flags).
    """

    # --- Identification ---------------------------------------------------
    image_a_path: Path
    image_b_path: Path

    # --- Algorithm snapshot -----------------------------------------------
    detector: str
    descriptor: str
    estimator: str
    mask_mode: str

    # --- Keypoint / match counts ------------------------------------------
    n_kp_a: int = 0
    n_kp_b: int = 0
    n_raw_matches: int = 0
    n_inliers: int = 0

    # --- Geometric verification output ------------------------------------
    affine_matrix: Optional[np.ndarray] = None
    """Estimated ``(2, 3)`` affine transformation matrix (``float64``),
    mapping image-A coordinates to image-B coordinates.
    ``None`` if verification failed."""
    inlier_mask: Optional[np.ndarray] = None

    # --- Overlap geometry -------------------------------------------------
    overlap_polygon_a: Optional[np.ndarray] = None
    overlap_polygon_b: Optional[np.ndarray] = None

    # --- Timing (seconds) -------------------------------------------------
    time_detection_s: float = 0.0
    time_description_s: float = 0.0
    time_matching_s: float = 0.0
    time_verification_s: float = 0.0
    time_geometry_s: float = 0.0
    time_total_s: float = 0.0

    # --- Status -----------------------------------------------------------
    error_message: Optional[str] = None

    result_label: str = "no_match"
    """Ordinal accuracy categorisation of this attempt.  Derived at metrics
    time from the configured ``RunConfig.accuracy_tiers_px`` and the measured
    mean corner error vs. ground truth.  Possible values:

    * ``"no_match"``    — no transform produced (insufficient keypoints,
                          insufficient matches, too few RANSAC inliers,
                          or the RANSAC call returned no transform).
    * ``"false_match"`` — transform produced but the mean corner error
                          exceeded the loosest accuracy tier.
    * ``"acc_at_<T>"``  — transform produced and mean corner error ≤ T px,
                          where T is the smallest tier the result cleared
                          (e.g. ``"acc_at_3"`` is strictly better than
                          ``"acc_at_5"`` which is strictly better than
                          ``"acc_at_10"``).
    """

    # --- Extensible metadata ---------------------------------------------
    extra: dict = field(default_factory=dict)
