"""
config.py — Centralised configuration for the overlap-detection pipeline.

All pipeline hyper-parameters live in :class:`RunConfig`.  Every field has a
sensible default but can be overridden at construction time, making it trivial
to sweep parameter grids without touching source code.

Module-level constants
----------------------
DETECTOR_NAMES   : ordered list of supported detector identifiers.
DESCRIPTOR_NAMES : ordered list of supported descriptor identifiers.
VALID_PAIRINGS   : mapping detector → [valid descriptors].

Pairing rules (derived from OpenCV / contrib constraints)
---------------------------------------------------------
* **MLDB descriptor** (M-LDB) officially benefits from AKAZE/KAZE scale
  information but accepts any detector because a fallback σ is supplied by
  the descriptor stage (see ``descriptor_default_sigma``).
* All other descriptors work with every detector via the same σ/θ fallback
  mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Supported algorithm identifiers
# ---------------------------------------------------------------------------

DETECTOR_NAMES: list[str] = [
    "Harris",
    "GFTT",
    "FAST",
    "AGAST",
    "BRISK",
    "SIFT",
    "USURF",
    "STAR",
    "KAZE",
    "AKAZE",
    "MSER",
]

DESCRIPTOR_NAMES: list[str] = [
    "SIFT",
    "RootSIFT",
    "USURF",
    "DAISY",
    "BRIEF",
    "BRISK",
    "SUFREAK",
    "MLDB",
    "LIOP",
]

# ---------------------------------------------------------------------------
# Valid detector → descriptor pairings
# ---------------------------------------------------------------------------
# Every detector in DETECTOR_NAMES is compatible with every descriptor in
# DESCRIPTOR_NAMES.  MLDB officially benefits from AKAZE/KAZE scale
# information but accepts any detector via the fallback σ supplied by the
# description stage.  No hard exclusions exist in this list.
# ---------------------------------------------------------------------------

_ALL_DESCRIPTORS: list[str] = DESCRIPTOR_NAMES.copy()

VALID_PAIRINGS: dict[str, list[str]] = {
    "Harris": _ALL_DESCRIPTORS,
    "GFTT":   _ALL_DESCRIPTORS,
    "FAST":   _ALL_DESCRIPTORS,
    "AGAST":  _ALL_DESCRIPTORS,
    "BRISK":  _ALL_DESCRIPTORS,
    "SIFT":   _ALL_DESCRIPTORS,
    "USURF":  _ALL_DESCRIPTORS,
    "STAR":   _ALL_DESCRIPTORS,
    "KAZE":   _ALL_DESCRIPTORS,
    "AKAZE":  _ALL_DESCRIPTORS,
    "MSER":   _ALL_DESCRIPTORS,
}

# Valid values for mask_mode / estimator string fields.  Consumed by CLI
# entrypoints (scripts/run_experiment.py) for upfront argument validation.
VALID_MASK_MODES: set[str] = {"no_mask", "mask", "both"}
VALID_ESTIMATORS: set[str] = {"PROSAC", "USAC_MAGSAC"}

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """All hyper-parameters for a single pipeline run.

    Parameters are grouped by pipeline stage.  Pass keyword arguments to
    override any subset of defaults::

        cfg = RunConfig(detector="KAZE", descriptor="KAZE", mask_mode="mask")
    """

    # ------------------------------------------------------------------
    # Mask / preprocessing
    # ------------------------------------------------------------------

    mask_mode: str = "both"
    """Masking strategy: ``"no_mask"`` | ``"mask"`` | ``"both"``.

    * ``no_mask`` — only band mask.
    * ``mask``    — band mask and tray mask.
    * ``both``    — run the pair twice (once with each mode) and emit a
      single CSV row whose ``no_mask_*`` and ``with_mask_*`` columns are
      both populated.  Per-attempt JSON files are still written separately.
    """

    mask_sat_threshold: float = 0.12
    """Saturation threshold for the cassette-frame mask.  Pixels whose
    relative saturation ``(max(R,G,B) - min(R,G,B)) / max(R,G,B)`` is
    **below** this value are candidates for exclusion (low-chroma plastic
    frame).  Combined with the brightness band below."""

    mask_brightness_lo: int = 15
    """Lower bound on ``max(R,G,B)`` for the frame band.  A pixel is treated
    as cassette frame only if its brightness is **at or above** this value
    (avoids masking out very dark plant/soil content)."""

    mask_brightness_hi: int = 180
    """Upper bound on ``max(R,G,B)`` for the frame band.  A pixel is treated
    as cassette frame only if its brightness is **at or below** this value
    (avoids masking out bright highlights)."""

    overlap_band_fraction: float = 0.20
    """Fraction of the image width retained on each edge as the candidate
    overlap search band (applied to both left and right images)."""

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    detector: str = "SIFT"
    """Feature detector algorithm name.  Must be a member of
    :data:`DETECTOR_NAMES`."""

    detector_params: dict = field(default_factory=dict)
    """Algorithm-specific constructor overrides forwarded to the OpenCV
    detector factory (e.g. ``{"nOctaveLayers": 6}`` for SIFT)."""

    max_keypoints: int = 5000
    """Upper bound on keypoints retained after detection.  Keypoints are
    sorted by response strength before truncation."""

    # ------------------------------------------------------------------
    # Description
    # ------------------------------------------------------------------

    descriptor: str = "SIFT"
    """Feature descriptor algorithm name.  Must be a member of
    :data:`DESCRIPTOR_NAMES` and compatible with ``detector`` according to
    :data:`VALID_PAIRINGS`."""

    descriptor_params: dict = field(default_factory=dict)
    """Algorithm-specific constructor overrides forwarded to the OpenCV
    descriptor factory."""

    descriptor_default_sigma: float = 4.0
    """Fallback scale (pixels) injected into keypoints whose detector does not
    provide a ``sigma`` value.  Used by scale-dependent descriptors such as
    DAISY, SIFT, LIOP."""

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    matcher_filter: str = "mnn_nndr"
    """Match filtering strategy: ``"mnn"`` (Mutual Nearest Neighbours) or
    ``"mnn_nndr"`` (MNN + Nearest Neighbour Distance Ratio test)."""

    nndr_threshold: float = 0.80
    """Lowe-ratio threshold for the NNDR filter.  A match is kept when
    ``d1 / d2 < nndr_threshold``."""

    # ------------------------------------------------------------------
    # Geometric verification
    # ------------------------------------------------------------------

    estimator: str = "PROSAC"
    """Robust homography estimator: ``"PROSAC"`` | ``"USAC_MAGSAC"``."""

    ransac_threshold_px: float = 5.0
    """Maximum reprojection error (pixels) for a point to be counted as an
    inlier during RANSAC / USAC."""

    ransac_max_iters: int = 10000
    """Maximum number of hypothesis iterations for the robust estimator."""

    ransac_confidence: float = 0.99
    """Desired probability that the estimated model is free of outliers."""

    # ------------------------------------------------------------------
    # Acceptance / categorisation
    # ------------------------------------------------------------------

    min_inliers: int = 8
    """Minimum inlier count for an estimated affine to be retained.  An
    estimate that fails this gate (or whose underlying RANSAC call returned
    no transform) yields the ``"no_match"`` categorical result."""

    pixel_correspondence_tolerance_px: float = 5.0
    """Per-pixel error budget (B-pixels) for the pixel-correspondence-rate
    metric.  A pixel inside the GT overlap region is counted as correctly
    placed when ``||M_est @ p − M_gt @ p|| ≤`` this value.  Default 1.0 —
    sub-pixel alignment.  Lower → stricter; higher → more forgiving."""

    accuracy_tiers_px: tuple[float, ...] = (3.0, 5.0, 10.0)
    """Mean-corner-error thresholds (px, B-frame) that define the ordinal
    accuracy tiers used to label each attempt's ``*_result`` column.  Error
    is the mean per-vertex distance between the estimated and ground-truth
    *clipped overlap polygons* in B's frame (vertex count 3–8, usually 3–5
    in practice).  Sorted ascending at use time.  A pair whose mean corner
    error is ≤ the smallest tier is labelled ``"acc_at_<smallest>"``; if
    greater than the largest tier it is labelled ``"false_match"``; if no
    transform was produced (or polygons have mismatched vertex counts / are
    empty) it is ``"no_match"``."""

