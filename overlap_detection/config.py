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
from pathlib import Path

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

    * ``no_mask`` — process full images.
    * ``mask``    — apply greenness / tray mask.
    * ``both``    — run the pair twice (once with each mode) and emit a
      single CSV row whose ``no_mask_*`` and ``with_mask_*`` columns are
      both populated.  Per-attempt JSON files are still written separately.
    """

    rgb_gray_threshold: int = 15
    """Pixels with ``max(R,G,B) - min(R,G,B) < rgb_gray_threshold`` are
    classified as achromatic (grey) and excluded from the greenness mask."""

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

    descriptor_default_sigma: float = 6.0
    """Fallback scale (pixels) injected into keypoints whose detector does not
    provide a ``sigma`` value.  Used by scale-dependent descriptors such as
    DAISY, SIFT, LIOP."""

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    matcher_filter: str = "mnn_nndr"
    """Match filtering strategy: ``"mnn"`` (Mutual Nearest Neighbours) or
    ``"mnn_nndr"`` (MNN + Nearest Neighbour Distance Ratio test)."""

    nndr_threshold: float = 0.90
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

    accuracy_tiers_px: tuple[float, ...] = (3.0, 5.0, 10.0)
    """Corner-RMS thresholds (px) that define the ordinal accuracy tiers
    used to label each attempt's ``*_result`` column.  Sorted ascending at
    use time.  A pair whose corner RMS is ≤ the smallest tier is labelled
    ``"acc_at_<smallest>"``; if greater than the largest tier it is labelled
    ``"false_match"``; if no transform was produced it is ``"no_match"``."""

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    output_dir: Path = Path("./results")
    """Root directory for all pipeline outputs (JSON, CSV, plots)."""

    save_intermediate: bool = False
    """When ``True``, persist intermediate artefacts (masked images, match
    visualisations) inside ``output_dir``."""

    random_seed: int = 42
    """Global RNG seed for reproducible RANSAC / PROSAC sampling."""
