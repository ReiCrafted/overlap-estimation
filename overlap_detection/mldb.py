"""mldb.py — MLDB (Modified Local Difference Binary) descriptor.

Standalone NumPy/OpenCV implementation of the M-LDB descriptor from:
    Alcantarilla P.F., Nuevo J., Bartoli A.  "Fast Explicit Diffusion for
    Accelerated Features in Nonlinear Scale Spaces."  BMVC 2013.

Why this exists
---------------
OpenCV's built-in MLDB is computed inside ``cv2.AKAZE_create().compute()``.
That code path reads derivative images directly from AKAZE's internal
nonlinear diffusion pyramid, indexed by ``kp.class_id``.  Keypoints produced
by any other detector (SIFT, Harris, GFTT, …) carry ``class_id = -1``, which
triggers a hard assertion:

    cv2.error: (-215) 0 <= kpts[i].class_id && … in Compute_Descriptors

This implementation substitutes Gaussian smoothing and Sobel gradients as
detector-agnostic approximations of AKAZE's internal images (Lt, Lx, Ly),
so MLDB descriptors can be computed for keypoints from **any** detector.

Descriptor layout (486 bits, packed into 61 bytes)
---------------------------------------------------
Three grid sizes are applied to the scale-normalised square patch:

  Grid   Cells    Cell pairs        Channels  Bits
  ─────  ──────   ──────────────    ────────  ────
  2 × 2    4      C(4,2)  =   6       3         18
  3 × 3    9      C(9,2)  =  36       3        108
  4 × 4   16      C(16,2) = 120       3        360
  ──────────────────────────────────────────────────
  Total                                        486  →  61 bytes (2 padding bits)

For each cell pair (a, b) and channel c ∈ {Lt, Lx, Ly}:
  bit = 1  iff  mean_c(cell_a) > mean_c(cell_b)

Bits from a given grid are interleaved as [Lt, Lx, Ly] per pair, grids are
concatenated in order 2×2 → 3×3 → 4×4.  The last 2 bits of byte 61 are
zero padding (np.packbits behaviour).
"""

from __future__ import annotations

import cv2
import numpy as np
from itertools import combinations

from overlap_detection.types import Keypoint

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

# Patch side length.  Must be divisible by 2, 3 and 4 for the three grids.
# 60 = LCM(2, 3, 4, 5) satisfies this and gives integer cell dimensions.
_PATCH_SIZE: int = 60

# Physical patch half-side = sigma × _SIGMA_SCALE.
# Matches AKAZE's pattern_size = 10 convention (half-side spans 10σ on each side).
_SIGMA_SCALE: float = 10.0

# Gaussian smoothing sigma applied to the normalised patch to approximate
# AKAZE's nonlinear diffusion image Lt.
_SMOOTH_SIGMA: float = 1.5

# Grid subdivisions: each entry (rows, cols) defines one partition of the patch.
_GRIDS: list[tuple[int, int]] = [(2, 2), (3, 3), (4, 4)]

# Descriptor dimensions — derived, not hand-coded.
DESC_BITS: int = sum(
    len(list(combinations(range(r * c), 2))) * 3
    for r, c in _GRIDS
)  # 18 + 108 + 360 = 486
DESC_BYTES: int = (DESC_BITS + 7) // 8   # ceil(486 / 8) = 61

# ---------------------------------------------------------------------------
# One-time precomputation of cell-pair index arrays
# ---------------------------------------------------------------------------
# Storing pairs as two index arrays (pa, pb) avoids rebuilding them per keypoint.

_PAIRS_A: dict[tuple[int, int], np.ndarray] = {}
_PAIRS_B: dict[tuple[int, int], np.ndarray] = {}

for _r, _c in _GRIDS:
    _all_pairs = list(combinations(range(_r * _c), 2))
    _PAIRS_A[(_r, _c)] = np.array([p[0] for p in _all_pairs], dtype=np.int32)
    _PAIRS_B[(_r, _c)] = np.array([p[1] for p in _all_pairs], dtype=np.int32)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_patch(gray: np.ndarray, kp: Keypoint, default_sigma: float) -> np.ndarray:
    """Warp a scale-normalised square patch into _PATCH_SIZE × _PATCH_SIZE.

    The physical half-side of the sampled region is ``sigma × _SIGMA_SCALE``,
    matching AKAZE's pattern_size = 10 convention.  Border pixels are filled
    by reflection so edge keypoints never receive zero padding.

    Parameters
    ----------
    gray          : float32 grayscale image (H, W)
    kp            : source keypoint
    default_sigma : fallback scale when kp.sigma is None
    """
    sigma = kp.sigma if kp.sigma is not None else default_sigma
    phys_half = max(sigma * _SIGMA_SCALE, 1.0)
    scale = (_PATCH_SIZE / 2.0) / phys_half
    M = np.array([
        [scale, 0.0, _PATCH_SIZE / 2.0 - scale * kp.x],
        [0.0, scale, _PATCH_SIZE / 2.0 - scale * kp.y],
    ], dtype=np.float32)
    patch = cv2.warpAffine(
        gray, M, (_PATCH_SIZE, _PATCH_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return patch.astype(np.float32)


def _compute_channels(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Lt, Lx, Ly: the smoothed image and its Sobel x/y gradients.

    These are detector-agnostic approximations of the three image channels
    that AKAZE's MLDB reads from its internal nonlinear diffusion pyramid.
    """
    Lt = cv2.GaussianBlur(patch, (0, 0), _SMOOTH_SIGMA, borderType=cv2.BORDER_REFLECT)
    Lx = cv2.Sobel(Lt, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Ly = cv2.Sobel(Lt, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    return Lt, Lx, Ly


def _grid_means(channel: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Compute per-cell mean values for a rows × cols partition of the patch.

    Uses reshape + mean so no Python loop over cells is needed.
    Requires ``_PATCH_SIZE`` divisible by ``rows`` and ``cols`` — guaranteed
    by the choice of _PATCH_SIZE = 60 and _GRIDS = [(2,2), (3,3), (4,4)].

    Returns float32 array of shape (rows * cols,).
    """
    h = _PATCH_SIZE // rows
    w = _PATCH_SIZE // cols
    return (
        channel
        .reshape(rows, h, cols, w)
        .mean(axis=(1, 3))
        .ravel()
        .astype(np.float32)
    )


def _mldb_single(patch: np.ndarray) -> np.ndarray:
    """Compute the 61-byte MLDB descriptor for one normalised patch.

    For each grid (2×2, 3×3, 4×4), three channels (Lt, Lx, Ly) are averaged
    over each cell, then all cell pairs are compared.  Each comparison yields
    three bits (one per channel).  All 486 bits are collected and packed.

    Returns uint8 array of shape (DESC_BYTES,) = (61,).
    """
    Lt, Lx, Ly = _compute_channels(patch)

    all_bits: list[np.ndarray] = []
    for rows, cols in _GRIDS:
        # Stack per-cell means: shape (n_cells, 3) — columns are Lt, Lx, Ly
        means = np.column_stack([
            _grid_means(Lt, rows, cols),
            _grid_means(Lx, rows, cols),
            _grid_means(Ly, rows, cols),
        ])
        pa = _PAIRS_A[(rows, cols)]
        pb = _PAIRS_B[(rows, cols)]
        # comparisons: (n_pairs, 3) bool — True where mean of cell a > mean of cell b
        comparisons = means[pa] > means[pb]
        # Flatten: [Lt0, Lx0, Ly0, Lt1, Lx1, Ly1, …] for this grid
        all_bits.append(comparisons.ravel())

    # Concatenate all grid bits, pack into bytes (last 2 bits are zero padding)
    return np.packbits(np.concatenate(all_bits))   # (61,) uint8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mldb_describe(
    gray: np.ndarray,
    keypoints: list[Keypoint],
    default_sigma: float,
) -> tuple[list[Keypoint], np.ndarray]:
    """Compute MLDB descriptors for a list of keypoints from any detector.

    Parameters
    ----------
    gray          : uint8 or float32 grayscale image, shape (H, W)
    keypoints     : keypoints produced by any detector
    default_sigma : fallback scale for keypoints whose sigma is None

    Returns
    -------
    keypoints   : same list as input (MLDB never rejects keypoints)
    descriptors : uint8 array, shape (N, 61)
    """
    if not keypoints:
        return [], np.empty((0, DESC_BYTES), dtype=np.uint8)

    gray_f = gray.astype(np.float32) if gray.dtype != np.float32 else gray

    rows = [
        _mldb_single(_extract_patch(gray_f, kp, default_sigma))
        for kp in keypoints
    ]
    return keypoints, np.vstack(rows)
