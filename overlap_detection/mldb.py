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
triggers a hard assertion.  This implementation substitutes Gaussian
smoothing and Sobel gradients as detector-agnostic approximations of
AKAZE's internal images (Lt, Lx, Ly), so MLDB descriptors can be computed
for keypoints from **any** detector.

Configurable hyper-parameters (forwarded from ``RunConfig.descriptor_params``)
-----------------------------------------------------------------------------
* ``patch_size``    (int,   default 60)  — warped-patch side length.  Must be
                                            divisible by every grid dimension.
* ``sigma_scale``   (float, default 10.0) — physical patch half-side =
                                            ``sigma * sigma_scale``.
* ``smooth_sigma``  (float, default 1.5) — Gaussian σ applied to the patch
                                            before Sobel (approximates AKAZE's
                                            nonlinear diffusion image Lt).
* ``grids`` (sequence of ``(rows, cols)``, default ``((2,2), (3,3), (4,4))``) —
                                            three subdivisions yield 486 bits
                                            at the defaults.

Descriptor layout at defaults (486 bits, packed into 61 bytes)
--------------------------------------------------------------
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
concatenated in the order they appear in ``grids``.  Trailing bits in the
final byte are zero padding (``np.packbits`` behaviour).
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import cv2
import numpy as np

from overlap_detection.types import Keypoint

# ---------------------------------------------------------------------------
# Default hyper-parameters (also kept as module constants for tests / docs)
# ---------------------------------------------------------------------------

DEFAULT_PATCH_SIZE: int = 60                   # LCM(2,3,4,5)
DEFAULT_SIGMA_SCALE: float = 10.0              # matches AKAZE pattern_size = 10
DEFAULT_SMOOTH_SIGMA: float = 1.5
DEFAULT_GRIDS: tuple[tuple[int, int], ...] = ((2, 2), (3, 3), (4, 4))

# Backwards-compat module constants — evaluated at the defaults.
_PATCH_SIZE: int = DEFAULT_PATCH_SIZE
_SIGMA_SCALE: float = DEFAULT_SIGMA_SCALE
_SMOOTH_SIGMA: float = DEFAULT_SMOOTH_SIGMA
_GRIDS: list[tuple[int, int]] = list(DEFAULT_GRIDS)
DESC_BITS: int = sum(
    len(list(combinations(range(r * c), 2))) * 3 for r, c in DEFAULT_GRIDS
)                                              # 18 + 108 + 360 = 486
DESC_BYTES: int = (DESC_BITS + 7) // 8         # 61


# ---------------------------------------------------------------------------
# Per-config precomputation (cached)
# ---------------------------------------------------------------------------

class _MldbLayout:
    """Cell-pair index arrays + derived bit/byte counts for a given config."""

    __slots__ = ("patch_size", "grids", "pairs_a", "pairs_b", "desc_bits", "desc_bytes")

    def __init__(self, patch_size: int, grids: tuple[tuple[int, int], ...]):
        self.patch_size = patch_size
        self.grids = grids
        self.pairs_a: dict[tuple[int, int], np.ndarray] = {}
        self.pairs_b: dict[tuple[int, int], np.ndarray] = {}
        bits = 0
        for r, c in grids:
            all_pairs = list(combinations(range(r * c), 2))
            self.pairs_a[(r, c)] = np.array([p[0] for p in all_pairs], dtype=np.int32)
            self.pairs_b[(r, c)] = np.array([p[1] for p in all_pairs], dtype=np.int32)
            bits += len(all_pairs) * 3
        self.desc_bits = bits
        self.desc_bytes = (bits + 7) // 8


@lru_cache(maxsize=16)
def _layout(patch_size: int, grids: tuple[tuple[int, int], ...]) -> _MldbLayout:
    if patch_size < 2:
        raise ValueError("mldb: patch_size must be >= 2")
    if not grids:
        raise ValueError("mldb: grids must be non-empty")
    for r, c in grids:
        if r < 1 or c < 1:
            raise ValueError(f"mldb: grid dims must be >= 1, got {(r, c)}")
        if patch_size % r != 0 or patch_size % c != 0:
            raise ValueError(
                f"mldb: patch_size={patch_size} must be divisible by every grid "
                f"dimension; failed on grid ({r}, {c})"
            )
    return _MldbLayout(patch_size, grids)


# Pre-warm so `from overlap_detection.mldb import DESC_BITS` etc. keep working.
_layout(DEFAULT_PATCH_SIZE, DEFAULT_GRIDS)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_patch(
    gray: np.ndarray, kp: Keypoint, default_sigma: float,
    patch_size: int, sigma_scale: float,
) -> np.ndarray:
    """Warp a scale-normalised square patch into ``patch_size × patch_size``."""
    sigma = kp.sigma if kp.sigma is not None else default_sigma
    phys_half = max(sigma * sigma_scale, 1.0)
    scale = (patch_size / 2.0) / phys_half
    M = np.array([
        [scale, 0.0, patch_size / 2.0 - scale * kp.x],
        [0.0, scale, patch_size / 2.0 - scale * kp.y],
    ], dtype=np.float32)
    patch = cv2.warpAffine(
        gray, M, (patch_size, patch_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return patch.astype(np.float32)


def _compute_channels(
    patch: np.ndarray, smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Lt (smoothed image), Lx, Ly (its Sobel x/y gradients)."""
    Lt = cv2.GaussianBlur(patch, (0, 0), smooth_sigma, borderType=cv2.BORDER_REFLECT)
    Lx = cv2.Sobel(Lt, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Ly = cv2.Sobel(Lt, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    return Lt, Lx, Ly


def _grid_means(channel: np.ndarray, rows: int, cols: int, patch_size: int) -> np.ndarray:
    """Per-cell mean values for a rows × cols partition of the patch.

    Requires ``patch_size % rows == 0`` and ``patch_size % cols == 0`` —
    enforced by ``_layout`` at config time.
    """
    h = patch_size // rows
    w = patch_size // cols
    return (
        channel
        .reshape(rows, h, cols, w)
        .mean(axis=(1, 3))
        .ravel()
        .astype(np.float32)
    )


def _mldb_single(patch: np.ndarray, smooth_sigma: float, layout: _MldbLayout) -> np.ndarray:
    """Compute the packed MLDB descriptor for one normalised patch."""
    Lt, Lx, Ly = _compute_channels(patch, smooth_sigma)

    all_bits: list[np.ndarray] = []
    for rows, cols in layout.grids:
        means = np.column_stack([
            _grid_means(Lt, rows, cols, layout.patch_size),
            _grid_means(Lx, rows, cols, layout.patch_size),
            _grid_means(Ly, rows, cols, layout.patch_size),
        ])
        pa = layout.pairs_a[(rows, cols)]
        pb = layout.pairs_b[(rows, cols)]
        comparisons = means[pa] > means[pb]
        all_bits.append(comparisons.ravel())

    return np.packbits(np.concatenate(all_bits))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mldb_describe(
    gray: np.ndarray,
    keypoints: list[Keypoint],
    default_sigma: float,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
    sigma_scale: float = DEFAULT_SIGMA_SCALE,
    smooth_sigma: float = DEFAULT_SMOOTH_SIGMA,
    grids: tuple[tuple[int, int], ...] | list[tuple[int, int]] = DEFAULT_GRIDS,
) -> tuple[list[Keypoint], np.ndarray]:
    """Compute MLDB descriptors for keypoints from any detector.

    Parameters
    ----------
    gray          : uint8 or float32 grayscale image, shape ``(H, W)``
    keypoints     : keypoints produced by any detector
    default_sigma : fallback scale for keypoints whose ``sigma`` is None
    patch_size    : warped-patch side length; must be divisible by every grid dim.
    sigma_scale   : physical patch half-side = ``sigma * sigma_scale``.
    smooth_sigma  : Gaussian σ applied to the patch before Sobel gradients.
    grids         : iterable of ``(rows, cols)`` subdivisions of the patch.

    Returns
    -------
    keypoints   : same list as input (MLDB never rejects keypoints)
    descriptors : uint8 array, shape ``(N, DESC_BYTES)`` where ``DESC_BYTES``
                  is derived from ``grids``: ``ceil(sum_grids(C(rows*cols, 2)) * 3 / 8)``
                  (= 61 at the defaults).
    """
    grids_tuple = tuple(tuple(g) for g in grids)
    layout = _layout(int(patch_size), grids_tuple)
    sigma_scale_f = float(sigma_scale)
    smooth_sigma_f = float(smooth_sigma)

    if not keypoints:
        return [], np.empty((0, layout.desc_bytes), dtype=np.uint8)

    gray_f = gray.astype(np.float32) if gray.dtype != np.float32 else gray

    rows = [
        _mldb_single(
            _extract_patch(gray_f, kp, default_sigma,
                           patch_size=layout.patch_size,
                           sigma_scale=sigma_scale_f),
            smooth_sigma_f, layout,
        )
        for kp in keypoints
    ]
    return keypoints, np.vstack(rows)
