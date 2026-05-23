"""liop.py — LIOP (Local Intensity Order Pattern) descriptor.

NumPy/OpenCV implementation of the descriptor from:
    Wang Z., Fan B., Wu F.  "Local Intensity Order Pattern for Feature
    Description."  ICCV 2011.

Works with any keypoints — no coupling to a specific detector.

Configurable hyper-parameters (forwarded from ``RunConfig.descriptor_params``)
-----------------------------------------------------------------------------
* ``n_neighbors``         (int,  default 4)   — K in the paper. Descriptor uses K! ordinal codes per point.
* ``n_bins``              (int,  default 6)   — number of equal-population ordinal intensity bins.
* ``patch_size``          (int,  default 41)  — warped-patch side length in pixels; must be odd.
* ``patch_radius_sigmas`` (float, default 3.0) — physical patch radius = ``sigma * patch_radius_sigmas``.

Descriptor dimension: ``n_bins * factorial(n_neighbors)``.  Defaults give 144.
Each ordinal-bin histogram is L2-normalised independently before concatenation.

Geometry precomputation (in-circle pixel mask + K-NN table + Lehmer factorials
+ bin edges) is cached via ``functools.lru_cache`` keyed on
``(n_neighbors, n_bins, patch_size)`` so repeat calls with the same config pay
the precomputation cost exactly once.
"""

from __future__ import annotations

import math
from functools import lru_cache
import cv2
import numpy as np

from overlap_detection.types import Keypoint

# ---------------------------------------------------------------------------
# Default hyper-parameters (kept as module constants for tests / docs)
# ---------------------------------------------------------------------------

DEFAULT_N_NEIGHBORS: int = 4
DEFAULT_N_BINS: int = 6
DEFAULT_PATCH_SIZE: int = 41
DEFAULT_PATCH_RADIUS_SIGMAS: float = 3.0

# Backwards-compat exports — match what computed at the defaults.
_N_NEIGHBORS: int = DEFAULT_N_NEIGHBORS
_N_BINS: int = DEFAULT_N_BINS
_N_CODES: int = math.factorial(DEFAULT_N_NEIGHBORS)   # 24
DESC_DIM: int = DEFAULT_N_BINS * _N_CODES             # 144 at defaults
_PATCH_SIZE: int = DEFAULT_PATCH_SIZE
_PATCH_RADIUS: int = DEFAULT_PATCH_SIZE // 2


# ---------------------------------------------------------------------------
# Per-config geometry precomputation (cached)
# ---------------------------------------------------------------------------

class _LiopGeometry:
    """Precomputed tables for a fixed ``(n_neighbors, n_bins, patch_size)``."""

    __slots__ = (
        "n_neighbors", "n_bins", "n_codes", "desc_dim",
        "patch_size", "patch_radius", "n_pts",
        "in_circle", "nn_idx", "factorials", "bin_edges",
    )

    def __init__(self, n_neighbors: int, n_bins: int, patch_size: int) -> None:
        self.n_neighbors = n_neighbors
        self.n_bins = n_bins
        self.n_codes = math.factorial(n_neighbors)
        self.desc_dim = n_bins * self.n_codes
        self.patch_size = patch_size
        self.patch_radius = patch_size // 2

        ys, xs = np.mgrid[0:patch_size, 0:patch_size]
        self.in_circle = (
            (xs - self.patch_radius) ** 2 + (ys - self.patch_radius) ** 2
        ) <= self.patch_radius ** 2

        circle_pts = np.column_stack(
            [xs[self.in_circle], ys[self.in_circle]]
        ).astype(np.float32)
        self.n_pts = len(circle_pts)

        # K-NN index table
        dists_sq = (
            (circle_pts[:, None, :] - circle_pts[None, :, :]) ** 2
        ).sum(axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        self.nn_idx = np.argpartition(
            dists_sq, n_neighbors, axis=1
        )[:, :n_neighbors]
        del dists_sq

        # Lehmer factorials
        self.factorials = np.array(
            [math.factorial(n_neighbors - 1 - k) for k in range(n_neighbors)],
            dtype=np.int32,
        )

        # Equal-population bin edges
        self.bin_edges = np.round(
            np.linspace(0, self.n_pts, n_bins + 1)
        ).astype(np.int32)


@lru_cache(maxsize=16)
def _geometry(n_neighbors: int, n_bins: int, patch_size: int) -> _LiopGeometry:
    if n_neighbors < 2:
        raise ValueError("liop: n_neighbors must be >= 2")
    if n_bins < 1:
        raise ValueError("liop: n_bins must be >= 1")
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("liop: patch_size must be an odd integer >= 3")
    return _LiopGeometry(n_neighbors, n_bins, patch_size)


# Pre-warm the default geometry so existing import-time module constants are
# meaningful (and so the first call at defaults doesn't pay the cost).
_default_geom = _geometry(_N_NEIGHBORS, _N_BINS, _PATCH_SIZE)
_in_circle = _default_geom.in_circle
_nn_idx = _default_geom.nn_idx
_FACTORIALS = _default_geom.factorials
_BIN_EDGES = _default_geom.bin_edges
_N_PTS = _default_geom.n_pts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_patch(
    gray: np.ndarray, kp: Keypoint, default_sigma: float,
    patch_size: int, patch_radius: int, patch_radius_sigmas: float,
) -> np.ndarray:
    """Warp a scale-normalised, axis-aligned patch into the configured window.

    The physical patch radius is ``patch_radius_sigmas * sigma`` (defaults to
    3σ on each side).  Bilinear interpolation; border pixels reflected so
    edge keypoints never produce zero-padded artefacts.
    """
    sigma = kp.sigma if kp.sigma is not None else default_sigma
    phys_radius = max(sigma * patch_radius_sigmas, 1.0)   # avoid divide-by-zero
    scale = patch_radius / phys_radius
    M = np.array([
        [scale, 0.0, patch_radius - scale * kp.x],
        [0.0, scale, patch_radius - scale * kp.y],
    ], dtype=np.float32)
    patch = cv2.warpAffine(
        gray, M, (patch_size, patch_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return patch.astype(np.float32)


def _liop_single(patch: np.ndarray, geom: _LiopGeometry) -> np.ndarray:
    """Compute one descriptor vector from a normalised patch."""
    intensities: np.ndarray = patch[geom.in_circle]                # (N_pts,)

    # --- LIOP codes via vectorised Lehmer encoding --------------------------
    neighbor_int = intensities[geom.nn_idx]                        # (N_pts, K)
    ranks = np.argsort(np.argsort(neighbor_int, axis=1), axis=1).astype(np.int32)

    L = np.zeros_like(ranks)
    for j in range(geom.n_neighbors - 1):
        L[:, j] = (ranks[:, j + 1:] < ranks[:, j : j + 1]).sum(axis=1)
    codes: np.ndarray = (L * geom.factorials).sum(axis=1)          # (N_pts,)

    # --- Ordinal bin assignment (equal-population) --------------------------
    sorted_idx = np.argsort(intensities)
    bin_ids = np.empty(geom.n_pts, dtype=np.int32)
    for b in range(geom.n_bins):
        bin_ids[sorted_idx[geom.bin_edges[b] : geom.bin_edges[b + 1]]] = b

    # --- Histogram accumulation (fully vectorised) --------------------------
    combined = bin_ids * geom.n_codes + codes
    hist_flat = np.bincount(
        combined, minlength=geom.desc_dim,
    ).astype(np.float32)
    hist = hist_flat.reshape(geom.n_bins, geom.n_codes)

    # --- Per-bin L2 normalisation -------------------------------------------
    norms = np.linalg.norm(hist, axis=1, keepdims=True)
    hist /= np.where(norms > 0.0, norms, 1.0)

    return hist.ravel()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def liop_describe(
    gray: np.ndarray,
    keypoints: list[Keypoint],
    default_sigma: float,
    *,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    n_bins: int = DEFAULT_N_BINS,
    patch_size: int = DEFAULT_PATCH_SIZE,
    patch_radius_sigmas: float = DEFAULT_PATCH_RADIUS_SIGMAS,
) -> tuple[list[Keypoint], np.ndarray]:
    """Compute LIOP descriptors for a list of keypoints.

    LIOP never rejects keypoints (unlike e.g. BRIEF which needs a minimum
    border distance), so the returned keypoint list is identical to the input.

    Parameters
    ----------
    gray                 : uint8 or float32 grayscale image, shape (H, W)
    keypoints            : keypoints from any detector
    default_sigma        : fallback scale for keypoints whose sigma is None
    n_neighbors          : K — neighbours per sample point.  K! ordinal codes.
    n_bins               : B — equal-population ordinal intensity bins.
    patch_size           : warped-patch diameter (odd integer ≥ 3).
    patch_radius_sigmas  : physical patch radius = ``sigma * this``.

    Returns
    -------
    keypoints   : same list as input (no filtering)
    descriptors : float32 array, shape ``(N, n_bins * factorial(n_neighbors))``
    """
    geom = _geometry(int(n_neighbors), int(n_bins), int(patch_size))

    if not keypoints:
        return [], np.empty((0, geom.desc_dim), dtype=np.float32)

    gray_f = gray.astype(np.float32) if gray.dtype != np.float32 else gray

    rows = [
        _liop_single(
            _extract_patch(
                gray_f, kp, default_sigma,
                patch_size=geom.patch_size,
                patch_radius=geom.patch_radius,
                patch_radius_sigmas=float(patch_radius_sigmas),
            ),
            geom,
        )
        for kp in keypoints
    ]
    return keypoints, np.vstack(rows).astype(np.float32)
