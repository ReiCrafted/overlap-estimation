"""liop.py — LIOP (Local Intensity Order Pattern) descriptor.

NumPy/OpenCV implementation of the descriptor from:
    Wang Z., Fan B., Wu F.  "Local Intensity Order Pattern for Feature
    Description."  ICCV 2011.

Works with any keypoints — no coupling to a specific detector.  All patches
are warped to a fixed _PATCH_SIZE window so the K-NN structure is precomputed
once at import time, keeping per-keypoint cost low.

Descriptor layout
-----------------
* K = 4 nearest neighbours  →  4! = 24 possible ordinal codes
* B = 6 ordinal intensity bins  (equal-population, sorted by intensity)
* Descriptor dim = B × 24 = 144 float32 values
* Each bin's histogram is L2-normalised independently before concatenation.
"""

from __future__ import annotations

import math
import cv2
import numpy as np

from overlap_detection.types import Keypoint

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

_N_NEIGHBORS: int = 4          # K nearest neighbours  →  K! = 24 codes
_N_BINS: int = 6               # ordinal intensity bins
_N_CODES: int = 24             # 4! permutation codes
DESC_DIM: int = _N_BINS * _N_CODES   # 144

_PATCH_SIZE: int = 41          # fixed patch diameter in pixels (must be odd)
_PATCH_RADIUS: int = _PATCH_SIZE // 2   # 20

# ---------------------------------------------------------------------------
# One-time geometry precomputation
# ---------------------------------------------------------------------------
# All patches are warped to the same _PATCH_SIZE × _PATCH_SIZE window, so
# the in-circle pixel mask and their K-NN table are constant and can be
# computed once at module import time.

_ys, _xs = np.mgrid[0:_PATCH_SIZE, 0:_PATCH_SIZE]
_in_circle: np.ndarray = (
    (_xs - _PATCH_RADIUS) ** 2 + (_ys - _PATCH_RADIUS) ** 2
) <= _PATCH_RADIUS ** 2                                         # bool (H, W)

_circle_pts: np.ndarray = np.column_stack(
    [_xs[_in_circle], _ys[_in_circle]]
).astype(np.float32)                                            # (N_pts, 2)
_N_PTS: int = len(_circle_pts)

# K-NN index table: _nn_idx[i] = indices of K nearest neighbours of point i
_dists_sq: np.ndarray = (
    (_circle_pts[:, None, :] - _circle_pts[None, :, :]) ** 2
).sum(axis=2)                                                   # (N_pts, N_pts)
np.fill_diagonal(_dists_sq, np.inf)
_nn_idx: np.ndarray = np.argpartition(
    _dists_sq, _N_NEIGHBORS, axis=1
)[:, :_N_NEIGHBORS]                                             # (N_pts, K)
del _dists_sq   # free ~12 MB; only the compact _nn_idx is kept

# Factorials for vectorised Lehmer encoding: [3!, 2!, 1!, 0!] = [6, 2, 1, 1]
_FACTORIALS: np.ndarray = np.array(
    [math.factorial(_N_NEIGHBORS - 1 - k) for k in range(_N_NEIGHBORS)],
    dtype=np.int32,
)

# Equal-population bin edges into the sorted-intensity index array
_BIN_EDGES: np.ndarray = np.round(
    np.linspace(0, _N_PTS, _N_BINS + 1)
).astype(np.int32)                                              # (N_BINS + 1,)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_patch(gray: np.ndarray, kp: Keypoint, default_sigma: float) -> np.ndarray:
    """Warp a scale-normalised, axis-aligned patch into the fixed window.

    The physical patch radius is 3σ on each side (covers ±3σ of the Gaussian
    scale associated with the keypoint).  The patch is mapped with bilinear
    interpolation; border pixels are filled by reflection so edge keypoints
    never produce zero-padded artefacts.

    Parameters
    ----------
    gray          : float32 grayscale image (H, W)
    kp            : source keypoint
    default_sigma : fallback scale when kp.sigma is None

    Returns
    -------
    patch : float32 array, shape (_PATCH_SIZE, _PATCH_SIZE)
    """
    sigma = kp.sigma if kp.sigma is not None else default_sigma
    phys_radius = max(sigma * 3.0, 1.0)          # avoid divide-by-zero
    scale = _PATCH_RADIUS / phys_radius
    M = np.array([
        [scale, 0.0, _PATCH_RADIUS - scale * kp.x],
        [0.0, scale, _PATCH_RADIUS - scale * kp.y],
    ], dtype=np.float32)
    patch = cv2.warpAffine(
        gray, M, (_PATCH_SIZE, _PATCH_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return patch.astype(np.float32)


def _liop_single(patch: np.ndarray) -> np.ndarray:
    """Compute the 144-dim LIOP descriptor for one normalised patch.

    Algorithm
    ---------
    1. Sample intensities at every in-circle pixel (precomputed mask).
    2. For each sample point, rank the intensities of its K nearest
       neighbours using the Lehmer code → integer in [0, 23].
    3. Assign each sample point to one of B ordinal bins (equal-population
       partition of the intensity-sorted point list).
    4. Accumulate a 24-bin count histogram per ordinal bin.
    5. L2-normalise each bin histogram and concatenate → 144-d vector.

    Parameters
    ----------
    patch : float32 array, shape (_PATCH_SIZE, _PATCH_SIZE)

    Returns
    -------
    descriptor : float32 array, shape (DESC_DIM,) == (144,)
    """
    intensities: np.ndarray = patch[_in_circle]               # (N_pts,)

    # --- LIOP codes via vectorised Lehmer encoding --------------------------
    neighbor_int = intensities[_nn_idx]                        # (N_pts, K)
    # ranks[i, j] = rank of neighbour j among the K neighbours of point i
    # (0 = smallest intensity, K-1 = largest)
    ranks = np.argsort(np.argsort(neighbor_int, axis=1), axis=1).astype(np.int32)

    # Lehmer digit L[i, j] = number of elements to the RIGHT of position j
    # in row i that are strictly less than ranks[i, j].
    L = np.zeros_like(ranks)
    for j in range(_N_NEIGHBORS - 1):
        L[:, j] = (ranks[:, j + 1:] < ranks[:, j : j + 1]).sum(axis=1)
    # Weighted sum gives a unique integer in [0, K!-1] = [0, 23]
    codes: np.ndarray = (L * _FACTORIALS).sum(axis=1)         # (N_pts,)

    # --- Ordinal bin assignment (equal-population) --------------------------
    sorted_idx = np.argsort(intensities)
    bin_ids = np.empty(_N_PTS, dtype=np.int32)
    for b in range(_N_BINS):
        bin_ids[sorted_idx[_BIN_EDGES[b] : _BIN_EDGES[b + 1]]] = b

    # --- Histogram accumulation (fully vectorised) --------------------------
    combined = bin_ids * _N_CODES + codes                      # flat bin index
    hist_flat = np.bincount(
        combined, minlength=_N_BINS * _N_CODES
    ).astype(np.float32)
    hist = hist_flat.reshape(_N_BINS, _N_CODES)

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
) -> tuple[list[Keypoint], np.ndarray]:
    """Compute LIOP descriptors for a list of keypoints.

    LIOP never rejects keypoints (unlike e.g. BRIEF which needs a minimum
    border distance), so the returned keypoint list is identical to the input.

    Parameters
    ----------
    gray          : uint8 or float32 grayscale image, shape (H, W)
    keypoints     : keypoints from any detector
    default_sigma : fallback scale for keypoints whose sigma is None

    Returns
    -------
    keypoints   : same list as input (no filtering)
    descriptors : float32 array, shape (N, 144)
    """
    if not keypoints:
        return [], np.empty((0, DESC_DIM), dtype=np.float32)

    gray_f = gray.astype(np.float32) if gray.dtype != np.float32 else gray

    rows = [
        _liop_single(_extract_patch(gray_f, kp, default_sigma))
        for kp in keypoints
    ]
    return keypoints, np.vstack(rows).astype(np.float32)
