# Project Overview

## Repository Directory Structure

```
Overlap_Estimation/
├── .gitignore                 # Specifies intentionally untracked files to ignore
├── pyproject.toml             # Configuration for packaging and building the project
├── project_overview.md        # This repository layout description
├── overlap_detection/         # Main Python source package
│   ├── __init__.py            # Marks the directory as a Python package
│   ├── config.py              # RunConfig dataclass for parameters
│   ├── types.py               # Shared dataclasses (Keypoint, PairResult, GroundTruth)
│   ├── preprocessing.py       # Masking and colorfulness/grayness test calculations
│   ├── detection.py           # Wrappers for feature detectors (SIFT, KAZE, etc.)
│   ├── description.py         # Descriptor dispatch; routes LIOP/MLDB to custom impls
│   ├── liop.py                # Custom NumPy LIOP descriptor (144-dim float32)
│   ├── mldb.py                # Custom NumPy MLDB descriptor (486-bit / 61-byte uint8)
│   ├── matching.py            # Keypoint matching logic (BFMatcher, NNDR, MNN)
│   ├── verification.py        # Robust geometric verification wrappers (USAC_MAGSAC)
│   ├── geometry.py            # Homography to overlap polygon transformation logic
│   ├── metrics.py             # Accuracy evaluation metrics (IoU, corner error)
│   ├── reporting.py           # Writers for JSON/CSV results and plot generation
│   ├── orchestrator.py        # Matrix/grid experiment runner
│   └── annotation_gui.py      # Standalone alignment/ground truth labeling GUI
├── tests/                     # Unit and integration tests
│   └── __init__.py            # Test package initializer
├── scripts/                   # CLI entrypoints for pipeline executions
│   ├── run_experiment.py      # Runs the experimental matrix benchmarking
│   ├── annotate_dataset.py    # Launches the Tkinter ground truth alignment GUI
│   └── generate_report.py     # Parses experimental results and generates plots
└── results/                   # Destination for CSV/JSON outputs and visualizations
    └── .gitkeep               # Directory placeholder
```

## Pipeline Architecture

The overlap detection system is built around a multi-stage sequential pipeline managed by distinct modules:

1. **Stage 1: Preprocessing & Masking** (`preprocessing.py`)
   - Inputs: Images A and B.
   - Outputs: Binary masks isolating regions of interest (e.g., green plants) using HSV-based color space thresholding or circular tray constraints.
   - Modes: `no_mask`, `mask`, `fallback`.

2. **Stage 2: Detection** (`detection.py`)
   - Inputs: Preprocessed/masked images.
   - Outputs: List of `Keypoint` dataclass instances with properties like sub-pixel coordinates $(x, y)$, scale $\sigma$, orientation $\theta$, and response strength.
   - **MSER note:** MSER uses `detectRegions()` rather than `detect()`. Each detected blob is converted to a single keypoint at the region centroid, with `response = pixel count` (larger region → stronger response). It produces no scale or orientation (`sigma=None`, `theta=None`); descriptors use `descriptor_default_sigma` as the fallback patch size. Default parameters are tuned away from OpenCV's built-in values: `min_area=20`, `max_area=8100`, `max_variation=0.5`. The original defaults (`min_area=60`, `max_area=14400`, `max_variation=0.25`) produced zero keypoints on the target dataset; the tuned values yield ~5000 keypoints on 2464×2056 images with region sizes in the 245–7980 px range.

3. **Stage 3: Description** (`description.py`)
   - Inputs: Images and keypoint lists.
   - Outputs: Descriptor matrices for both images. Handles scale-less/orientation-less keypoints where needed.

4. **Stage 4: Matching & Filtering** (`matching.py`)
   - Inputs: Descriptor matrices.
   - Outputs: Sorted list of matches ranked by distance, filtered by Mutual Nearest Neighbors (MNN) or Nearest Neighbor Distance Ratio (NNDR).

5. **Stage 5: Geometric Verification** (`verification.py`, `orchestrator.py`)
   - Inputs: Keypoint lists and matches.
   - Outputs: Inliers mask and estimated affine transformation matrix (`verify_affine`) using robust estimators like `PROSAC` or `USAC_MAGSAC`.
   - After RANSAC, `orchestrator.py` applies an **affine sanity filter** before accepting the result: the estimated matrix is decomposed into scale (column-0 norm of the 2×2 sub-matrix) and rotation (`atan2(a10, a00)`). If `|scale − 1| > 10%` or `|rotation| > 3°` the result is rejected with a descriptive error message and the pair is counted as failed. This catches geometrically implausible transforms that accumulate enough inliers to pass RANSAC but are physically impossible for near-planar, similarly-scaled adjacent image pairs. Thresholds are defined as `_MAX_SCALE_DIFF = 0.10` and `_MAX_ROTATION_DEG = 3.0` in `orchestrator.py`.

6. **Stage 6: Overlap Geometry** (`geometry.py`)
   - Inputs: Estimated transformation matrix and image shapes.
   - Outputs: Intersection bounding polygons of overlap regions and their corner coordinates.

7. **Stage 7: Metrics & Reporting** (`metrics.py`, `reporting.py`)
   - Inputs: Transformation results, keypoint matches, and ground truth polygons.
   - Outputs: Computes IoU (Intersection over Union), corner error, aggregates metrics, and saves results (JSON/CSV) along with visualizations.

8. **Orchestration** (`orchestrator.py`)
   - Manages matrix runs, pipeline execution loops, error fallbacks, and multi-configuration sweeps.

9. **Ground Truth Annotation** (`annotation_gui.py`)
   - Provides a standalone Tkinter GUI allowing manual alignment and corner-picking to establish ground-truth homographies.

---

## Data Contracts between Pipeline Stages

The pipeline defines strict contracts for structures passed between stages to maintain modular decoupling:

### 1. Image
- **Format**: `numpy.ndarray`
- **Shape**: `(H, W, 3)`
- **Dtype**: `uint8`
- **Details**: RGB color channel order.

### 4. Mask
- **Format**: `numpy.ndarray`
- **Shape**: `(H, W)`
- **Dtype**: `uint8`
- **Details**: Binary mask where values are `0` (ignore) or `255` (keep).

### 3. Keypoint List
- **Format**: `list` of `Keypoint` dataclass instances:
```python
@dataclass
class Keypoint:
    x: float                # Sub-pixel x coordinate
    y: float                # Sub-pixel y coordinate
    response: float         # Detector response strength
    sigma: float | None     # Scale (None if detector doesn't provide)
    theta: float | None     # Orientation in radians (None if absent)
    octave: int | None      # Pyramidal scale octave layer
```

### 4. Descriptor Matrix
- **Format**: `numpy.ndarray`
- **Shape**: `(N, D)` where $N$ is the number of keypoints and $D$ is the descriptor dimensionality.
- **Dtype**: `float32` (for float descriptors like SIFT/KAZE) or `uint8` (for packed binary descriptors like ORB/BRISK).

### 5. Match List
- **Format**: `numpy.ndarray`
- **Shape**: `(M, 3)` where $M$ is the number of matches.
- **Columns**: `[idx_in_A, idx_in_B, distance]` where columns 0 and 1 are integer indices and column 2 is float distance.

### 6. Transformation Matrix
- **Format**: `numpy.ndarray`
- **Shape**: `(2, 3)` (Affine transformation)
- **Dtype**: `float64`

### 7. Overlap Polygon
- **Format**: `numpy.ndarray`
- **Shape**: `(K, 2)` (typically `(4, 2)` representing four corners of the overlapping region in clockwise order).

### 8. Pair Result Record
- **Format**: A `PairResult` dataclass containing complete run metadata, configuration snapshot, timing, matching statistics, estimated affine matrix, and geometric error metrics.

---

## Configuration System

All components utilize a single centralized configuration object `RunConfig` defined in `overlap_detection/config.py`:

```python
@dataclass
class RunConfig:
    # Mask mode
    mask_mode: str = "fallback"  # "no_mask" | "mask" | "fallback"
    rgb_gray_threshold: int = 15
    overlap_band_fraction: float = 0.20

    # Detector
    detector: str = "SIFT"  # see DETECTOR_NAMES below
    detector_params: dict = field(default_factory=dict)  # detector-specific overrides
    max_keypoints: int = 5000

    # Descriptor
    descriptor: str = "SIFT"  # see DESCRIPTOR_NAMES below
    descriptor_params: dict = field(default_factory=dict)

    # Matching
    matcher_filter: str = "mnn_nndr"  # "mnn" | "mnn_nndr"
    nndr_threshold: float = 0.90  # Lowe ratio threshold

    # Verification
    estimator: str = "PROSAC"  # "PROSAC" | "USAC_MAGSAC"
    ransac_threshold_px: float = 5.0
    ransac_max_iters: int = 10000
    ransac_confidence: float = 0.99

    # Fallback logic
    fallback_min_inliers: int = 8

    # Output
    output_dir: Path = Path("./results")
    save_intermediate: bool = False
    random_seed: int = 42
```

---

## Custom Descriptor Implementations

Two descriptors in the pipeline are implemented from scratch in NumPy rather
than relying on OpenCV, each for a distinct reason.

---

### LIOP — Local Intensity Order Pattern

**Source file:** `overlap_detection/liop.py`  
**Reference:** Wang Z., Fan B., Wu F. "Local Intensity Order Pattern for Feature Description." ICCV 2011.

**Why custom:** `cv2.xfeatures2d.LIOP_create` does not exist in any released
version of opencv-contrib.  Despite being referenced in the xfeatures2d
documentation, the C++ source was never contributed to the repository.  No
pip-installable Python binding exists for any other LIOP implementation.

**Algorithm summary:**

1. A circular patch of radius 3σ is extracted around each keypoint using
   bilinear interpolation and warped into a fixed 41 × 41 pixel window.
2. All ~1 257 in-circle pixels are treated as sample points.  Because every
   patch has the same fixed size, the K-nearest-neighbour table for those
   points is precomputed once at module import time and reused for every
   keypoint.
3. For each sample point, the intensities of its K = 4 nearest neighbours are
   ranked.  The rank permutation is encoded as an integer 0–23 (4! codes)
   using the Lehmer factoradic representation — fully vectorised with NumPy.
4. Sample points are partitioned into B = 6 equal-population ordinal bins by
   sorting all points by their own intensity and dividing evenly.
5. A 24-bin count histogram is accumulated for each ordinal bin using
   `np.bincount`, then each bin's histogram is L2-normalised independently.
6. The 6 histograms are concatenated to form the final descriptor.

**Descriptor properties:**

| Property | Value |
|---|---|
| Dimension | 144 (6 bins × 24 codes) |
| Dtype | float32 |
| Binary? | No — use BFMatcher with NORM_L2 |
| Detector coupling | None — works with any detector |
| Patch size | 41 × 41 px (fixed) |
| Physical patch radius | 3σ |
| Import-time precomputation | ~12 MB temporary (KNN distances), ~20 KB kept (index table) |

**Matching override (orchestrator.py):**

LIOP's purely ordinal encoding makes all pairwise L2 distances cluster in a
narrow band (~0.95–1.05 out of a maximum of √6 ≈ 2.45) on low-texture
datasets (e.g. bare soil without vegetation).  When all descriptor distances
are similar, the NNDR ratio d₁/d₂ ≈ 1.0 everywhere, causing zero tentative
matches regardless of inlier presence.

`run_single_pair` in `orchestrator.py` therefore overrides `matcher_filter`
to `"mnn"` (mutual nearest neighbour, no ratio test) whenever
`config.descriptor == "LIOP"`.  The noisier match set that results is handled
downstream by PROSAC/RANSAC.  `RunConfig`, the matching module, and all
reporting paths are unmodified — the override is a single conditional at the
call site.

**Discriminability note:** This is a dataset-characteristic limitation, not an
implementation bug.  On high-texture images (e.g. plant canopy), LIOP's
ordinal coding is more distinctive and NNDR filtering may be viable.  On
near-uniform content the ratio test must be bypassed.

---

### MLDB — Modified Local Difference Binary

**Source file:** `overlap_detection/mldb.py`  
**Reference:** Alcantarilla P.F., Nuevo J., Bartoli A. "Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces." BMVC 2013.

**Why custom:** OpenCV's built-in MLDB is computed inside
`cv2.AKAZE_create().compute()`, which reads derivative images from AKAZE's
internal nonlinear diffusion pyramid indexed by `kp.class_id`.  Keypoints
from every other detector carry `class_id = -1` (the cv2.KeyPoint default),
causing a hard C++ assertion:

```
cv2.error: (-215) 0 <= kpts[i].class_id && kpts[i].class_id
           < static_cast<int>(evolution_.size())
           in AKAZEFeatures::Compute_Descriptors
```

This is a load-bearing architectural coupling in AKAZE, not a soft
limitation.  The custom implementation substitutes Gaussian smoothing and
Sobel gradients as detector-agnostic approximations of the diffusion images,
enabling MLDB on keypoints from any detector.

**Algorithm summary:**

1. A square patch spanning ±10σ on each side (matching AKAZE's
   `pattern_size = 10` convention) is extracted via bilinear warp into a
   fixed 60 × 60 pixel window.  60 is divisible by 2, 3, and 4 — the three
   grid sizes used — so all cell dimensions are integers.
2. Three channels are computed from the patch:
   - **Lt**: Gaussian-smoothed image (σ = 1.5 px), approximating AKAZE's
     nonlinear diffusion image at the keypoint's scale level.
   - **Lx**: Sobel x-gradient of Lt.
   - **Ly**: Sobel y-gradient of Lt.
3. The patch is divided three times using different grids (2×2, 3×3, 4×4).
   For each grid, per-cell means of Lt, Lx, Ly are computed using a single
   `reshape + mean` operation (no Python loops over cells).
4. All distinct cell pairs within each grid are compared for each channel.
   Each comparison yields one bit: 1 if the first cell's mean exceeds the
   second's, 0 otherwise.
5. Bit counts by grid:

   | Grid | Cell pairs C(n,2) | × 3 channels | Bits |
   |------|-------------------|--------------|------|
   | 2×2  | C(4,2) = 6        | × 3          | 18   |
   | 3×3  | C(9,2) = 36       | × 3          | 108  |
   | 4×4  | C(16,2) = 120     | × 3          | 360  |
   | **Total** |              |              | **486** |

6. All 486 bits are concatenated and packed with `np.packbits` (2 zero
   padding bits in the final byte).

**Descriptor routing (in `description.py`):**

The pipeline automatically selects the best available MLDB path per run:

| Detector | MLDB path used | Reason |
|---|---|---|
| AKAZE | **OpenCV native** | `class_id` indexes AKAZE's own nonlinear diffusion pyramid — exact match |
| KAZE | **Custom NumPy** | KAZE's `class_id` indexes a *linear* diffusion pyramid; passing it to AKAZE's `compute()` would index the wrong scale level in the wrong diffusion type |
| All others | **Custom NumPy** | No valid `class_id` available |

The routing is explicit: `describe()` accepts an optional `detector_name`
parameter; the native path is only taken when `detector_name == "AKAZE"` and
the keypoints carry a valid `class_id`.  The `class_id` field was added to
the `Keypoint` dataclass to survive the detect → describe round-trip:
AKAZE sets it during `detect()`, `_cv_kp_to_keypoint` preserves it, and
`describe()` passes it back to `cv2.AKAZE_create().compute()` via a
reconstructed `cv2.KeyPoint`.

**Descriptor properties:**

| Property | Value |
|---|---|
| Dimension | 486 bits = 61 bytes |
| Dtype | uint8 (packed bits) |
| Binary? | Yes — use BFMatcher with NORM_HAMMING |
| Detector coupling | None (custom path) / AKAZE+KAZE (native path) |
| Patch size | 60 × 60 px (fixed, custom path only) |
| Physical patch half-side | 10σ (custom path only) |
| Import-time precomputation | Cell-pair index arrays (trivial, < 1 KB) |

**Approximation note (custom path only):** The NumPy implementation substitutes
Gaussian smoothing and Sobel gradients for AKAZE's true nonlinear diffusion
images.  Descriptor values will not match OpenCV's MLDB bit-for-bit, but the
behaviour is consistent across all non-AKAZE detectors, which is what matters
for the comparative experiment.

