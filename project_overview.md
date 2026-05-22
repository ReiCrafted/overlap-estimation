# Project Overview

## Repository Directory Structure

```
Overlap_Estimation/
├── .gitignore                 # Specifies intentionally untracked files to ignore
├── LICENSE                    # MIT
├── README.md                  # Top-level entry point with quick-start commands
├── pyproject.toml             # Packaging metadata and dependencies
├── project_overview.md        # This document — architecture, data contracts, outputs
├── overlap_detection/         # Main Python source package
│   ├── __init__.py            # Public-API re-exports
│   ├── config.py              # RunConfig, VALID_MASK_MODES, VALID_ESTIMATORS
│   ├── types.py               # Shared dataclasses (Keypoint, PairResult, GroundTruth)
│   ├── preprocessing.py       # Masking (overlap band + greenness/grayness)
│   ├── detection.py           # OpenCV feature detector wrappers
│   ├── description.py         # Descriptor dispatch; routes LIOP/MLDB to custom impls
│   ├── liop.py                # Custom NumPy LIOP descriptor (144-dim float32)
│   ├── mldb.py                # Custom NumPy MLDB descriptor (486-bit / 61-byte uint8)
│   ├── matching.py            # Keypoint matching (BFMatcher, NNDR, MNN)
│   ├── verification.py        # Robust affine estimation (PROSAC / USAC_MAGSAC)
│   ├── geometry.py            # Overlap-polygon computation from affine
│   ├── metrics.py             # IoU, mean corner error, result categorisation
│   ├── reporting.py           # JSON/CSV writers + markdown summary report
│   ├── orchestrator.py        # Pipeline runner + experiment matrix (multi-core)
│   ├── auto_aligner.py        # Background auto-alignment for the annotation GUI
│   └── annotation_gui.py      # Standalone Tkinter ground-truth labelling GUI
├── tests/                     # Pytest suite (per-module)
├── scripts/                   # CLI entrypoints
│   ├── run_experiment.py      # Runs the experimental matrix
│   ├── annotate_dataset.py    # Launches the annotation GUI
│   └── generate_report.py     # Renders the markdown summary from aggregate CSV
├── reports/                   # Committed example reports (markdown + future plots)
└── results/                   # Per-pair JSONs + aggregate CSV (gitignored)
```

## Pipeline Architecture

The overlap detection system is built around a multi-stage sequential pipeline managed by distinct modules:

1. **Stage 1: Preprocessing & Masking** (`preprocessing.py`)
   - Inputs: Images A and B.
   - Outputs: Binary masks isolating regions of interest (e.g., green plants) using RGB-channel-range thresholding combined with an overlap-band mask along the gantry motion axis.
   - Modes consumed by the per-attempt pipeline: `"no_mask"` (band-only) and `"mask"` (band ∧ grayness exclusion).
   - The user-facing `RunConfig.mask_mode = "both"` is **scheduling-only** and does not reach this stage — the orchestrator expands it into one `"no_mask"` invocation and one `"mask"` invocation per pair (see Stage 8).

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
   - Outputs: Inliers mask and estimated affine transformation matrix (`verify_affine`) using `PROSAC` or `USAC_MAGSAC`.
   - **Acceptance gates** (both applied before the affine is accepted):
     1. **Min-inliers**: `n_inliers ≥ RunConfig.min_inliers` (default `8`).
     2. **Affine sanity**: scale and rotation are decomposed from the 2×2 sub-matrix; the affine is rejected if `|scale − 1| > 10 %` or `|rotation| > 3°`. Catches geometrically implausible transforms that accumulate enough inliers to pass RANSAC but are impossible for near-planar, similarly-scaled adjacent image pairs. Thresholds: `_MAX_SCALE_DIFF = 0.10`, `_MAX_ROTATION_DEG = 3.0` in `orchestrator.py`.
   - A failing attempt sets `result.affine_matrix = None`, `error_message`, and the categorisation stage labels it `"no_match"`.

6. **Stage 6: Overlap Geometry** (`geometry.py`)
   - Inputs: Accepted affine matrix and the two image shapes.
   - Outputs: Overlap polygon corners in both image frames.

7. **Stage 7: Metrics & Categorisation** (`metrics.py`)
   - Inputs: `PairResult` (timings, counts, affine, polygons) and `GroundTruth`.
   - Outputs: per-stage timings, keypoint/match/inlier counts, IoU vs. ground truth, **mean corner error** (mean Euclidean distance across the four overlap-polygon corners), and the **ordinal `result_label`** assigned by `categorize_result`.
   - **Error metric.** Mean corner reprojection error (`metrics.mean_corner_error`), matching the HPatches / SuperGlue / LoFTR convention. This makes our acc@T rates and mAA directly comparable to numbers in published benchmark tables.
   - **`result_label` values** (driven by `RunConfig.accuracy_tiers_px`, default `(3, 5, 10)` px):
     - `"acc_at_<T>"` where T is the smallest configured tier the pair cleared (e.g. `"acc_at_3"` is strictly better than `"acc_at_5"` is strictly better than `"acc_at_10"`).
     - `"false_match"` — pipeline produced an accepted affine but its corner error exceeded every configured tier.
     - `"no_match"` — no affine was produced (insufficient keypoints/matches, RANSAC failure, or affine-sanity rejection).

8. **Orchestration** (`orchestrator.py`)
   - **Per-attempt pipeline.** `_execute_pipeline(...)` runs the seven stages above with a concrete `mask_mode ∈ {"no_mask", "mask"}` and returns one `(PairResult, metrics)` tuple. It never sees the user-facing `"both"` value.
   - **Public entrypoint `run_single_pair`** dispatches based on `config.mask_mode`:
     - `"no_mask"` or `"mask"` → list of 1 `(result, metrics)` tuple.
     - `"both"` → list of 2 tuples, `no_mask` first then `mask`. Both attempts are run unconditionally — no early exit on first success.
   - **Why `"both"` instead of fallback?** Any "try X first, fall back to Y" policy can be derived post-hoc from the paired columns produced by `"both"`. Running both unconditionally costs one extra pipeline pass per pair but yields a richer dataset: per-attempt accuracy distributions, false-match shares per mask mode, and a free `best_of_both` analysis in reporting (see Outputs section). The main pipeline therefore has no fallback machinery; the only consumer that needs short-circuit fallback semantics is the auto-aligner, which implements them locally (Stage 9).
   - **Experiment matrix.** `run_experiment_matrix` schedules a Cartesian product of `(pair, detector, descriptor, estimator, mask_mode_spec)` across a `multiprocessing` pool. Each worker handles one image pair, loads the images once, and runs every pending attempt for that pair, writing per-attempt JSON files as it goes. Resume is **per-attempt**: an existing JSON for `(pair, det, desc, est, no_mask)` is reused without rerunning that specific attempt, even if its sibling `_mask.json` is missing.
   - **Pool details.** Start method is `"spawn"` (Windows-friendly; avoids inheriting OpenCV/GUI state from the parent). Worker count picked by `default_experiment_workers()` — `cpu_count - 1`, capped at `_DEFAULT_EXPERIMENT_WORKER_CAP = 8`. `--workers 1` forces serial execution for debugging.
   - **CSV row aggregation.** For each `(pair, det, desc, est, mask_mode_spec)`, one CSV row is emitted with paired `no_mask_*` and `with_mask_*` columns (NaN on whichever side wasn't run). See the Outputs section for the full column list.

9. **Ground Truth Annotation** (`annotation_gui.py`, `auto_aligner.py`)
   - Standalone Tkinter GUI for manually aligning each pair and saving a `GroundTruth` JSON.
   - **Background auto-alignment.** `AutoAligner` runs in a `multiprocessing.Pool` while the annotator reviews. Worker count from `default_auto_align_workers()` — `cpu_count - 1`, capped at `_DEFAULT_AUTO_ALIGN_WORKER_CAP = 6`.
   - **Self-contained fallback.** The auto-aligner needs a fast best-effort alignment, not the full both-attempts dataset that the experiment runner wants. It therefore implements its own `_try_with_fallback(cfg_template)` helper that calls `run_single_pair` with `mask_mode = "no_mask"` first and `"mask"` only if the first attempt produced no transform. This is the only place in the codebase that has fallback semantics — by design, so the main pipeline stays simple.

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
    # Mask scheduling
    mask_mode: str = "both"        # "no_mask" | "mask" | "both"
    rgb_gray_threshold: int = 15
    overlap_band_fraction: float = 0.20

    # Detector
    detector: str = "SIFT"
    detector_params: dict = field(default_factory=dict)
    max_keypoints: int = 5000

    # Descriptor
    descriptor: str = "SIFT"
    descriptor_params: dict = field(default_factory=dict)
    descriptor_default_sigma: float = 6.0

    # Matching
    matcher_filter: str = "mnn_nndr"
    nndr_threshold: float = 0.90

    # Verification
    estimator: str = "PROSAC"      # "PROSAC" | "USAC_MAGSAC"
    ransac_threshold_px: float = 5.0
    ransac_max_iters: int = 10000
    ransac_confidence: float = 0.99

    # Acceptance / categorisation
    min_inliers: int = 8                              # affine acceptance gate
    accuracy_tiers_px: tuple[float, ...] = (3.0, 5.0, 10.0)  # tier thresholds for result_label

    # I/O
    output_dir: Path = Path("./results")
    save_intermediate: bool = False
    random_seed: int = 42
```

`VALID_MASK_MODES = {"no_mask", "mask", "both"}` and `VALID_ESTIMATORS = {"PROSAC", "USAC_MAGSAC"}` live alongside `DETECTOR_NAMES` / `DESCRIPTOR_NAMES` in `config.py`. CLI entrypoints validate against these sets so unknown strings fail upfront.

---

## Experiment Outputs

An experiment writes three kinds of artefact into `output_dir`. Knowing **what** each one contains and **why** makes it easier to consume them programmatically without reading the code.

### 1. Per-attempt JSON files

```
{pair_id}_{detector}_{descriptor}_{estimator}_{no_mask|mask}.json
```

One file per `(pair, det, desc, est, attempt_mode)`. The filename's last segment is always the **concrete** mask mode that ran — never `"both"`. When `mask_mode = "both"`, a pair contributes two files (`..._no_mask.json` and `..._mask.json`).

Top-level shape: `{"result": {...PairResult fields...}, "metrics": {...}}`.

**Why store raw measurements (not the categorical label)?** Because the categorical depends on `accuracy_tiers_px`. Keeping raw `iou`, `mean_corner_error`, and `corner_error_{0..3}` in the JSON means changing the tier set between report generations requires **zero pipeline reruns** — the CSV is rebuilt from the JSONs with the new tiers applied.

Per-file contents:

| Block        | Field                       | Type / units            | Why we keep it |
|--------------|-----------------------------|--------------------------|----------------|
| `result`     | `image_a_path`, `image_b_path` | str (filesystem path) | Provenance |
|              | `detector`, `descriptor`, `estimator`, `mask_mode` | str  | Config snapshot |
|              | `n_kp_a`, `n_kp_b`          | int                      | Per-image keypoint counts (post-mask, pre-description) |
|              | `n_raw_matches`             | int                      | Tentative matches before RANSAC |
|              | `n_inliers`                 | int                      | Affine-supporting matches |
|              | `affine_matrix`             | list (2×3 float)         | Estimated transform, A → B. `None` on failure |
|              | `inlier_mask`               | list (bool)              | Match-level inlier indicator |
|              | `overlap_polygon_a`, `overlap_polygon_b` | list (N×2 float) | Computed overlap region in each frame |
|              | `time_{detection,description,matching,verification,geometry}_s` | float | Per-stage wall-clock |
|              | `time_total_s`              | float                    | End-to-end wall-clock |
|              | `error_message`             | str \| null              | Human-readable failure reason |
|              | `result_label`              | str                      | Echoed from metrics (see below) |
| `metrics`    | `iou`                       | float \| null            | Overlap-polygon IoU vs. GT |
|              | `mean_corner_error`         | float \| null, px        | Mean Euclidean distance across the four overlap-polygon corners — the canonical HPatches-style accuracy metric |
|              | `corner_error_{0..3}`       | float \| null, px        | Per-corner errors (useful for debugging lopsided drift) |
|              | `result_label`              | str                      | Categorical (see Stage 7) |
|              | `num_keypoints_A/B`, `num_inliers`, `inlier_ratio` | numeric | CSV-friendly aliases of the result counts |
|              | `*_ms`                      | float, ms                | Per-stage timings in milliseconds |
|              | `pair_id`, `detector`, `descriptor`, `estimator`, `mask_mode_spec`, `attempt_mode` | str | Identifying fields stamped at write time |

### 2. Aggregate CSV (`aggregate_results.csv`)

One row per `(pair_id, detector, descriptor, estimator, mask_mode_spec)`. When `mask_mode_spec = "both"`, both column families are populated; otherwise the unrun side is NaN.

Identifying columns:

| Column           | Description |
|------------------|-------------|
| `pair_id`        | `"{stem_A}_{stem_B}"` |
| `detector`, `descriptor`, `estimator` | Config snapshot |
| `mask_mode`      | The user-requested spec (`"no_mask"`, `"mask"`, or `"both"`). Distinct from the per-attempt `attempt_mode` inside each JSON. |

Per-attempt columns (suffix is the canonical attempt name — `no_mask` or `with_mask`):

| Column                            | Description |
|-----------------------------------|-------------|
| `{prefix}_result`                 | Friendly alias of `{prefix}_result_label` — `acc_at_<T>` / `false_match` / `no_match` |
| `{prefix}_result_label`           | Same value; long name for explicit code paths |
| `{prefix}_err`                    | Friendly alias of `{prefix}_mean_corner_error` (px) |
| `{prefix}_mean_corner_error`      | Mean corner reprojection error in pixels |
| `{prefix}_iou`                    | Overlap-polygon IoU vs. GT |
| `{prefix}_num_keypoints_A/B`      | Keypoint counts per image |
| `{prefix}_num_tentative_matches`  | Pre-RANSAC matches |
| `{prefix}_num_inliers`            | Post-RANSAC inliers |
| `{prefix}_inlier_ratio`           | `num_inliers / num_tentative_matches` |
| `{prefix}_{detection,description,matching,verification,geometry,total}_ms` | Per-stage runtimes in milliseconds |

**Why both `*_result` and `*_result_label`** — short alias keeps interactive pandas terse; long name is what the JSON also writes, so code that reads either source can use a single key.

### 3. Markdown summary report (`report.md`, written separately by `scripts/generate_report.py`)

Contents (all derived from the CSV; no per-attempt JSONs needed at this stage):

| Section                 | What it shows | Why |
|-------------------------|---------------|-----|
| **Overall**             | Number of CSV rows, the configured `accuracy_tiers_px`, and per-attempt mAA + per-tier acc rates + `false_match` / `no_match` shares | Single-glance sanity check on the whole run |
| **Per-configuration scoreboard** | One row per `(det, desc, est)` with paired `mAA`, `acc@T`, `false_match`, `no_match` columns for each attempt, plus `mAA<sub>best</sub>` and `acc@T<sub>best</sub>` columns derived from a per-pair "pick the better attempt" policy | Direct comparison of pipelines, with the fallback-vs-single-attempt question answered in-place |
| **mAA matrices**        | Detector × descriptor table of mAA, one per attempt | Compact pipeline-vs-pipeline view; designed so it can be re-rendered as a heatmap |
| **Fallback benefit**    | Per-config table of `mAA_best − mAA_no_mask` and `mAA_best − mAA_with_mask` | Quantifies how much running `mask_mode = "both"` would actually buy you over running a single mask attempt — answered post-hoc, no extra runs |

The report is **markdown-only in the current revision**. Heatmap and curve visualisations are intentionally deferred until the report-visualisation design is settled separately.

### Metric definitions (canonical)

- **Mean corner error** — `mean(||p_pred_i − p_gt_i||₂  for i in 0..3)` over the four overlap-polygon corners, in pixels. Matches the HPatches / SuperGlue / LoFTR convention so our numbers are directly comparable to published tables.
- **IoU** — `intersection_area / union_area` between the predicted and ground-truth overlap polygons in image-A coordinates.
- **`acc@T`** (per attempt, per config) — fraction of rows whose `result_label` is some `"acc_at_<t>"` with `t ≤ T`. Cumulative: hitting a tighter tier implies clearing every looser one.
- **mAA** — mean over the configured `accuracy_tiers_px` of the per-tier `acc@T` rate. Equivalently: for each pair, count how many tiers it cleared and divide by the tier count; average across pairs. A pipeline that consistently lands at 2 px (clears all tiers) scores 1.00; one that consistently lands at 11 px (clears no tier) scores 0.00.
- **`best_of_both`** — per-pair pick of the better attempt by tier rank (smaller-tier label > larger-tier label > `false_match` > `no_match`). Used to compute the fallback-benefit lift; **not** something the pipeline can produce online — it's a post-hoc analysis on the paired columns.

All markdown / JSON I/O is UTF-8.

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

