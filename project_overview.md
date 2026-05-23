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
   - Outputs: Binary masks isolating regions of interest using RGB-channel-range thresholding combined with an overlap-band mask along the gantry motion axis.
   - **Primitive masks:**
     - `make_overlap_band_mask(image_shape, band_fraction, side)` — sets a vertical strip of width `int(W * band_fraction)` on each requested edge to `255`, the rest to `0`. `side ∈ {"left", "right", "both"}`, default `"both"`. Top and bottom edges are never masked (motion is horizontal).
     - `make_grayness_mask(image, gray_threshold)` — `255` where `np.ptp(rgb, axis=2) > gray_threshold`, `0` elsewhere. Operates on uint8 RGB; classifies a pixel as achromatic when its per-channel range is small.
     - `combine_masks(*masks)` — bitwise AND of all inputs; raises `ValueError` if no masks provided.
   - **Composite per attempt mode** (`apply_mask_mode(image, mode, band_fraction, gray_threshold, side)`):
     | `mode`        | Returned mask                                |
     |---------------|----------------------------------------------|
     | `"no_mask"`   | band mask only                               |
     | `"mask"`      | band mask ∧ grayness mask                    |
     | `"fallback"`  | same as `"no_mask"` (legacy alias)           |
     - Any other value → `ValueError(f"Unknown mask mode: {mode}")`.
   - **Orchestrator usage of `side`:** image A always receives `side="right"` (its right edge overlaps B) and image B always receives `side="left"`. Hard-coded in `_execute_pipeline`; not exposed in `RunConfig`.
   - The user-facing `RunConfig.mask_mode = "both"` is **scheduling-only** and does not reach this stage — the orchestrator expands it into one `"no_mask"` invocation and one `"mask"` invocation per pair (see Stage 8).

2. **Stage 2: Detection** (`detection.py`)
   - Inputs: Preprocessed image (uint8 RGB; internally converted to grayscale) and the mask from Stage 1.
   - Outputs: List of `Keypoint` dataclass instances with sub-pixel coordinates `(x, y)`, scale `sigma`, orientation `theta`, response strength, and (where set) `octave` and `class_id`.
   - **Truncation.** After the chosen detector runs, keypoints are sorted by descending `response` and truncated to at most `RunConfig.max_keypoints` (default `5000`).
   - **Per-detector OpenCV builder + tuned defaults** (overlaid by `RunConfig.detector_params`):

     | Name    | Backing OpenCV call                              | Default parameters (`_DETECTOR_DEFAULTS`) |
     |---------|--------------------------------------------------|-------------------------------------------|
     | Harris  | `cv2.goodFeaturesToTrack(useHarrisDetector=True)` | `maxCorners=5000, qualityLevel=0.01, minDistance=10, k=0.04` |
     | GFTT    | `cv2.goodFeaturesToTrack(useHarrisDetector=False)`| `maxCorners=5000, qualityLevel=0.01, minDistance=10` |
     | FAST    | `cv2.FastFeatureDetector_create`                 | `threshold=10, nonmaxSuppression=True, type=FAST_FEATURE_DETECTOR_TYPE_9_16` |
     | AGAST   | `cv2.AgastFeatureDetector_create`                | `threshold=10, nonmaxSuppression=True` |
     | BRISK   | `cv2.BRISK_create`                               | `thresh=30, octaves=4` |
     | SIFT    | `cv2.SIFT_create`                                | `nfeatures=0, contrastThreshold=0.04, edgeThreshold=10` |
     | USURF   | `cv2.xfeatures2d.SURF_create`                    | `hessianThreshold=100, upright=True` |
     | STAR    | `cv2.xfeatures2d.StarDetector_create`            | `maxSize=45, responseThreshold=30` |
     | KAZE    | `cv2.KAZE_create`                                | `threshold=0.001` |
     | AKAZE   | `cv2.AKAZE_create`                               | `threshold=0.001` |
     | MSER    | `cv2.MSER_create` + `detectRegions()`            | `min_area=20, max_area=8100, max_variation=0.5` |

   - **Detector property categorisations** (used to overwrite values OpenCV reports for detectors that don't truly produce them):
     - `_SCALELESS_DETECTORS = {"Harris", "GFTT", "FAST", "AGAST", "MSER"}` — `sigma` forced to `None` even if `cv_kp.size > 0`.
     - `_ORIENTATIONLESS_DETECTORS = {"Harris", "GFTT", "FAST", "AGAST", "MSER", "USURF"}` — `theta` forced to `None`. USURF is in here because it is configured `upright=True`.
   - **`class_id` round-trip.** AKAZE (and KAZE) write the scale-space evolution layer into `cv_kp.class_id`. The detection wrapper preserves it when `≥ 0`, otherwise stores `None`. This is the only piece of information AKAZE→native-MLDB needs from the detector (see Stage 3 / §Custom Descriptor Implementations / MLDB).
   - **Harris / GFTT path.** These return `(x, y)` only — every keypoint is constructed with `response=1.0` (no per-corner score from `goodFeaturesToTrack`).
   - **MSER path.** Uses `detectRegions()` rather than `detect()`. Each detected blob is converted to a single keypoint at the region centroid, with `response = pixel count` (larger region → stronger). The mask is consulted at the centroid pixel; out-of-bounds centroids are clipped before the lookup. Tuned defaults differ from OpenCV's: `min_area=20, max_area=8100, max_variation=0.5` (originals `60 / 14400 / 0.25` produced zero keypoints on the target dataset; tuned values yield ~5000 keypoints on 2464×2056 images with region sizes in the 245–7980 px range).
   - **Unknown detector name** → `ValueError(f"Unknown detector: {name}")`.

3. **Stage 3: Description** (`description.py`)
   - Inputs: Image (uint8 RGB; internally converted to grayscale) and the keypoint list from Stage 2.
   - Outputs: `(filtered_keypoints, descriptor_matrix)`. OpenCV's `compute()` may drop keypoints too close to the image edge; the returned keypoint list is the surviving subset and indexes the descriptor rows 1:1.
   - **Upright by default.** Every keypoint is passed to OpenCV with `angle=0.0` regardless of what the detector reported. Scale `sigma` is taken from the keypoint, falling back to `RunConfig.descriptor_default_sigma` (default `6.0`); the cv2 keypoint receives `size = sigma * 2`. Per-descriptor upright overrides applied on top of `RunConfig.descriptor_params`:

     | Name      | Backing call                                       | Forced parameters (override `descriptor_params`) | Dim × dtype |
     |-----------|----------------------------------------------------|--------------------------------------------------|-------------|
     | SIFT      | `cv2.SIFT_create`                                  | (none)                                           | 128 float32 |
     | RootSIFT  | `cv2.SIFT_create` + L1-norm + element-wise sqrt    | (none)                                           | 128 float32, L2-unit |
     | USURF     | `cv2.xfeatures2d.SURF_create`                      | `upright=True`                                   | 64 float32 |
     | DAISY     | `cv2.xfeatures2d.DAISY_create`                     | `use_orientation=False`                          | 200 float32 |
     | BRIEF     | `cv2.xfeatures2d.BriefDescriptorExtractor_create`  | (none)                                           | 32 bytes uint8 |
     | BRISK     | `cv2.BRISK_create`                                 | (none)                                           | 64 bytes uint8 |
     | SUFREAK   | `cv2.xfeatures2d.FREAK_create`                     | `orientationNormalized=False, scaleNormalized=False` | 64 bytes uint8 |
     | MLDB      | see routing below                                  | (none)                                           | 61 bytes uint8 |
     | LIOP      | `overlap_detection.liop.liop_describe` (NumPy)     | n/a                                              | 144 float32, per-bin L2-unit |

   - **Binary vs. float discrimination.** `is_binary_descriptor(name)` returns `True` for `{BRIEF, BRISK, SUFREAK, MLDB}`. Consumed by `matching.py` to pick `NORM_HAMMING` vs. `NORM_L2`.
   - **MLDB routing decision** (`description.py`):
     - **Native AKAZE path** (`cv2.AKAZE_create(descriptor_type=AKAZE_DESCRIPTOR_MLDB_UPRIGHT).compute`) is taken **iff** all of:
       - `detector_name == "AKAZE"`, and
       - the keypoint list is non-empty, and
       - the first keypoint has `class_id is not None` and `class_id >= 0`.
     - Otherwise the custom NumPy MLDB (`overlap_detection.mldb.mldb_describe`) runs. KAZE keypoints take this route too — they also carry `class_id`, but it indexes a *linear* diffusion pyramid while AKAZE's MLDB expects a *nonlinear* one, so handing KAZE class_ids to AKAZE's `compute()` would be semantically wrong.
   - **LIOP routing.** The OpenCV path is bypassed entirely; `liop_describe` is called directly.
   - **Empty-input behaviour.** If `keypoints` is empty, `(list(), np.array([]))` is returned without invoking OpenCV.
   - **Unknown descriptor name** → `ValueError(f"Unknown descriptor: {name}")`.

4. **Stage 4: Matching & Filtering** (`matching.py`)
   - Inputs: Descriptor matrices `desc_A` and `desc_B`, a `is_binary` flag (from `is_binary_descriptor` of Stage 3), a `filter_mode`, and the NNDR threshold.
   - Outputs: `np.ndarray` of shape `(M, 3)` with columns `[idx_A, idx_B, distance]`, **sorted ascending by distance** (this ordering is consumed by PROSAC as the quality ranking).
   - **Distance metric**: `cv2.NORM_HAMMING` when `is_binary`, else `cv2.NORM_L2`.
   - **Filter modes** (`RunConfig.matcher_filter`):
     | Mode         | Behaviour |
     |--------------|-----------|
     | `"mnn"`      | `cv2.BFMatcher(norm_type, crossCheck=True).match(...)`. Returns one best mutual match per query keypoint. |
     | `"mnn_nndr"` | `cv2.BFMatcher(norm_type, crossCheck=False)` with `knnMatch(k=2)` in both directions. A match `(a, b)` is kept iff (i) `d(a, b) < nndr_threshold · d(a, second_best_for_a)`, (ii) the same inequality holds in the B→A direction with the symmetric second-best, and (iii) the two top picks form a mutual pair. |
     - Unknown mode → `ValueError(f"Unknown filter mode: {filter_mode}")`.
   - **NNDR threshold**: `RunConfig.nndr_threshold` (default `0.90`).
   - **Empty-input handling.** If either descriptor matrix is `None`/empty, an `(0, 3)` empty array is returned.
   - **LIOP descriptor override** (applied in `orchestrator._execute_pipeline`, not in `matching.py`): whenever `RunConfig.descriptor == "LIOP"`, the matcher filter is forced to `"mnn"` regardless of `RunConfig.matcher_filter` (LIOP's L2 distances cluster too tightly to support NNDR). RunConfig is not mutated; the override is a single conditional at the call site.

5. **Stage 5: Geometric Verification** (`verification.py`, `orchestrator.py`)
   - Inputs: Match array from Stage 4 and the keypoint lists for both images.
   - Outputs: `(affine_matrix, inlier_mask)` — `affine_matrix` is `(2, 3) float64` mapping A→B (or `None` on failure), `inlier_mask` is a boolean array of length `len(matches)`.
   - **Estimator selection.** `RunConfig.estimator ∈ {"PROSAC", "USAC_MAGSAC"}` → `_METHOD_MAP = {"PROSAC": cv2.USAC_PROSAC, "USAC_MAGSAC": cv2.USAC_MAGSAC}`; an unknown value silently falls back to `cv2.USAC_MAGSAC`. Estimation is via `cv2.estimateAffine2D(src=A, dst=B, method=…, ransacReprojThreshold=RunConfig.ransac_threshold_px, maxIters=RunConfig.ransac_max_iters, confidence=RunConfig.ransac_confidence)` (defaults `5.0 px / 10000 / 0.99`).
   - **Early-out in `verify_affine` itself:**
     1. `len(matches) < 3` → returns `(None, zeros(0,))` before calling cv2.
     2. `cv2.estimateAffine2D` returned `affine_matrix is None` → returns `(None, zeros(len(matches),))`.
     3. `inlier_count < 3` after RANSAC → returns `(None, inlier_mask)`. This is a hard lower bound for any affine; the orchestrator's `min_inliers` gate (below) is the tunable, stricter check.
   - **Orchestrator acceptance gates** (applied in order in `_execute_pipeline`, after `verify_affine` returns a non-None matrix):
     1. **Min-inliers gate** — `n_inliers ≥ RunConfig.min_inliers` (default `8`). Failure sets `error_message = "Too few inliers (N < M)"` and leaves `result.affine_matrix = None`.
     2. **Affine sanity gate** — `_affine_is_sane` decomposes the 2×2 sub-matrix as `scale = sqrt(a00² + a10²)`, `rotation_deg = degrees(atan2(a10, a00))`. The affine is rejected if `|scale − 1| > _MAX_SCALE_DIFF` (10 %) **or** `|rotation_deg| > _MAX_ROTATION_DEG` (3°). Thresholds are module-level constants in `orchestrator.py`, **not** in `RunConfig` (see §Hard-coded constants).
   - **On rejection** by any of the above, `result.affine_matrix` is set to `None`, `result.error_message` records the specific reason, downstream stages (overlap geometry, GT-comparison) are skipped, and `categorize_result` produces `"no_match"`.
   - **On acceptance**, `result.affine_matrix` and `result.inlier_mask` are stored on the `PairResult`.

6. **Stage 6: Overlap Geometry** (`geometry.py`)
   - Inputs: Accepted affine matrix (2×3, A→B) and the two image shapes.
   - Outputs: `(overlap_in_A, overlap_in_B)` — the same overlap polygon expressed in each frame as an `(N, 2) float32` array.
   - **Construction.** A's four image-rectangle corners `[(0,0), (W_A,0), (W_A,H_A), (0,H_A)]` are warped into B's frame via the affine; that parallelogram is intersected (shapely `Polygon.intersection`) with B's image rectangle; the resulting polygon is pulled back to A's frame via the inverse affine. Vertex count is 3–8 depending on how A's warped rectangle clips against B.
   - **Canonical vertex ordering** (applied to both returned polygons):
     1. **Clockwise.** Shapely's `exterior` is counter-clockwise by default; if so, the array is reversed.
     2. **Top-left first.** The polygon is `np.roll`-ed so the vertex with the smallest `x + y` (top-left under the standard image axes) is at index 0.
   - **Degenerate-affine guard.** Before inverting, `invert_affine` is wrapped in `try/except np.linalg.LinAlgError`. A singular 2×2 sub-matrix (zero/near-zero scale, collinear axes) silently returns `(empty, empty)` so the metrics stage can still grade the attempt as `no_match` without crashing.
   - **Empty / non-polygon intersection** (`is_empty` or geometry type outside `{Polygon, MultiPolygon}`) → `(empty, empty)`.
   - **MultiPolygon intersection** → the component with the largest area is kept; the rest are discarded.
   - **Helpers.** `apply_affine(points, M)` does the homogeneous-coords multiply and returns `(N, 2)`; `invert_affine(M)` analytically inverts the 2×2 sub-matrix and propagates the translation. Both are exported and reused by `metrics.corner_errors_hpatches` and the annotation GUI.

7. **Stage 7: Metrics & Categorisation** (`metrics.py`)
   - Inputs: `PairResult` (timings, counts, affine, polygons) and `GroundTruth`.
   - Outputs: per-stage timings, keypoint/match/inlier counts, IoU vs. ground truth, **mean corner error** (HPatches convention — see below), and the **ordinal `result_label`** assigned by `categorize_result`.
   - **Error metric (HPatches convention).** `corner_errors_hpatches` warps image-A's **four unclipped image-rectangle corners** (in order TL, TR, BR, BL) through both the estimated and the ground-truth affines, then takes the Euclidean distance between the two sets — measured in **B-pixels**. Always 4 corners, no polygon clipping; projected corners are allowed to land outside B's image rectangle. This matches what SuperGlue / glue-factory / LoFTR report, so the numbers and the AUC-form mAA derived from them are directly comparable to published tables.
   - **Why decoupled from IoU.** IoU still uses the *clipped* overlap polygon (since IoU is fundamentally an area metric and needs the actually-overlapping region). Corner error doesn't need a polygon at all — it's a pure 4-point reprojection check. Decoupling them removed a pre-existing edge case where vertex-count mismatches between estimated and GT overlap polygons would silently downgrade a successful pair to `no_match`.
   - mAA is then computed in the standard AUC form by the **reporting** stage (see §Metric definitions) — `metrics.py` only emits the raw `mean_corner_error` and the ordinal label; `reporting.py` derives mAA / Precision / acc@T / matrices from the CSV.
   - **`result_label` values** (assigned by `categorize_result(has_transform, mean_corner_error, accuracy_tiers_px)`, driven by `RunConfig.accuracy_tiers_px`, default `(3, 5, 10)` px). Decision rules in order:
     1. `has_transform is False` → `"no_match"`.
     2. `mean_corner_error is None` **or** not finite → `"no_match"` (a transform was produced but cannot be graded — e.g. no ground truth supplied).
     3. For each tier `T` in **ascending** order: if `mean_corner_error ≤ T`, return `f"acc_at_{T:g}"`. The `:g` format strips trailing zeros (`3.0 → "3"`, `2.5 → "2.5"`). Comparison is **inclusive** at the boundary.
     4. Otherwise → `"false_match"` (transform produced but error exceeded every tier).

8. **Orchestration** (`orchestrator.py`)
   - **Per-attempt pipeline.** `_execute_pipeline(...)` runs the seven stages above with a concrete `mask_mode ∈ {"no_mask", "mask"}` and returns one `(PairResult, metrics)` tuple. It never sees the user-facing `"both"` value.
   - **Public entrypoint `run_single_pair`** dispatches based on `config.mask_mode`:
     - `"no_mask"` or `"mask"` → list of 1 `(result, metrics)` tuple.
     - `"both"` → list of 2 tuples, `no_mask` first then `mask`. Both attempts are run unconditionally — no early exit on first success.
     - Any other value → `ValueError(f"Unknown mask_mode: {mask_mode!r}")`.
   - **Why `"both"` instead of fallback?** Any "try X first, fall back to Y" policy can be derived post-hoc from the paired columns produced by `"both"`. Running both unconditionally costs one extra pipeline pass per pair but yields a richer dataset: per-attempt accuracy distributions, false-match shares per mask mode, and a free `best_of_both` analysis in reporting (see Outputs section). The main pipeline therefore has no fallback machinery; the only consumer that needs short-circuit fallback semantics is the auto-aligner, which implements them locally (Stage 9).
   - **Image-pair discovery** (`list_image_pairs(dataset_dir)`):
     - Looks for `*.jpg` and `*.png` under `dataset_dir`.
     - Filename convention: `{x}_{y}_{timestamp}.ext`. The first two underscore-separated tokens of the stem are parsed as integer coordinates by `_parse_coords`; files that don't match this scheme are silently skipped.
     - Two images are considered **adjacent** when they share one coordinate and are consecutive in the other (sorted numerically). This handles 1-D strips and 2-D grids.
     - Pairs are returned in deterministic order: ascending `(_parse_coords(a), _parse_coords(b))`.
   - **Experiment matrix.** `run_experiment_matrix` schedules a Cartesian product of `(pair, detector, descriptor, estimator, mask_mode_spec)` across a `multiprocessing` pool. Each worker handles one image pair, loads the images once, and runs every pending attempt for that pair, writing per-attempt JSON files as it goes. Matrix construction (`build_full_matrix`) skips combinations where `descriptor ∉ VALID_PAIRINGS[detector]`.
   - **Per-attempt cache resume.** Filename pattern is `{pair_id}_{detector}_{descriptor}_{estimator}_{concrete_mask_mode}.json` where `concrete_mask_mode ∈ {"no_mask", "mask"}` (never `"both"`). On startup, every attempt is probed against `_load_cached_metrics`: a present JSON with a readable `"metrics"` block is reused as-is and never re-executed. Cache misses load the image lazily (only the first miss per pair pays the I/O cost). Resume is **per-attempt**, not per-pair — an existing `_no_mask.json` is reused even when its sibling `_mask.json` is missing.
   - **Worker pool.**
     - Start method: `"spawn"` (`multiprocessing.get_context("spawn")`) — Windows-friendly, avoids inheriting parent OpenCV/GUI state.
     - Worker count: `default_experiment_workers()` → `max(1, min(cpu_count() − 1, _DEFAULT_EXPERIMENT_WORKER_CAP))`, where `_DEFAULT_EXPERIMENT_WORKER_CAP = 8`. CLI `--workers 1` forces a serial loop in the main process (no pool spawned) for debugging.
     - Per-worker initialiser (`_worker_init`): calls `cv2.setNumThreads(0)` and `cv2.ocl.setUseOpenCL(False)`. Disables OpenCV's internal TBB/OpenMP and OpenCL paths so the only parallelism is the process pool itself; without this, workers can deadlock on Windows before returning their first result.
   - **CSV row aggregation.** For each `(pair, det, desc, est, mask_mode_spec)`, one CSV row is emitted with paired `no_mask_*` and `with_mask_*` columns (NaN on whichever side wasn't run). Stat keys merged per attempt (`_ATTEMPT_STAT_KEYS`): `result_label, iou, mean_corner_error, num_keypoints_A, num_keypoints_B, num_tentative_matches, num_inliers, inlier_ratio, detection_ms, description_ms, matching_ms, verification_ms, geometry_ms, total_ms`. Plus the short aliases `{prefix}_result` and `{prefix}_err`. The column prefix is **`no_mask`** for the no-mask attempt and **`with_mask`** for the mask attempt (asymmetric on purpose — `mask_*` would collide with `mask_mode` column in pandas filters).

9. **Ground Truth Annotation** (`annotation_gui.py`, `auto_aligner.py`)
   - **Annotation GUI.** Standalone OpenCV `cv2.namedWindow` UI; not Tkinter. Hotkeys: left-drag = translate, right-drag = rotate, Ctrl+left-drag = scale, middle-drag or Shift+left-drag = pan, wheel = zoom (centred on cursor), `m` = cycle render mode (normal blend ↔ anaglyph), `r` = reset alignment, `a`/`d` (or arrow keys) = previous/next pair, `s` = save current alignment, `Esc` = quit.
   - **Saved GroundTruth JSON** (per pair, written as `{annotations_dir}/{pair_id}_groundtruth.json`):
     - `image_A_path`, `image_B_path` — filesystem paths as strings.
     - `affine_matrix_A_to_B` — 2×3 list, A→B (the GUI works with B→A internally and inverts before saving).
     - `image_a_shape`, `image_b_shape` — `[H, W, 3]` lists.
     - `annotator` — string passed via `--annotator`.
     - `annotation_date` — ISO-8601 timestamp from `datetime.datetime.now().isoformat()`.
   - **User input freeze.** Until the auto-aligner produces a result for the current pair (or the user explicitly accepts the default), mouse input is ignored — prevents the user from starting from the GUI's `_reset_alignment` guess only to have it overwritten when the auto-align finishes.
   - **Background auto-alignment.**
     - `AutoAligner` runs in a `multiprocessing.get_context("spawn").Pool` while the annotator reviews. Worker count from `default_auto_align_workers()` → `max(1, min(cpu_count() − 1, _DEFAULT_AUTO_ALIGN_WORKER_CAP))`, where `_DEFAULT_AUTO_ALIGN_WORKER_CAP = 6`.
     - **Two-config attempt cascade** per pair (in `_align_worker`):
       1. Primary: `RunConfig(detector="GFTT", descriptor="BRISK", estimator="PROSAC")`.
       2. Secondary: `RunConfig(detector="FAST", descriptor="SIFT", estimator="PROSAC")`.
     - Each config is tried through `_try_with_fallback`: `mask_mode = "no_mask"` first, `"mask"` only if no transform was produced. First successful affine wins; both configs returning None marks the pair as failed.
   - **Self-contained fallback.** The auto-aligner is the only consumer that wants short-circuit fallback semantics (instead of the experiment runner's run-both-and-pair). It owns the logic locally so the main pipeline stays simple.

---

## Data Contracts between Pipeline Stages

The pipeline defines strict contracts for structures passed between stages to maintain modular decoupling:

### 1. Image
- **Format**: `numpy.ndarray`
- **Shape**: `(H, W, 3)`
- **Dtype**: `uint8`
- **Details**: RGB color channel order. Internally converted to grayscale (`COLOR_RGB2GRAY`) at the detection and description stages.

### 2. Mask
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
    octave: int | None      # Pyramidal scale octave layer (None if absent)
    class_id: int | None    # AKAZE/KAZE scale-space evolution layer index;
                            # consumed only by the native MLDB path.
                            # None for every other detector.
```

### 4. Descriptor Matrix
- **Format**: `numpy.ndarray`
- **Shape**: `(N, D)` where $N$ is the number of keypoints and $D$ is the descriptor dimensionality.
- **Dtype**: `float32` (for float descriptors like SIFT/KAZE/USURF/DAISY/LIOP) or `uint8` (for packed binary descriptors BRIEF/BRISK/SUFREAK/MLDB).

### 5. Match List
- **Format**: `numpy.ndarray`
- **Shape**: `(M, 3)` where $M$ is the number of matches.
- **Columns**: `[idx_in_A, idx_in_B, distance]` where columns 0 and 1 are integer indices and column 2 is float distance.
- **Ordering**: rows are sorted by **ascending distance**; PROSAC consumes this ordering as its quality ranking.

### 6. Transformation Matrix
- **Format**: `numpy.ndarray`
- **Shape**: `(2, 3)` (affine transformation; full perspective homographies are *not* used)
- **Dtype**: `float64`
- **Direction**: maps **A → B** (a point `p_A` in A's frame lands at `M @ [p_A.x, p_A.y, 1]` in B's frame).

### 7. Overlap Polygon
- **Format**: `numpy.ndarray`
- **Shape**: `(K, 2)` where `K ∈ [3, 8]`.
- **Dtype**: `float32`
- **Ordering**: clockwise, with the vertex of minimum `x + y` (top-left under image-pixel axes) at index 0.

### 8. Pair Result Record
- **Format**: A `PairResult` dataclass containing complete run metadata, configuration snapshot, timing, matching statistics, estimated affine matrix, and geometric error metrics. See [`overlap_detection/types.py`](overlap_detection/types.py) for the canonical field list and docstrings.

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

`DETECTOR_NAMES = ["Harris", "GFTT", "FAST", "AGAST", "BRISK", "SIFT", "USURF", "STAR", "KAZE", "AKAZE", "MSER"]` (11). `DESCRIPTOR_NAMES = ["SIFT", "RootSIFT", "USURF", "DAISY", "BRIEF", "BRISK", "SUFREAK", "MLDB", "LIOP"]` (9). `VALID_PAIRINGS` maps each detector to **every** descriptor — no hard exclusions exist; LIOP and MLDB cover the cases where OpenCV's native paths would otherwise fail (see §Custom Descriptor Implementations).

**Vestigial RunConfig fields** (defined for API stability / forward-compat, but currently not read by any pipeline stage):
- `output_dir` — the orchestrator's `run_experiment_matrix` takes `output_dir` as a direct function parameter; the `RunConfig.output_dir` field is unused.
- `save_intermediate` — no stage consumes this; intermediate artefacts (masked images, match visualisations) are not currently persisted.
- `random_seed` — no stage seeds an RNG from this; the OpenCV USAC estimators have no `seed` parameter, and the rest of the pipeline is deterministic.

---

## Hard-coded constants outside RunConfig

The constants listed below influence pipeline behaviour but are deliberately **not** exposed through `RunConfig`. Changing them requires editing source.

### `orchestrator.py`

| Constant                            | Value | Effect |
|-------------------------------------|-------|--------|
| `_MAX_SCALE_DIFF`                   | `0.10` | Affine-sanity gate rejects an estimate when `|scale − 1| > 0.10`. |
| `_MAX_ROTATION_DEG`                 | `3.0`  | Affine-sanity gate rejects an estimate when `|rotation_deg| > 3.0`. |
| `_DEFAULT_EXPERIMENT_WORKER_CAP`    | `8`    | Upper bound for `default_experiment_workers()`; actual worker count is `min(cpu_count − 1, 8)` floored at 1. |

`_ATTEMPT_COLUMN_PREFIX = {"no_mask": "no_mask", "mask": "with_mask"}` controls the CSV column prefixes per attempt (the asymmetric "mask" → "with_mask" rename keeps `mask_*` from colliding with the `mask_mode` identifier column).

`_ATTEMPT_STAT_KEYS` enumerates which metrics-dict keys get the per-attempt prefix in the aggregate CSV: `result_label, iou, mean_corner_error, num_keypoints_A, num_keypoints_B, num_tentative_matches, num_inliers, inlier_ratio, detection_ms, description_ms, matching_ms, verification_ms, geometry_ms, total_ms`.

### `auto_aligner.py`

| Constant                          | Value | Effect |
|-----------------------------------|-------|--------|
| `_DEFAULT_AUTO_ALIGN_WORKER_CAP`  | `6`   | Upper bound for `default_auto_align_workers()` (lower than the experiment cap to leave headroom for the GUI process). |

### `detection.py`

| Constant                      | Value | Effect |
|-------------------------------|-------|--------|
| `_DETECTOR_DEFAULTS`          | (per-detector dict — see Stage 2 table) | Baseline `**params` for each OpenCV detector factory; overlaid by `RunConfig.detector_params`. |
| `_SCALELESS_DETECTORS`        | `{Harris, GFTT, FAST, AGAST, MSER}` | These detectors' keypoints always carry `sigma=None`. |
| `_ORIENTATIONLESS_DETECTORS`  | `{Harris, GFTT, FAST, AGAST, MSER, USURF}` | These detectors' keypoints always carry `theta=None`. |

### `verification.py`

| Constant       | Value | Effect |
|----------------|-------|--------|
| `_METHOD_MAP`  | `{"PROSAC": cv2.USAC_PROSAC, "USAC_MAGSAC": cv2.USAC_MAGSAC}` | Maps the user-facing estimator string to the cv2 method enum. Unknown values silently fall back to `cv2.USAC_MAGSAC`. |

### `mldb.py` (custom MLDB descriptor — non-AKAZE path)

| Constant      | Value | Effect |
|---------------|-------|--------|
| `_PATCH_SIZE` | `60` px | Side length of the warped patch; divisible by 2, 3, and 4 so every grid cell is integer-sized. |
| `_SIGMA_SCALE`| `10.0` | Physical patch half-side = `sigma * 10` (matches AKAZE's `pattern_size = 10`). |
| `_SMOOTH_SIGMA` | `1.5` | Gaussian sigma applied to the patch before computing Lx/Ly (approximates AKAZE's nonlinear diffusion image Lt). |
| `_GRIDS`      | `[(2,2), (3,3), (4,4)]` | Three subdivision passes; their cell-pair counts × 3 channels sum to `DESC_BITS = 486`. |
| `DESC_BITS` / `DESC_BYTES` | `486` / `61` | Derived dimensions (`(486+7)//8`). |

### `liop.py` (LIOP descriptor)

| Constant     | Value | Effect |
|--------------|-------|--------|
| `_PATCH_SIZE` / `_PATCH_RADIUS` | `41` / `20` px | Fixed warped-patch dimensions; the in-circle pixel set is precomputed at import time. |
| `_N_NEIGHBORS` | `4`  | K nearest neighbours per sample point; `K! = 24` distinct ordinal codes. |
| `_N_BINS`    | `6`   | Equal-population ordinal intensity bins. |
| `_N_CODES`   | `24` (= `4!`) | Codes per bin. |
| `DESC_DIM`   | `144` (`6 × 24`) | Final descriptor length. |
| Patch physical radius | `3 * sigma` | Set inside `_extract_patch`; falls back to `default_sigma` from RunConfig when the keypoint has none. |

### `reporting.py`

| Constant / function           | Value / behaviour | Effect |
|--------------------------------|--------------------|--------|
| `_MAA_CMAP_NAME`              | `"bwr_r"` (matplotlib reversed blue-white-red) | mAA heatmap colour: low → red, mid → white, high → blue. |
| `_precision_cmap()`           | `LinearSegmentedColormap.from_list("rwg", ["red", "white", "green"])` | Precision heatmap colour: low → red, mid → white, high → green. |
| `_section_vmax(tables, "maa")` | `min(1.0, max(0.3, ceil(observed_max * 10) / 10))` | Shared vmax for all mAA heatmaps in one section. NaN-aware (uses `np.isfinite` masking, not `nanmax`). |
| `_section_vmax(tables, "precision")` | `1.0` (always) | Precision is bounded in `[0, 1]`. |
| `_ATTEMPTS`                   | `("no_mask", "with_mask", "best_of_both")` | The three attempt slices each report section breaks out separately. |

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
| `metrics`    | `iou`                       | float \| null            | Clipped overlap-polygon IoU vs. GT (uses the actual visible overlap region) |
|              | `mean_corner_error`         | float \| null, B-px      | HPatches-convention mean reprojection error: image A's four image-rectangle corners warped by the estimated vs. GT affine, averaged. Not clipped to B's bounds. |
|              | `corner_error_{0..3}`       | float \| null, B-px      | Per-corner errors in fixed order **TL, TR, BR, BL** (matches the construction order in `corner_errors_hpatches`). Useful for debugging lopsided drift. |
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

Every section is split by **estimator** (`PROSAC`, `USAC_MAGSAC`) and by **mask attempt** (`no_mask`, `with_mask`, `best_of_both`), because the two estimators behave categorically differently enough that pooling them obscures the picture, and each attempt answers a different operational question.

`best_of_both` is a derived per-row label: for each pair, the report builder picks whichever single attempt landed in the better tier (`acc_at_3 > acc_at_5 > acc_at_10 > false_match > no_match`). It's only populated for rows where `mask_mode_spec = "both"`. From that point on it's treated identically to the other two attempts — it appears in every overall, scoreboard, matrix, and benefit table.

| Section                 | Granularity | What it shows | Why |
|-------------------------|-------------|---------------|-----|
| **Overall**             | One row per `(estimator × attempt)` = up to 6 rows | Headline mAA, **Precision**, per-tier `acc@T`, `false_match` / `no_match` shares | Single-glance summary of the whole experiment with the two key axes already separated |
| **Per-configuration scoreboard** | One table per estimator; rows are `(detector + descriptor)` | All three attempts side-by-side with mAA, Precision, per-tier rates, false/no_match shares | Comparing pipelines within an estimator, with the fallback-vs-single-attempt question answered in-place |
| **mAA matrices (detector × descriptor)** | One **heatmap PNG + numeric table** per `(estimator × attempt)` = up to 6 of each | mAA across the full detector/descriptor grid | Compact pipeline-vs-pipeline view that survives an 11×9 sweep cleanly. Rows and columns sorted by descending mean mAA so the strongest configurations sit top-left. Colour scale: **red → white → blue** (matplotlib `bwr_r`) |
| **Precision matrices**  | Same layout as mAA matrices                       | Per-emission correctness rate                                                                          | Catches the failure mode mAA can hide: a pipeline that emits confident-but-wrong transforms (high `false_match`) scores low here even when mAA looks decent. Colour scale: **red → white → green** (custom diverging palette) |
| **Fallback benefit** | One table per estimator; rows are `(detector + descriptor)` | `mAA_best − mAA_no_mask` and `mAA_best − mAA_with_mask` lift columns | Quantifies how much running `mask_mode = "both"` would actually buy you over a single mask attempt — answered post-hoc, no extra runs needed |

Heatmap PNG filenames: `heatmap_{maa|precision}_{estimator}_{attempt}.png`, written alongside `report.md`. The numeric table beneath each PNG uses the same row/column ordering, so visual scan and precise lookup stay aligned.

### Metric definitions added in this section

- **Precision** = `(emitted ∧ not false_match) / emitted`, where `emitted = (label != "no_match")`. Equivalently `1 − false_match / (1 − no_match)`. Answers: *when the pipeline does emit a transform, how often is it at least within the loosest configured tier (default 10 px)?* Undefined (NaN) for slices where every attempt was `no_match`.
- This complements mAA rather than replacing it: mAA conflates `false_match` and `no_match` (both score 0), Precision distinguishes "wrong" from "abstained". A high-mAA / low-Precision pipeline is dangerous — it lands accurately most of the time but emits confidently-wrong answers in the failure cases.

### Metric definitions (canonical)

- **Mean corner error** — `mean(||p_pred_i − p_gt_i||₂  for i in 0..3)` where `p_pred_i = est_affine @ A_corner_i` and `p_gt_i = gt_affine @ A_corner_i`, with `A_corner_i ∈ {(0,0), (W_A,0), (W_A,H_A), (0,H_A)}`. In other words: take image A's four image-rectangle corners, warp them with both the estimated and the GT affine, and average the four Euclidean distances. Measured in B-pixels. Matches the HPatches / SuperGlue / LoFTR convention exactly — the corners are *not* clipped to B's image bounds, so reported values are directly comparable to published tables. CSV columns `corner_error_{0..3}` correspond to TL, TR, BR, BL in that order.
- **IoU** — `intersection_area / union_area` between the predicted and ground-truth overlap polygons in image-A coordinates.
- **`acc@T`** (per attempt, per config) — fraction of rows whose `result_label` is some `"acc_at_<t>"` with `t ≤ T`. Cumulative: hitting a tighter tier implies clearing every looser one.
- **mAA** (standard AUC form, matching SuperGlue / glue-factory) — computed from the raw `mean_corner_error` values, not the ordinal labels. For each configured tier threshold T, `AUC@T = (1/T) ∫₀ᵀ recall(ε) dε` where `recall(ε)` is the fraction of pairs with corner error ≤ ε; the integral is evaluated via the trapezoidal rule on the sorted error values. `mAA = mean(AUC@T₁, AUC@T₂, …)`. Failures (NaN corner error — no transform produced) are treated as infinite error: they enter the recall denominator but never reach any threshold, so each failure reduces the score proportionally. This is the same convention as glue-factory (`error = inf` for estimation failures). Unlike binary acc@T averaging, the AUC form rewards accuracy *within* the threshold window — a pair at 0.5 px contributes more than one at 2.9 px even though both clear a 3 px threshold.
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

