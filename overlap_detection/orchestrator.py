"""orchestrator.py — Pipeline execution and experimental matrix runner.

Runs one image pair through the full pipeline (preprocess → detect →
describe → match → verify → geometry → metrics) and iterates over the
experimental matrix (detector × descriptor × mask_mode × estimator).

The ``mask_mode = "both"`` value schedules two pipeline attempts per pair —
one without mask, one with mask — and merges their results into a single
CSV row whose ``no_mask_*`` / ``with_mask_*`` columns are paired.  Each
attempt still produces its own JSON file.
"""

import json
import os
import time
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import product
from multiprocessing import get_context
from typing import Optional
from dataclasses import replace

from overlap_detection.config import RunConfig, VALID_PAIRINGS
from overlap_detection.types import PairResult, GroundTruth
from overlap_detection.metrics import compute_pair_metrics
from overlap_detection.preprocessing import apply_mask_mode
from overlap_detection.detection import detect
from overlap_detection.description import describe, is_binary_descriptor
from overlap_detection.matching import match
from overlap_detection.verification import verify_affine
from overlap_detection.geometry import compute_overlap_polygon
from overlap_detection.reporting import write_pair_json, write_aggregate_csv


# ---------------------------------------------------------------------------
# Multi-core defaults
# ---------------------------------------------------------------------------
_DEFAULT_EXPERIMENT_WORKER_CAP = 8


def default_experiment_workers() -> int:
    """Pick a worker count for ``run_experiment_matrix``: ``cpu_count - 1``
    capped at ``_DEFAULT_EXPERIMENT_WORKER_CAP`` and floored at 1."""
    return max(1, min((os.cpu_count() or 1) - 1, _DEFAULT_EXPERIMENT_WORKER_CAP))


def _parse_coords(path: Path) -> tuple[int, int]:
    parts = path.stem.split('_')
    return int(parts[0]), int(parts[1])


def list_image_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    """Return all adjacent image pairs from a dataset directory.

    Images are named ``{x}_{y}_{timestamp}.ext``.  Two images are adjacent
    when they share one coordinate and are consecutive in the other,
    sorted numerically.  Handles both 1-D strips and 2-D grids.
    Files that do not match the naming scheme are silently skipped.
    """
    images = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png"))

    coord_map: dict[tuple[int, int], Path] = {}
    for img in images:
        try:
            coord_map[_parse_coords(img)] = img
        except (ValueError, IndexError):
            continue

    pairs: list[tuple[Path, Path]] = []

    # Same x, consecutive y
    for x in sorted(set(x for x, _ in coord_map)):
        col_ys = sorted(y for (cx, y) in coord_map if cx == x)
        for i in range(len(col_ys) - 1):
            pairs.append((coord_map[(x, col_ys[i])], coord_map[(x, col_ys[i + 1])]))

    # Same y, consecutive x
    for y in sorted(set(y for _, y in coord_map)):
        row_xs = sorted(x for (x, cy) in coord_map if cy == y)
        for i in range(len(row_xs) - 1):
            pairs.append((coord_map[(row_xs[i], y)], coord_map[(row_xs[i + 1], y)]))

    # Decorate-sort-undecorate: parse each path once instead of per comparison.
    decorated = [((_parse_coords(a), _parse_coords(b)), (a, b)) for a, b in pairs]
    decorated.sort(key=lambda item: item[0])
    return [pair for _, pair in decorated]


# ---------------------------------------------------------------------------
# Single-attempt pipeline
# ---------------------------------------------------------------------------


def _execute_pipeline(
    image_A: np.ndarray,
    image_B: np.ndarray,
    config: RunConfig,
    mask_mode_to_use: str,
    ground_truth: Optional[GroundTruth],
    image_a_path: Path,
    image_b_path: Path,
) -> tuple[PairResult, dict]:
    """Run the full pipeline once with a concrete mask mode.

    Always returns ``(result, metrics)``.  ``result.affine_matrix`` is set
    only when the affine passes both the inlier-count gate and the sanity
    check; ``metrics["result_label"]`` is assigned by ``compute_pair_metrics``.
    """
    t_start_total = time.perf_counter()

    result = PairResult(
        image_a_path=image_a_path,
        image_b_path=image_b_path,
        detector=config.detector,
        descriptor=config.descriptor,
        estimator=config.estimator,
        mask_mode=mask_mode_to_use,
    )

    try:
        # 1. Preprocessing (masking)
        mask_A = apply_mask_mode(image_A, mask_mode_to_use,
                                 config.overlap_band_fraction,
                                 side="right",
                                 sat_threshold=config.mask_sat_threshold,
                                 brightness_lo=config.mask_brightness_lo,
                                 brightness_hi=config.mask_brightness_hi)
        mask_B = apply_mask_mode(image_B, mask_mode_to_use,
                                 config.overlap_band_fraction,
                                 side="left",
                                 sat_threshold=config.mask_sat_threshold,
                                 brightness_lo=config.mask_brightness_lo,
                                 brightness_hi=config.mask_brightness_hi)

        # 2. Detection
        t0 = time.perf_counter()
        kps_A = detect(image_A, mask_A, config.detector,
                       config.detector_params, config.max_keypoints)
        kps_B = detect(image_B, mask_B, config.detector,
                       config.detector_params, config.max_keypoints)
        result.time_detection_s = time.perf_counter() - t0
        result.n_kp_a = len(kps_A)
        result.n_kp_b = len(kps_B)

        if len(kps_A) < 3 or len(kps_B) < 3:
            result.error_message = "Not enough keypoints detected."
        else:
            # 3. Description
            t0 = time.perf_counter()
            filtered_kps_A, desc_A = describe(
                image_A, kps_A, config.descriptor, config.descriptor_params,
                config.descriptor_default_sigma, config.detector)
            filtered_kps_B, desc_B = describe(
                image_B, kps_B, config.descriptor, config.descriptor_params,
                config.descriptor_default_sigma, config.detector)
            result.time_description_s = time.perf_counter() - t0

            if len(filtered_kps_A) < 3 or len(filtered_kps_B) < 3:
                result.error_message = "Not enough keypoints after description."
            else:
                # 4. Matching
                # LIOP descriptors cluster tightly in distance space, making
                # the NNDR ratio ≈ 1.0 and killing all matches.  Use plain
                # MNN so PROSAC/RANSAC can filter the noisier set itself.
                t0 = time.perf_counter()
                is_bin = is_binary_descriptor(config.descriptor)
                matcher_filter = ("mnn" if config.descriptor == "LIOP"
                                  else config.matcher_filter)
                matches = match(desc_A, desc_B, is_bin,
                                matcher_filter, config.nndr_threshold)
                result.time_matching_s = time.perf_counter() - t0
                result.n_raw_matches = len(matches)

                if len(matches) < 3:
                    result.error_message = "Not enough tentative matches."
                else:
                    # 5. Verification
                    t0 = time.perf_counter()
                    affine_mat, inliers = verify_affine(
                        matches, filtered_kps_A, filtered_kps_B,
                        config.estimator, config.ransac_threshold_px,
                        config.ransac_max_iters, config.ransac_confidence)
                    result.time_verification_s = time.perf_counter() - t0

                    if affine_mat is None:
                        result.error_message = "Affine estimation failed."
                    else:
                        result.n_inliers = int(np.sum(inliers))
                        if result.n_inliers < config.min_inliers:
                            result.error_message = (
                                f"Too few inliers ({result.n_inliers} "
                                f"< {config.min_inliers})")
                        else:
                            result.affine_matrix = affine_mat
                            result.inlier_mask = inliers

                            # 6. Geometry
                            t0 = time.perf_counter()
                            poly_A, poly_B = compute_overlap_polygon(
                                affine_mat, image_A.shape, image_B.shape)
                            result.time_geometry_s = time.perf_counter() - t0
                            result.overlap_polygon_a = poly_A
                            result.overlap_polygon_b = poly_B

    except Exception as e:
        result.error_message = f"Exception: {str(e)}"

    result.time_total_s = time.perf_counter() - t_start_total
    metrics = compute_pair_metrics(
        result, ground_truth,
        accuracy_tiers_px=config.accuracy_tiers_px,
        pixel_correspondence_tolerance_px=config.pixel_correspondence_tolerance_px,
    )
    return result, metrics


def run_single_pair(
    image_A: np.ndarray,
    image_B: np.ndarray,
    config: RunConfig,
    ground_truth: Optional[GroundTruth] = None,
    image_a_path: Path = Path("A.jpg"),
    image_b_path: Path = Path("B.jpg"),
) -> list[tuple[PairResult, dict]]:
    """Execute the pipeline for one pair.

    Returns a list of ``(PairResult, metrics)`` tuples — one per mask attempt:

    * ``mask_mode == "no_mask"`` / ``"mask"``: list of length 1.
    * ``mask_mode == "both"``: list of length 2 (no_mask first, then mask).

    Each attempt is an independent pipeline invocation: no shared state, no
    early-exit between attempts.
    """
    if config.mask_mode == "both":
        modes = ("no_mask", "mask")
    elif config.mask_mode in ("no_mask", "mask"):
        modes = (config.mask_mode,)
    else:
        raise ValueError(f"Unknown mask_mode: {config.mask_mode!r}")

    return [
        _execute_pipeline(
            image_A, image_B, config, m, ground_truth,
            image_a_path, image_b_path,
        )
        for m in modes
    ]


def build_full_matrix(
    detectors: list[str],
    descriptors: list[str],
    mask_modes: list[str],
    estimators: list[str],
    base_config: RunConfig,
) -> list[RunConfig]:
    """Cartesian product of options, respecting VALID_PAIRINGS.
    Returns list of RunConfig instances ready to execute."""
    configs = []
    for det, desc, mode, est in product(detectors, descriptors,
                                         mask_modes, estimators):
        if desc not in VALID_PAIRINGS.get(det, []):
            continue
        cfg = replace(
            base_config,
            detector=det,
            descriptor=desc,
            mask_mode=mode,
            estimator=est,
        )
        configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# Experiment matrix runner
# ---------------------------------------------------------------------------


def _read_rgb(path: Path) -> Optional[np.ndarray]:
    """Read an image from disk and convert BGR→RGB.  Returns ``None`` on
    any failure (missing file, unsupported codec, decode error)."""
    try:
        img = cv2.imread(str(path))
    except Exception:
        return None
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _attempt_filename(pair_id: str, cfg: RunConfig, attempt_mode: str) -> str:
    """JSON filename for a single mask attempt.  ``attempt_mode`` is always
    the concrete mode that ran (``"no_mask"`` or ``"mask"``), never
    ``"both"``."""
    return (f"{pair_id}_{cfg.detector}_{cfg.descriptor}"
            f"_{cfg.estimator}_{attempt_mode}.json")


def _attempt_modes_for(cfg: RunConfig) -> tuple[str, ...]:
    if cfg.mask_mode == "both":
        return ("no_mask", "mask")
    return (cfg.mask_mode,)


_ATTEMPT_COLUMN_PREFIX = {"no_mask": "no_mask", "mask": "with_mask"}

# Per-attempt stat keys that get suffixed and merged into the CSV row.
_ATTEMPT_STAT_KEYS = (
    "result_label", "pixel_correspondence_rate", "mean_corner_error",
    "num_keypoints_A", "num_keypoints_B",
    "num_tentative_matches", "num_inliers", "inlier_ratio",
    "detection_ms", "description_ms", "matching_ms",
    "verification_ms", "geometry_ms", "total_ms",
)


def _attempt_row_fragment(attempt_mode: str, metrics: dict) -> dict:
    """Project ``metrics`` into per-attempt CSV columns, prefixed with
    ``"no_mask_"`` or ``"with_mask_"``."""
    prefix = _ATTEMPT_COLUMN_PREFIX[attempt_mode]
    out: dict = {}
    for k in _ATTEMPT_STAT_KEYS:
        if k in metrics:
            out[f"{prefix}_{k}"] = metrics[k]
    # Friendly aliases for the most-used columns (saves a join in pandas).
    if "result_label" in metrics:
        out[f"{prefix}_result"] = metrics["result_label"]
    if "mean_corner_error" in metrics:
        out[f"{prefix}_err"] = metrics["mean_corner_error"]
    return out


def _worker_init():
    """Per-worker initialiser.  Disable cv2's internal thread pool and OpenCL
    so cv2 doesn't deadlock under ``multiprocessing`` on Windows.

    Without this, each worker keeps cv2's TBB/OpenMP threads alive on top of
    multiprocessing's own process pool, which on Windows can wedge the
    worker before it ever returns its first result.  We pin each worker to
    single-threaded cv2 so the only parallelism is at the process level.
    """
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def _load_cached_metrics(json_path: Path) -> Optional[dict]:
    """Read the ``metrics`` block from a previously-written per-attempt JSON.
    Returns ``None`` if the file is missing or unreadable."""
    if not json_path.exists():
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("metrics") if isinstance(data, dict) else None


def _pair_worker(args):
    """Run a chunk of pending configs for one image pair.

    Each worker task handles `len(configs)` configs for the pair — that may
    be all of them (chunk size = total config count, legacy behaviour, best
    image-I/O amortisation) or just one (chunk size = 1, maximum parallelism,
    enables multiple workers to attack the same pair concurrently). Chunk
    size is set by `run_experiment_matrix(configs_per_task=…)`.

    Returns ``(pair_id, rows, error)`` where ``rows`` is a list of merged
    CSV row dicts (one per config in the chunk) and ``error`` is a short
    string when the images could not be read.
    """
    image_A_path, image_B_path, gt, configs, output_dir = args
    pair_id = f"{image_A_path.stem}_{image_B_path.stem}"
    n_in_chunk = len(configs)

    # Lazy image load: only read if at least one attempt is missing a cached JSON.
    img_A: Optional[np.ndarray] = None
    img_B: Optional[np.ndarray] = None
    rows: list[dict] = []

    for cfg in configs:
        attempt_modes = _attempt_modes_for(cfg)
        row: dict = {
            "pair_id": pair_id,
            "detector": cfg.detector,
            "descriptor": cfg.descriptor,
            "estimator": cfg.estimator,
            "mask_mode": cfg.mask_mode,
        }

        for attempt_mode in attempt_modes:
            json_path = output_dir / _attempt_filename(pair_id, cfg, attempt_mode)
            cached = _load_cached_metrics(json_path)
            if cached is not None:
                row.update(_attempt_row_fragment(attempt_mode, cached))
                continue

            # Need to actually run — load images on first miss for this pair.
            if img_A is None or img_B is None:
                img_A = _read_rgb(image_A_path)
                img_B = _read_rgb(image_B_path)
                if img_A is None or img_B is None:
                    return pair_id, [], (
                        f"could not read {image_A_path.name} / {image_B_path.name}"
                    ), n_in_chunk

            try:
                result, metrics = _execute_pipeline(
                    img_A, img_B, cfg, attempt_mode, gt,
                    image_A_path, image_B_path,
                )
            except Exception as e:
                print(f"Error: pipeline crashed on {pair_id} "
                      f"({cfg.detector}+{cfg.descriptor}+{attempt_mode}): {e}")
                continue

            # Stamp identifying fields on the metrics dict (echoed into JSON).
            metrics["pair_id"] = pair_id
            metrics["detector"] = cfg.detector
            metrics["descriptor"] = cfg.descriptor
            metrics["estimator"] = cfg.estimator
            metrics["mask_mode_spec"] = cfg.mask_mode
            metrics["attempt_mode"] = attempt_mode

            try:
                write_pair_json(result, metrics, output_dir)
            except Exception as e:
                print(f"Warning: could not write JSON for {pair_id} "
                      f"({cfg.detector}+{cfg.descriptor}+{attempt_mode}): {e}")

            row.update(_attempt_row_fragment(attempt_mode, metrics))

        rows.append(row)

    return pair_id, rows, None, n_in_chunk


def run_experiment_matrix(
    dataset_pairs: list[tuple[Path, Path, Optional[GroundTruth]]],
    configs: list[RunConfig],
    output_dir: Path,
    n_workers: Optional[int] = None,
    configs_per_task: int = 1,
) -> None:
    """Run all combinations of pairs × configs.  Write per-attempt JSONs and
    one aggregated CSV row per ``(pair, detector, descriptor, estimator,
    mask_mode)`` into ``output_dir``.  Resumes by re-using any per-attempt
    JSON that already exists.

    Work units are ``(pair, chunk-of-configs)`` tuples dispatched across a
    process pool.  Multiple workers can therefore process configs for the
    *same* pair concurrently — useful when ``n_pairs < n_workers`` (small
    datasets / smoke tests) where the legacy "one worker per pair" scheme
    left workers idle.

    Parameters
    ----------
    n_workers
        Number of worker processes.  ``None`` (default) picks
        :func:`default_experiment_workers`.  Pass ``1`` to force serial
        execution (useful for debugging — the worker still runs in the same
        process, no pool is spawned).
    configs_per_task
        Number of configs each worker task handles for a single pair.
        Default ``1`` (one task per ``(pair, config)``) maximises parallelism
        at the cost of one image read per task.  Pass ``len(configs)`` to
        recover the legacy behaviour (one worker handles a pair's entire
        config list, amortising image I/O across all configs).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = default_experiment_workers()
    n_workers = max(1, n_workers)
    if configs_per_task < 1:
        raise ValueError(f"configs_per_task must be ≥ 1, got {configs_per_task}")

    # Chunk configs per pair.  Each task is (pair, sub-list of configs).
    # The worker still loads images lazily and only once per task, so a small
    # chunk size pays at most one image read per chunk (mitigated by the OS
    # file cache when many workers hit the same pair concurrently).
    worker_args = [
        (img_a, img_b, gt, list(configs[i:i + configs_per_task]), output_dir)
        for img_a, img_b, gt in dataset_pairs
        for i in range(0, len(configs), configs_per_task)
    ]

    # Progress bar counts CSV rows — one per (pair, config) — which matches
    # the scoreboard size the user actually sees.
    total_rows = len(dataset_pairs) * len(configs)
    all_rows: list[dict] = []

    def consume(pair_id: str, rows: list[dict], error: Optional[str]) -> None:
        if error:
            print(f"Warning: skipping pair {pair_id} — {error}.")
        all_rows.extend(rows)

    with tqdm(total=total_rows, desc="Running Matrix") as pbar:
        if n_workers == 1:
            for args in worker_args:
                pair_id, rows, error, n_in_chunk = _pair_worker(args)
                consume(pair_id, rows, error)
                pbar.update(n_in_chunk)
        else:
            ctx = get_context("spawn")
            # initializer disables cv2 threading per worker; without it the
            # pool can deadlock on Windows before any worker returns a result.
            with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
                for pair_id, rows, error, n_in_chunk in pool.imap_unordered(
                    _pair_worker, worker_args
                ):
                    consume(pair_id, rows, error)
                    pbar.update(n_in_chunk)

    if all_rows:
        try:
            write_aggregate_csv(all_rows, output_dir / "aggregate_results.csv")
        except Exception as e:
            print(f"Error: could not write aggregate CSV: {e}")
