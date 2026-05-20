"""orchestrator.py — Pipeline execution and experimental matrix runner.

Runs one image pair through the full pipeline (preprocess → detect →
describe → match → verify → geometry → metrics) and iterates over the
experimental matrix (detector × descriptor × mask_mode × estimator).
"""

import json
import math
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

# ---------------------------------------------------------------------------
# Affine sanity filter
# ---------------------------------------------------------------------------
_MAX_SCALE_DIFF = 0.10   # reject if |scale - 1| > 10 %
_MAX_ROTATION_DEG = 3.0  # reject if |rotation| > 3 °


def _affine_is_sane(affine_mat: np.ndarray) -> tuple[bool, float, float]:
    """Check scale and rotation of a 2×3 affine matrix.

    Returns (is_sane, scale, rotation_deg).
    Scale is the column-0 norm of the 2×2 sub-matrix; rotation is atan2(a10, a00).
    A pure translation has scale=1.0, rotation=0.0.
    """
    a00, a10 = float(affine_mat[0, 0]), float(affine_mat[1, 0])
    scale = math.sqrt(a00 ** 2 + a10 ** 2)
    rotation_deg = math.degrees(math.atan2(a10, a00))
    is_sane = (abs(scale - 1.0) <= _MAX_SCALE_DIFF and
               abs(rotation_deg) <= _MAX_ROTATION_DEG)
    return is_sane, scale, rotation_deg


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
# Quality gates
# ---------------------------------------------------------------------------


def _apply_quality_gates(
    result: PairResult,
    metrics: dict,
    iou_threshold: float,
    rms_threshold_px: float,
) -> None:
    """Demote ``result.success`` to False if GT-derived gates are violated.

    Mutates ``result`` and ``metrics`` in place.  Gates that require ground
    truth (IoU, corner RMS) are skipped when those metrics are ``None`` — in
    that case only the pipeline's existing gates (affine sanity, min inliers)
    determine success.
    """
    if not result.success:
        return

    iou = metrics.get("iou")
    if iou is not None and iou < iou_threshold:
        result.success = False
        result.error_message = (
            f"IoU {iou:.3f} < {iou_threshold:.2f} (quality gate)"
        )
        return

    rms = metrics.get("rms_corner_error")
    if rms is not None and rms > rms_threshold_px:
        result.success = False
        result.error_message = (
            f"Corner RMS {rms:.2f} px > {rms_threshold_px:.2f} px (quality gate)"
        )
        return


def run_single_pair(
    image_A: np.ndarray,
    image_B: np.ndarray,
    config: RunConfig,
    ground_truth: Optional[GroundTruth] = None,
    image_a_path: Path = Path("A.jpg"),
    image_b_path: Path = Path("B.jpg"),
) -> tuple[PairResult, dict]:
    """Execute pipeline on one pair. Returns (result, metrics).

    Quality gates (applied after metrics are computed): affine scale ±10 %,
    rotation ±3 °, ``n_inliers ≥ fallback_min_inliers``, ``iou ≥ iou_threshold``,
    ``rms_corner_error ≤ rms_error_threshold_px``.  IoU/RMS gates are skipped
    when no ground truth is provided.

    Sets ``result.quality_flag`` to one of:
      * ``"true"``               — passed on the primary attempt.
      * ``"false"``              — failed (non-fallback config).
      * ``"true after false"``   — fallback re-run with mask succeeded after
                                   the no-mask attempt failed.
      * ``"false after false"``  — both fallback attempts failed.

    For ``mask_mode == "fallback"``: try ``no_mask`` first, and if it does not
    pass the gates, re-run with ``mask``.  Whichever attempt is returned has
    ``result.mask_mode`` set to the mode that actually ran."""

    def execute_pipeline(mask_mode_to_use: str) -> tuple[PairResult, dict]:
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
            # 1. Preprocessing (Masking)
            mask_A = apply_mask_mode(image_A, mask_mode_to_use,
                                     config.overlap_band_fraction,
                                     config.rgb_gray_threshold,
                                     side="right")
            mask_B = apply_mask_mode(image_B, mask_mode_to_use,
                                     config.overlap_band_fraction,
                                     config.rgb_gray_threshold,
                                     side="left")

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
                return result, compute_pair_metrics(result, ground_truth)

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
                return result, compute_pair_metrics(result, ground_truth)

            # 4. Matching
            # LIOP descriptors cluster tightly in distance space, making the
            # NNDR ratio always ≈ 1.0 and killing all matches.  Use plain MNN
            # (no ratio test) so PROSAC/RANSAC can filter the noisier match set.
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
                return result, compute_pair_metrics(result, ground_truth)

            # 5. Verification
            t0 = time.perf_counter()
            affine_mat, inliers = verify_affine(
                matches, filtered_kps_A, filtered_kps_B,
                config.estimator, config.ransac_threshold_px,
                config.ransac_max_iters, config.ransac_confidence)
            result.time_verification_s = time.perf_counter() - t0

            if affine_mat is not None:
                result.n_inliers = int(np.sum(inliers))
                if result.n_inliers >= config.fallback_min_inliers:
                    # Sanity-check scale and rotation before accepting
                    sane, scale, rot_deg = _affine_is_sane(affine_mat)
                    if not sane:
                        result.error_message = (
                            f"Affine rejected: scale={scale:.3f} "
                            f"(diff={abs(scale-1)*100:.1f}%), "
                            f"rotation={rot_deg:.2f}°")
                    else:
                        result.affine_matrix = affine_mat
                        result.inlier_mask = inliers
                        result.success = True

                        # 6. Geometry
                        t0 = time.perf_counter()
                        poly_A, poly_B = compute_overlap_polygon(
                            affine_mat, image_A.shape, image_B.shape)
                        result.time_geometry_s = time.perf_counter() - t0
                        result.overlap_polygon_a = poly_A
                        result.overlap_polygon_b = poly_B
                else:
                    result.error_message = (
                        f"Too few inliers ({result.n_inliers} "
                        f"< {config.fallback_min_inliers})")
            else:
                result.error_message = "Affine estimation failed."

        except Exception as e:
            result.error_message = f"Exception: {str(e)}"

        result.time_total_s = time.perf_counter() - t_start_total
        metrics = compute_pair_metrics(result, ground_truth)
        # GT-dependent quality gates may demote success after the fact.
        _apply_quality_gates(
            result, metrics,
            config.iou_threshold, config.rms_error_threshold_px,
        )
        return result, metrics

    def _stamp(result: PairResult, metrics: dict, flag: str) -> None:
        result.quality_flag = flag
        metrics["quality_flag"] = flag

    if config.mask_mode == "fallback":
        # Try no_mask first
        res_no_mask, met_no_mask = execute_pipeline("no_mask")
        if res_no_mask.success:
            _stamp(res_no_mask, met_no_mask, "true")
            return res_no_mask, met_no_mask
        # Fallback to stricter mask
        res_mask, met_mask = execute_pipeline("mask")
        res_mask.fallback_triggered = True
        res_mask.extra["fallback_reason"] = res_no_mask.error_message
        res_mask.extra["no_mask_metrics"] = met_no_mask
        _stamp(res_mask, met_mask,
               "true after false" if res_mask.success else "false after false")
        return res_mask, met_mask
    else:
        res, met = execute_pipeline(config.mask_mode)
        _stamp(res, met, "true" if res.success else "false")
        return res, met


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


def _config_filename(pair_id: str, cfg: RunConfig) -> str:
    return (f"{pair_id}_{cfg.detector}_{cfg.descriptor}"
            f"_{cfg.estimator}_{cfg.mask_mode}.json")


def _pair_worker(args):
    """Run all pending configs for one image pair.

    Returns ``(pair_id, metrics_list, error)``.  ``metrics_list`` contains the
    metrics dict for each config that ran (or was loaded from cache); ``error``
    is ``None`` on success or a short string when the pair could not be read.
    """
    image_A_path, image_B_path, gt, configs, output_dir = args
    pair_id = f"{image_A_path.stem}_{image_B_path.stem}"

    img_A = _read_rgb(image_A_path)
    img_B = _read_rgb(image_B_path)
    if img_A is None or img_B is None:
        return pair_id, [], f"could not read {image_A_path.name} / {image_B_path.name}"

    metrics_list: list[dict] = []
    for cfg in configs:
        json_path = output_dir / _config_filename(pair_id, cfg)
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                metrics_list.append(data.get("metrics", {}))
            except (OSError, json.JSONDecodeError):
                pass
            continue

        try:
            result, metrics = run_single_pair(
                img_A, img_B, cfg, gt, image_A_path, image_B_path)
        except Exception as e:
            print(f"Error: run_single_pair crashed on {pair_id} "
                  f"({cfg.detector}+{cfg.descriptor}): {e}")
            continue

        metrics["pair_id"] = pair_id
        metrics["detector"] = cfg.detector
        metrics["descriptor"] = cfg.descriptor
        metrics["estimator"] = cfg.estimator
        metrics["mask_mode"] = cfg.mask_mode
        metrics_list.append(metrics)

        try:
            write_pair_json(result, metrics, output_dir)
        except Exception as e:
            print(f"Warning: could not write result for {pair_id} "
                  f"({cfg.detector}+{cfg.descriptor}): {e}")

    return pair_id, metrics_list, None


def run_experiment_matrix(
    dataset_pairs: list[tuple[Path, Path, Optional[GroundTruth]]],
    configs: list[RunConfig],
    output_dir: Path,
    n_workers: Optional[int] = None,
) -> None:
    """Run all combinations of pairs × configs.  Write per-pair JSONs and an
    aggregate CSV into ``output_dir``.  Pairs are dispatched across a process
    pool (each worker handles one pair × all its pending configs, loading the
    images once).  Resumes by skipping any (pair, config) combination whose
    JSON already exists.

    Parameters
    ----------
    n_workers
        Number of worker processes.  ``None`` (default) picks
        :func:`default_experiment_workers`.  Pass ``1`` to force serial
        execution (useful for debugging — the worker still runs in the same
        process, no pool is spawned).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = default_experiment_workers()
    n_workers = max(1, n_workers)

    worker_args = [
        (img_a, img_b, gt, configs, output_dir)
        for img_a, img_b, gt in dataset_pairs
    ]

    # Progress bar counts (pair, config) runs — including cached/skipped ones
    # — so the user sees the full matrix size, not just what's left to do.
    total_runs = sum(len(configs) for _ in dataset_pairs)
    all_metrics: list[dict] = []

    def consume(pair_id: str, metrics_list: list[dict], error: Optional[str]) -> None:
        if error:
            print(f"Warning: skipping pair {pair_id} — {error}.")
        all_metrics.extend(metrics_list)

    with tqdm(total=total_runs, desc="Running Matrix") as pbar:
        if n_workers == 1:
            for args in worker_args:
                pair_id, metrics_list, error = _pair_worker(args)
                consume(pair_id, metrics_list, error)
                pbar.update(len(configs))
        else:
            # Use "spawn" so workers don't inherit OpenCV state / GUI handles
            # from the parent — required on Windows and safer on POSIX.
            ctx = get_context("spawn")
            with ctx.Pool(processes=n_workers) as pool:
                for pair_id, metrics_list, error in pool.imap_unordered(
                    _pair_worker, worker_args
                ):
                    consume(pair_id, metrics_list, error)
                    pbar.update(len(configs))

    if all_metrics:
        try:
            write_aggregate_csv(all_metrics, output_dir / "aggregate_results.csv")
        except Exception as e:
            print(f"Error: could not write aggregate CSV: {e}")
