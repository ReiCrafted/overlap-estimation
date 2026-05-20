"""orchestrator.py — Pipeline execution and experimental matrix runner.

Runs one image pair through the full pipeline (preprocess → detect →
describe → match → verify → geometry → metrics) and iterates over the
experimental matrix (detector × descriptor × mask_mode × estimator).
"""

import json
import math
import time
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import product
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

    pairs.sort(key=lambda p: (_parse_coords(p[0]), _parse_coords(p[1])))
    return pairs


def run_single_pair(
    image_A: np.ndarray,
    image_B: np.ndarray,
    config: RunConfig,
    ground_truth: Optional[GroundTruth] = None,
    image_a_path: Path = Path("A.jpg"),
    image_b_path: Path = Path("B.jpg"),
) -> tuple[PairResult, dict]:
    """Execute pipeline on one pair. Returns (result, metrics).
    Handles fallback mask mode internally:
    - If config.mask_mode == "fallback": try with no_mask first.
      If inlier count < config.fallback_min_inliers, re-run with mask.
      Record which mode actually succeeded in result.mask_mode."""

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
        return result, metrics

    if config.mask_mode == "fallback":
        # Try no_mask first
        res_no_mask, met_no_mask = execute_pipeline("no_mask")
        if res_no_mask.success:
            return res_no_mask, met_no_mask
        # Fallback to stricter mask
        res_mask, met_mask = execute_pipeline("mask")
        res_mask.fallback_triggered = True
        res_mask.extra["fallback_reason"] = res_no_mask.error_message
        res_mask.extra["no_mask_metrics"] = met_no_mask
        return res_mask, met_mask
    else:
        return execute_pipeline(config.mask_mode)


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


def run_experiment_matrix(
    dataset_pairs: list[tuple[Path, Path, Optional[GroundTruth]]],
    configs: list[RunConfig],
    output_dir: Path,
) -> None:
    """Run all combinations of pairs × configs. Write per-pair JSONs
    and aggregate CSV to output_dir. Progress bar via tqdm.
    Resumes if partial results exist (skip already-completed runs)."""

    all_metrics = []
    output_dir.mkdir(parents=True, exist_ok=True)
    total_runs = len(dataset_pairs) * len(configs)

    with tqdm(total=total_runs, desc="Running Matrix") as pbar:
        for image_A_path, image_B_path, gt in dataset_pairs:
            pair_id = f"{image_A_path.stem}_{image_B_path.stem}"

            # Preload images once per pair (avoid re-reading per config)
            try:
                img_A = cv2.imread(str(image_A_path))
                if img_A is not None:
                    img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
                img_B = cv2.imread(str(image_B_path))
                if img_B is not None:
                    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Warning: could not load {image_A_path} or {image_B_path}: {e}")
                img_A = None
                img_B = None

            if img_A is None or img_B is None:
                print(f"Warning: skipping pair {pair_id} — image could not be read.")

            for config in configs:
                filename = (f"{pair_id}_{config.detector}_{config.descriptor}"
                            f"_{config.estimator}_{config.mask_mode}.json")
                json_path = output_dir / filename

                if json_path.exists():
                    # Resume: load existing metrics
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            all_metrics.append(data.get("metrics", {}))
                    except (json.JSONDecodeError, KeyError):
                        pass
                    pbar.update(1)
                    continue

                if img_A is None or img_B is None:
                    pbar.update(1)
                    continue

                try:
                    result, metrics = run_single_pair(
                        img_A, img_B, config, gt, image_A_path, image_B_path)
                except Exception as e:
                    print(f"Error: run_single_pair crashed on {pair_id} "
                          f"({config.detector}+{config.descriptor}): {e}")
                    pbar.update(1)
                    continue

                # Add identification fields for the CSV
                metrics["pair_id"] = pair_id
                metrics["detector"] = config.detector
                metrics["descriptor"] = config.descriptor
                metrics["estimator"] = config.estimator
                metrics["mask_mode"] = config.mask_mode

                all_metrics.append(metrics)
                try:
                    write_pair_json(result, metrics, output_dir)
                except Exception as e:
                    print(f"Warning: could not write result for {pair_id} "
                          f"({config.detector}+{config.descriptor}): {e}")
                pbar.update(1)

    if all_metrics:
        try:
            write_aggregate_csv(all_metrics, output_dir / "aggregate_results.csv")
        except Exception as e:
            print(f"Error: could not write aggregate CSV: {e}")
