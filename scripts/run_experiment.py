import argparse
import sys
from pathlib import Path

# Add project root to path so 'overlap_detection' is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from overlap_detection.config import (
    RunConfig, DETECTOR_NAMES, DESCRIPTOR_NAMES,
    VALID_MASK_MODES, VALID_ESTIMATORS,
)
from overlap_detection.orchestrator import (
    run_experiment_matrix, build_full_matrix, list_image_pairs,
    default_experiment_workers,
)
from overlap_detection.types import GroundTruth
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Run image registration experimental matrix.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing image pairs.")
    parser.add_argument("--groundtruth-dir", type=Path, required=True, help="Directory containing ground truth annotations.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save results.")
    parser.add_argument("--detectors", type=str, default=",".join(DETECTOR_NAMES), help="Comma-separated list of detectors to evaluate.")
    parser.add_argument("--descriptors", type=str, default=",".join(DESCRIPTOR_NAMES), help="Comma-separated list of descriptors to evaluate.")
    parser.add_argument("--mask-modes", type=str, default="both", help="Comma-separated list of mask modes (no_mask, mask, both). 'both' runs each pair twice and pairs the results in one CSV row.")
    parser.add_argument("--estimators", type=str, default="PROSAC", help="Comma-separated list of estimators (e.g. PROSAC,USAC_MAGSAC).")
    parser.add_argument("--max-pairs", type=int, default=None, help="Limit to the first N pairs (useful for quick tests).")
    parser.add_argument(
        "--workers", type=int, default=None,
        help=(f"Number of worker processes (default: cpu_count-1 up to 8, "
              f"resolved to {default_experiment_workers()} on this machine). "
              "Pass 1 for serial execution."),
    )
    parser.add_argument(
        "--configs-per-task", type=int, default=1,
        help=("Configs handled per worker task (default 1 = max parallelism, "
              "multiple workers can share one pair).  Pass a larger value "
              "to amortise image I/O across more configs per task; pass "
              "len(configs) to recover the legacy 'one worker per pair' scheme."),
    )
    return parser.parse_args()

def load_groundtruth(gt_dir: Path, img_a_stem: str, img_b_stem: str) -> GroundTruth | None:
    pair_id = f"{img_a_stem}_{img_b_stem}"
    gt_path = gt_dir / f"{pair_id}_groundtruth.json"
    if not gt_path.exists():
        return None
    try:
        with open(gt_path, 'r') as f:
            data = json.load(f)
        return GroundTruth(
            image_A_path=Path(data["image_A_path"]),
            image_B_path=Path(data["image_B_path"]),
            affine_matrix_A_to_B=np.array(data["affine_matrix_A_to_B"], dtype=np.float64),
            image_a_shape=tuple(data["image_a_shape"]),
            image_b_shape=tuple(data["image_b_shape"]),
            annotator=data["annotator"],
            annotation_date=data["annotation_date"],
        )
    except Exception as e:
        print(f"Warning: could not load ground truth for {pair_id}: {e}")
        return None

def main():
    args = parse_args()

    detectors  = args.detectors.split(",")
    descriptors = args.descriptors.split(",")
    mask_modes  = args.mask_modes.split(",")
    estimators  = args.estimators.split(",")

    # --- Upfront parameter validation ---
    bad_det  = [d for d in detectors  if d not in DETECTOR_NAMES]
    bad_desc = [d for d in descriptors if d not in DESCRIPTOR_NAMES]
    bad_mask = [m for m in mask_modes  if m not in VALID_MASK_MODES]
    bad_est  = [e for e in estimators  if e not in VALID_ESTIMATORS]
    errors = []
    if bad_det:  errors.append(f"Unknown detectors: {bad_det}")
    if bad_desc: errors.append(f"Unknown descriptors: {bad_desc}")
    if bad_mask: errors.append(f"Unknown mask modes: {bad_mask}")
    if bad_est:  errors.append(f"Unknown estimators: {bad_est}")
    if errors:
        for e in errors:
            print(f"Error: {e}")
        print(f"Valid detectors:  {DETECTOR_NAMES}")
        print(f"Valid descriptors: {DESCRIPTOR_NAMES}")
        return

    # --- Load pairs ---
    dataset_pairs = []
    for img_a, img_b in list_image_pairs(args.dataset_dir):
        gt = load_groundtruth(args.groundtruth_dir, img_a.stem, img_b.stem)
        dataset_pairs.append((img_a, img_b, gt))

    if not dataset_pairs:
        print(f"Error: no image pairs found in {args.dataset_dir}.")
        return

    if args.max_pairs is not None:
        dataset_pairs = dataset_pairs[:args.max_pairs]

    # --- Build config matrix ---
    base_config = RunConfig()
    configs = build_full_matrix(detectors, descriptors, mask_modes, estimators, base_config)

    if not configs:
        print("Error: no valid detector/descriptor configurations generated. "
              "Check VALID_PAIRINGS in config.py.")
        return

    n_workers = args.workers if args.workers is not None else default_experiment_workers()
    print(f"Running {len(configs)} configurations across {len(dataset_pairs)} image pairs "
          f"({len(configs) * len(dataset_pairs)} total runs) on {n_workers} worker(s).")
    run_experiment_matrix(
        dataset_pairs, configs, args.output_dir,
        n_workers=n_workers,
        configs_per_task=args.configs_per_task,
    )

if __name__ == "__main__":
    main()
