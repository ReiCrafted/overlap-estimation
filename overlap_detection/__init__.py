from .config import RunConfig, DETECTOR_NAMES, DESCRIPTOR_NAMES, VALID_PAIRINGS
from .types import Image, Mask, DescriptorMatrix, Keypoint, PairResult, GroundTruth
from .preprocessing import (
    make_overlap_band_mask,
    make_grayness_mask,
    combine_masks,
    apply_mask_mode,
)
from .detection import detect
from .description import describe, is_binary_descriptor
from .matching import match
from .verification import verify_affine
from .geometry import compute_overlap_polygon, apply_affine, invert_affine
from .metrics import (
    per_corner_errors,
    mean_corner_error,
    overlap_iou,
    compute_pair_metrics,
    categorize_result,
)
from .reporting import write_pair_json, write_aggregate_csv, write_summary_report
from .orchestrator import run_single_pair, run_experiment_matrix, build_full_matrix
from .annotation_gui import AnnotationGUI

__all__ = [
    # config
    "RunConfig",
    "DETECTOR_NAMES",
    "DESCRIPTOR_NAMES",
    "VALID_PAIRINGS",
    # types
    "Image",
    "Mask",
    "DescriptorMatrix",
    "Keypoint",
    "PairResult",
    "GroundTruth",
    # preprocessing
    "make_overlap_band_mask",
    "make_grayness_mask",
    "combine_masks",
    "apply_mask_mode",
    # detection
    "detect",
    # description
    "describe",
    "is_binary_descriptor",
    # matching
    "match",
    # verification
    "verify_affine",
    # geometry
    "compute_overlap_polygon",
    "apply_affine",
    "invert_affine",
    # metrics
    "per_corner_errors",
    "mean_corner_error",
    "overlap_iou",
    "compute_pair_metrics",
    "categorize_result",
    # reporting
    "write_pair_json",
    "write_aggregate_csv",
    "write_summary_report",
    # orchestrator
    "run_single_pair",
    "run_experiment_matrix",
    "build_full_matrix",
    # annotation
    "AnnotationGUI",
]
