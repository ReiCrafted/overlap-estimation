import pytest
import numpy as np
from pathlib import Path
from overlap_detection.orchestrator import run_single_pair, build_full_matrix
from overlap_detection.config import RunConfig


def test_build_full_matrix():
    base = RunConfig(detector="SIFT", descriptor="SIFT", estimator="PROSAC")
    configs = build_full_matrix(
        detectors=["FAST", "SIFT"],
        descriptors=["BRIEF", "SIFT"],
        mask_modes=["no_mask"],
        estimators=["PROSAC"],
        base_config=base,
    )
    # Should yield 4 valid configs since all pairings are valid
    assert len(configs) == 4
    for c in configs:
        assert c.mask_mode == "no_mask"
        assert c.estimator == "PROSAC"


def test_run_single_pair_synthetic():
    img_A = np.zeros((100, 100, 3), dtype=np.uint8)
    img_A[20:80, 20:80] = 255
    img_B = img_A.copy()

    cfg = RunConfig(
        detector="FAST", descriptor="BRIEF",
        estimator="PROSAC", mask_mode="no_mask",
    )

    attempts = run_single_pair(img_A, img_B, cfg)
    assert len(attempts) == 1
    res, metrics = attempts[0]
    assert res.detector == "FAST"
    assert res.estimator == "PROSAC"
    assert res.mask_mode == "no_mask"
    assert metrics["result_label"] in {"no_match", "false_match"} or metrics["result_label"].startswith("acc_at_")


def test_run_single_pair_both_runs_two_attempts():
    img_A = np.zeros((100, 100, 3), dtype=np.uint8)
    img_A[20:80, 20:80] = 255
    img_B = img_A.copy()

    cfg = RunConfig(
        detector="FAST", descriptor="BRIEF",
        estimator="PROSAC", mask_mode="both",
    )

    attempts = run_single_pair(img_A, img_B, cfg)
    assert len(attempts) == 2
    modes = [r.mask_mode for r, _ in attempts]
    assert modes == ["no_mask", "mask"]
