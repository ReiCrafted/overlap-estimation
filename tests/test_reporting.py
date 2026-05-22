import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from overlap_detection.types import PairResult
from overlap_detection.reporting import (
    write_pair_json, write_aggregate_csv, write_summary_report,
)


def test_reporting_flow(tmp_path):
    result = PairResult(
        image_a_path=Path("img1.jpg"),
        image_b_path=Path("img2.jpg"),
        detector="FAST",
        descriptor="BRIEF",
        estimator="PROSAC",
        mask_mode="no_mask",
        result_label="acc_at_5",
    )
    metrics = {
        "pair_id": "img1_img2",
        "detector": "FAST",
        "descriptor": "BRIEF",
        "estimator": "PROSAC",
        "mask_mode": "no_mask",
        "result_label": "acc_at_5",
        "mean_corner_error": 4.2,
        "iou": 0.93,
        "inlier_ratio": 0.5,
        "total_ms": 100.0,
    }

    # JSON round trip
    json_path = write_pair_json(result, metrics, tmp_path)
    assert json_path.exists()

    # Build CSV rows in the format the orchestrator now emits.
    rows = [
        {
            "pair_id": "img1_img2",
            "detector": "FAST", "descriptor": "BRIEF",
            "estimator": "PROSAC", "mask_mode": "both",
            "no_mask_result": "acc_at_5",   "with_mask_result": "acc_at_3",
            "no_mask_err": 4.2,             "with_mask_err": 2.1,
            "no_mask_iou": 0.93,            "with_mask_iou": 0.97,
        },
        {
            "pair_id": "img3_img4",
            "detector": "FAST", "descriptor": "BRIEF",
            "estimator": "PROSAC", "mask_mode": "both",
            "no_mask_result": "false_match", "with_mask_result": "acc_at_10",
            "no_mask_err": 14.0,             "with_mask_err": 8.5,
            "no_mask_iou": 0.5,              "with_mask_iou": 0.85,
        },
        {
            "pair_id": "img5_img6",
            "detector": "FAST", "descriptor": "BRIEF",
            "estimator": "PROSAC", "mask_mode": "both",
            "no_mask_result": "no_match",   "with_mask_result": "acc_at_5",
            "no_mask_err": None,            "with_mask_err": 4.0,
        },
    ]
    csv_path = tmp_path / "aggregate.csv"
    write_aggregate_csv(rows, csv_path)
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 3
    assert "no_mask_result" in df.columns
    assert "with_mask_result" in df.columns

    report_dir = tmp_path / "report"
    write_summary_report(csv_path, report_dir)
    report_md = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "mAA" in report_md
    assert "acc_at_3" in report_md or "acc@3" in report_md
    assert "Fallback benefit" in report_md
