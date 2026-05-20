import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from overlap_detection.types import PairResult
from overlap_detection.reporting import write_pair_json, write_aggregate_csv, write_summary_report


def test_reporting_flow(tmp_path):
    result = PairResult(
        image_a_path=Path("img1.jpg"),
        image_b_path=Path("img2.jpg"),
        detector="FAST",
        descriptor="BRIEF",
        estimator="PROSAC",
        mask_mode="no_mask",
    )
    metrics = {
        "pair_id": "img1_img2",
        "detector": "FAST",
        "descriptor": "BRIEF",
        "estimator": "PROSAC",
        "mask_mode": "no_mask",
        "rms_corner_error": 5.0,
        "inlier_ratio": 0.5,
        "total_ms": 100.0,
        "estimation_succeeded": True,
    }

    # JSON round trip
    json_path = write_pair_json(result, metrics, tmp_path)
    assert json_path.exists()

    # CSV
    csv_path = tmp_path / "aggregate.csv"
    write_aggregate_csv([metrics], csv_path)
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["detector"] == "FAST"

    # Plots
    plot_dir = tmp_path / "plots"
    write_summary_report(csv_path, plot_dir)
    assert (plot_dir / "report.md").exists()
