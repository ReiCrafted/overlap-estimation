import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from overlap_detection.types import PairResult
from overlap_detection.reporting import (
    write_pair_json, write_aggregate_csv, write_summary_report,
    _maa, _add_best_of_both_err_column, _add_best_of_both_column,
    _section_vmax,
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

    # Build CSV rows in the format the orchestrator now emits — cover BOTH
    # estimators so the per-estimator splits in the report are exercised.
    rows = []
    for est in ("PROSAC", "USAC_MAGSAC"):
        rows.extend([
            {
                "pair_id": f"img1_img2_{est}",
                "detector": "FAST", "descriptor": "BRIEF",
                "estimator": est, "mask_mode": "both",
                "no_mask_result": "acc_at_5",   "with_mask_result": "acc_at_3",
                "no_mask_err": 4.2,             "with_mask_err": 2.1,
                "no_mask_iou": 0.93,            "with_mask_iou": 0.97,
            },
            {
                "pair_id": f"img3_img4_{est}",
                "detector": "FAST", "descriptor": "BRIEF",
                "estimator": est, "mask_mode": "both",
                "no_mask_result": "false_match", "with_mask_result": "acc_at_10",
                "no_mask_err": 14.0,             "with_mask_err": 8.5,
                "no_mask_iou": 0.5,              "with_mask_iou": 0.85,
            },
            {
                "pair_id": f"img5_img6_{est}",
                "detector": "FAST", "descriptor": "BRIEF",
                "estimator": est, "mask_mode": "both",
                "no_mask_result": "no_match",   "with_mask_result": "acc_at_5",
                "no_mask_err": None,            "with_mask_err": 4.0,
            },
        ])
    csv_path = tmp_path / "aggregate.csv"
    write_aggregate_csv(rows, csv_path)
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 6
    assert "no_mask_result" in df.columns
    assert "with_mask_result" in df.columns

    report_dir = tmp_path / "report"
    write_summary_report(csv_path, report_dir)
    report_md = (report_dir / "report.md").read_text(encoding="utf-8")
    # Headline content
    assert "mAA" in report_md
    assert "Precision" in report_md
    assert "acc_at_3" in report_md or "acc@3" in report_md
    # Both estimators get their own splits
    assert "PROSAC" in report_md
    assert "USAC_MAGSAC" in report_md
    # All three attempt slices appear
    assert "no_mask" in report_md
    assert "with_mask" in report_md
    assert "best_of_both" in report_md
    # Per-estimator sections exist
    assert "Per-configuration scoreboard" in report_md
    assert "mAA matrices" in report_md
    assert "Precision matrices" in report_md
    assert "Fallback benefit" in report_md
    # Heatmap PNGs were written, one per (metric, estimator, attempt) slice
    # that has data.  Our fixture has both estimators × all three attempts.
    for metric in ("maa", "precision"):
        for est in ("PROSAC", "USAC_MAGSAC"):
            for attempt in ("no_mask", "with_mask", "best_of_both"):
                png = report_dir / f"heatmap_{metric}_{est}_{attempt}.png"
                assert png.exists(), f"missing {png.name}"


# ---------------------------------------------------------------------------
# AUC mAA — direct unit tests with hand-computable inputs
# ---------------------------------------------------------------------------


def test_maa_single_error_known_value():
    """One pair at 2 px against tiers [3, 5, 10] gives a closed-form mAA."""
    errors = pd.Series([2.0])
    tiers = [3.0, 5.0, 10.0]
    # Per-threshold AUC by hand:
    #   sorted_err=[0,2], recall=[0,1]
    #   t=3 : e=[0,2,3], r=[0,1,1]  → trapz = 1 + 1   = 2.0 ; AUC = 2/3
    #   t=5 : e=[0,2,5], r=[0,1,1]  → trapz = 1 + 3   = 4.0 ; AUC = 4/5
    #   t=10: e=[0,2,10],r=[0,1,1]  → trapz = 1 + 8   = 9.0 ; AUC = 9/10
    # mean = (2/3 + 4/5 + 9/10) / 3
    expected = (2 / 3 + 4 / 5 + 9 / 10) / 3
    assert np.isclose(_maa(errors, tiers), expected, atol=1e-9)


def test_maa_all_failures_is_zero():
    """NaN errors map to inf and never reach any threshold → mAA = 0."""
    errors = pd.Series([np.nan, np.nan, np.nan])
    assert _maa(errors, [3.0, 5.0, 10.0]) == 0.0


def test_maa_all_perfect_is_one():
    """All-zero errors clear every threshold completely → mAA = 1.0."""
    errors = pd.Series([0.0, 0.0, 0.0, 0.0])
    assert np.isclose(_maa(errors, [3.0, 5.0, 10.0]), 1.0)


def test_maa_failures_penalised_proportionally():
    """One perfect pair + one failure: per-threshold AUC = 0.5 → mAA = 0.5."""
    errors = pd.Series([0.0, np.nan])
    assert np.isclose(_maa(errors, [3.0, 5.0, 10.0]), 0.5)


def test_maa_empty_or_no_tiers_returns_nan():
    assert np.isnan(_maa(pd.Series([], dtype=float), [3.0]))
    assert np.isnan(_maa(pd.Series([1.0]), []))


# ---------------------------------------------------------------------------
# best_of_both error column — tie-breaking and missing-column behaviour
# ---------------------------------------------------------------------------


def test_best_of_both_err_picks_winning_attempt():
    df = pd.DataFrame([
        # no_mask wins outright (lower-tier label)
        {"no_mask_result": "acc_at_3", "with_mask_result": "acc_at_10",
         "no_mask_err": 2.0,           "with_mask_err": 8.0},
        # with_mask wins outright
        {"no_mask_result": "acc_at_10", "with_mask_result": "acc_at_3",
         "no_mask_err": 8.0,            "with_mask_err": 2.0},
        # Same label → tie → no_mask wins (matches _best_of_both ordering)
        {"no_mask_result": "acc_at_5",  "with_mask_result": "acc_at_5",
         "no_mask_err": 4.0,            "with_mask_err": 4.5},
        # one no_match — the other side wins
        {"no_mask_result": "no_match",  "with_mask_result": "acc_at_5",
         "no_mask_err": np.nan,         "with_mask_err": 4.0},
        # One side is NaN (attempt didn't run) → result is NaN
        {"no_mask_result": "acc_at_3",  "with_mask_result": np.nan,
         "no_mask_err": 2.0,            "with_mask_err": np.nan},
    ])
    df = _add_best_of_both_err_column(df)
    assert df["best_of_both_err"].iloc[0] == 2.0   # no_mask wins
    assert df["best_of_both_err"].iloc[1] == 2.0   # with_mask wins
    assert df["best_of_both_err"].iloc[2] == 4.0   # tie → no_mask
    assert df["best_of_both_err"].iloc[3] == 4.0   # with_mask wins
    assert pd.isna(df["best_of_both_err"].iloc[4]) # one side missing


def test_best_of_both_err_handles_missing_columns():
    """When *_err columns aren't in the CSV at all, the helper degrades
    gracefully (no exception, column populated with NA)."""
    df = pd.DataFrame([{"no_mask_result": "acc_at_3", "with_mask_result": "acc_at_5"}])
    df = _add_best_of_both_err_column(df)
    assert "best_of_both_err" in df.columns
    assert df["best_of_both_err"].isna().all()


# ---------------------------------------------------------------------------
# Resilience — old CSVs without *_err columns shouldn't crash the report
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared-scale heatmap vmax
# ---------------------------------------------------------------------------


def test_section_vmax_precision_is_one():
    """Precision colour scale is always [0, 1] regardless of observed range."""
    t = pd.DataFrame([[0.1, 0.2]])
    assert _section_vmax([t], "precision") == 1.0


def test_section_vmax_floors_at_03():
    """Tiny-mAA datasets don't amplify noise below the 0.3 floor."""
    t = pd.DataFrame([[0.05, 0.10]])
    assert _section_vmax([t], "maa") == 0.3


def test_section_vmax_rounds_up_to_next_tenth():
    """0.42 → 0.5 (the colorbar should not end mid-tenth)."""
    t = pd.DataFrame([[0.42, 0.31]])
    assert _section_vmax([t], "maa") == 0.5


def test_section_vmax_ignores_nan_cells():
    """A real value of 0.7 must beat the floor even when other cells are NaN
    in the same table (regression: ndarray.max propagates NaN)."""
    t = pd.DataFrame([[0.7, np.nan], [np.nan, 0.6]])
    assert _section_vmax([t], "maa") == 0.7


def test_section_vmax_capped_at_one():
    """Theoretical max for AUC is 1.0; the scale should never exceed it."""
    t = pd.DataFrame([[0.99, 1.0]])
    assert _section_vmax([t], "maa") == 1.0


def test_report_runs_without_err_columns(tmp_path):
    """Pre-AUC CSVs only had *_result columns. The report builder must still
    produce a markdown file (mAA columns will be N/A but Precision survives)."""
    rows = []
    for est in ("PROSAC",):
        rows.extend([
            {"pair_id": "a_b", "detector": "FAST", "descriptor": "BRIEF",
             "estimator": est, "mask_mode": "no_mask",
             "no_mask_result": "acc_at_5"},
            {"pair_id": "c_d", "detector": "FAST", "descriptor": "BRIEF",
             "estimator": est, "mask_mode": "no_mask",
             "no_mask_result": "false_match"},
        ])
    csv_path = tmp_path / "aggregate.csv"
    write_aggregate_csv(rows, csv_path)
    report_dir = tmp_path / "report"
    write_summary_report(csv_path, report_dir)
    assert (report_dir / "report.md").exists()
