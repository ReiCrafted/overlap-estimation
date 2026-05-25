"""regenerate_report_with_tiers.py — rebuild a report from an existing
aggregate CSV using a different accuracy_tiers_px set, without re-running
the experiment.

The CSV already stores the raw per-pair ``no_mask_err`` / ``with_mask_err``
and pixel correspondence rates.  Only the categorical ``*_result`` labels
depend on the tier set, so we recompute them in-place and then re-render
the report.

Usage::

    # Regenerate with the canonical final-test tiers (default):
    python scripts/regenerate_report_with_tiers.py \
        --results-dir results/full_final_prosac \
        --output-dir reports/full_final_prosac

    # Regenerate with the earlier literature-standard tiers for comparison:
    python scripts/regenerate_report_with_tiers.py \
        --results-dir results/full_final_prosac \
        --output-dir reports/full_final_prosac_tiers_3_5_10 \
        --tiers 3,5,10
"""

import argparse
import sys
from pathlib import Path

# Add project root so ``overlap_detection`` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from overlap_detection.metrics import categorize_result
from overlap_detection.reporting import write_summary_report


def parse_args():
    p = argparse.ArgumentParser(
        description="Rebuild report from CSV with a different tier set."
    )
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Directory containing aggregate_results.csv")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory to write the regenerated report + heatmaps")
    p.add_argument("--tiers", type=str, default="3,10,22",
                   help="Comma-separated tier thresholds in pixels (e.g. '3,10,22')")
    return p.parse_args()


def _relabel(err, tiers):
    """Return the categorical label for a single pair given the new tier set."""
    if err is None or (isinstance(err, float) and np.isnan(err)):
        # NaN err means either no transform or ungradable polygon — both stay no_match
        return "no_match"
    return categorize_result(
        has_transform=True,
        mean_corner_error=float(err),
        accuracy_tiers_px=tiers,
    )


def main():
    args = parse_args()
    tiers = tuple(float(t) for t in args.tiers.split(","))
    print(f"Regenerating report with tier set: {tiers}")

    src_csv = args.results_dir / "aggregate_results.csv"
    if not src_csv.exists():
        sys.exit(f"Error: {src_csv} not found")

    df = pd.read_csv(src_csv)

    for attempt in ("no_mask", "with_mask"):
        err_col = f"{attempt}_err"
        result_col = f"{attempt}_result"
        result_label_col = f"{attempt}_result_label"
        if err_col not in df.columns:
            print(f"  Skipping {attempt} (no '{err_col}' column)")
            continue

        old_result = df[result_col] if result_col in df.columns else None
        new_result = []
        for i in range(len(df)):
            old = old_result.iat[i] if old_result is not None else None
            # Preserve no_match where the original was no_match (no transform produced).
            # For non-no_match rows, re-categorise from the raw err.
            if old == "no_match" or pd.isna(old):
                new_result.append("no_match")
            else:
                new_result.append(_relabel(df[err_col].iat[i], tiers))

        df[result_col] = new_result
        if result_label_col in df.columns:
            df[result_label_col] = new_result

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "aggregate_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Wrote relabelled CSV -> {out_csv}")

    print(f"  Generating report -> {args.output_dir}")
    write_summary_report(out_csv, args.output_dir)
    print("  Done.")


if __name__ == "__main__":
    main()
