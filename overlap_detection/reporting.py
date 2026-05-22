"""reporting.py — Per-pair JSON, aggregate CSV, and markdown summary.

Plots are intentionally omitted in this revision; the only visualisation
that scales to an 11×9 detector/descriptor matrix is a heatmap, and we
design those separately.  The markdown report covers everything textually.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from overlap_detection.types import PairResult


# ---------------------------------------------------------------------------
# Writers (per-pair JSON + aggregate CSV)
# ---------------------------------------------------------------------------


def write_pair_json(
    result: PairResult,
    metrics: dict,
    output_dir: Path,
) -> Path:
    """Write a full per-attempt result + metrics blob to JSON.

    Filename: ``{pair_id}_{detector}_{descriptor}_{estimator}_{mask_mode}.json``
    where ``mask_mode`` is the concrete mode that actually ran (never
    ``"both"``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_id = f"{result.image_a_path.stem}_{result.image_b_path.stem}"
    filename = (
        f"{pair_id}_{result.detector}_{result.descriptor}"
        f"_{result.estimator}_{result.mask_mode}.json"
    )
    filepath = output_dir / filename

    data = {"result": asdict(result), "metrics": metrics}

    for k, v in data["result"].items():
        if isinstance(v, np.ndarray):
            data["result"][k] = v.tolist()
        elif isinstance(v, Path):
            data["result"][k] = str(v)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return filepath


def write_aggregate_csv(rows: list[dict], output_path: Path) -> None:
    """Write one CSV row per ``(pair, detector, descriptor, estimator,
    mask_mode_spec)`` experiment unit.  Columns may include both
    ``no_mask_*`` and ``with_mask_*`` blocks when ``mask_mode == "both"``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Categorical helpers
# ---------------------------------------------------------------------------


def _parse_tier(label: str) -> Optional[float]:
    """Extract the tier threshold from an ``"acc_at_<T>"`` label.

    Returns ``None`` for ``"false_match"``, ``"no_match"``, or anything
    that doesn't parse.
    """
    if not isinstance(label, str) or not label.startswith("acc_at_"):
        return None
    try:
        return float(label[len("acc_at_"):])
    except ValueError:
        return None


def _tiers_present(series: pd.Series) -> list[float]:
    """Discover the accuracy tiers that appear in a column of result labels.
    Returns them sorted ascending."""
    tiers: set[float] = set()
    for v in series.dropna().unique():
        t = _parse_tier(v)
        if t is not None:
            tiers.add(t)
    return sorted(tiers)


def _cumulative_acc_rate(series: pd.Series, threshold: float) -> float:
    """Fraction of rows whose label is ``"acc_at_<t>"`` with ``t ≤ threshold``.

    Cumulative because hitting a tighter tier implies clearing a looser one.
    """
    def cleared(label) -> bool:
        t = _parse_tier(label)
        return t is not None and t <= threshold
    cleared_mask = series.apply(cleared)
    return float(cleared_mask.mean()) if len(series) else 0.0


def _maa(series: pd.Series, tiers: list[float]) -> float:
    """Mean Average Accuracy: mean across the configured tiers of the
    per-tier cumulative pass rate.  Equivalent to averaging, over both
    pairs and thresholds, the boolean "this pair cleared this threshold"."""
    if not tiers:
        return float("nan")
    return float(np.mean([_cumulative_acc_rate(series, t) for t in tiers]))


def _share(series: pd.Series, value: str) -> float:
    return float((series == value).mean()) if len(series) else 0.0


def _best_of_both(no_mask: pd.Series, with_mask: pd.Series) -> pd.Series:
    """Per row, pick the label of the better attempt.

    Ordering (best → worst): smallest tier value (best) > larger tier values >
    ``"false_match"`` > ``"no_match"``.  Useful for answering "what would a
    fallback policy that picks the better of the two attempts achieve?".
    """
    def rank(label) -> tuple[int, float]:
        if label == "no_match" or label is None or (isinstance(label, float) and np.isnan(label)):
            return (3, 0.0)
        if label == "false_match":
            return (2, 0.0)
        t = _parse_tier(label)
        if t is None:
            return (3, 0.0)
        return (1, t)   # smaller tier value → better

    out = []
    for a, b in zip(no_mask, with_mask):
        a_rank = rank(a)
        b_rank = rank(b)
        out.append(a if a_rank <= b_rank else b)
    return pd.Series(out, index=no_mask.index)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


_ATTEMPTS = (("no_mask", "no_mask"), ("with_mask", "with_mask"))


def _config_id(df: pd.DataFrame) -> pd.Series:
    return df["detector"] + "+" + df["descriptor"] + "+" + df["estimator"]


def _per_attempt_summary(df: pd.DataFrame, attempt_col: str,
                         tiers: list[float]) -> dict:
    """Compute a one-attempt summary block (mAA, per-tier rates, false/no_match
    shares) over the full dataframe."""
    if attempt_col not in df.columns:
        return {}
    series = df[attempt_col].dropna()
    if series.empty:
        return {}
    out = {
        "n": int(len(series)),
        "maa": _maa(series, tiers),
        "false_match": _share(series, "false_match"),
        "no_match": _share(series, "no_match"),
    }
    for t in tiers:
        out[f"acc_at_{t:g}"] = _cumulative_acc_rate(series, t)
    return out


def _per_config_summary(df: pd.DataFrame, tiers: list[float]) -> pd.DataFrame:
    """One row per ``(detector, descriptor, estimator)`` config, columns
    summarising both attempts (mAA, per-tier rates, false/no_match)."""
    df = df.copy()
    df["config"] = _config_id(df)

    records: list[dict] = []
    for config, group in df.groupby("config", sort=False):
        rec: dict = {"config": config, "n_pairs": int(len(group))}
        for attempt_label, prefix in _ATTEMPTS:
            col = f"{prefix}_result"
            if col not in group.columns:
                continue
            series = group[col].dropna()
            if series.empty:
                continue
            rec[f"maa_{attempt_label}"] = _maa(series, tiers)
            for t in tiers:
                rec[f"acc_at_{t:g}_{attempt_label}"] = _cumulative_acc_rate(series, t)
            rec[f"false_match_{attempt_label}"] = _share(series, "false_match")
            rec[f"no_match_{attempt_label}"] = _share(series, "no_match")

        # Best-of-both: only meaningful when both attempts ran in lockstep.
        if {"no_mask_result", "with_mask_result"}.issubset(group.columns):
            paired = group.dropna(subset=["no_mask_result", "with_mask_result"])
            if not paired.empty:
                best = _best_of_both(paired["no_mask_result"], paired["with_mask_result"])
                rec["maa_best_of_both"] = _maa(best, tiers)
                for t in tiers:
                    rec[f"acc_at_{t:g}_best_of_both"] = _cumulative_acc_rate(best, t)
        records.append(rec)

    return pd.DataFrame(records)


def _heatmap_table(df: pd.DataFrame, attempt: str, tiers: list[float]) -> pd.DataFrame:
    """Detector × descriptor matrix of mAA for the chosen attempt
    (``"no_mask"`` or ``"with_mask"``).  Returned as a wide DataFrame so it
    can be rendered as a markdown table now and later piped into a heatmap."""
    col = f"{attempt}_result"
    if col not in df.columns:
        return pd.DataFrame()
    rows = []
    for (det, desc), group in df.groupby(["detector", "descriptor"], sort=False):
        series = group[col].dropna()
        rows.append({
            "detector": det,
            "descriptor": desc,
            "maa": _maa(series, tiers) if not series.empty else float("nan"),
        })
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    return long.pivot(index="detector", columns="descriptor", values="maa")


def _render_overall(df: pd.DataFrame, tiers: list[float]) -> list[str]:
    out: list[str] = ["## Overall\n"]
    out.append(f"Total CSV rows: **{len(df)}**\n")
    out.append(
        f"Accuracy tiers (px): **{', '.join(f'{t:g}' for t in tiers)}**\n"
    )
    for attempt_label, prefix in _ATTEMPTS:
        summary = _per_attempt_summary(df, f"{prefix}_result", tiers)
        if not summary:
            continue
        out.append(f"\n### {attempt_label}\n")
        out.append(f"Attempts scored: {summary['n']}")
        out.append(f"- **mAA**: {summary['maa']:.3f}")
        for t in tiers:
            out.append(f"- acc@{t:g} px: {summary[f'acc_at_{t:g}']:.3f}")
        out.append(f"- false_match: {summary['false_match']:.3f}")
        out.append(f"- no_match: {summary['no_match']:.3f}")
    return out


def _render_scoreboard(per_cfg: pd.DataFrame, tiers: list[float]) -> list[str]:
    if per_cfg.empty:
        return []
    sort_key = "maa_with_mask" if "maa_with_mask" in per_cfg.columns else "maa_no_mask"
    per_cfg = per_cfg.sort_values(sort_key, ascending=False, na_position="last")

    header_cells = ["Configuration", "Pairs"]
    cols: list[tuple[str, str]] = []   # (column, header)
    for attempt_label, _ in _ATTEMPTS:
        for stat in ["maa", *(f"acc_at_{t:g}" for t in tiers), "false_match", "no_match"]:
            colname = f"{stat}_{attempt_label}"
            if colname in per_cfg.columns:
                cols.append((colname, f"{stat}<sub>{attempt_label}</sub>"))
    if "maa_best_of_both" in per_cfg.columns:
        cols.append(("maa_best_of_both", "mAA<sub>best</sub>"))
        for t in tiers:
            colname = f"acc_at_{t:g}_best_of_both"
            if colname in per_cfg.columns:
                cols.append((colname, f"acc@{t:g}<sub>best</sub>"))

    header_cells.extend(h for _, h in cols)

    lines = ["## Per-configuration scoreboard\n"]
    lines.append("Sorted by mAA on the with_mask attempt (descending).\n")
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "---|" * len(header_cells))
    for _, row in per_cfg.iterrows():
        cells = [str(row["config"]), str(int(row["n_pairs"]))]
        for colname, _ in cols:
            v = row.get(colname)
            cells.append("N/A" if pd.isna(v) else f"{v:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _render_heatmap_table(df: pd.DataFrame, tiers: list[float],
                          attempt: str) -> list[str]:
    table = _heatmap_table(df, attempt, tiers)
    if table.empty:
        return []
    descriptors = list(table.columns)
    out = [f"\n## mAA matrix — {attempt}\n"]
    out.append("| Detector | " + " | ".join(descriptors) + " |")
    out.append("|" + "---|" * (len(descriptors) + 1))
    for det, row in table.iterrows():
        cells = [
            f"{row[d]:.3f}" if d in row and not pd.isna(row[d]) else "N/A"
            for d in descriptors
        ]
        out.append(f"| {det} | " + " | ".join(cells) + " |")
    return out


def _render_fallback_benefit(per_cfg: pd.DataFrame) -> list[str]:
    if "maa_best_of_both" not in per_cfg.columns:
        return []
    rows = []
    for _, row in per_cfg.iterrows():
        best = row.get("maa_best_of_both")
        wm = row.get("maa_with_mask")
        nm = row.get("maa_no_mask")
        if pd.isna(best):
            continue
        rows.append({
            "config": row["config"],
            "maa_no_mask": nm,
            "maa_with_mask": wm,
            "maa_best": best,
            "lift_vs_no_mask": (best - nm) if pd.notna(nm) else float("nan"),
            "lift_vs_with_mask": (best - wm) if pd.notna(wm) else float("nan"),
        })
    if not rows:
        return []
    table = pd.DataFrame(rows).sort_values(
        "lift_vs_with_mask", ascending=False, na_position="last",
    )
    out = ["\n## Fallback benefit (best-of-both vs. single attempt)\n"]
    out.append(
        "If a pipeline ran `mask_mode = both` and the policy picked whichever "
        "attempt landed in the better tier, how much would mAA improve over "
        "running just one attempt? Sorted by lift over the with_mask attempt."
        "\n"
    )
    out.append("| Configuration | mAA<sub>no_mask</sub> | mAA<sub>with_mask</sub> "
               "| mAA<sub>best</sub> | Δ vs. no_mask | Δ vs. with_mask |")
    out.append("|---|---|---|---|---|---|")
    for _, row in table.iterrows():
        def fmt(v):
            return "N/A" if pd.isna(v) else f"{v:.3f}"
        out.append(
            f"| {row['config']} | {fmt(row['maa_no_mask'])} | "
            f"{fmt(row['maa_with_mask'])} | {fmt(row['maa_best'])} | "
            f"{fmt(row['lift_vs_no_mask'])} | {fmt(row['lift_vs_with_mask'])} |"
        )
    return out


def write_summary_report(csv_path: Path, output_dir: Path) -> None:
    """Read ``aggregate_results.csv`` and emit ``report.md``.

    No plots are written — this revision is markdown-only.  The report
    covers:

    * Overall mAA, per-tier accuracy rates, and false/no_match shares per
      mask attempt.
    * Per-configuration scoreboard with paired columns for each attempt and
      a derived ``best_of_both`` column.
    * Detector × descriptor mAA matrix for each attempt.
    * "Fallback benefit" table: lift of best-of-both over each single
      attempt, per configuration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Discover the accuracy tiers present in the data; gives a stable header
    # set without requiring the report to be told them up-front.
    tiers: list[float] = []
    for col in ("no_mask_result", "with_mask_result"):
        if col in df.columns:
            for t in _tiers_present(df[col]):
                if t not in tiers:
                    tiers.append(t)
    tiers.sort()

    per_cfg = _per_config_summary(df, tiers)

    report: list[str] = ["# Overlap Detection Summary Report\n"]
    report.extend(_render_overall(df, tiers))
    report.extend(_render_scoreboard(per_cfg, tiers))
    report.extend(_render_heatmap_table(df, tiers, "no_mask"))
    report.extend(_render_heatmap_table(df, tiers, "with_mask"))
    report.extend(_render_fallback_benefit(per_cfg))

    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
