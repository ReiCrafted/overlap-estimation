"""reporting.py — Per-pair JSON, aggregate CSV, markdown summary, and heatmaps.

The markdown report covers headline numbers (overall mAA/Precision split by
estimator × attempt, per-config scoreboards, fallback benefit) as tables.
Heatmap PNGs render the detector × descriptor matrices visually: one PNG per
(metric × estimator × attempt) = 12 files for a full PROSAC + USAC_MAGSAC sweep.

Row/column ordering on every heatmap is by descending **per-axis average** of
the metric being plotted, so the strongest configurations sit in the top-left.
Colour scales:
  * mAA       → blue–white–red  (matplotlib "bwr")
  * Precision → green–white–red (custom LinearSegmentedColormap)
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from overlap_detection.types import PairResult


_PRECISION_CMAP = LinearSegmentedColormap.from_list(
    "gwr", ["green", "white", "red"],
)
_MAA_CMAP = "bwr"   # built-in blue-white-red


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

# Three attempt slices the report breaks out separately.  The third is
# derived (per-row pick of the better attempt) and only meaningful when
# both single-attempt columns are populated.
_ATTEMPTS: tuple[str, ...] = ("no_mask", "with_mask", "best_of_both")


def _add_best_of_both_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``best_of_both_result`` column to ``df`` (in place / returned).

    Computed only for rows where both ``no_mask_result`` and
    ``with_mask_result`` are non-null.  Other rows get NaN, which makes all
    downstream helpers skip them correctly.
    """
    if "best_of_both_result" in df.columns:
        return df
    if not {"no_mask_result", "with_mask_result"}.issubset(df.columns):
        df["best_of_both_result"] = pd.NA
        return df
    paired = df.dropna(subset=["no_mask_result", "with_mask_result"])
    df["best_of_both_result"] = pd.NA
    if not paired.empty:
        df.loc[paired.index, "best_of_both_result"] = _best_of_both(
            paired["no_mask_result"], paired["with_mask_result"]
        ).values
    return df


def _attempt_col(attempt: str) -> str:
    return f"{attempt}_result"


def _precision(series: pd.Series) -> float:
    """Fraction of *emitted* transforms that cleared the loosest accuracy tier.

    `precision = (acc_at_T for any T) / (acc_at_T for any T  +  false_match)`
    Equivalently `1 - false_match / (1 - no_match)`.  ``no_match`` rows are
    excluded from both numerator and denominator — they're abstentions, not
    wrong answers.  Returns ``nan`` when nothing was emitted (precision
    undefined).
    """
    series = series.dropna()
    if series.empty:
        return float("nan")
    emitted = series != "no_match"
    n_emitted = int(emitted.sum())
    if n_emitted == 0:
        return float("nan")
    n_correct = int((emitted & (series != "false_match")).sum())
    return n_correct / n_emitted


def _maa_for(series: pd.Series, tiers: list[float]) -> dict:
    """Convenience: a single attempt's slice → {n, maa, precision, acc@T..., false, no}."""
    series = series.dropna()
    if series.empty:
        return {"n": 0}
    out: dict = {
        "n": int(len(series)),
        "maa": _maa(series, tiers),
        "precision": _precision(series),
        "false_match": _share(series, "false_match"),
        "no_match": _share(series, "no_match"),
    }
    for t in tiers:
        out[f"acc_at_{t:g}"] = _cumulative_acc_rate(series, t)
    return out


# ---------- Section: overall (estimator × attempt) ----------


def _render_overall(df: pd.DataFrame, estimators: list[str],
                    tiers: list[float]) -> list[str]:
    """Single cross-table: rows = (estimator, attempt), columns = headline
    stats.  One look summarises the entire experiment."""
    out: list[str] = ["## Overall\n"]
    out.append(f"Total CSV rows: **{len(df)}**.")
    out.append(
        f"Accuracy tiers (px): **{', '.join(f'{t:g}' for t in tiers)}**.\n"
    )

    headers = ["Estimator", "Attempt", "Pairs", "mAA", "Precision",
               *(f"acc@{t:g}" for t in tiers),
               "false_match", "no_match"]
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "---|" * len(headers))

    for est in estimators:
        sub = df[df["estimator"] == est]
        for attempt in _ATTEMPTS:
            col = _attempt_col(attempt)
            if col not in sub.columns:
                continue
            stats = _maa_for(sub[col], tiers)
            if stats.get("n", 0) == 0:
                continue
            prec = stats["precision"]
            cells = [
                est, attempt, str(stats["n"]),
                f"{stats['maa']:.3f}",
                "N/A" if pd.isna(prec) else f"{prec:.3f}",
                *(f"{stats[f'acc_at_{t:g}']:.3f}" for t in tiers),
                f"{stats['false_match']:.3f}",
                f"{stats['no_match']:.3f}",
            ]
            out.append("| " + " | ".join(cells) + " |")
    return out


# ---------- Section: per-config scoreboard (one table per estimator) ----------


def _per_config_summary(df: pd.DataFrame, estimator: str,
                        tiers: list[float]) -> pd.DataFrame:
    """One row per (detector, descriptor) within a fixed estimator, columns
    summarising each of the three attempts (mAA, per-tier rates, false/no)."""
    sub = df[df["estimator"] == estimator]
    if sub.empty:
        return pd.DataFrame()

    records: list[dict] = []
    for (det, desc), group in sub.groupby(["detector", "descriptor"], sort=False):
        rec: dict = {
            "config": f"{det}+{desc}",
            "n_pairs": int(len(group)),
        }
        for attempt in _ATTEMPTS:
            col = _attempt_col(attempt)
            if col not in group.columns:
                continue
            stats = _maa_for(group[col], tiers)
            if stats.get("n", 0) == 0:
                continue
            rec[f"maa_{attempt}"] = stats["maa"]
            rec[f"precision_{attempt}"] = stats["precision"]
            for t in tiers:
                rec[f"acc_at_{t:g}_{attempt}"] = stats[f"acc_at_{t:g}"]
            rec[f"false_match_{attempt}"] = stats["false_match"]
            rec[f"no_match_{attempt}"] = stats["no_match"]
        records.append(rec)

    return pd.DataFrame(records)


def _render_scoreboard_block(df: pd.DataFrame, estimator: str,
                             tiers: list[float]) -> list[str]:
    per_cfg = _per_config_summary(df, estimator, tiers)
    if per_cfg.empty:
        return []
    # Sort by best_of_both mAA when available, else with_mask, else no_mask.
    for key in ("maa_best_of_both", "maa_with_mask", "maa_no_mask"):
        if key in per_cfg.columns:
            per_cfg = per_cfg.sort_values(key, ascending=False, na_position="last")
            sort_label = key.removeprefix("maa_")
            break
    else:
        sort_label = "(unsorted)"

    def header_label(stat: str) -> str:
        if stat == "maa":
            return "mAA"
        if stat == "precision":
            return "Prec"
        if stat.startswith("acc_at_"):
            return f"acc@{stat[len('acc_at_'):]}"
        return stat

    header_cells = ["Configuration", "Pairs"]
    cols: list[tuple[str, str]] = []
    for attempt in _ATTEMPTS:
        for stat in ["maa", "precision",
                     *(f"acc_at_{t:g}" for t in tiers),
                     "false_match", "no_match"]:
            colname = f"{stat}_{attempt}"
            if colname in per_cfg.columns:
                cols.append((colname, f"{header_label(stat)}<sub>{attempt}</sub>"))
    header_cells.extend(h for _, h in cols)

    out = [f"\n### {estimator}\n",
           f"Sorted by mAA<sub>{sort_label}</sub> (descending). "
           f"{len(per_cfg)} detector+descriptor combinations.\n"]
    out.append("| " + " | ".join(header_cells) + " |")
    out.append("|" + "---|" * len(header_cells))
    for _, row in per_cfg.iterrows():
        cells = [str(row["config"]), str(int(row["n_pairs"]))]
        for colname, _ in cols:
            v = row.get(colname)
            cells.append("N/A" if pd.isna(v) else f"{v:.3f}")
        out.append("| " + " | ".join(cells) + " |")
    return out


def _render_scoreboards(df: pd.DataFrame, estimators: list[str],
                        tiers: list[float]) -> list[str]:
    out = ["\n## Per-configuration scoreboard"]
    out.append("One table per estimator. Configurations are detector+descriptor; "
               "each attempt gets its own mAA / acc@T / false_match / no_match columns.\n")
    for est in estimators:
        out.extend(_render_scoreboard_block(df, est, tiers))
    return out


# ---------- Section: detector × descriptor mAA matrices ----------


def _metric_matrix(df: pd.DataFrame, estimator: str, attempt: str,
                   metric: str, tiers: list[float]) -> pd.DataFrame:
    """Wide DataFrame: index=detector, columns=descriptor, values=<metric>.

    ``metric`` is either ``"maa"`` or ``"precision"``.
    """
    col = _attempt_col(attempt)
    if col not in df.columns:
        return pd.DataFrame()
    sub = df[df["estimator"] == estimator]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for (det, desc), group in sub.groupby(["detector", "descriptor"], sort=False):
        series = group[col].dropna()
        if series.empty:
            value = float("nan")
        elif metric == "maa":
            value = _maa(series, tiers)
        elif metric == "precision":
            value = _precision(series)
        else:
            raise ValueError(f"Unknown metric: {metric!r}")
        rows.append({"detector": det, "descriptor": desc, "value": value})
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    return long.pivot(index="detector", columns="descriptor", values="value")


def _order_by_average(table: pd.DataFrame) -> pd.DataFrame:
    """Reorder rows and columns by descending mean of the metric.

    NaN cells are ignored when computing the averages.  Rows/columns whose
    average is entirely NaN sink to the bottom/right.
    """
    if table.empty:
        return table
    row_order = (
        table.mean(axis=1, skipna=True)
        .sort_values(ascending=False, na_position="last")
        .index
    )
    col_order = (
        table.mean(axis=0, skipna=True)
        .sort_values(ascending=False, na_position="last")
        .index
    )
    return table.loc[row_order, col_order]


def _save_heatmap(table: pd.DataFrame, path: Path, *, title: str,
                  cmap, vmin: float = 0.0, vmax: float = 1.0) -> None:
    """Render a heatmap with annotated cell values and save to ``path``.

    Cell text colour adapts to the underlying value — extreme-end values
    (where the colormap is darkest) get white text; mid-range cells get
    black text — keeps annotations legible on both diverging palettes.
    """
    if table.empty:
        return
    n_rows, n_cols = table.shape
    fig, ax = plt.subplots(
        figsize=(max(6, n_cols * 0.95 + 1.5), max(4, n_rows * 0.55 + 1.0))
    )
    im = ax.imshow(table.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(table.columns, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(table.index)
    for i in range(n_rows):
        for j in range(n_cols):
            v = table.iat[i, j]
            if pd.isna(v):
                ax.text(j, i, "—", ha="center", va="center",
                        color="black", fontsize=8)
                continue
            # Use white text on dark extremes, black near the colormap centre.
            text_color = "white" if abs(v - (vmin + vmax) / 2) > 0.35 * (vmax - vmin) else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=text_color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Descriptor")
    ax.set_ylabel("Detector")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _heatmap_filename(metric: str, estimator: str, attempt: str) -> str:
    return f"heatmap_{metric}_{estimator}_{attempt}.png"


def _render_matrices_section(df: pd.DataFrame, estimators: list[str],
                             tiers: list[float], metric: str,
                             output_dir: Path) -> list[str]:
    """One unified renderer for both mAA and Precision matrix sections.

    Writes one heatmap PNG per (estimator × attempt) into ``output_dir`` and
    emits markdown that embeds each PNG followed by the same data as a table
    (table uses the same row/column ordering, so the heatmap and the lookup
    table line up cell-for-cell).
    """
    pretty_metric = "mAA" if metric == "maa" else "Precision"
    cmap = _MAA_CMAP if metric == "maa" else _PRECISION_CMAP

    out: list[str] = [f"\n## {pretty_metric} matrices (detector × descriptor)"]
    out.append(
        f"One heatmap per (estimator × attempt). Rows/columns are sorted by "
        f"descending mean {pretty_metric}, so the strongest detectors sit at "
        f"the top and the strongest descriptors at the left. "
        + ("Colour: blue (low) → white → red (high)."
           if metric == "maa"
           else "Colour: green (low) → white → red (high).")
        + "\n"
    )

    any_section = False
    for est in estimators:
        for attempt in _ATTEMPTS:
            table = _metric_matrix(df, est, attempt, metric, tiers)
            if table.empty or table.isna().all().all():
                continue
            ordered = _order_by_average(table)

            png_name = _heatmap_filename(metric, est, attempt)
            _save_heatmap(
                ordered, output_dir / png_name,
                title=f"{pretty_metric} — {est} / {attempt}",
                cmap=cmap, vmin=0.0, vmax=1.0,
            )

            out.append(f"\n### {est} — {attempt}\n")
            out.append(f"![{pretty_metric} {est} {attempt}](./{png_name})\n")
            # Embed the same numbers as a table for precise lookup.
            descriptors = list(ordered.columns)
            out.append("| Detector | " + " | ".join(descriptors) + " |")
            out.append("|" + "---|" * (len(descriptors) + 1))
            for det, row in ordered.iterrows():
                cells = [
                    f"{row[d]:.3f}" if d in row and not pd.isna(row[d]) else "N/A"
                    for d in descriptors
                ]
                out.append(f"| {det} | " + " | ".join(cells) + " |")
            any_section = True

    return out if any_section else []


# ---------- Section: fallback benefit (one table per estimator) ----------


def _fallback_benefit_table(df: pd.DataFrame, estimator: str,
                            tiers: list[float]) -> pd.DataFrame:
    sub = df[df["estimator"] == estimator]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for (det, desc), group in sub.groupby(["detector", "descriptor"], sort=False):
        if not {"no_mask_result", "with_mask_result", "best_of_both_result"}.issubset(group.columns):
            continue
        paired = group.dropna(subset=["no_mask_result", "with_mask_result"])
        if paired.empty:
            continue
        nm = _maa(paired["no_mask_result"], tiers)
        wm = _maa(paired["with_mask_result"], tiers)
        bo = _maa(paired["best_of_both_result"].dropna(), tiers)
        rows.append({
            "config": f"{det}+{desc}",
            "maa_no_mask": nm,
            "maa_with_mask": wm,
            "maa_best": bo,
            "lift_vs_no_mask": bo - nm,
            "lift_vs_with_mask": bo - wm,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        "lift_vs_with_mask", ascending=False, na_position="last",
    )


def _render_fallback_benefit(df: pd.DataFrame, estimators: list[str],
                             tiers: list[float]) -> list[str]:
    out = ["\n## Fallback benefit (best_of_both vs. single attempt)"]
    out.append(
        "For each detector+descriptor combo, how much would mAA improve if "
        "the policy ran `mask_mode = both` and kept the better of the two "
        "attempts per pair? Δ < 0 means a single attempt is already as good "
        "as the picker. One table per estimator.\n"
    )
    any_table = False
    for est in estimators:
        table = _fallback_benefit_table(df, est, tiers)
        if table.empty:
            continue
        any_table = True
        out.append(f"\n### {est}\n")
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
    return out if any_table else []


# ---------- Entry point ----------


def write_summary_report(csv_path: Path, output_dir: Path) -> None:
    """Read ``aggregate_results.csv`` and emit ``report.md`` plus heatmap PNGs.

    Every section is split by **estimator** (``PROSAC``, ``USAC_MAGSAC``) and
    by **mask attempt** (``no_mask``, ``with_mask``, ``best_of_both``), on the
    assumption that PROSAC and USAC_MAGSAC are categorically different enough
    that pooling them obscures the picture.

    Sections:

    * **Overall** — single cross-table over (estimator × attempt) of mAA,
      Precision, per-tier accuracy rates, and false/no_match shares.
    * **Per-configuration scoreboard** — one table per estimator. Rows are
      detector+descriptor combos; columns are per-attempt mAA, Precision,
      acc@T, false_match, no_match.
    * **mAA matrices** — heatmap PNG + numeric table per (estimator × attempt),
      up to 6 of each. Detectors (rows) and descriptors (columns) sorted by
      descending mean mAA. Colour: blue → white → red.
    * **Precision matrices** — same layout as mAA matrices but coloured
      green → white → red.
    * **Fallback benefit** — one table per estimator. Per detector+descriptor
      combo, lift of the best_of_both policy over each single attempt.

    Heatmap PNGs are saved alongside ``report.md`` as
    ``heatmap_{maa|precision}_{estimator}_{attempt}.png``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _add_best_of_both_column(df)

    # Discover the accuracy tiers present anywhere in the data.
    tiers: list[float] = []
    for col in ("no_mask_result", "with_mask_result", "best_of_both_result"):
        if col in df.columns:
            for t in _tiers_present(df[col]):
                if t not in tiers:
                    tiers.append(t)
    tiers.sort()

    estimators = sorted(df["estimator"].dropna().unique().tolist()) if "estimator" in df.columns else []

    report: list[str] = ["# Overlap Detection Summary Report\n"]
    report.extend(_render_overall(df, estimators, tiers))
    report.extend(_render_scoreboards(df, estimators, tiers))
    report.extend(_render_matrices_section(df, estimators, tiers, "maa", output_dir))
    report.extend(_render_matrices_section(df, estimators, tiers, "precision", output_dir))
    report.extend(_render_fallback_benefit(df, estimators, tiers))

    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
