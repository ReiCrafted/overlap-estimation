"""reporting.py — Per-pair JSON, aggregate CSV, markdown summary, and heatmaps.

The markdown report covers headline numbers (overall mAA-OP / Precision split
by estimator × attempt, per-config scoreboards, fallback benefit) as tables.
Heatmap PNGs render the detector × descriptor matrices visually: one PNG per
(metric × estimator × attempt) = 12 files for a full PROSAC + USAC_MAGSAC sweep.

Row/column ordering on every heatmap is by descending **per-axis average** of
the metric being plotted, so the strongest configurations sit in the top-left.
Colour scales:
  * mAA-OP    → red–white–blue  (matplotlib "bwr_r")
  * Precision → red–white–green (custom LinearSegmentedColormap)

mAA-OP definition (AUC form, overlap-polygon variant)
-----------------------------------------------------
"mAA-OP" stands for **mean Average Accuracy on the Overlap Polygon**.  Same
AUC aggregation as the SuperGlue / glue-factory mAA, but the underlying
per-pair error is the mean per-vertex distance on the **clipped overlap
polygon** (see `metrics.corner_errors_overlap_polygon`), not the
HPatches-convention four-image-corner reprojection.  Numbers are therefore
NOT directly comparable to published image-matching tables — see
project_overview.md §Stage 7 for the rationale.

mAA-OP is computed from the per-pair ``{attempt}_err`` columns in the CSV,
not from the ordinal result labels.

For each configured accuracy tier threshold T (default 3, 10, 22 px):

    AUC@T = (1/T) * ∫₀ᵀ recall(ε) dε

where recall(ε) is the fraction of pairs whose corner error ≤ ε.  The
integral is evaluated exactly via the trapezoidal rule on the sorted error
values.  mAA-OP = mean(AUC@T₁, AUC@T₂, …).

Failures (no transform produced → NaN corner error) are mapped to infinite
error before sorting.  They count in the recall denominator but never reach
any finite threshold, so each failure reduces the score proportionally.
This is identical to the glue-factory convention (``error = float("inf")``
for estimation failures).

Difference from binary-accuracy averaging
  The standard AUC rewards accuracy *within* each threshold window: a pair
  with error 0.5 px contributes more than one with error 2.9 px even though
  both clear a 3 px threshold in a binary sense.  Binary averaging
  (mean of acc@T₁, acc@T₂, …) loses this within-tier information.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from overlap_detection.types import PairResult


# matplotlib is intentionally NOT imported at module load — it's only needed
# by the heatmap renderer, and importing it eagerly adds ~1-2 s to every
# orchestrator worker startup (workers only ever call write_pair_json).
# The lazy-import helpers below return matplotlib objects on first call.

def _matplotlib_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _precision_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("rwg", ["red", "white", "green"])


def _pcr_cmap():
    """Diverging colormap red → white → magenta for the PCR matrices.
    Used with a linear ``Normalize(0, 1)``, so white sits at exactly 0.5
    (the fixed midpoint of the metric's bounded range)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("rwm", ["red", "white", "magenta"])


def _match_rate_cmap():
    """Diverging colormap red → white → azure (dodgerblue) for the
    match-rate matrices.  Used with a linear ``Normalize(0, 1)``, so
    white sits at exactly 0.5."""
    from matplotlib.colors import LinearSegmentedColormap
    # "azure" in CSS is near-white (#F0FFFF); use a saturated azure-blue
    # so the high end of the colormap is visually distinct from the white
    # midpoint.
    return LinearSegmentedColormap.from_list("rwa", ["red", "white", "dodgerblue"])


_MAA_CMAP_NAME = "bwr_r"   # matplotlib built-in: red-white-blue


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


def _maa(errors: pd.Series, tiers: list[float]) -> float:
    """mean Average Accuracy on the Overlap Polygon (mAA-OP).

    Same AUC aggregation as SuperGlue / glue-factory mAA, but the underlying
    per-pair error is the mean per-vertex distance on the clipped overlap
    polygon (not the HPatches-convention four-image-corner error).  For
    each tier threshold T, computes AUC@T — the area under the
    recall-vs-error curve from 0 to T, normalised by T.  mAA-OP is the mean
    of these per-threshold AUC values.

    Failures (NaN error — no transform produced) map to infinite error:
    they count in the recall denominator but never reach any threshold,
    so each failure reduces the score proportionally.  This is identical
    to how glue-factory handles estimation failures (error = inf).
    """
    if not tiers or errors.empty:
        return float("nan")
    err = errors.fillna(float("inf")).to_numpy(dtype=float)
    sorted_err = np.r_[0.0, np.sort(err)]
    recall     = np.r_[0.0, (np.arange(len(err)) + 1) / len(err)]
    aucs = []
    for t in tiers:
        last = int(np.searchsorted(sorted_err, t))
        e = np.r_[sorted_err[:last], t]
        r = np.r_[recall[:last],     recall[last - 1]]
        aucs.append(float(np.trapezoid(r, x=e) / t))
    return float(np.mean(aucs))


def _share(series: pd.Series, value: str) -> float:
    return float((series == value).mean()) if len(series) else 0.0


def _best_of_both(no_mask: pd.Series, with_mask: pd.Series) -> pd.Series:
    """Per row, pick the label of the better attempt.

    Ordering (best → worst): smallest tier value (best) > larger tier values >
    ``"no_match"`` > ``"false_match"``.  ``no_match`` ranks above
    ``"false_match"`` because ``no_match`` is neutral for Precision (excluded
    from both numerator and denominator), whereas ``false_match`` is an
    incorrect emission that actively lowers Precision.  For mAA-OP both
    contribute zero AUC at the standard tier sizes (> 10 px error / inf), so
    this ordering has no effect on mAA-OP.
    """
    def rank(label) -> tuple[int, float]:
        if pd.isna(label) or label == "no_match":
            return (3, 0.0)
        if label == "false_match":
            return (3, 1.0)   # worse than no_match for Precision; same for mAA-OP
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


def _error_col(attempt: str) -> str:
    return f"{attempt}_err"


def _pcr_col(attempt: str) -> str:
    return f"{attempt}_pixel_correspondence_rate"


def _match_rate(label_series: pd.Series) -> float:
    """Fraction of rows whose label is **anything other than** ``no_match`` —
    i.e. the share of pairs for which the pipeline produced some transform,
    whether or not that transform was accurate.  ``acc_at_*`` and
    ``false_match`` both count as 'a transform was produced'.  Returns 0 on
    an empty series."""
    if label_series.empty:
        return 0.0
    return float((label_series != "no_match").mean())


def _add_best_of_both_err_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``best_of_both_err``: the smaller of (no_mask_err, with_mask_err)
    per row, with NaN treated as infinity.

    Picked by raw error value, **not** by label rank, because labels are
    tier-level (``acc_at_3``/``acc_at_5``/…) while mAA-OP cares about
    continuous within-tier accuracy.  Two attempts that both land in
    ``acc_at_3`` (e.g. 2.5 px vs. 0.5 px) tie on label rank, and the
    label-picker would default to no_mask's 2.5 — discarding the 0.5
    even though it contributes more to AUC@3.  Picking ``min(err)``
    instead guarantees ``mAA-OP_best_of_both ≥ max(mAA-OP_no_mask,
    mAA-OP_with_mask)`` for every cell.

    Rows where one attempt did not run (label NaN) get NaN; rows where
    both errs are NaN also get NaN.  Otherwise NaN is filled with inf so
    a real measurement always beats a missing one in the min.
    """
    if "best_of_both_err" in df.columns:
        return df
    needed = {"no_mask_err", "with_mask_err", "no_mask_result", "with_mask_result"}
    if not needed.issubset(df.columns):
        df["best_of_both_err"] = pd.NA
        return df

    nm = pd.to_numeric(df["no_mask_err"], errors="coerce")
    wm = pd.to_numeric(df["with_mask_err"], errors="coerce")
    both_ran = df["no_mask_result"].notna() & df["with_mask_result"].notna()
    both_err_nan = nm.isna() & wm.isna()

    picked = np.minimum(nm.fillna(float("inf")), wm.fillna(float("inf")))
    # Rows where neither attempt produced an error → keep NaN (not inf).
    picked = picked.where(~both_err_nan, other=float("nan"))
    # Rows where only one attempt ran → NaN (not a valid best_of_both).
    picked = picked.where(both_ran, other=float("nan"))

    df["best_of_both_err"] = picked
    return df


def _add_best_of_both_pcr_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``best_of_both_pixel_correspondence_rate``: the larger of
    (no_mask_pcr, with_mask_pcr) per row, with NaN treated as ``-inf``.

    Picked by raw PCR value, **not** by label rank, for the same reason
    as :func:`_add_best_of_both_err_column`: labels are tier-level and
    discard the sub-label PCR ordering that the metric actually cares
    about.  Picking ``max(pcr)`` guarantees ``PCR_best_of_both ≥
    max(PCR_no_mask, PCR_with_mask)`` for every cell.

    Rows where one attempt did not run get NaN; rows where both PCRs are
    NaN also get NaN.  Otherwise NaN is filled with -inf so a real
    measurement always beats a missing one in the max.
    """
    out_col = "best_of_both_pixel_correspondence_rate"
    if out_col in df.columns:
        return df
    no_mask_pcr = "no_mask_pixel_correspondence_rate"
    with_mask_pcr = "with_mask_pixel_correspondence_rate"
    needed = {no_mask_pcr, with_mask_pcr, "no_mask_result", "with_mask_result"}
    if not needed.issubset(df.columns):
        df[out_col] = pd.NA
        return df

    nm = pd.to_numeric(df[no_mask_pcr], errors="coerce")
    wm = pd.to_numeric(df[with_mask_pcr], errors="coerce")
    both_ran = df["no_mask_result"].notna() & df["with_mask_result"].notna()
    both_pcr_nan = nm.isna() & wm.isna()

    picked = np.maximum(nm.fillna(float("-inf")), wm.fillna(float("-inf")))
    # Rows where neither attempt produced a PCR → keep NaN (not -inf).
    picked = picked.where(~both_pcr_nan, other=float("nan"))
    # Rows where only one attempt ran → NaN.
    picked = picked.where(both_ran, other=float("nan"))

    df[out_col] = picked
    return df


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


def _maa_for(label_series: pd.Series, err_series: pd.Series,
             tiers: list[float]) -> dict:
    """Convenience: a single attempt's label + error slices → {n, maa, …}."""
    label_series = label_series.dropna()
    if label_series.empty:
        return {"n": 0}
    err_series = err_series.reindex(label_series.index)
    out: dict = {
        "n": int(len(label_series)),
        "maa": _maa(err_series, tiers),
        "precision": _precision(label_series),
        "false_match": _share(label_series, "false_match"),
        "no_match": _share(label_series, "no_match"),
    }
    for t in tiers:
        out[f"acc_at_{t:g}"] = _cumulative_acc_rate(label_series, t)
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

    headers = ["Estimator", "Attempt", "Pairs", "mAA-OP", "Precision",
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
            ecol = _error_col(attempt)
            err = sub[ecol] if ecol in sub.columns else pd.Series(dtype=float)
            stats = _maa_for(sub[col], err, tiers)
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
    summarising each of the three attempts (mAA-OP, per-tier rates, false/no)."""
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
            ecol = _error_col(attempt)
            err = group[ecol] if ecol in group.columns else pd.Series(dtype=float)
            stats = _maa_for(group[col], err, tiers)
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
    # Sort by best_of_both mAA-OP when available, else with_mask, else no_mask.
    for key in ("maa_best_of_both", "maa_with_mask", "maa_no_mask"):
        if key in per_cfg.columns:
            per_cfg = per_cfg.sort_values(key, ascending=False, na_position="last")
            sort_label = key.removeprefix("maa_")
            break
    else:
        sort_label = "(unsorted)"

    def header_label(stat: str) -> str:
        if stat == "maa":
            return "mAA-OP"
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
           f"Sorted by mAA-OP<sub>{sort_label}</sub> (descending). "
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
               "each attempt gets its own mAA-OP / acc@T / false_match / no_match columns.\n")
    for est in estimators:
        out.extend(_render_scoreboard_block(df, est, tiers))
    return out


# ---------- Section: detector × descriptor mAA-OP matrices ----------


def _metric_matrix(df: pd.DataFrame, estimator: str, attempt: str,
                   metric: str, tiers: list[float]) -> pd.DataFrame:
    """Wide DataFrame: index=detector, columns=descriptor, values=<metric>.

    ``metric`` is one of ``"maa"``, ``"precision"``, ``"pcr"``, ``"match_rate"``.
    """
    col = _attempt_col(attempt)
    if col not in df.columns:
        return pd.DataFrame()
    sub = df[df["estimator"] == estimator]
    if sub.empty:
        return pd.DataFrame()
    ecol = _error_col(attempt)
    pcr_col = _pcr_col(attempt)
    rows = []
    for (det, desc), group in sub.groupby(["detector", "descriptor"], sort=False):
        label_s = group[col].dropna()
        if label_s.empty:
            value = float("nan")
        elif metric == "maa":
            err_s = group[ecol].reindex(label_s.index) if ecol in group.columns else pd.Series(dtype=float)
            value = _maa(err_s, tiers)
        elif metric == "precision":
            value = _precision(label_s)
        elif metric == "pcr":
            if pcr_col in group.columns:
                pcr_s = pd.to_numeric(group[pcr_col], errors="coerce").dropna()
                value = float(pcr_s.mean()) if not pcr_s.empty else float("nan")
            else:
                value = float("nan")
        elif metric == "match_rate":
            if attempt == "best_of_both":
                # Match-rate for best_of_both = "did EITHER attempt produce a
                # transform?".  Bypass the best_of_both_result label column,
                # which is picked Precision-optimally (prefers no_match over
                # false_match to avoid wrong emissions) and would therefore
                # under-count match_rate.  Restricted to rows where both
                # attempts actually ran (label NaN means that attempt was
                # never executed for this row's mask_mode_spec).
                nm = group.get("no_mask_result")
                wm = group.get("with_mask_result")
                if nm is None or wm is None:
                    value = float("nan")
                else:
                    both_ran = nm.notna() & wm.notna()
                    if not both_ran.any():
                        value = float("nan")
                    else:
                        either_emitted = (
                            (nm[both_ran] != "no_match")
                            | (wm[both_ran] != "no_match")
                        )
                        value = float(either_emitted.mean())
            else:
                value = _match_rate(label_s)
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
                  cmap, vmin: float = 0.0, vmax: float = 1.0,
                  norm=None) -> None:
    """Render a heatmap with annotated cell values and save to ``path``.

    Cell text colour adapts to the underlying value — extreme-end cells
    (where the colormap is darkest) get white text; mid-range cells get
    black text — keeps annotations legible on both linear and
    median-centred (``TwoSlopeNorm``) palettes.

    Pass ``norm`` to use a non-linear normalisation (e.g.
    ``TwoSlopeNorm`` for median-centred diverging colormaps); when
    ``norm`` is given, ``vmin`` / ``vmax`` are ignored.
    """
    if table.empty:
        return
    plt = _matplotlib_pyplot()
    from matplotlib.colors import Normalize
    if norm is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    n_rows, n_cols = table.shape
    fig, ax = plt.subplots(
        figsize=(max(6, n_cols * 0.95 + 1.5), max(4, n_rows * 0.55 + 1.0))
    )
    im = ax.imshow(table.values, cmap=cmap, norm=norm, aspect="auto")
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
            # Normalised position in the colormap (0 = low extreme, 1 = high).
            # White text on dark extremes; black text near the centre (0.5).
            nv = float(norm(v))
            text_color = "white" if abs(nv - 0.5) > 0.35 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=text_color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Descriptor")
    ax.set_ylabel("Detector")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    _matplotlib_pyplot().close(fig)


def _heatmap_filename(metric: str, estimator: str, attempt: str) -> str:
    return f"heatmap_{metric}_{estimator}_{attempt}.png"


def _section_vmax(tables: list[pd.DataFrame], metric: str) -> float:
    """Return the shared ``vmax`` for every heatmap in one section.

    All four matrix types (mAA-OP, Precision, PCR, match_rate) are bounded
    in ``[0, 1]`` and share a fixed top of ``1.0`` so panels stay directly
    comparable both within a section and across sections.  ``tables`` is
    accepted for backward-compatibility but is no longer consulted.
    """
    return 1.0


def _render_matrices_section(df: pd.DataFrame, estimators: list[str],
                             tiers: list[float], metric: str,
                             output_dir: Path) -> list[str]:
    """One unified renderer for all four matrix sections (mAA-OP, Precision,
    PCR, match_rate).

    Writes one heatmap PNG per (estimator × attempt) into ``output_dir`` and
    emits markdown that embeds each PNG followed by the same data as a table
    (table uses the same row/column ordering, so the heatmap and the lookup
    table line up cell-for-cell).

    All heatmaps across every section share a single fixed linear colour
    scale `[0, 1]` (via :func:`_section_vmax`).  All four metrics
    (mAA-OP, Precision, PCR, match_rate) are bounded in `[0, 1]`, so a
    fixed top keeps panels directly comparable both within and across
    sections — and diverging colormaps put white at exactly 0.5.
    """
    if metric == "maa":
        pretty_metric = "mAA-OP"
        cmap = _MAA_CMAP_NAME
        colour_desc = "red (low) → white → blue (high)"
    elif metric == "precision":
        pretty_metric = "Precision"
        cmap = _precision_cmap()
        colour_desc = "red (0) → white (0.5) → green (1)"
    elif metric == "pcr":
        pretty_metric = "PCR"
        cmap = _pcr_cmap()
        colour_desc = "red (0) → white (0.5) → magenta (1)"
    elif metric == "match_rate":
        pretty_metric = "Match rate"
        cmap = _match_rate_cmap()
        colour_desc = "red (0) → white (0.5) → azure (1)"
    else:
        raise ValueError(f"Unknown matrix metric: {metric!r}")

    # Precision and PCR are both quality-given-emission metrics: each cell's
    # value is averaged only over pairs where that attempt actually emitted.
    # The per-pair best_of_both pick can therefore drag the cell mean below an
    # individual attempt's mean (different denominators), and the metric is an
    # oracle-policy view rather than something the runtime can produce.  Hide
    # the best_of_both panel for these two metrics — mAA-OP and match_rate
    # cover the "overall ceiling" question.  The scoreboard and Overall tables
    # still display best_of_both Precision for users who want the detail.
    if metric in ("precision", "pcr"):
        attempts_to_show = tuple(a for a in _ATTEMPTS if a != "best_of_both")
    else:
        attempts_to_show = _ATTEMPTS

    # Pass 1: collect all the ordered tables so we can pick a shared scale.
    populated: list[tuple[str, str, pd.DataFrame]] = []
    for est in estimators:
        for attempt in attempts_to_show:
            table = _metric_matrix(df, est, attempt, metric, tiers)
            if table.empty or table.isna().all().all():
                continue
            populated.append((est, attempt, _order_by_average(table)))

    if not populated:
        return []

    vmax = _section_vmax([t for _, _, t in populated], metric)

    out: list[str] = [f"\n## {pretty_metric} matrices (detector × descriptor)"]
    out.append(
        f"One heatmap per (estimator × attempt). Rows/columns are sorted by "
        f"descending mean {pretty_metric}, so the strongest detectors sit at "
        f"the top and the strongest descriptors at the left. "
        f"Colour: {colour_desc}. Colour scale: 0.0 → {vmax:.1f}.\n"
    )

    # Pass 2: render each table against the shared scale.
    for est, attempt, ordered in populated:
        png_name = _heatmap_filename(metric, est, attempt)
        _save_heatmap(
            ordered, output_dir / png_name,
            title=f"{pretty_metric} — {est} / {attempt}",
            cmap=cmap, vmin=0.0, vmax=vmax,
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

    return out


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
        nm = _maa(paired["no_mask_err"]       if "no_mask_err"       in paired.columns else pd.Series(dtype=float), tiers)
        wm = _maa(paired["with_mask_err"]     if "with_mask_err"     in paired.columns else pd.Series(dtype=float), tiers)
        bo = _maa(paired["best_of_both_err"]  if "best_of_both_err"  in paired.columns else pd.Series(dtype=float), tiers)
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
        "For each detector+descriptor combo, how much would mAA-OP improve if "
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
        out.append("| Configuration | mAA-OP<sub>no_mask</sub> | mAA-OP<sub>with_mask</sub> "
                   "| mAA-OP<sub>best</sub> | Δ vs. no_mask | Δ vs. with_mask |")
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

    * **Overall** — single cross-table over (estimator × attempt) of mAA-OP,
      Precision, per-tier accuracy rates, and false/no_match shares.
    * **Per-configuration scoreboard** — one table per estimator. Rows are
      detector+descriptor combos; columns are per-attempt mAA-OP, Precision,
      acc@T, false_match, no_match.
    * **mAA-OP matrices** — heatmap PNG + numeric table per (estimator × attempt),
      up to 6 of each. Detectors (rows) and descriptors (columns) sorted by
      descending mean mAA-OP. Colour: red (low) → white → blue (high),
      linear ``Normalize(0, 1)`` — same fixed `[0, 1]` scale as every other
      matrix section.
    * **Precision matrices** — same layout as mAA-OP matrices but **only
      shows `no_mask` and `with_mask` panels** (best_of_both is omitted
      from this matrix section because Precision is an emission-conditional
      diagnostic and best_of_both Precision represents an oracle abstention
      policy that adds little over the individual views; best_of_both
      Precision is still shown in the scoreboard and Overall tables).
      Coloured red (low) → white → green (high).
    * **PCR matrices** — same layout as Precision (no_mask and with_mask
      only — best_of_both omitted), aggregating per-pair
      ``pixel_correspondence_rate`` by mean within each (detector,
      descriptor) cell.  PCR is conditioned on emission and is **not**
      monotonic under per-pair best_of_both selection because each
      attempt averages over a different set of pairs (denominator
      mismatch).  Colour: red (0) → white (0.5) → magenta (1),
      linear ``Normalize(0, 1)``.
    * **Match-rate matrices** — same layout, cell value = fraction of pairs
      whose result is anything other than ``no_match`` (i.e. the pipeline
      produced *some* transform, accurate or not). Colour: red (0) →
      white (0.5) → azure (1), linear ``Normalize(0, 1)``.
    * **Fallback benefit** — one table per estimator. Per detector+descriptor
      combo, lift of the best_of_both policy over each single attempt.

    Heatmap PNGs are saved alongside ``report.md`` as
    ``heatmap_{maa|precision|pcr|match_rate}_{estimator}_{attempt}.png``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _add_best_of_both_column(df)
    df = _add_best_of_both_err_column(df)
    df = _add_best_of_both_pcr_column(df)

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
    report.extend(_render_matrices_section(df, estimators, tiers, "pcr", output_dir))
    report.extend(_render_matrices_section(df, estimators, tiers, "match_rate", output_dir))
    report.extend(_render_fallback_benefit(df, estimators, tiers))

    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
