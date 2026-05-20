import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import asdict
from overlap_detection.types import PairResult
import numpy as np

def write_pair_json(
    result: PairResult,
    metrics: dict,
    output_dir: Path,
) -> Path:
    """Write full per-pair result + metrics to JSON. Filename:
    {pair_id}_{detector}_{descriptor}_{estimator}_{mask_mode}.json
    Returns path written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pair_id = f"{result.image_a_path.stem}_{result.image_b_path.stem}"
    filename = f"{pair_id}_{result.detector}_{result.descriptor}_{result.estimator}_{result.mask_mode}.json"
    filepath = output_dir / filename
    
    data = {
        "result": asdict(result),
        "metrics": metrics
    }
    
    # Convert numpy arrays to lists for JSON serialization
    for k, v in data["result"].items():
        if isinstance(v, np.ndarray):
            data["result"][k] = v.tolist()
        elif isinstance(v, Path):
            data["result"][k] = str(v)
            
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    return filepath

def write_aggregate_csv(
    all_pair_metrics: list[dict],
    output_path: Path,
) -> None:
    """Write all per-pair metrics as a single CSV. Columns include
    pair_id, detector, descriptor, estimator, mask_mode, and all
    metric fields from compute_pair_metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_pair_metrics)
    df.to_csv(output_path, index=False)

_SUCCESS_FLAGS = {"true", "true after false"}


def _derive_success_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``passed`` (bool) and ``maa`` (float in [0, 1] per row) columns.

    Both are derived from ``quality_flag`` when present so reports written
    after the gate change reflect the new pass criteria.  Falls back to the
    legacy ``estimation_succeeded`` column for runs written before
    ``quality_flag`` existed.
    """
    if 'quality_flag' in df.columns and df['quality_flag'].notna().any():
        df['passed'] = df['quality_flag'].isin(_SUCCESS_FLAGS)
    elif 'estimation_succeeded' in df.columns:
        df['passed'] = df['estimation_succeeded'].astype(bool)
    else:
        df['passed'] = False
    df['maa'] = df['passed'].astype(float)
    return df


def write_summary_report(
    csv_path: Path,
    output_dir: Path,
) -> None:
    """Read aggregate CSV, generate summary plots and a markdown report.

    Plots written:
      * ``rms_error_boxplot.png``   — RMS corner error by detector+descriptor
      * ``inlier_ratio_barplot.png``— mean inlier ratio by configuration
      * ``runtime_boxplot.png``     — total runtime by configuration
      * ``inlier_vs_rms_scatter.png`` — inlier ratio vs RMS (per pair)
      * ``success_rate_heatmap.png``— mAA by detector × descriptor
      * ``iou_boxplot.png``         — IoU by detector+descriptor
      * ``maa_barplot.png``         — mAA by configuration

    The markdown report also tabulates median IoU and mAA per configuration.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _derive_success_columns(df)

    # Create configuration identifier
    df['config_name'] = df['detector'] + '+' + df['descriptor'] + '+' + df['estimator'] + '+' + df['mask_mode']
    df['det_desc'] = df['detector'] + '+' + df['descriptor']

    sns.set_theme(style="whitegrid")

    has_rms = 'rms_corner_error' in df.columns and not df['rms_corner_error'].isna().all()
    has_iou = 'iou' in df.columns and not df['iou'].isna().all()

    # 1. Box plot of RMS corner error by detector+descriptor combination
    if has_rms:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='det_desc', y='rms_corner_error')
        plt.xticks(rotation=45, ha='right')
        plt.title('RMS Corner Error by Detector+Descriptor')
        plt.tight_layout()
        plt.savefig(output_dir / 'rms_error_boxplot.png', dpi=150)
        plt.close()

    # 2. Bar chart of mean inlier ratio by configuration
    if 'inlier_ratio' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='config_name', y='inlier_ratio')
        plt.xticks(rotation=45, ha='right')
        plt.title('Mean Inlier Ratio by Configuration')
        plt.tight_layout()
        plt.savefig(output_dir / 'inlier_ratio_barplot.png', dpi=150)
        plt.close()

    # 3. Box plot of total runtime by configuration
    if 'total_ms' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='config_name', y='total_ms')
        plt.xticks(rotation=45, ha='right')
        plt.title('Total Runtime (ms) by Configuration')
        plt.tight_layout()
        plt.savefig(output_dir / 'runtime_boxplot.png', dpi=150)
        plt.close()

    # 4. Scatter plot: inlier ratio vs RMS error (per pair)
    if 'inlier_ratio' in df.columns and has_rms:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='inlier_ratio', y='rms_corner_error', hue='det_desc', alpha=0.7)
        plt.title('Inlier Ratio vs RMS Error')
        plt.tight_layout()
        plt.savefig(output_dir / 'inlier_vs_rms_scatter.png', dpi=150)
        plt.close()

    # 5. Heatmap: mAA by detector × descriptor
    if 'detector' in df.columns and 'descriptor' in df.columns:
        plt.figure(figsize=(10, 8))
        maa_matrix = df.groupby(['detector', 'descriptor'])['maa'].mean().unstack()
        sns.heatmap(maa_matrix, annot=True, cmap='viridis', fmt='.2f', vmin=0.0, vmax=1.0)
        plt.title('mAA by Detector and Descriptor')
        plt.tight_layout()
        plt.savefig(output_dir / 'success_rate_heatmap.png', dpi=150)
        plt.close()

    # 6. Box plot of IoU by detector+descriptor
    if has_iou:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='det_desc', y='iou')
        plt.xticks(rotation=45, ha='right')
        plt.title('IoU by Detector+Descriptor')
        plt.tight_layout()
        plt.savefig(output_dir / 'iou_boxplot.png', dpi=150)
        plt.close()

    # 7. Bar chart of mAA by configuration
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='config_name', y='maa')
    plt.xticks(rotation=45, ha='right')
    plt.title('Mean Average Accuracy (mAA) by Configuration')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / 'maa_barplot.png', dpi=150)
    plt.close()

    # ---- Markdown report ----
    report: list[str] = ["# Overlap Detection Summary Report\n"]
    report.append(f"Total runs executed: {len(df)}\n")
    report.append(f"Overall mAA (mean Average Accuracy): {df['maa'].mean():.2%}\n")

    if 'quality_flag' in df.columns and df['quality_flag'].notna().any():
        flag_counts = df['quality_flag'].value_counts()
        report.append("\n## Quality flag distribution\n")
        report.append("| Flag | Count | Share |")
        report.append("|------|-------|-------|")
        for flag in ["true", "true after false", "false", "false after false"]:
            count = int(flag_counts.get(flag, 0))
            share = count / len(df) if len(df) else 0.0
            report.append(f"| {flag} | {count} | {share:.2%} |")

    if has_rms:
        best_rms = df.groupby('config_name')['rms_corner_error'].median().idxmin()
        report.append(f"\nBest configuration by median RMS error: **{best_rms}**\n")

    if 'total_ms' in df.columns:
        best_speed = df.groupby('config_name')['total_ms'].median().idxmin()
        report.append(f"Best configuration by speed: **{best_speed}**\n")

    if has_iou:
        best_iou = df.groupby('config_name')['iou'].median().idxmax()
        report.append(f"Best configuration by median IoU: **{best_iou}**\n")

    best_maa = df.groupby('config_name')['maa'].mean().idxmax()
    report.append(f"Best configuration by mAA: **{best_maa}**\n")

    if 'inlier_ratio' in df.columns:
        low_inlier = df[df['inlier_ratio'] < 0.1]
        if not low_inlier.empty:
            report.append("\n## Warning: Low Inlier Ratio Runs (Potential Periodicity Failures)\n")
            report.append(f"{len(low_inlier)} runs had <10% inliers.\n")

    if 'detector' in df.columns and 'descriptor' in df.columns:
        maa_matrix = df.groupby(['detector', 'descriptor'])['maa'].mean().unstack()
        descriptors = list(maa_matrix.columns)
        report.append("\n## mAA by Detector × Descriptor\n")
        report.append("| Detector | " + " | ".join(descriptors) + " |")
        report.append("|" + "---|" * (len(descriptors) + 1))
        for detector, row in maa_matrix.iterrows():
            cells = [f"{row[d]:.2f}" if d in row and not pd.isna(row[d]) else "N/A" for d in descriptors]
            report.append(f"| {detector} | " + " | ".join(cells) + " |")

    # Combined per-config table: mAA, median IoU, median RMS
    grouped = df.groupby('config_name')
    maa_per_cfg = grouped['maa'].mean()
    iou_per_cfg = grouped['iou'].median() if 'iou' in df.columns else None
    rms_per_cfg = grouped['rms_corner_error'].median() if has_rms else None

    report.append("\n## Per-configuration scoreboard\n")
    report.append("Sorted by mAA (descending), then by median RMS (ascending).\n")
    report.append("| Configuration | mAA | Median IoU | Median RMS (px) |")
    report.append("|---------------|-----|------------|------------------|")
    order = maa_per_cfg.sort_values(ascending=False).index
    for cfg in order:
        maa_val = maa_per_cfg.loc[cfg]
        iou_val = iou_per_cfg.loc[cfg] if iou_per_cfg is not None and cfg in iou_per_cfg.index else float('nan')
        rms_val = rms_per_cfg.loc[cfg] if rms_per_cfg is not None and cfg in rms_per_cfg.index else float('nan')
        iou_cell = f"{iou_val:.3f}" if not pd.isna(iou_val) else "N/A"
        rms_cell = f"{rms_val:.2f}" if not pd.isna(rms_val) else "N/A"
        report.append(f"| {cfg} | {maa_val:.2f} | {iou_cell} | {rms_cell} |")

    report.append("\n## Visualizations\n")
    report.append("![mAA by Configuration](./maa_barplot.png)\n")
    report.append("![mAA Heatmap](./success_rate_heatmap.png)\n")
    if has_iou:
        report.append("![IoU Boxplot](./iou_boxplot.png)\n")
    if has_rms:
        report.append("![RMS Error Boxplot](./rms_error_boxplot.png)\n")
        report.append("![Inlier vs RMS Scatter](./inlier_vs_rms_scatter.png)\n")
    report.append("![Inlier Ratio Barplot](./inlier_ratio_barplot.png)\n")
    report.append("![Runtime Boxplot](./runtime_boxplot.png)\n")

    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
