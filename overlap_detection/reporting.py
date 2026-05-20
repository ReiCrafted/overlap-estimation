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
            
    with open(filepath, 'w') as f:
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

def write_summary_report(
    csv_path: Path,
    output_dir: Path,
) -> None:
    """Read aggregate CSV, generate summary plots:
    - Box plot of RMS corner error by detector+descriptor combination
    - Bar chart of mean inlier ratio by configuration
    - Box plot of total runtime by configuration
    - Scatter plot: inlier ratio vs RMS error (per pair)
    - Heatmap: success rate by detector × descriptor
    Save as PNG + a markdown report linking them."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Create configuration identifier
    df['config_name'] = df['detector'] + '+' + df['descriptor'] + '+' + df['estimator'] + '+' + df['mask_mode']
    df['det_desc'] = df['detector'] + '+' + df['descriptor']
    
    sns.set_theme(style="whitegrid")
    
    # 1. Box plot of RMS corner error by detector+descriptor combination
    plt.figure(figsize=(12, 6))
    if 'rms_corner_error' in df.columns and not df['rms_corner_error'].isna().all():
        sns.boxplot(data=df, x='det_desc', y='rms_corner_error')
        plt.xticks(rotation=45, ha='right')
        plt.title('RMS Corner Error by Detector+Descriptor')
        plt.tight_layout()
        plt.savefig(output_dir / 'rms_error_boxplot.png', dpi=150)
    plt.close()
    
    # 2. Bar chart of mean inlier ratio by configuration
    plt.figure(figsize=(12, 6))
    if 'inlier_ratio' in df.columns:
        sns.barplot(data=df, x='config_name', y='inlier_ratio')
        plt.xticks(rotation=45, ha='right')
        plt.title('Mean Inlier Ratio by Configuration')
        plt.tight_layout()
        plt.savefig(output_dir / 'inlier_ratio_barplot.png', dpi=150)
    plt.close()
    
    # 3. Box plot of total runtime by configuration
    plt.figure(figsize=(12, 6))
    if 'total_ms' in df.columns:
        sns.boxplot(data=df, x='config_name', y='total_ms')
        plt.xticks(rotation=45, ha='right')
        plt.title('Total Runtime (ms) by Configuration')
        plt.tight_layout()
        plt.savefig(output_dir / 'runtime_boxplot.png', dpi=150)
    plt.close()
    
    # 4. Scatter plot: inlier ratio vs RMS error (per pair)
    plt.figure(figsize=(8, 6))
    if 'inlier_ratio' in df.columns and 'rms_corner_error' in df.columns and not df['rms_corner_error'].isna().all():
        sns.scatterplot(data=df, x='inlier_ratio', y='rms_corner_error', hue='det_desc', alpha=0.7)
        plt.title('Inlier Ratio vs RMS Error')
        plt.tight_layout()
        plt.savefig(output_dir / 'inlier_vs_rms_scatter.png', dpi=150)
    plt.close()
    
    # 5. Heatmap: success rate by detector × descriptor
    plt.figure(figsize=(10, 8))
    if 'estimation_succeeded' in df.columns and 'detector' in df.columns and 'descriptor' in df.columns:
        success_rate = df.groupby(['detector', 'descriptor'])['estimation_succeeded'].mean().unstack()
        sns.heatmap(success_rate, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Success Rate by Detector and Descriptor')
        plt.tight_layout()
        plt.savefig(output_dir / 'success_rate_heatmap.png', dpi=150)
    plt.close()
    
    # Generate Markdown Report
    report = ["# Overlap Detection Summary Report\n"]
    report.append(f"Total runs executed: {len(df)}\n")
    if 'estimation_succeeded' in df.columns:
        report.append(f"Overall success rate: {df['estimation_succeeded'].mean():.2%}\n")
        
    if 'rms_corner_error' in df.columns and not df['rms_corner_error'].isna().all():
        best_rms = df.groupby('config_name')['rms_corner_error'].median().idxmin()
        report.append(f"Best configuration by median RMS error: **{best_rms}**\n")
        
    if 'total_ms' in df.columns:
        best_speed = df.groupby('config_name')['total_ms'].median().idxmin()
        report.append(f"Best configuration by speed: **{best_speed}**\n")
        
    if 'iou' in df.columns and not df['iou'].isna().all():
        best_iou = df.groupby('config_name')['iou'].median().idxmax()
        report.append(f"Best configuration by IoU: **{best_iou}**\n")
        
    if 'inlier_ratio' in df.columns:
        low_inlier = df[df['inlier_ratio'] < 0.1]
        if not low_inlier.empty:
            report.append("\n## Warning: Low Inlier Ratio Runs (Potential Periodicity Failures)\n")
            report.append(f"{len(low_inlier)} runs had <10% inliers.\n")
            
    if 'estimation_succeeded' in df.columns and 'detector' in df.columns and 'descriptor' in df.columns:
        success_matrix = df.groupby(['detector', 'descriptor'])['estimation_succeeded'].mean().unstack()
        descriptors = list(success_matrix.columns)
        report.append("\n## Success Rate by Detector × Descriptor\n")
        report.append("| Detector | " + " | ".join(descriptors) + " |")
        report.append("|" + "---|" * (len(descriptors) + 1))
        for detector, row in success_matrix.iterrows():
            cells = [f"{row[d]:.2f}" if d in row and not pd.isna(row[d]) else "N/A" for d in descriptors]
            report.append(f"| {detector} | " + " | ".join(cells) + " |")

    report.append("\n## Median RMS Error Table\n")
    if 'rms_corner_error' in df.columns and not df['rms_corner_error'].isna().all():
        table = df.groupby('config_name')['rms_corner_error'].median().sort_values()
        report.append("| Configuration | Median RMS Error (px) |")
        report.append("|--------------|----------------------|")
        for idx, val in table.items():
            report.append(f"| {idx} | {val:.2f} |")
            
    report.append("\n## Visualizations\n")
    report.append("![RMS Error Boxplot](./rms_error_boxplot.png)\n")
    report.append("![Inlier Ratio Barplot](./inlier_ratio_barplot.png)\n")
    report.append("![Runtime Boxplot](./runtime_boxplot.png)\n")
    report.append("![Inlier vs RMS Scatter](./inlier_vs_rms_scatter.png)\n")
    report.append("![Success Rate Heatmap](./success_rate_heatmap.png)\n")
    
    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report))
