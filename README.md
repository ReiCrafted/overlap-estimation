# Overlap Estimation

Comparative evaluation framework for feature-based image-pair overlap
detection. Sweeps a full matrix of detectors × descriptors × estimators ×
mask modes over a labelled image dataset and reports per-configuration
accuracy, IoU, mAA, and runtime.

## Quick start

```powershell
# Install (editable)
pip install -e .[dev]

# Annotate ground truth for a dataset (Tkinter GUI)
python scripts/annotate_dataset.py --dataset-dir path/to/images --annotator you

# Run the experimental matrix (parallel by default; cpu_count-1 up to 8)
python scripts/run_experiment.py `
    --dataset-dir     path/to/images `
    --groundtruth-dir path/to/images/annotations `
    --output-dir      results/my_run

# Generate plots and a markdown report
python scripts/generate_report.py --results-dir results/my_run --output-dir reports/my_run
```

## Pipeline at a glance

```
preprocess (mask) → detect → describe → match → verify (RANSAC) →
    affine sanity check → overlap geometry → metrics → quality gates →
    quality_flag ∈ {true, true after false, false, false after false}
```

11 detectors, 9 descriptors, every pairing is valid (custom NumPy fallbacks
for LIOP and MLDB make this work). Robust verification via PROSAC or
USAC_MAGSAC. Geometric verification is followed by a scale/rotation sanity
filter and post-metrics IoU/RMS gates.

For full architecture, data contracts, custom descriptor algorithms, and
the quality-gate semantics see [project_overview.md](project_overview.md).

## Layout

* `overlap_detection/` — library package (one module per pipeline stage)
* `scripts/` — CLI entrypoints
* `tests/` — pytest suite
* `reports/` — generated reports + plots committed to the repo

## License

MIT — see [LICENSE](LICENSE).
