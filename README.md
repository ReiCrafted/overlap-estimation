# Overlap Estimation

Comparative evaluation framework for feature-based image-pair overlap
detection. Sweeps a full matrix of detectors × descriptors × estimators ×
mask modes over a labelled image dataset and reports per-configuration
accuracy, pixel-correspondence rate, mAA-OP, and runtime.

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
    min-inliers gate → overlap geometry → metrics →
    result_label ∈ {acc_at_<T>, false_match, no_match}
```

11 detectors, 9 descriptors, every pairing is valid (custom NumPy fallbacks
for LIOP and MLDB make this work). Robust verification via PROSAC or
USAC_MAGSAC, gated only by `RunConfig.min_inliers`; metrics produce an
ordinal `result_label` per the configured accuracy tiers, plus a per-pixel
correspondence rate (PCR) and the AUC-form **mAA-OP** ("mean Average
Accuracy on the Overlap Polygon") in reporting.

For full architecture, data contracts, custom descriptor algorithms, and
the quality-gate semantics see [project_overview.md](project_overview.md).

## Layout

* `overlap_detection/` — library package (one module per pipeline stage)
* `scripts/` — CLI entrypoints
* `tests/` — pytest suite
* `reports/` — generated reports + plots committed to the repo

## License

MIT — see [LICENSE](LICENSE).
