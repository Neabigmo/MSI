# Current manuscript figures

The five main figures are rebuilt with `src/draw_figures.py` from the formal CSV results in `results/generated`. No raw expression matrices are needed for plotting. The compatibility entry point `src/figures.py` calls the same implementation.

Run from the release directory:

```powershell
python -B src/draw_figures.py
```

The script exports vector PDF, editable SVG text, and 400-dpi PNG to `figures/generated`. It also recomputes 475 AUROC, locked balanced accuracy, Brier, Brier skill and ICI estimates from the formal predictions, checks them against `metrics.csv`, and writes `prediction_metric_checks.csv`. This is a plotting-stage consistency check, not an independent retraining experiment.

| Figure/panel | Input files | Display |
| --- | --- | --- |
| 1a | cohort_manifest.csv | Observed MSI/MSS sample counts, in fixed cohort order |
| 1b | Study protocol | Schematic only; not a quantitative result |
| 1c–d | metrics.csv | Paired fixed/rank Elastic Net AUROC and locked balanced accuracy |
| 2a–b | source_pca_coordinates.csv | Saved PCA coordinates after the formally defined representations; rank is computed on 11,316 genes before display-feature reduction |
| 2c | source_classifier_stratified.csv | Pooled out-of-fold source-classification balanced accuracy, separately within MSI and MSS |
| 3a–c | metrics.csv | Equal-weight five-cohort means; fixed row/column order |
| 3d | predictions.csv | LOWESS (span 0.75, no robust iterations), recomputed with calibration_metrics.py; lines only over observed probability support |
| 3e | predictions.csv | Calibration slopes on a logarithmic display with 500 patient-bootstrap intervals; all finite bootstrap slopes enter interval estimation |
| 4a | rank_svm_kernel_distribution.csv; rank_svm_loco_kernel_distribution.csv | Saved direct log-kernels divided by ln(10), pooled by design and pair type |
| 4b | predictions.csv | All external rank-SVM probabilities from TCGA training; deterministic horizontal and vertical display jitter, fixed seed 20261008 |
| 4c–e | metrics.csv | Paired TCGA-to-GEO and GEO-LOCO evaluation, rank SVM and Elastic Net |
| 5a | compact_gene_stability.csv | First 20 selected genes, 500 full-pipeline discovery bootstraps |
| 5b | compact_rank_frozen_coefficients.csv | Final compact-model standardized coefficients; NOT bootstrap medians |
| 5c | stable_gene_cohort_directions.csv | Median MSI–MSS rank differences, divided by each gene's maximum absolute difference across six cohorts |
| 5d–f | metrics.csv | Full rank Elastic Net versus compact-20, paired by cohort |

All paired plots use cohort-specific marker shapes, grey connecting lines, and black squares for unweighted means. Small deterministic horizontal offsets separate markers with the same x position; grey lines connect the corresponding offset points. They show cohort estimates, not confidence intervals. No uncomputed uncertainty or significance symbols are added. Kernel boxes show quartiles, median and 1.5-IQR whiskers; outliers are omitted from the box plot but remain in the supplied data and min/max summaries. These pairs are descriptive, not independent statistical replicates.

`kernel_distribution_summary.csv` preserves group size, log-kernel ranges and quartiles, arithmetic mean of saved kernels and the fraction stored as zero. Direct log-kernels retain finite values even when exponentiation underflows. The dotted line in Figure 4a denotes the smallest positive float64 subnormal value on the log10 scale; it is not used to replace data. Figure 4a places the training-within scale in a separate right-hand axis. Figure 4b directly uses the expanded local probability scale with deterministic display jitter; `rank_svm_probability_summary.csv` records the exact probability differences that cannot be resolved visually on the full 0–1 axis.

The compact model requires ranks across all 11,316 reference genes and is not a 20-gene measurement assay. Final coefficients and bootstrap sign consistency are distinct quantities; full sign-consistency values remain in the stability CSV. All three top panels in Figure 5 share exactly the CSV gene order, including ties.

The five main figures have been visually inspected. Numerical plotting inputs are unchanged; model training was not repeated for this redraw. Earlier graphics and the old plotting script are retained in the research workspace archive. Supplementary graphics were not redesigned in this task.
