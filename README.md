# Cross-cohort transcriptomic prediction of microsatellite instability

This repository contains the public code and derived data release for “Representation and Feature Selection Shape Cross-Cohort Transcriptomic Prediction of Microsatellite Instability.” It is the reproducibility companion for the manuscript; the editable manuscript source and submission files are distributed separately. Raw expression matrices, CEL files, and restricted clinical records are deliberately excluded.

Repository: https://github.com/Neabigmo/MSI

## Reproduction

Create the environment with `conda env create -f environment.yml`. Place the six processed cohort matrices and label files under the layout described in `docs/DATA_INTERFACE.md`, then run `scripts/01_run_analysis.ps1` followed by `scripts/02_postprocess_and_figures.ps1`. The formal entry point uses the nine-candidate grids, `xgboost.XGBClassifier`, 500 compact-model bootstrap fits, nested fold-wise compact threshold selection, and a LOWESS-based Integrated Calibration Index. The second command also runs the nested fixed-200-gene control and the out-of-bag threshold-stability analysis before rebuilding the figures.

The figure-only workflow does not require raw data when the supplied derived tables are present. Run `python src/figures.py`; it reads only the supplied derived tables and recreates the five main figures plus the nested fixed-panel supplementary figure under `figures/generated/`. The fixed-panel and threshold-stability scripts write to `results/generated/` and are already included in the numbered run order.

## Contents

- `configs/main.json`: locked analysis configuration.
- `src/`: analysis, calibration, postprocessing, and figure code.
- `scripts/`: numbered command wrappers and run order.
- `data/metadata/cohort_manifest.csv`: cohort-level public metadata.
- `data/metadata/coad_msi_labels/`: sample-level public accession IDs, binary labels, and label-source files for the six included cohorts; these contain no direct personal identifiers or expression matrices.
- `results/generated/`: the authoritative metrics, prediction-level tables, calibration curves, case-mix summaries, fixed-feature control outputs, threshold-stability outputs, and PCA/source-classifier inputs used by the figures.
- `results/supplementary/`: calibration, uncertainty, stability, comparator, and sensitivity tables.
- `figures/`: publication figures in PDF and PNG.
- `docs/`: data interface, provenance, result dictionary, figure-to-data map, and the BibTeX verification report generated with `bibtex-verifier` 0.2.0.

The current manuscript contains five main figures and supplementary calibration, nested fixed-panel, stability, and exploratory-enrichment figures. All are rebuilt from the same formal output tables; no older manuscript or reviewer-version files are part of this release.

No target outcome is used during feature selection, tuning, probability fitting, or threshold selection. Target-cohort adaptive standardization is explicitly transductive and should not be presented as single-sample deployment.

For this retrospective benchmark, the 11,316-gene reference space was defined once from feature availability across the six included studies, without labels or expression-distribution summaries, and then frozen. This availability-based alignment is not equivalent to specifying the feature universe before any target dataset is known.

The rank workflow is sample-wise only from the published series-matrix input onward. Public series matrices may already incorporate cohort-level preprocessing, so this repository does not claim a complete raw-CEL-to-prediction single-sample workflow. The TCGA expression reconstruction and the clinical-label join require source-specific verification against the released manifest; see `DATA_SOURCES.md` and `REPRODUCE.md`.

This repository contains one current public release state. Older internal versions, reviewer files, local paths, caches, and raw data are not included.

