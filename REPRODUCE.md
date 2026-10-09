# Reproducibility

The public repository is https://github.com/Neabigmo/MSI. It contains the code, anonymous/public-accession metadata and derived result tables; it does not contain raw expression matrices or restricted clinical data.

## Requirements
Environment dependencies are recorded in `environment.yml`. They describe software versions, NOT the author's local machine. You can use your own compatible environment.

## Figures without raw expression matrices
The figure source `src/figures.py` reads eleven provided machine-readable derived data tables and validates primary metrics against stored prediction records. Run from repository root:

```bash
python src/figures.py
```

Outputs are written to the ignored `figures/generated/` directory. This figure-only workflow is distinct from full model retraining.

## Full retraining from public data
Construct processed expression TSV files at `data/processed/coad_msi_probe_mean/<COHORT>_shared_expression.tsv`, with samples in rows, the frozen 11,316 gene symbols in columns, and `sample_id` as row identifier. Associated `<COHORT>_labels.csv` must contain exactly the frozen accession and binary label records. Obtain data from `DATA_SOURCES.md` and validate cohort counts before running:

```bash
python src/run_formal.py
python src/postprocess_formal.py
python src/fixed_feature_control.py
python src/threshold_stability.py
python src/figures.py
```

The GEO helper `python src/data_ingest.py --output-root data/processed/coad_msi_probe_mean` downloads public GEO series matrices and writes audit/label files. After checking each downloaded cohort, copy the frozen `<COHORT>_labels.csv` files from `data/metadata/coad_msi_labels/` into `data/processed/coad_msi_probe_mean/`. The TCGA expression matrix itself must still be obtained and aligned separately; full TCGA cohort construction is not automated. Confirm model search, time/memory requirements, seed, and output hashes against frozen artifacts. Do not claim full raw-data reproducibility until it has actually been tested with the public inputs.

## Study safeguards
- All model selection, feature selection, threshold and probability fitting use development labels only.
- Target-adaptive Z is deliberately transductive on an unlabeled target batch.
- The full rank representation requires 11,316 genes even when only 20 selected ranks are used in the compact classifier.
- Five GEO cohorts were inspected during model iteration, so this is not a previously untouched prospective validation.
