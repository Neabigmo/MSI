# Public data sources and processing

This file documents the inputs needed to reproduce the results in the public repository at https://github.com/Neabigmo/MSI.

## Discovery
TCGA-COAD: GDC (https://portal.gdc.cancer.gov/projects/TCGA-COAD) STAR-count-derived TPM; MSI annotations joined to the frozen GDC/cBioPortal sample roster. cBioPortal study browser: https://www.cbioportal.org/ . Download each source and reconcile sample IDs against `data/metadata/coad_msi_labels/TCGA_COAD_labels.csv`. The GDC-to-MSI join is not yet fully automated in the supplied code and requires source-specific verification. Do not substitute another clinical label field without recording and validating the mapping.

## External GEO cohorts
- GSE13294: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13294
- GSE13067: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13067
- GSE18088: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE18088
- GSE26682: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE26682 (GPL570 branch only; exclude 23 MSI-L samples)
- GSE39084: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39084 (67 selected samples)

Array annotation: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL570

Follow the source series-matrix annotation and the `data/metadata/coad_msi_labels/` accession manifest. The source data may already include cohort-level preprocessing. Fixed annotation maps probes to the first valid gene symbol; multiple probes per gene are averaged within each sample. The 11,316-gene feature space is fixed by `data/metadata/shared_gene_set.csv` and must not be recomputed from GEO cohorts alone. MSI-L and unknown cases are excluded. All sample IDs and labels must match the frozen manifest exactly.

## Important limits
This project does not mirror raw CEL files, expression matrices or controlled-access clinical records. The public release provides the fixed sample manifest, labels, gene space, derived results and the processing code, but full end-to-end reconstruction is not claimed to be turnkey: TCGA expression retrieval and the source-specific MSI label join still require independent verification against the released manifest. Do not substitute another clinical label field or silently change cohort inclusion rules.
