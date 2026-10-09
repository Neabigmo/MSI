# MSIsensor-RNA comparator

The official academic repository was retrieved from `https://github.com/xjtu-omics/msisensor-rna` at commit `0fc3b7874349e9b182b86ca006fc965b856d8761`. The published `TCGA.CRC.model.pkl` requires 397 genes and performs within-sample z standardization before applying its frozen SVM classifier.

Four GPL570 matrices exposed 390 required genes; the GSE26682 GPL570 branch exposed 291. The adapter assigns unavailable genes zero on the standardized scale and records coverage by cohort. Consequently, this is an explicitly interface-limited sensitivity comparison, not a protocol-equivalent benchmark. No external label was used to fit or retune the model. Predictions, metrics, model metadata, coverage, upstream commit, and the exact adapter are included in the release.
