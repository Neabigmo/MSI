"""Single authoritative runner for the final benchmark and publication figures.

All downstream figures read only ``results/generated`` and
``results/generated`` produced here.  The compact-model bootstrap
repeats feature screening, scaling, and sparse fitting on the full gene space.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import mechanism_diagnostics as mechanism
import transportability as core
import tuning_robustness as expanded

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "coad_msi_probe_mean"
OUT = ROOT / "results" / "generated"
TAB = ROOT / "results" / "generated"


def save_calibration_curves(predictions: pd.DataFrame) -> None:
    curves = []
    keys = ["design", "cohort", "representation", "model"]
    for values, group in predictions.groupby(keys):
        curve = core.calibration_curve_frame(group.label, group.probability)
        for key, value in zip(keys, values):
            curve[key] = value
        curves.append(curve)
    pd.concat(curves, ignore_index=True).to_csv(OUT / "calibration_curves.csv", index=False)


def save_pca_coordinates(data, cfg) -> None:
    cohorts = [cfg["discovery"], *cfg["external"]]
    matrix = pd.concat([data[c][0] for c in cohorts], axis=0)
    labels = np.asarray([c for c in cohorts for _ in range(len(data[c][0]))])
    rows = []
    for rep in ["fixed_log_expression", "single_sample_rank"]:
        values = core.represent(matrix, rep).to_numpy(dtype=np.float32)
        if rep == "fixed_log_expression":
            values = StandardScaler().fit_transform(values)
        pca = PCA(2, random_state=cfg["seed"])
        coords = pca.fit_transform(values)
        ev = pca.explained_variance_ratio_ * 100
        rows.extend({"cohort": cohort, "representation": rep, "pc1": float(row[0]),
                     "pc2": float(row[1]), "pc1_variance_pct": float(ev[0]),
                     "pc2_variance_pct": float(ev[1])}
                    for cohort, row in zip(labels, coords))
    pd.DataFrame(rows).to_csv(TAB / "source_pca_coordinates.csv", index=False)


def save_manifest(data, cfg) -> None:
    expression_inputs = {
        "TCGA_COAD": "GDC STAR-count TPM; log2(TPM+1)",
        "GSE13294": "GEO series matrix; fixed GPL570 mapping; sample-wise probe aggregation",
        "GSE13067": "GEO series matrix; fixed GPL570 mapping; sample-wise probe aggregation",
        "GSE18088": "GEO series matrix; fixed array annotation; sample-wise probe aggregation",
        "GSE26682": "GEO series matrix; fixed GPL570 mapping; sample-wise probe aggregation",
        "GSE39084": "GEO series matrix; fixed GPL570 mapping; sample-wise probe aggregation",
    }
    label_sources = {
        "TCGA_COAD": "GDC roster joined to cBioPortal MSI_STATUS",
        "GSE13294": "Direct MSI/MSS labels in GEO and published cohort annotations",
        "GSE13067": "Direct MSI/MSS labels in GEO and published cohort annotations",
        "GSE18088": "Direct MSI/MSS labels in published CRC cohort annotations",
        "GSE26682": "Direct MSI/MSS labels in published CRC cohort compendium annotations",
        "GSE39084": "Direct MSI/MSS labels in published CRC cohort annotations",
    }
    rows = []
    for cohort in [cfg["discovery"], *cfg["external"]]:
        x, y = data[cohort]
        rows.append({"cohort": cohort, "role": "discovery" if cohort == cfg["discovery"] else "external",
                     "platform": "RNA-seq" if cohort == cfg["discovery"] else (
                         "Affymetrix HG-U133 Plus 2.0 family" if cohort == "GSE18088"
                         else "Affymetrix GPL570-family microarray"),
                     "n": len(y), "msi": int(y.sum()), "mss": int((1 - y).sum()),
                     "prevalence": float(y.mean()), "label_source": label_sources[cohort],
                     "expression_input": expression_inputs[cohort]})
    pd.DataFrame(rows).to_csv(TAB / "cohort_manifest.csv", index=False)


def save_case_mix(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    keys = ["design", "cohort", "representation", "model"]
    p = predictions[predictions.design.eq("TCGA_to_GEO")].copy()
    p_summary = p.groupby(keys).agg(
        n=("label", "size"), prevalence=("label", "mean"),
        mean_prediction=("probability", "mean"), sd_prediction=("probability", "std"),
        min_prediction=("probability", "min"), max_prediction=("probability", "max"),
    ).reset_index()
    cols = keys + ["calibration_intercept", "calibration_slope", "brier", "brier_skill", "ici"]
    out = p_summary.merge(metrics[metrics.design.eq("TCGA_to_GEO")][cols], on=keys, how="left")
    out["prevalence_prediction_gap"] = out["mean_prediction"] - out["prevalence"]
    out.to_csv(TAB / "calibration_case_mix.csv", index=False)


def save_directions(data, cfg, stability) -> None:
    genes = stability.head(cfg["compact_gene_count"])["gene"].tolist()
    rows = []
    for cohort in [cfg["discovery"], *cfg["external"]]:
        x, y = data[cohort]
        rank_x = core.represent(x, "single_sample_rank")
        for gene in genes:
            rows.append({"gene": gene, "cohort": cohort,
                         "median_msi_minus_mss": float(x.loc[y == 1, gene].median() -
                                                        x.loc[y == 0, gene].median()),
                         "median_rank_msi_minus_mss": float(
                             rank_x.loc[y == 1, gene].median() - rank_x.loc[y == 0, gene].median())})
    pd.DataFrame(rows).to_csv(TAB / "stable_gene_cohort_directions.csv", index=False)


def main() -> None:
    cfg = core.load_config()
    core.DATA = DATA
    core.model_grid = expanded.expanded_grid
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    cohorts = [cfg["discovery"], *cfg["external"]]
    data = {cohort: core.load_cohort(cohort) for cohort in cohorts}

    cross, cross_pred, cross_tuning = core.run_cross_platform(data, cfg)
    loco, loco_pred, loco_tuning = core.run_loco(data, cfg)
    compact, compact_pred, stability, frozen, metadata = core.fit_compact(data, cfg)
    metrics = pd.DataFrame(cross + loco + compact)
    predictions = pd.DataFrame(cross_pred + loco_pred + compact_pred)
    predictions["locked_prediction"] = (predictions["probability"] >= predictions["threshold"]).astype(int)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(cross_tuning + loco_tuning).to_csv(OUT / "tuning_results.csv", index=False)
    stability.to_csv(OUT / "compact_gene_stability.csv", index=False)
    frozen.to_csv(OUT / "compact_rank_frozen_coefficients.csv", index=False)
    (OUT / "compact_rank_model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_calibration_curves(predictions)
    save_manifest(data, cfg)
    save_case_mix(predictions, metrics)
    save_directions(data, cfg, stability)
    save_pca_coordinates(data, cfg)
    core.paired_contrasts(metrics, cfg).to_csv(TAB / "paired_bootstrap_contrasts.csv", index=False)
    source = mechanism.source_accuracy(data, cfg, 0) + mechanism.source_accuracy(data, cfg, 1)
    pd.DataFrame(source).to_csv(OUT / "source_classifier_stratified.csv", index=False)
    kernels, scores = mechanism.kernel_geometry(data, cfg)
    pd.DataFrame(kernels).to_csv(ROOT / "results" / "generated" /
                                 "rank_svm_kernel_distribution.csv", index=False)
    pd.DataFrame(scores).to_csv(ROOT / "results" / "generated" /
                                "rank_svm_external_scores.csv", index=False)
    loco_kernels, loco_scores = mechanism.kernel_geometry_loco(data, cfg)
    pd.DataFrame(loco_kernels).to_csv(OUT / "rank_svm_loco_kernel_distribution.csv", index=False)
    pd.DataFrame(loco_scores).to_csv(OUT / "rank_svm_loco_scores.csv", index=False)
    print(metrics[metrics.design.eq("TCGA_to_GEO")].groupby(["representation", "model"])
          [["auroc", "balanced_accuracy", "brier_skill", "ici"]].mean().to_string())
    print("Wrote authoritative outputs to", OUT)


if __name__ == "__main__":
    main()

