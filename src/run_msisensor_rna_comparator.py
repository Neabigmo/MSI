"""Evaluate the published MSIsensor-RNA CRC model on the five GEO cohorts.

The upstream detector standardizes the genes required by its model within
each sample before prediction.  This adapter preserves that behaviour, adds
cohort labels, and writes prediction- and cohort-level audit tables.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score


COHORTS = ["GSE13294", "GSE13067", "GSE18088", "GSE26682", "GSE39084"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true",
                        help="Impute unavailable model genes at zero after within-sample standardization.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.model.open("rb") as handle:
        classifier = pickle.load(handle)
        description = pickle.load(handle)
        genes = list(pickle.load(handle))

    prediction_rows = []
    metric_rows = []
    for cohort in COHORTS:
        shared_path = args.data_dir / f"{cohort}_shared_expression.tsv"
        full_path = args.data_dir / f"{cohort}_expression.tsv"
        expression_path = full_path if full_path.exists() else shared_path
        expression = pd.read_csv(expression_path, sep="\t", index_col=0)
        labels_dir = args.labels_dir or args.data_dir
        labels = pd.read_csv(labels_dir / f"{cohort}_labels.csv").set_index("sample_id")
        missing = sorted(set(genes) - set(expression.columns))
        if missing and not args.allow_missing:
            raise ValueError(f"{cohort}: {len(missing)} model genes missing: {missing[:12]}")
        common = expression.index.intersection(labels.index)
        available_genes = [gene for gene in genes if gene in expression.columns]
        x = expression.loc[common, available_genes].astype(float)
        # Exact normalization used in upstream detection.py (pandas sample SD).
        x = x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1, ddof=1), axis=0).fillna(0)
        for gene in missing:
            x[gene] = 0.0
        x = x.loc[:, genes]
        probability = classifier.predict_proba(x)[:, 1]
        predicted = classifier.predict(x).astype(int)
        y = labels.loc[common, "label"].replace({"MSI": 1, "MSS": 0}).astype(int).to_numpy()
        for sample_id, label, prob, pred in zip(common, y, probability, predicted):
            prediction_rows.append({
                "cohort": cohort, "sample_id": sample_id, "label": label,
                "probability": prob, "prediction": pred,
            })
        prevalence = y.mean()
        brier = brier_score_loss(y, probability)
        metric_rows.append({
            "cohort": cohort, "n": len(y), "msi": int(y.sum()), "mss": int((1-y).sum()),
            "available_model_genes": len(available_genes), "missing_model_genes": len(missing),
            "auroc": roc_auc_score(y, probability),
            "balanced_accuracy": balanced_accuracy_score(y, predicted),
            "brier": brier,
            "brier_skill": 1 - brier / (prevalence * (1 - prevalence)),
        })

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    predictions.to_csv(args.output_dir / "msisensor_rna_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "msisensor_rna_metrics.csv", index=False)
    provenance = {
        "tool": "MSIsensor-RNA", "upstream_commit": "0fc3b7874349e9b182b86ca006fc965b856d8761",
        "model": args.model.name, "model_description": description,
        "required_gene_count": len(genes), "normalization": "within-sample z score, upstream implementation",
        "missing_gene_handling": "zero on standardized scale" if args.allow_missing else "none permitted",
        "note": "Published frozen CRC model; no study labels used for fitting or threshold selection.",
    }
    (args.output_dir / "msisensor_rna_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print("Macro mean:", metrics[["auroc", "balanced_accuracy", "brier_skill"]].mean().to_dict())


if __name__ == "__main__":
    main()
