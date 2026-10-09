"""Nested fixed-200-gene control for representation--model interactions.

For each outer discovery fold, the 200-gene panel is selected from the outer
training IDs only. Hyperparameter tuning and fitting then use that outer
training set, and the held-out IDs are predicted without their labels entering
feature selection. The discovery threshold is locked from these nested OOF
predictions. A final 200-gene panel is selected on all discovery samples for
the frozen external fits; external labels are used only for evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold

import transportability as core
import tuning_robustness as expanded

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "coad_msi_probe_mean"
OUT = ROOT / "results" / "generated"


def select_panel(x: pd.DataFrame, y: pd.Series | np.ndarray, k: int):
    selector = SelectKBest(f_classif, k=min(k, x.shape[1])).fit(x, y)
    mask = selector.get_support()
    genes = x.columns[mask].tolist()
    return selector, genes


def panel_rows(selector, genes, representation, panel_id):
    return [{
        "panel_id": panel_id,
        "representation": representation,
        "gene": gene,
        "f_statistic": float(selector.scores_[selector.get_support()][j]),
        "raw_pvalue": float(selector.pvalues_[selector.get_support()][j]),
        "panel_size": len(genes),
    } for j, gene in enumerate(genes)]


def main() -> None:
    cfg = core.load_config()
    core.DATA = DATA
    core.model_grid = expanded.expanded_grid
    core._REP_CACHE.clear()
    OUT.mkdir(parents=True, exist_ok=True)

    cohorts = [cfg["discovery"], *cfg["external"]]
    data = {cohort: core.load_cohort(cohort) for cohort in cohorts}
    dx, dy = data[cfg["discovery"]]
    y = dy.to_numpy(dtype=int)
    fixed = core.represent(dx, "fixed_log_expression")
    represented = {rep: core.represent(dx, rep) for rep in cfg["representations"]}
    outer = StratifiedKFold(cfg["cv_splits"], shuffle=True, random_state=cfg["seed"])

    rows, predictions, tuning, panel_records = [], [], [], []
    for rep in cfg["representations"]:
        rx = represented[rep]
        for model_name in cfg["models"]:
            print(f"Nested fixed feature control: {rep} / {model_name}", flush=True)
            oof = np.full(len(dy), np.nan, dtype=float)
            fold_settings = []
            for fold, (tr, va) in enumerate(outer.split(fixed, y)):
                selector, genes = select_panel(fixed.iloc[tr], y[tr], cfg["feature_budget"])
                panel_records.extend(panel_rows(selector, genes, rep, f"outer_fold_{fold}"))
                train_x = rx.iloc[tr].loc[:, genes]
                best, candidates = core.tune_oof(train_x, dy.iloc[tr].to_numpy(), model_name, cfg)
                _, setting, estimator, _ = best
                fitted = clone(estimator).fit(train_x, dy.iloc[tr].to_numpy())
                oof[va] = core.predict_score(fitted, rx.iloc[va].loc[:, genes])
                fold_settings.append({"fold": fold, "setting": setting})
                tuning.extend({"design": "nested_outer_fold", "outer_fold": fold,
                                "representation": rep, "model": model_name, **rec}
                               for rec in candidates)

            if not np.isfinite(oof).all():
                raise ValueError(f"Missing nested OOF predictions for {rep}/{model_name}")
            threshold = core.lock_threshold(y, oof)
            rows.append({
                "design": "discovery_nested_oof", "training": cfg["discovery"],
                "cohort": cfg["discovery"], "representation": rep,
                "model": model_name, "setting": "outer-fold-specific",
                "threshold": threshold, **core.metrics(y, oof, threshold),
                "feature_panel": "outer_fold_nested_fixed_200",
                "threshold_source": "nested_outer_oof",
            })
            predictions.extend({
                "design": "discovery_nested_oof", "cohort": cfg["discovery"],
                "sample_id": sid, "representation": rep, "model": model_name,
                "label": int(yv), "probability": float(pv), "threshold": threshold,
                "feature_panel": "outer_fold_nested_fixed_200",
            } for sid, yv, pv in zip(dy.index, y, oof))

            final_selector, final_genes = select_panel(fixed, y, cfg["feature_budget"])
            panel_records.extend(panel_rows(final_selector, final_genes, rep, "full_discovery"))
            final_x = rx.loc[:, final_genes]
            best, candidates = core.tune_oof(final_x, dy.to_numpy(), model_name, cfg)
            _, setting, estimator, _ = best
            fitted = clone(estimator).fit(final_x, dy.to_numpy())
            tuning.extend({"design": "TCGA_to_GEO_fixed_feature_panel",
                           "representation": rep, "model": model_name, **rec}
                          for rec in candidates)
            for cohort in cfg["external"]:
                tx, ty = data[cohort]
                test_x = core.represent(tx, rep).loc[:, final_genes]
                p = core.predict_score(fitted, test_x)
                rows.append({
                    "design": "TCGA_to_GEO", "training": cfg["discovery"],
                    "cohort": cohort, "representation": rep, "model": model_name,
                    "setting": setting, "threshold": threshold,
                    **core.metrics(ty, p, threshold),
                    "feature_panel": "full_discovery_fixed_200",
                    "threshold_source": "nested_outer_oof",
                })
                predictions.extend({
                    "design": "TCGA_to_GEO", "cohort": cohort, "sample_id": sid,
                    "representation": rep, "model": model_name, "label": int(yv),
                    "probability": float(pv), "threshold": threshold,
                    "feature_panel": "full_discovery_fixed_200",
                } for sid, yv, pv in zip(ty.index, ty, p))

    control = pd.DataFrame(rows)
    pred = pd.DataFrame(predictions)
    panels = pd.DataFrame(panel_records).drop_duplicates(
        ["panel_id", "representation", "gene"])
    control.to_csv(OUT / "fixed_feature_control_metrics.csv", index=False)
    pred.to_csv(OUT / "fixed_feature_control_predictions.csv", index=False)
    pd.DataFrame(tuning).to_csv(OUT / "fixed_feature_control_tuning.csv", index=False)
    panels.to_csv(OUT / "fixed_feature_control_panels.csv", index=False)
    panels[panels.panel_id.eq("full_discovery")].sort_values(
        "f_statistic", ascending=False).to_csv(OUT / "fixed_feature_control_panel.csv", index=False)

    formal = pd.read_csv(OUT / "metrics.csv")
    formal = formal[(formal.design == "TCGA_to_GEO") &
                    formal.representation.isin(cfg["representations"]) &
                    formal.model.isin(cfg["models"])].copy()
    control_ext = control[control.design == "TCGA_to_GEO"].copy()
    keys = ["cohort", "representation", "model"]
    comparison = formal.merge(control_ext, on=keys, suffixes=("_formal", "_fixed_panel"))
    for metric in ["auroc", "balanced_accuracy", "brier_skill", "ici"]:
        comparison[f"delta_{metric}"] = comparison[f"{metric}_fixed_panel"] - comparison[f"{metric}_formal"]
    comparison.to_csv(OUT / "fixed_feature_control_comparison.csv", index=False)
    summary = comparison.groupby(["representation", "model"])[
        ["auroc_formal", "auroc_fixed_panel", "balanced_accuracy_formal",
         "balanced_accuracy_fixed_panel", "brier_skill_formal",
         "brier_skill_fixed_panel", "ici_formal", "ici_fixed_panel"]
    ].mean().reset_index()
    summary.to_csv(OUT / "fixed_feature_control_summary.csv", index=False)
    metadata = {
        "panel_size": cfg["feature_budget"],
        "outer_cv": cfg["cv_splits"],
        "panel_selection": "SelectKBest ANOVA on each outer training fold",
        "external_panel": "SelectKBest ANOVA on all discovery samples after nested OOF evaluation",
        "threshold_selection": "locked from nested outer-fold OOF predictions",
        "seed": cfg["seed"],
        "candidate_grid": "expanded nine-candidate grid",
        "representations": cfg["representations"],
        "models": cfg["models"],
    }
    (OUT / "fixed_feature_control_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Nested fixed-panel macro external results:")
    print(control_ext.groupby(["representation", "model"])[
        ["auroc", "balanced_accuracy", "brier_skill", "ici"]].mean().to_string())


if __name__ == "__main__":
    main()
