from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, pdist

import transportability as core
import tuning_robustness as expanded

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "supplementary"


def source_accuracy(data, cfg, label_value):
    core._REP_CACHE.clear()
    cohorts = [cfg["discovery"], *cfg["external"]]
    parts, source = [], []
    for cohort in cohorts:
        x, y = data[cohort]
        keep = y.to_numpy() == label_value
        parts.append(x.loc[keep])
        source.extend([cohort] * int(keep.sum()))
    source = np.asarray(source)
    rows = []
    for rep in ["fixed_log_expression", "single_sample_rank"]:
        x = pd.concat([core.represent(z, rep) for z in parts])
        pred = np.empty(len(x), dtype=object)
        cv = StratifiedKFold(5, shuffle=True, random_state=cfg["seed"])
        for tr, va in cv.split(x, source):
            model = Pipeline([("select", SelectKBest(f_classif, k=min(cfg["feature_budget"], x.shape[1]))),
                              ("scale", StandardScaler()),
                              ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                           random_state=cfg["seed"]))])
            model.fit(x.iloc[tr], source[tr])
            pred[va] = model.predict(x.iloc[va])
        rows.append({"label_stratum": "MSI" if label_value else "MSS", "representation": rep,
                     "n": len(x), "source_balanced_accuracy": balanced_accuracy_score(source, pred),
                     "chance_balanced_accuracy": 1 / len(cohorts)})
    return rows


def _log_kernel_values(train, query, gamma, rng, max_values=4000):
    """Return RBF log-similarities without exponentiating underflowing values."""
    distances = cdist(query, train, metric="sqeuclidean")
    values = (-float(gamma) * distances).ravel()
    if len(values) > max_values:
        values = rng.choice(values, max_values, replace=False)
    return values


def _within_log_kernel_values(train, gamma, rng, max_values=10000):
    values = -float(gamma) * pdist(train, metric="sqeuclidean")
    if len(values) > max_values:
        values = rng.choice(values, max_values, replace=False)
    return values


def _kernel_rows(values, cohort, kernel_type, design, held_out=None, gamma=None):
    return [{"design": design, "held_out": held_out or "", "cohort": cohort,
             "kernel_type": kernel_type, "log_kernel_similarity": float(v),
             "kernel_similarity": float(np.exp(v)) if v > -745 else 0.0,
             "gamma": float(gamma) if gamma is not None else np.nan}
            for v in values]


def kernel_geometry(data, cfg):
    core._REP_CACHE.clear()
    x, y = data[cfg["discovery"]]
    x = core.represent(x, "single_sample_rank")
    core.model_grid = expanded.expanded_grid
    best, _ = core.tune_oof(x, y, "svm_rbf", cfg)
    _, setting, estimator, _ = best
    fitted = clone(estimator).fit(x, y)
    train = fitted.named_steps["scale"].transform(fitted.named_steps["select"].transform(x))
    svm = fitted.named_steps["model"]
    rng = np.random.default_rng(cfg["seed"])
    gamma = float(svm._gamma)
    tk = _within_log_kernel_values(train, gamma, rng)
    rows = _kernel_rows(tk, "TCGA_COAD", "discovery within", "TCGA_to_GEO", gamma=gamma)
    scores = []
    for cohort in cfg["external"]:
        tx, ty = data[cohort]
        rx = core.represent(tx, "single_sample_rank")
        scaled = fitted.named_steps["scale"].transform(fitted.named_steps["select"].transform(rx))
        ck = _log_kernel_values(train, scaled, gamma, rng)
        rows.extend(_kernel_rows(ck, cohort, "external cross", "TCGA_to_GEO", gamma=gamma))
        decision = fitted.decision_function(rx)
        probability = fitted.predict_proba(rx)[:, 1]
        scores.extend({"cohort": cohort, "sample_id": sid, "label": int(label),
                       "decision_score": float(d), "probability": float(p), "setting": setting}
                      for sid, label, d, p in zip(rx.index, ty, decision, probability))
    return rows, scores


def kernel_geometry_loco(data, cfg):
    """Repeat the rank-SVM geometry diagnostic for each GEO-LOCO training set."""
    core._REP_CACHE.clear()
    rows, scores = [], []
    rng = np.random.default_rng(cfg["seed"] + 17)
    for held in cfg["external"]:
        sources = [c for c in cfg["external"] if c != held]
        parts = [core.represent(data[c][0], "single_sample_rank") for c in sources]
        train_x = pd.concat(parts)
        train_y = pd.concat([data[c][1] for c in sources])
        best, _ = core.tune_oof(train_x, train_y, "svm_rbf", cfg, np.asarray(
            [c for c in sources for _ in range(len(data[c][0]))]))
        _, setting, estimator, _ = best
        fitted = clone(estimator).fit(train_x, train_y)
        selector = fitted.named_steps["select"]
        scaler = fitted.named_steps["scale"]
        train = scaler.transform(selector.transform(train_x))
        svm = fitted.named_steps["model"]
        gamma = float(svm._gamma)
        rows.extend(_kernel_rows(_within_log_kernel_values(train, gamma, rng),
                                 "+".join(sources), "LOCO training within", "GEO_LOCO",
                                 held_out=held, gamma=gamma))
        tx, ty = data[held]
        rx = core.represent(tx, "single_sample_rank")
        scaled = scaler.transform(selector.transform(rx))
        rows.extend(_kernel_rows(_log_kernel_values(train, scaled, gamma, rng), held,
                                 "LOCO external cross", "GEO_LOCO", held_out=held,
                                 gamma=gamma))
        decision = fitted.decision_function(rx)
        probability = fitted.predict_proba(rx)[:, 1]
        scores.extend({"design": "GEO_LOCO", "held_out": held, "cohort": held,
                       "sample_id": sid, "label": int(label), "decision_score": float(d),
                       "probability": float(p), "setting": setting}
                      for sid, label, d, p in zip(rx.index, ty, decision, probability))
    return rows, scores


def main():
    cfg = core.load_config()
    core.DATA = ROOT / "data" / "processed" / "coad_msi_probe_mean"
    data = {c: core.load_cohort(c) for c in [cfg["discovery"], *cfg["external"]]}
    source = source_accuracy(data, cfg, 0) + source_accuracy(data, cfg, 1)
    kernels, scores = kernel_geometry(data, cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source).to_csv(OUT / "source_classifier_stratified.csv", index=False)
    pd.DataFrame(kernels).to_csv(OUT / "rank_svm_kernel_distribution.csv", index=False)
    pd.DataFrame(scores).to_csv(OUT / "rank_svm_external_scores.csv", index=False)
    print(pd.DataFrame(source).to_string(index=False))
    print(pd.DataFrame(scores).groupby("cohort")[["decision_score", "probability"]].agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()

