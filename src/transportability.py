from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, recall_score, roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from calibration_metrics import (calibration_curve_frame, calibration_intercept_slope,
                                 expected_calibration_error, integrated_calibration_index)

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "coad_msi_probe_mean"
CFG_PATH = ROOT / "configs" / "main.json"
OUT = ROOT / "results" / "generated"
TABLES = ROOT / "results" / "generated"
_REP_CACHE = {}


class StableXGBClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost with a deterministic internal validation split for early stopping."""
    def __init__(self, max_depth=3, learning_rate=0.05, n_estimators=400,
                 early_stopping_rounds=20, n_jobs=2, random_state=20261008):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        xfit, xval, yfit, yval = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=self.random_state)
        neg, pos = np.bincount(yfit, minlength=2)
        self.model_ = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            max_depth=self.max_depth, learning_rate=self.learning_rate,
            n_estimators=self.n_estimators, min_child_weight=3, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=neg / max(pos, 1),
            early_stopping_rounds=self.early_stopping_rounds, n_jobs=self.n_jobs,
            random_state=self.random_state)
        self.model_.fit(xfit, yfit, eval_set=[(xval, yval)], verbose=False)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X))


def load_config():
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def load_cohort(cohort: str):
    x = pd.read_csv(DATA / f"{cohort}_shared_expression.tsv", sep="\t", index_col=0)
    lab = pd.read_csv(DATA / f"{cohort}_labels.csv").set_index("sample_id")
    if x.index.astype(str).duplicated().any() or lab.index.astype(str).duplicated().any():
        raise ValueError(f"{cohort}: duplicate sample IDs are not allowed")
    labels = lab["label"].astype(str).str.upper()
    allowed = {"MSI", "MSI-H", "MSI_H", "MSS"}
    unknown = sorted(set(labels) - allowed)
    if unknown:
        raise ValueError(f"{cohort}: unrecognized labels {unknown}; labels are never coerced to MSS")
    if set(x.index.astype(str)) != set(lab.index.astype(str)):
        raise ValueError(f"{cohort}: expression and frozen-label sample sets differ")
    frozen_genes = pd.read_csv(DATA / "shared_gene_set.csv").iloc[:, 0].astype(str).tolist()
    if x.columns.astype(str).tolist() != frozen_genes or len(frozen_genes) != 11316:
        raise ValueError(f"{cohort}: expected the frozen ordered 11,316-gene interface")
    ordered = lab.index.astype(str).tolist()
    x = x.loc[ordered].astype("float32")
    y = labels.loc[ordered].isin(["MSI", "MSI-H", "MSI_H"]).astype(int)
    expected_counts = {
        "TCGA_COAD": (160, 37, 123), "GSE13294": (155, 78, 77),
        "GSE13067": (74, 11, 63), "GSE18088": (53, 19, 34),
        "GSE26682": (137, 18, 119), "GSE39084": (67, 16, 51),
    }
    observed = (len(y), int(y.sum()), int((1-y).sum()))
    if cohort in expected_counts and observed != expected_counts[cohort]:
        raise ValueError(f"{cohort}: observed N/MSI/MSS {observed}, expected {expected_counts[cohort]}")
    return x, y


def represent(x: pd.DataFrame, name: str) -> pd.DataFrame:
    key = (id(x), name)
    if key in _REP_CACHE:
        return _REP_CACHE[key]
    a = x.to_numpy(dtype=np.float32, copy=True)
    if name == "target_cohort_adaptive_zscore":
        mu, sd = np.nanmean(a, axis=0), np.nanstd(a, axis=0)
        sd[sd < 1e-6] = 1.0
        a = (a - mu) / sd
    elif name == "single_sample_rank":
        a = np.apply_along_axis(lambda z: rankdata(z, method="average") / len(z), 1, a).astype("float32")
    represented = pd.DataFrame(a, index=x.index, columns=x.columns)
    _REP_CACHE[key] = represented
    return represented


def model_grid(name: str, k: int, seed: int):
    if name == "elastic_net":
        settings = [(0.3, .5), (1.0, .75)]
        return [(f"C={c},l1={l1}", Pipeline([("select", SelectKBest(f_classif, k=k)), ("scale", StandardScaler()),
                 ("model", LogisticRegression(C=c, penalty="elasticnet", l1_ratio=l1, solver="saga", max_iter=1500,
                                                tol=1e-3, class_weight="balanced", random_state=seed))])) for c, l1 in settings]
    if name == "svm_rbf":
        settings = [(0.3, "scale"), (1.0, "scale")]
        return [(f"C={c},gamma={g}", Pipeline([("select", SelectKBest(f_classif, k=k)), ("scale", StandardScaler()),
                 ("model", SVC(C=c, gamma=g, probability=True, class_weight="balanced", random_state=seed))])) for c, g in settings]
    settings = [(3, .03, 500), (5, .05, 400)]
    return [(f"depth={depth},lr={lr},trees={trees}", Pipeline([("select", SelectKBest(f_classif, k=k)),
             ("model", StableXGBClassifier(max_depth=depth, learning_rate=lr, n_estimators=trees,
                                            early_stopping_rounds=20, n_jobs=2,
                                            random_state=seed))])) for depth, lr, trees in settings]


def predict_score(model, x):
    return model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x)


def tune_oof(x, y, model_name, cfg, groups=None):
    splitter = LeaveOneGroupOut().split(x, y, groups) if groups is not None else StratifiedKFold(
        cfg["cv_splits"], shuffle=True, random_state=cfg["seed"]).split(x, y)
    splits = list(splitter)
    candidates = model_grid(model_name, min(cfg["feature_budget"], x.shape[1]), cfg["seed"])
    best = None
    records = []
    for label, estimator in candidates:
        oof = np.full(len(y), np.nan)
        for tr, va in splits:
            fitted = clone(estimator).fit(x.iloc[tr], y.iloc[tr])
            oof[va] = predict_score(fitted, x.iloc[va])
        score = roc_auc_score(y, oof)
        records.append({"candidate": label, "cv_auroc": score})
        if best is None or score > best[0]:
            best = (score, label, estimator, oof)
    return best, records


def lock_threshold(y, p):
    candidates = np.unique(np.r_[0.0, p, 1.0])
    rows = []
    for t in candidates:
        pred = p >= t
        rows.append((balanced_accuracy_score(y, pred), f1_score(y, pred, zero_division=0), -abs(t-.5), t))
    return max(rows)[-1]


def metrics(y, p, threshold):
    pred = p >= threshold
    prevalence = float(np.mean(y))
    null_brier = prevalence * (1 - prevalence)
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p), "brier_skill": 1 - brier_score_loss(y, p) / null_brier,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y, pred, pos_label=0, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "calibration_intercept": intercept, "calibration_slope": slope,
        "ici": integrated_calibration_index(y, p),
        "ece_10bin": expected_calibration_error(y, p),
    }


def run_cross_platform(data, cfg):
    dx, dy = data[cfg["discovery"]]
    result, predictions, tuning = [], [], []
    for rep in cfg["representations"]:
        xtrain = represent(dx, rep)
        for model_name in cfg["models"]:
            print(f"Cross-platform: {rep} / {model_name}", flush=True)
            best, candidates = tune_oof(xtrain, dy, model_name, cfg)
            _, setting, estimator, oof = best
            threshold = lock_threshold(dy.to_numpy(), oof)
            fitted = clone(estimator).fit(xtrain, dy)
            for rec in candidates:
                tuning.append({"design": "TCGA_to_GEO", "representation": rep, "model": model_name, **rec})
            result.append({"design": "discovery_oof", "training": "TCGA_COAD", "cohort": "TCGA_COAD",
                           "representation": rep, "model": model_name, "setting": setting,
                           "threshold": threshold, **metrics(dy, oof, threshold)})
            for cohort in cfg["external"]:
                tx, ty = data[cohort]
                p = predict_score(fitted, represent(tx, rep))
                result.append({"design": "TCGA_to_GEO", "training": "TCGA_COAD", "cohort": cohort,
                               "representation": rep, "model": model_name, "setting": setting,
                               "threshold": threshold, **metrics(ty, p, threshold)})
                predictions.extend({"design": "TCGA_to_GEO", "cohort": cohort, "sample_id": sid,
                                    "representation": rep, "model": model_name, "label": int(yv),
                                    "probability": float(pv), "threshold": threshold}
                                   for sid, yv, pv in zip(ty.index, ty, p))
    return result, predictions, tuning


def run_loco(data, cfg):
    rows, preds, tuning = [], [], []
    for held in cfg["external"]:
        sources = [c for c in cfg["external"] if c != held]
        for rep in cfg["representations"]:
            xs, ys, gs = [], [], []
            for c in sources:
                x, y = data[c]
                xs.append(represent(x, rep)); ys.append(y); gs.extend([c] * len(y))
            train_x, train_y = pd.concat(xs), pd.concat(ys)
            for model_name in cfg["models"]:
                print(f"GEO LOCO {held}: {rep} / {model_name}", flush=True)
                best, candidates = tune_oof(train_x, train_y, model_name, cfg, np.asarray(gs))
                _, setting, estimator, oof = best
                threshold = lock_threshold(train_y.to_numpy(), oof)
                fitted = clone(estimator).fit(train_x, train_y)
                tx, ty = data[held]
                p = predict_score(fitted, represent(tx, rep))
                rows.append({"design": "GEO_LOCO", "training": "+".join(sources), "cohort": held,
                             "representation": rep, "model": model_name, "setting": setting,
                             "threshold": threshold, **metrics(ty, p, threshold)})
                tuning.extend({"design": f"GEO_LOCO:{held}", "representation": rep, "model": model_name, **rec}
                              for rec in candidates)
                preds.extend({"design": "GEO_LOCO", "cohort": held, "sample_id": sid,
                              "representation": rep, "model": model_name, "label": int(yv),
                              "probability": float(pv), "threshold": threshold}
                             for sid, yv, pv in zip(ty.index, ty, p))
    return rows, preds, tuning


def _bootstrap_stability(rx, y, cfg, seed):
    """Repeat the complete screen -> scale -> sparse-fit pipeline on all genes."""
    yarr = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(yarr == 1), np.flatnonzero(yarr == 0)
    n_genes = rx.shape[1]
    coefficients = np.zeros((cfg["stability_bootstraps"], n_genes), dtype=np.float32)
    for b in range(cfg["stability_bootstraps"]):
        idx = np.r_[rng.choice(pos, len(pos), replace=True),
                    rng.choice(neg, len(neg), replace=True)]
        screen = SelectKBest(f_classif, k=min(cfg["feature_budget"], n_genes)).fit(rx.iloc[idx], yarr[idx])
        keep = screen.get_support()
        scaler = StandardScaler().fit(rx.iloc[idx, keep])
        z = scaler.transform(rx.iloc[idx, keep])
        model = LogisticRegression(C=.15, penalty="l1", solver="liblinear", class_weight="balanced",
                                   random_state=seed + b).fit(z, yarr[idx])
        coefficients[b, keep] = model.coef_[0]
    selected_mask = np.abs(coefficients) > 1e-8
    signs = np.sign(coefficients)
    stability = pd.DataFrame({"gene": rx.columns,
                              "selection_frequency": selected_mask.mean(axis=0),
                              "median_coefficient": np.median(coefficients, axis=0),
                              "median_abs_coefficient": np.median(np.abs(coefficients), axis=0),
                              "bootstrap_count": cfg["stability_bootstraps"]})
    stability["sign_consistency"] = [
        float(max((signs[:, j] > 0).sum(), (signs[:, j] < 0).sum()) /
              max(selected_mask[:, j].sum(), 1))
        for j in range(n_genes)]
    stability = stability.sort_values(["selection_frequency", "median_abs_coefficient"], ascending=False)
    return stability.reset_index(drop=True)


def fit_compact(data, cfg):
    dx, dy = data[cfg["discovery"]]
    rx = represent(dx, "single_sample_rank")
    yarr = dy.to_numpy()
    stability = _bootstrap_stability(rx, dy, cfg, cfg["seed"])
    selected = stability.head(cfg["compact_gene_count"])["gene"].tolist()
    estimator = Pipeline([("scale", StandardScaler()),
                          ("model", LogisticRegression(C=1., class_weight="balanced", max_iter=3000,
                                                       random_state=cfg["seed"]))])
    skf = StratifiedKFold(cfg["cv_splits"], shuffle=True, random_state=cfg["seed"])
    oof = np.zeros(len(dy))
    nested_selection = []
    for fold, (tr, va) in enumerate(skf.split(rx, dy)):
        fold_stability = _bootstrap_stability(rx.iloc[tr], dy.iloc[tr], cfg,
                                              cfg["seed"] + 100000 + fold)
        fold_selected = fold_stability.head(cfg["compact_gene_count"])["gene"].tolist()
        nested_selection.extend({"fold": fold, "gene": gene, "selected": True} for gene in fold_selected)
        m = clone(estimator).fit(rx.iloc[tr][fold_selected], dy.iloc[tr])
        oof[va] = predict_score(m, rx.iloc[va][fold_selected])
    threshold = lock_threshold(yarr, oof)
    fitted = clone(estimator).fit(rx[selected], dy)
    rows, preds = [], []
    rows.append({"design": "discovery_oof", "training": "TCGA_COAD", "cohort": "TCGA_COAD",
                 "representation": "single_sample_rank", "model": "compact_rank_20", "setting": "20 stable genes",
                 "threshold": threshold, **metrics(dy, oof, threshold)})
    for cohort in cfg["external"]:
        x, y = data[cohort]
        p = predict_score(fitted, represent(x, "single_sample_rank")[selected])
        rows.append({"design": "TCGA_to_GEO", "training": "TCGA_COAD", "cohort": cohort,
                     "representation": "single_sample_rank", "model": "compact_rank_20", "setting": "20 stable genes",
                     "threshold": threshold, **metrics(y, p, threshold)})
        preds.extend({"design": "TCGA_to_GEO", "cohort": cohort, "sample_id": sid,
                      "representation": "single_sample_rank", "model": "compact_rank_20", "label": int(yv),
                      "probability": float(pv), "threshold": threshold}
                     for sid, yv, pv in zip(y.index, y, p))
    scaler, lm = fitted.named_steps["scale"], fitted.named_steps["model"]
    frozen = pd.DataFrame({"gene": selected, "coefficient_scaled": lm.coef_[0],
                           "rank_mean": scaler.mean_, "rank_scale": scaler.scale_})
    frozen["coefficient_rank"] = frozen["coefficient_scaled"] / frozen["rank_scale"]
    intercept = float(lm.intercept_[0] - np.sum(lm.coef_[0] * scaler.mean_ / scaler.scale_))
    metadata = {"intercept_rank": intercept, "threshold": threshold, "genes": selected,
                "stability_scope": "all genes -> 200-feature screen -> scaling -> L1 fit",
                "nested_oof_scope": "each outer fold repeated the complete 500-bootstrap stability procedure"}
    return rows, preds, stability, frozen, metadata


def run_domain(data, cfg):
    rows, distances = [], []
    cohorts = np.concatenate([[c] * len(data[c][0]) for c in [cfg["discovery"], *cfg["external"]]])
    for rep in cfg["representations"]:
        parts = [represent(data[c][0], rep) for c in [cfg["discovery"], *cfg["external"]]]
        x = pd.concat(parts)
        variances = x.var().nlargest(min(cfg["feature_budget"], x.shape[1])).index
        xr = x[variances]
        splitter = StratifiedKFold(5, shuffle=True, random_state=cfg["seed"])
        proba = np.zeros((len(xr), len(np.unique(cohorts))))
        predicted = np.empty(len(xr), dtype=object)
        labels = np.unique(cohorts)
        for tr, va in splitter.split(xr, cohorts):
            clf = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
            clf.fit(xr.iloc[tr], cohorts[tr])
            predicted[va] = clf.predict(xr.iloc[va])
            for j, label in enumerate(labels):
                proba[va, j] = clf.predict_proba(xr.iloc[va])[:, list(clf.classes_).index(label)]
        ybin = label_binarize(cohorts, classes=labels)
        rows.append({"representation": rep,
                     "source_macro_ovr_auroc": roc_auc_score(ybin, proba, average="macro", multi_class="ovr"),
                     "source_balanced_accuracy": balanced_accuracy_score(cohorts, predicted),
                     "chance_balanced_accuracy": 1 / len(labels)})
        tc = xr.loc[cohorts == cfg["discovery"]].mean().to_numpy()
        for c in cfg["external"]:
            cc = xr.loc[cohorts == c].mean().to_numpy()
            distances.append({"representation": rep, "cohort": c,
                              "standardized_centroid_distance": float(np.linalg.norm(tc - cc) / np.sqrt(len(cc)))})
    return rows, distances


def summarize(metrics_df, cfg):
    ext = metrics_df[(metrics_df.design.isin(["TCGA_to_GEO", "GEO_LOCO"]))]
    measure = ["auroc", "auprc", "brier", "brier_skill", "balanced_accuracy", "sensitivity", "specificity", "f1",
               "calibration_intercept", "calibration_slope", "ici", "ece_10bin"]
    return ext.groupby(["design", "representation", "model"])[measure].agg(["mean", "std", "median", "min"]).reset_index()


def paired_contrasts(metrics_df, cfg):
    rng = np.random.default_rng(cfg["seed"])
    ext = metrics_df[metrics_df.design == "TCGA_to_GEO"].copy()
    score = ext.groupby(["representation", "model"]).auroc.mean()
    best_model = {r: score.loc[r].idxmax() for r in cfg["representations"]}
    comparisons = [
        ("adaptive_vs_fixed", ("target_cohort_adaptive_zscore", best_model["target_cohort_adaptive_zscore"]),
         ("fixed_log_expression", best_model["fixed_log_expression"])),
        ("rank_vs_fixed", ("single_sample_rank", best_model["single_sample_rank"]),
         ("fixed_log_expression", best_model["fixed_log_expression"])),
        ("compact_rank_vs_complex_fixed", ("single_sample_rank", "compact_rank_20"),
         ("fixed_log_expression", best_model["fixed_log_expression"])),
    ]
    rows = []
    for name, a, b in comparisons:
        for metric in ["auroc", "balanced_accuracy", "brier_skill"]:
            av = ext[(ext.representation == a[0]) & (ext.model == a[1])].set_index("cohort")[metric]
            bv = ext[(ext.representation == b[0]) & (ext.model == b[1])].set_index("cohort")[metric]
            d = av - bv
            boots = np.array([rng.choice(d.to_numpy(), len(d), replace=True).mean() for _ in range(cfg["bootstrap_replicates"])])
            rows.append({"contrast": name, "metric": metric, "model_a": a[1], "model_b": b[1],
                         "mean_difference": d.mean(), "ci_low": np.quantile(boots, .025), "ci_high": np.quantile(boots, .975),
                         "wins": int((d > 1e-8).sum()), "ties": int((abs(d) <= 1e-8).sum()), "losses": int((d < -1e-8).sum())})
    return pd.DataFrame(rows)


def main():
    cfg = load_config(); OUT.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    cohorts = [cfg["discovery"], *cfg["external"]]
    data = {c: load_cohort(c) for c in cohorts}
    cross, cross_pred, tune1 = run_cross_platform(data, cfg)
    pd.DataFrame(cross).to_csv(OUT / "cross_platform_metrics.csv", index=False)
    pd.DataFrame(cross_pred).to_csv(OUT / "cross_platform_predictions.csv", index=False)
    loco, loco_pred, tune2 = run_loco(data, cfg)
    compact, compact_pred, stability, frozen, metadata = fit_compact(data, cfg)
    domain, distances = run_domain(data, cfg)
    metrics_df = pd.DataFrame(cross + loco + compact)
    predictions_df = pd.DataFrame(cross_pred + loco_pred + compact_pred)
    metrics_df.to_csv(OUT / "all_metrics.csv", index=False)
    predictions_df.to_csv(OUT / "all_predictions.csv", index=False)
    curves = []
    for keys, group in predictions_df.groupby(["design", "cohort", "representation", "model"]):
        curve = calibration_curve_frame(group.label, group.probability)
        for name, value in zip(["design", "cohort", "representation", "model"], keys):
            curve[name] = value
        curves.append(curve)
    pd.concat(curves, ignore_index=True).to_csv(OUT / "calibration_curves.csv", index=False)
    pd.DataFrame(tune1 + tune2).to_csv(OUT / "tuning_results.csv", index=False)
    stability.to_csv(OUT / "compact_gene_stability.csv", index=False)
    frozen.to_csv(OUT / "compact_rank_frozen_coefficients.csv", index=False)
    (OUT / "compact_rank_model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(domain).to_csv(OUT / "domain_classifier.csv", index=False)
    pd.DataFrame(distances).to_csv(OUT / "domain_distances.csv", index=False)
    summary = summarize(metrics_df, cfg); summary.to_csv(TABLES / "transportability_summary.csv", index=False)
    contrasts = paired_contrasts(metrics_df, cfg); contrasts.to_csv(TABLES / "paired_bootstrap_contrasts.csv", index=False)
    print(metrics_df.groupby(["design", "representation", "model"])[["auroc", "balanced_accuracy", "brier_skill"]].mean().to_string())
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()

