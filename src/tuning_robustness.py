from __future__ import annotations

import warnings

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import transportability as core

warnings.filterwarnings("ignore")


def expanded_grid(name: str, k: int, seed: int):
    if name == "elastic_net":
        settings = [(c, l1) for c in (.1, .5, 2.) for l1 in (.2, .5, .8)]
        return [(f"C={c},l1={l1}", Pipeline([
            ("select", SelectKBest(f_classif, k=k)), ("scale", StandardScaler()),
            ("model", LogisticRegression(C=c, l1_ratio=l1, solver="saga", max_iter=2000,
                                         tol=1e-3, class_weight="balanced", random_state=seed))]))
                for c, l1 in settings]
    if name == "svm_rbf":
        settings = [(c, g) for c in (.1, 1., 10.) for g in (1e-3, 1e-2, 1e-1)]
        return [(f"C={c},gamma={g}", Pipeline([
            ("select", SelectKBest(f_classif, k=k)), ("scale", StandardScaler()),
            ("model", SVC(C=c, gamma=g, probability=True, class_weight="balanced", random_state=seed))]))
                for c, g in settings]
    settings = [(depth, lr, trees) for depth in (2, 3, 5) for lr, trees in ((.02, 600), (.05, 400), (.10, 250))]
    return [(f"depth={depth},lr={lr},trees={trees}", Pipeline([
        ("select", SelectKBest(f_classif, k=k)),
        ("model", core.StableXGBClassifier(max_depth=depth, learning_rate=lr, n_estimators=trees,
                                            early_stopping_rounds=20, n_jobs=2, random_state=seed))]))
            for depth, lr, trees in settings]


def calibration(y: np.ndarray, p: np.ndarray):
    intercept, slope = core.calibration_intercept_slope(y, p)
    return intercept, slope, core.integrated_calibration_index(y, p), core.expected_calibration_error(y, p)
