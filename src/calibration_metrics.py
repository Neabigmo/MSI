"""Calibration estimands used by the transportability analysis.

ICI follows Austin and Steyerberg (2019): a LOWESS estimate of observed risk is
evaluated at every subject's predicted risk and the mean absolute difference is
reported. Quantile-binned ECE is retained under its own name as a sensitivity
analysis; it is never labelled ICI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from statsmodels.nonparametric.smoothers_lowess import lowess


def calibration_intercept_slope(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000).fit(logit, y)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def lowess_calibration(y, p, frac=0.75):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if np.ptp(p) < 1e-12:
        return np.full_like(p, np.mean(y), dtype=float)
    order = np.argsort(p, kind="mergesort")
    fitted = lowess(y[order], p[order], frac=frac, it=0, return_sorted=False)
    fitted = np.clip(fitted, 0.0, 1.0)
    observed = np.empty_like(fitted)
    observed[order] = fitted
    return observed


def integrated_calibration_index(y, p, frac=0.75):
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(lowess_calibration(y, p, frac=frac) - p)))


def expected_calibration_error(y, p, n_bins=10):
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    if frame.p.nunique() < 2:
        return float(abs(frame.y.mean() - frame.p.mean()))
    frame["bin"] = pd.qcut(frame.p, q=min(n_bins, max(2, len(frame) // 10)), duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
    return float(np.average(np.abs(grouped.observed - grouped.predicted), weights=grouped.n))


def calibration_curve_frame(y, p, grid_size=101, frac=0.75):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if np.ptp(p) < 1e-12:
        grid = np.linspace(0, 1, grid_size)
        return pd.DataFrame({"predicted_probability": grid,
                             "smoothed_observed_probability": np.repeat(np.mean(y), grid_size)})
    order = np.argsort(p, kind="mergesort")
    fitted = np.clip(lowess(y[order], p[order], frac=frac, it=0, return_sorted=False), 0, 1)
    unique_p, unique_idx = np.unique(p[order], return_index=True)
    unique_fit = fitted[unique_idx]
    grid = np.linspace(0, 1, grid_size)
    curve = np.interp(grid, unique_p, unique_fit, left=unique_fit[0], right=unique_fit[-1])
    return pd.DataFrame({"predicted_probability": grid, "smoothed_observed_probability": curve})
