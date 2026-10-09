"""Joint model-and-threshold variability under discovery-data resampling.

Each repetition resamples the original discovery sample IDs within label
strata, deduplicates the in-bag IDs before fitting, and obtains predictions
only for the complementary out-of-bag IDs. The threshold is therefore
locked on observations that were not used to fit that repetition's model.
External probabilities remain those from the formal full-discovery fit; the
analysis estimates joint development and threshold variability without reusing external
labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score

import transportability as core
import tuning_robustness as expanded

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "coad_msi_probe_mean"
OUT = ROOT / "results" / "generated"


def _oob_resample(rng: np.random.Generator, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unique in-bag and complementary OOB original-row indices."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    inbag_draw = np.r_[rng.choice(pos, len(pos), replace=True),
                       rng.choice(neg, len(neg), replace=True)]
    inbag = np.unique(inbag_draw)
    all_rows = np.arange(len(y), dtype=int)
    oob = np.setdiff1d(all_rows, inbag, assume_unique=True)
    return inbag, oob


def main() -> None:
    cfg = core.load_config()
    core.DATA = DATA
    core.model_grid = expanded.expanded_grid
    core._REP_CACHE.clear()
    OUT.mkdir(parents=True, exist_ok=True)

    cohorts = [cfg["discovery"], *cfg["external"]]
    data = {cohort: core.load_cohort(cohort) for cohort in cohorts}
    dx, dy = data[cfg["discovery"]]
    rx = core.represent(dx, "single_sample_rank")

    formal_metrics = pd.read_csv(OUT / "metrics.csv")
    selected = formal_metrics[(formal_metrics.design == "discovery_oof") &
                              (formal_metrics.representation == "single_sample_rank") &
                              (formal_metrics.model == "elastic_net")]
    if len(selected) != 1:
        raise ValueError("Expected exactly one formal rank Elastic Net discovery row")
    selected = selected.iloc[0]
    setting = selected.setting
    estimator = dict(core.model_grid("elastic_net", cfg["feature_budget"], cfg["seed"]))[setting]

    external_predictions = pd.read_csv(OUT / "predictions.csv")
    external_predictions = external_predictions[
        (external_predictions.design == "TCGA_to_GEO") &
        (external_predictions.representation == "single_sample_rank") &
        (external_predictions.model == "elastic_net")
    ]

    y = dy.to_numpy(dtype=int)
    rng = np.random.default_rng(cfg["seed"] + 909)
    rows = []
    for b in range(cfg["stability_bootstraps"]):
        inbag, oob = _oob_resample(rng, y)
        attempts = 0
        while len(oob) < 20 or len(np.unique(y[oob])) < 2:
            inbag, oob = _oob_resample(rng, y)
            attempts += 1
            if attempts > 100:
                raise RuntimeError("Could not obtain a two-class OOB sample")
        fitted = clone(estimator).fit(rx.iloc[inbag], y[inbag])
        oob_probability = core.predict_score(fitted, rx.iloc[oob])
        threshold = core.lock_threshold(y[oob], oob_probability)
        row = {
            "bootstrap": b,
            "threshold": float(threshold),
            "setting": setting,
            "n_total": len(y),
            "n_inbag_unique": len(inbag),
            "n_oob": len(oob),
            "oob_msi": int(y[oob].sum()),
            "oob_mss": int((1 - y[oob]).sum()),
            "sample_unit": "original discovery sample ID",
            "fit_oob_overlap": 0,
        }
        for cohort in cfg["external"]:
            z = external_predictions[external_predictions.cohort == cohort]
            row[f"balanced_accuracy_{cohort}"] = balanced_accuracy_score(
                z.label, z.probability >= threshold)
        rows.append(row)
        if (b + 1) % 50 == 0:
            print(f"Threshold stability OOB: {b + 1}/{cfg['stability_bootstraps']}", flush=True)

    draws = pd.DataFrame(rows)
    draws.to_csv(OUT / "threshold_stability_draws.csv", index=False)
    summary_rows = []
    locked = float(selected.threshold)
    for cohort in cfg["external"]:
        values = draws[f"balanced_accuracy_{cohort}"]
        z = external_predictions[external_predictions.cohort == cohort]
        summary_rows.append({
            "cohort": cohort,
            "locked_threshold": locked,
            "threshold_q025": draws.threshold.quantile(.025),
            "threshold_median": draws.threshold.median(),
            "threshold_q975": draws.threshold.quantile(.975),
            "threshold_min": draws.threshold.min(),
            "threshold_max": draws.threshold.max(),
            "balanced_accuracy_at_locked": balanced_accuracy_score(
                z.label, z.probability >= locked),
            "balanced_accuracy_q025": values.quantile(.025),
            "balanced_accuracy_median": values.median(),
            "balanced_accuracy_q975": values.quantile(.975),
            "balanced_accuracy_min": values.min(),
            "balanced_accuracy_max": values.max(),
        })
    pd.DataFrame(summary_rows).to_csv(OUT / "threshold_stability_summary.csv", index=False)
    metadata = {
        "bootstrap_count": cfg["stability_bootstraps"],
        "resampling": "class-stratified bootstrap of original IDs; duplicate in-bag draws deduplicated",
        "threshold_predictions": "out-of-bag predictions only; each OOB ID absent from its fitted model",
        "fit_oob_overlap": 0,
        "threshold_rule": "balanced accuracy, F1, then proximity to 0.5",
        "model": "single_sample_rank / elastic_net",
        "setting": setting,
        "external_probabilities": "formal full-discovery fit; external labels used only for descriptive evaluation",
    }
    (OUT / "threshold_stability_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
