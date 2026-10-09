"""Small MSI-publication plotting palette shared by optional downstream scripts."""

from __future__ import annotations

import matplotlib as mpl

COHORT_ORDER = ["TCGA_COAD", "GSE13294", "GSE13067", "GSE18088", "GSE26682", "GSE39084"]
MODEL_ORDER = ["elastic_net", "svm_rbf", "xgboost"]
REPRESENTATION_ORDER = ["fixed_log_expression", "target_cohort_adaptive_zscore", "single_sample_rank"]
COLORS = {
    "ink": "#243746",
    "blue": "#0072B2",
    "green": "#009E73",
    "orange": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#E69F00",
}


def apply_publication_style() -> None:
    """Apply the typography and line-weight defaults used by the release figures."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#82909C",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
