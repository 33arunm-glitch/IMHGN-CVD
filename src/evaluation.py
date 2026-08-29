from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    out["roc_auc"] = (
        float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["tn"], out["fp"], out["fn"], out["tp"] = [int(v) for v in cm.ravel()]
    return out


def calibration_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame({"mean_predicted_probability": mean_pred, "observed_fraction_positive": frac_pos})


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(int(n_boot)):
        sample = rng.choice(arr, size=len(arr), replace=True)
        stats.append(np.mean(sample))
    lower = np.quantile(stats, alpha / 2)
    upper = np.quantile(stats, 1 - alpha / 2)
    return {"mean": float(np.mean(arr)), "lower": float(lower), "upper": float(upper)}


def paired_wilcoxon(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> Dict[str, float | bool]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return {"statistic": float("nan"), "p_value": float("nan"), "significant": False}
    if np.allclose(a, b):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return {"statistic": float(stat), "p_value": float(p), "significant": bool(p < alpha)}
