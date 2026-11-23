"""
Evaluation metrics for PD models: discrimination and calibration, reported separately.

Discrimination : AUC, Gini (=2AUC-1), KS, PR-AUC (average precision)
Calibration    : Brier score, log-loss, ECE (equal-mass bins),
                 calibration-in-the-large (mean PD - observed default rate),
                 calibration slope (logistic recalibration of logit(p) on y; ideal = 1)
"""
from __future__ import annotations

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score, roc_curve)

EPS = 1e-6
HIGHER_IS_BETTER = {"auc": True, "gini": True, "ks": True, "pr_auc": True,
                    "brier": False, "logloss": False, "ece": False,
                    "cal_large_abs": False, "cal_slope_dev": False}


def ks_statistic(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def ece_equal_mass(y, p, n_bins=10):
    """Expected calibration error with equal-frequency (quantile) bins."""
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    err = 0.0
    for b in bins:
        if len(b):
            err += len(b) * abs(p[b].mean() - y[b].mean())
    return float(err / len(p))


def calibration_slope(y, p):
    z = logit(np.clip(p, EPS, 1 - EPS)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0])


def evaluate(y, p) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    auc = roc_auc_score(y, p)
    slope = calibration_slope(y, p)
    return dict(
        auc=auc, gini=2 * auc - 1, ks=ks_statistic(y, p),
        pr_auc=average_precision_score(y, p), base_rate=float(y.mean()),
        brier=brier_score_loss(y, p), logloss=log_loss(y, p), ece=ece_equal_mass(y, p),
        cal_large=float(p.mean() - y.mean()), cal_large_abs=float(abs(p.mean() - y.mean())),
        cal_slope=slope, cal_slope_dev=float(abs(slope - 1)),
    )
