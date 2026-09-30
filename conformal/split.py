"""Split (inductive) conformal prediction and its variants.

All methods here fit on a training fold, compute nonconformity scores on a
disjoint calibration fold, and use the k-th smallest score with
k = ceil((n + 1)(1 - alpha)). If the scores are exchangeable and continuous,
    1 - alpha <= P(Y in C(X)) <= 1 - alpha + 1 / (n + 1),
and, conditional on the calibration set, coverage ~ Beta(k, n + 1 - k).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta as beta_dist


def conformal_rank(n: int, alpha: float) -> int:
    """k = ceil((n + 1)(1 - alpha)). Guards against float error, e.g. 1000 * 0.9."""
    return int(math.ceil((n + 1) * (1.0 - alpha) - 1e-9))


def kth_smallest(scores: np.ndarray, k: int) -> float:
    n = len(scores)
    if k > n:
        return math.inf
    if k < 1:
        return -math.inf
    return float(np.partition(scores, k - 1)[k - 1])


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    return kth_smallest(scores, conformal_rank(len(scores), alpha))


# ------------------------------------------------------------ calibration-conditional law


def coverage_law(n: int, alpha: float, k: int | None = None):
    """Frozen Beta(k, n + 1 - k), the law of coverage given the calibration set."""
    k = conformal_rank(n, alpha) if k is None else k
    return beta_dist(k, n + 1 - k)


def pac_rank(n: int, alpha: float, delta: float) -> int:
    """Smallest k with P(coverage < 1 - alpha) <= delta over the calibration draw.

    This gives training-conditional (PAC) validity: with probability >= 1 - delta
    over the calibration set, the realized coverage is at least 1 - alpha.
    Returns n + 1 (an infinite interval) if even k = n is not enough.
    """
    for k in range(conformal_rank(n, alpha), n + 1):
        if beta_dist.cdf(1.0 - alpha, k, n + 1 - k) <= delta:
            return k
    return n + 1


# ------------------------------------------------------------ interval constructors


def split_abs(model, X_tr, y_tr, X_cal, y_cal, X_te, alpha, k=None):
    """Absolute-residual score |y - mu(x)|. Constant width everywhere."""
    model.fit(X_tr, y_tr)
    scores = np.abs(y_cal - model.predict(X_cal))
    q = kth_smallest(scores, conformal_rank(len(scores), alpha) if k is None else k)
    mu = model.predict(X_te)
    return mu - q, mu + q


def split_normalized(model, scale_model, X_tr, y_tr, X_cal, y_cal, X_te, alpha):
    """Locally weighted score |y - mu(x)| / s(x), with s fit to |residuals| on the training fold."""
    model.fit(X_tr, y_tr)
    abs_res = np.abs(y_tr - model.predict(X_tr))
    scale_model.fit(X_tr, abs_res)
    floor = 1e-3 * max(float(abs_res.mean()), 1e-12)

    def s(X):
        return np.maximum(scale_model.predict(X), floor)

    scores = np.abs(y_cal - model.predict(X_cal)) / s(X_cal)
    q = conformal_quantile(scores, alpha)
    mu, sc = model.predict(X_te), s(X_te)
    return mu - q * sc, mu + q * sc


def cqr(lo_model, hi_model, X_tr, y_tr, X_cal, y_cal, X_te, alpha):
    """Conformalized quantile regression (Romano, Patterson & Candes 2019).

    Score max(q_lo(x) - y, y - q_hi(x)). The calibrated interval is
    [q_lo - Q, q_hi + Q]. Q may be negative, which shrinks the interval.
    """
    lo_model.fit(X_tr, y_tr)
    hi_model.fit(X_tr, y_tr)
    scores = np.maximum(lo_model.predict(X_cal) - y_cal, y_cal - hi_model.predict(X_cal))
    q = conformal_quantile(scores, alpha)
    return lo_model.predict(X_te) - q, hi_model.predict(X_te) + q


def weighted_quantile(scores: np.ndarray, w_cal: np.ndarray, w_test: np.ndarray, alpha: float) -> np.ndarray:
    """Per-test-point weighted conformal quantile (Tibshirani et al. 2019).

    This is the (1 - alpha) quantile of
        sum_i p_i delta_{V_i} + p_test delta_{+inf},
    where p_i = w_i / (sum_j w_j + w_test). Returns +inf when the calibration
    mass is not enough.
    """
    order = np.argsort(scores)
    s, cw = scores[order], np.cumsum(w_cal[order])
    need = (1.0 - alpha) * (cw[-1] + w_test) * (1.0 - 1e-12)
    j = np.searchsorted(cw, need, side="left")
    out = np.full(len(w_test), math.inf)
    ok = j < len(s)
    out[ok] = s[j[ok]]
    return out


def split_weighted(model, X_tr, y_tr, X_cal, y_cal, X_te, w_cal, w_te, alpha):
    """Weighted split conformal for known covariate shift. w = dP_test/dP_train at each point."""
    model.fit(X_tr, y_tr)
    scores = np.abs(y_cal - model.predict(X_cal))
    q = weighted_quantile(scores, w_cal, w_te, alpha)
    mu = model.predict(X_te)
    return mu - q, mu + q


def effective_sample_size(w: np.ndarray) -> float:
    """Kish ESS, (sum w)^2 / sum w^2."""
    return float(w.sum() ** 2 / (w**2).sum())
