"""Online conformal inference under distribution shift.

Each method takes a precomputed sequence of nonconformity scores s_t (e.g.
|y_t - yhat_t| from an online predictor that only sees data before time t) and
outputs a threshold q_t chosen *before* s_t is revealed. The interval at time
t is yhat_t +- q_t, and err_t = 1{s_t > q_t}.

Adaptive conformal inference (ACI, Gibbs & Candes 2021):
    alpha_{t+1} = alpha_t + gamma (alpha - err_t),   q_t = conformal quantile at level alpha_t.
For every score sequence, including adversarial ones,
    | (1/T) sum err_t - alpha | <= (max(alpha_1, 1 - alpha_1) + gamma) / (gamma T).

Quantile tracking (the "P" part of conformal PID control, Angelopoulos et al. 2023):
    q_{t+1} = q_t + eta (err_t - alpha).
If scores lie in [0, B] and q_1 is in [0, B], then
    | (1/T) sum err_t - alpha | <= (B + eta) / (eta T).
"""
from __future__ import annotations

import math

import numpy as np

from .split import conformal_quantile


def rls_one_step(X: np.ndarray, y: np.ndarray, forget: float = 0.99, delta: float = 100.0) -> np.ndarray:
    """Recursive least squares with exponential forgetting. pred[t] uses data < t only."""
    T, d = X.shape
    P = delta * np.eye(d)
    w = np.zeros(d)
    pred = np.empty(T)
    for t in range(T):
        x = X[t]
        pred[t] = x @ w
        Px = P @ x
        g = Px / (forget + x @ Px)
        w = w + g * (y[t] - pred[t])
        P = (P - np.outer(g, Px)) / forget
    return pred


def aci(scores: np.ndarray, alpha: float, gamma: float, window: int | None = None, alpha_1: float | None = None):
    """Adaptive conformal inference. gamma = 0 gives plain online conformal with fixed alpha.

    Rules that keep alpha_t bounded: alpha_t <= 0 -> q = +inf (never an error);
    alpha_t >= 1 -> empty set (always an error).
    Returns (q, err, alpha_path).
    """
    T = len(scores)
    a = alpha if alpha_1 is None else alpha_1
    q = np.empty(T)
    err = np.empty(T, dtype=bool)
    path = np.empty(T)
    for t in range(T):
        path[t] = a
        if a >= 1.0:
            qt = -math.inf
        elif a <= 0.0:
            qt = math.inf
        else:
            lo = 0 if window is None else max(0, t - window)
            qt = conformal_quantile(scores[lo:t], a)
        q[t] = qt
        err[t] = scores[t] > qt
        a = a + gamma * (alpha - float(err[t]))
    return q, err, path


def aci_bound(alpha: float, gamma: float, T: int, alpha_1: float | None = None) -> float:
    a1 = alpha if alpha_1 is None else alpha_1
    return (max(a1, 1.0 - a1) + gamma) / (gamma * T)


def quantile_tracker(scores: np.ndarray, alpha: float, eta: float, q_1: float = 0.0):
    """Online gradient descent on the pinball loss. Returns (q, err)."""
    T = len(scores)
    q = np.empty(T)
    err = np.empty(T, dtype=bool)
    qt = q_1
    for t in range(T):
        q[t] = qt
        err[t] = scores[t] > qt
        qt = qt + eta * (float(err[t]) - alpha)
    return q, err


def static_split(scores: np.ndarray, alpha: float, n_cal: int):
    """Calibrate once on the first n_cal scores and freeze. Returns (q, err) for t >= n_cal."""
    qv = conformal_quantile(scores[:n_cal], alpha)
    s = scores[n_cal:]
    return np.full(len(s), qv), s > qv


def rolling_coverage(err: np.ndarray, window: int) -> np.ndarray:
    c = np.cumsum(1.0 - err.astype(float))
    out = np.full(len(err), np.nan)
    out[window - 1 :] = (c[window - 1 :] - np.concatenate([[0.0], c[:-window]])) / window
    return out
