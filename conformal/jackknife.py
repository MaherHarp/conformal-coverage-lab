"""Leave-one-out and cross-validation conformal methods (Barber et al. 2021).

Jackknife: mu(x) +- q+(|R_i|). No coverage guarantee; it can fail when the
  base model is unstable.
Jackknife+: [q-(mu_{-i}(x) - |R_i|), q+(mu_{-i}(x) + |R_i|)]. Guarantees
  coverage >= 1 - 2 alpha for any symmetric algorithm, and is typically
  close to 1 - alpha.
Jackknife-minmax: [min_i mu_{-i}(x) - q+(|R|), max_i mu_{-i}(x) + q+(|R|)].
  Guarantees >= 1 - alpha, but is conservative.
CV+: the K-fold analogue of jackknife+. Guarantees >= 1 - 2 alpha - sqrt(2/n).

For ridge, all LOO models come from one fit via the Sherman-Morrison identity
(see models.ridge_loo), so jackknife+ costs O(n d^2 + m n d) rather than n refits.
"""
from __future__ import annotations

import numpy as np

from .models import ridge_loo
from .split import conformal_rank


def q_plus(v: np.ndarray, alpha: float) -> np.ndarray:
    """Row-wise ceil((1 - alpha)(n + 1))-th smallest value along the last axis (+inf if out of range)."""
    n = v.shape[-1]
    k = conformal_rank(n, alpha)
    if k > n:
        return np.full(v.shape[:-1], np.inf)
    return np.partition(v, k - 1, axis=-1)[..., k - 1]


def q_minus(v: np.ndarray, alpha: float) -> np.ndarray:
    """Row-wise floor(alpha (n + 1))-th smallest value (-inf if out of range)."""
    return -q_plus(-v, alpha)


def naive_insample(X, y, X_te, lam, alpha):
    """Uses in-sample residuals. Undercovers whenever the model overfits."""
    o = ridge_loo(X, y, lam, X_te)
    q = q_plus(np.abs(o["resid"]), alpha)
    return o["mu_test"] - q, o["mu_test"] + q


def jackknife(X, y, X_te, lam, alpha):
    o = ridge_loo(X, y, lam, X_te)
    q = q_plus(np.abs(o["loo_resid"]), alpha)
    return o["mu_test"] - q, o["mu_test"] + q


def jackknife_plus(X, y, X_te, lam, alpha):
    o = ridge_loo(X, y, lam, X_te)
    R = np.abs(o["loo_resid"])[None, :]
    M = o["mu_loo_test"]
    return q_minus(M - R, alpha), q_plus(M + R, alpha)


def jackknife_minmax(X, y, X_te, lam, alpha):
    o = ridge_loo(X, y, lam, X_te)
    q = q_plus(np.abs(o["loo_resid"]), alpha)
    M = o["mu_loo_test"]
    return M.min(axis=1) - q, M.max(axis=1) + q


def cv_plus(model_factory, X, y, X_te, alpha, K: int, rng: np.random.Generator):
    """K-fold CV+ for any model with fit/predict."""
    n = len(y)
    folds = np.array_split(rng.permutation(n), K)
    R = np.empty(n)
    M = np.empty((len(X_te), n))
    for idx in folds:
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        m = model_factory().fit(X[mask], y[mask])
        R[idx] = np.abs(y[idx] - m.predict(X[idx]))
        M[:, idx] = m.predict(X_te)[:, None]
    return q_minus(M - R[None, :], alpha), q_plus(M + R[None, :], alpha)
