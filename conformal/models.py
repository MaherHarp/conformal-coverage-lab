"""Base regressors wrapped by the conformal methods.

Conformal prediction only assumes the base model is a symmetric function of
its training data. These are deliberately simple models whose failure modes
are easy to understand: ridge regression (closed form, with exact
leave-one-out shortcuts) and linear quantile regression (a linear program).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


class Ridge:
    """Ridge regression with a uniform L2 penalty on all coefficients."""

    def __init__(self, lam: float = 1e-2):
        self.lam = lam
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        d = X.shape[1]
        self.coef_ = np.linalg.solve(X.T @ X + self.lam * np.eye(d), X.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_


class QuantileLinear:
    """Linear tau-quantile regression. Minimizes the pinball loss as an LP.

    min_b  sum_i tau * u_i + (1 - tau) * v_i
    s.t.   X b + u - v = y,   u, v >= 0
    """

    def __init__(self, tau: float):
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must be in (0, 1)")
        self.tau = tau
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileLinear":
        n, d = X.shape
        c = np.concatenate([np.zeros(d), np.full(n, self.tau), np.full(n, 1.0 - self.tau)])
        eye = sparse.identity(n, format="csr")
        A = sparse.hstack([sparse.csr_matrix(X), eye, -eye], format="csr")
        bounds = [(None, None)] * d + [(0.0, None)] * (2 * n)
        # The RBF designs are nearly collinear, and HiGHS' default path occasionally
        # stops with a numerical-difficulty status. Interior point solves those cases.
        for method in ("highs", "highs-ipm"):
            res = linprog(c, A_eq=A, b_eq=y, bounds=bounds, method=method)
            if res.status == 0:
                break
        else:
            raise RuntimeError(f"quantile LP failed: {res.message}")
        self.coef_ = res.x[:d]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_


def ridge_loo(X: np.ndarray, y: np.ndarray, lam: float, X_test: np.ndarray | None = None) -> dict:
    """Exact leave-one-out ridge quantities without n refits.

    With G = X^T X + lam I, h_i = x_i^T G^{-1} x_i and e_i = y_i - x_i^T beta,
    Sherman-Morrison gives
        beta_{-i} = beta - G^{-1} x_i e_i / (1 - h_i),
    so the LOO residual is y_i - mu_{-i}(x_i) = e_i / (1 - h_i), and the LOO
    prediction at a new point is mu(x) - (x^T G^{-1} x_i) e_i / (1 - h_i).
    """
    n, d = X.shape
    G = X.T @ X + lam * np.eye(d)
    Ginv_Xt = np.linalg.solve(G, X.T)  # d x n
    beta = Ginv_Xt @ y
    h = np.einsum("ij,ji->i", X, Ginv_Xt)
    e = y - X @ beta
    c = e / (1.0 - h)
    out = {"beta": beta, "leverage": h, "resid": e, "loo_resid": c}
    if X_test is not None:
        mu = X_test @ beta
        K = X_test @ Ginv_Xt  # m x n, entries x^T G^{-1} x_i
        out["mu_test"] = mu
        out["mu_loo_test"] = mu[:, None] - K * c[None, :]
    return out
