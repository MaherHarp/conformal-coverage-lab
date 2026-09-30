"""Exact full (transductive) conformal prediction for ridge regression.

Full conformal accepts a candidate label y for a test point x if, after
refitting on the n training points plus (x, y), the test point's residual is
not unusually large:

    p(y) = #{i <= n+1 : |r_i(y)| >= |r_{n+1}(y)|} / (n + 1),    C(x) = {y : p(y) > alpha}.

Naively this needs one refit per candidate y. Ridge fitted values are linear in
the response vector, so with H the hat matrix of the augmented design,

    r(y) = (I - H) [y_1, ..., y_n, 0]^T + y (I - H) e_{n+1} = a + b y.

Each comparison |a_i + b_i y| >= |a_{n+1} + b_{n+1} y| is a quadratic
inequality in y, with at most 2 roots. Between consecutive roots the count
is constant, so the prediction set is an exact union of intervals found by
one sort and one vectorized count. There is no grid and no refit.
(This follows Vovk, Gammerman & Shafer's ridge-regression confidence machine.)
"""
from __future__ import annotations

import math

import numpy as np


def _affine_residuals(X: np.ndarray, y: np.ndarray, x_new: np.ndarray, lam: float):
    n, d = X.shape
    Xa = np.vstack([X, x_new[None, :]])
    G = Xa.T @ Xa + lam * np.eye(d)
    H = Xa @ np.linalg.solve(G, Xa.T)
    ya = np.append(y, 0.0)
    a = ya - H @ ya
    b = -H[:, n].copy()
    b[n] += 1.0
    return a, b


def _real_roots(q2: np.ndarray, q1: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """All real roots of q2 y^2 + q1 y + q0 = 0, vectorized, using the stable formula."""
    scale = np.maximum(np.maximum(np.abs(q2), np.abs(q1)), np.abs(q0)) + 1e-300
    quad = np.abs(q2) > 1e-12 * scale
    roots = []
    # Degenerate (linear) comparisons.
    lin = ~quad & (np.abs(q1) > 1e-12 * scale)
    roots.append(-q0[lin] / q1[lin])
    # Proper quadratics.
    A, B, C = q2[quad], q1[quad], q0[quad]
    disc = B * B - 4.0 * A * C
    ok = disc >= 0.0
    A, B, C, sq = A[ok], B[ok], C[ok], np.sqrt(disc[ok])
    t = -0.5 * (B + np.copysign(sq, B))
    nz = t != 0.0
    roots.append(t[nz] / A[nz])
    roots.append(C[nz] / t[nz])
    roots.append(np.zeros(int((~nz).sum())))  # B = 0 and disc = 0 -> double root at 0
    return np.concatenate(roots)


def _accepted(a, b, ys: np.ndarray, alpha: float) -> np.ndarray:
    n = len(a) - 1
    r = np.abs(a[:, None] + b[:, None] * ys[None, :])  # (n+1) x m
    count = (r[:n] >= r[n][None, :]).sum(axis=0) + 1  # the test point always counts itself
    return count > alpha * (n + 1)


def full_conformal_ridge(X: np.ndarray, y: np.ndarray, x_new: np.ndarray, lam: float, alpha: float) -> np.ndarray:
    """Exact full-conformal ridge prediction set as an array of [lo, hi] rows.

    Endpoints may be +-inf. Adjacent accepted regions are merged. Isolated
    accepted points (measure zero) are dropped.
    """
    a, b = _affine_residuals(X, y, x_new, lam)
    n = len(y)
    an, bn = a[n], b[n]
    q2 = b[:n] ** 2 - bn**2
    q1 = 2.0 * (a[:n] * b[:n] - an * bn)
    q0 = a[:n] ** 2 - an**2
    bps = np.unique(_real_roots(q2, q1, q0))
    if len(bps) == 0:
        return np.array([[-math.inf, math.inf]]) if _accepted(a, b, np.array([0.0]), alpha)[0] else np.empty((0, 2))
    pad = max(1.0, float(np.abs(bps).max()))
    probes = np.concatenate([[bps[0] - pad], 0.5 * (bps[1:] + bps[:-1]), [bps[-1] + pad]])
    acc = _accepted(a, b, probes, alpha)
    edges = np.concatenate([[-math.inf], bps, [math.inf]])  # region j spans edges[j], edges[j+1]
    out = []
    j = 0
    while j < len(acc):
        if acc[j]:
            start = j
            while j + 1 < len(acc) and acc[j + 1]:
                j += 1
            out.append([edges[start], edges[j + 1]])
        j += 1
    return np.array(out) if out else np.empty((0, 2))


def full_conformal_bruteforce(X, y, x_new, lam, alpha, grid: np.ndarray) -> np.ndarray:
    """Reference implementation: refit ridge for every grid candidate. Returns a boolean mask."""
    n, d = X.shape
    Xa = np.vstack([X, x_new[None, :]])
    G = Xa.T @ Xa + lam * np.eye(d)
    acc = np.empty(len(grid), dtype=bool)
    for j, yc in enumerate(grid):
        ya = np.append(y, yc)
        beta = np.linalg.solve(G, Xa.T @ ya)
        r = np.abs(ya - Xa @ beta)
        acc[j] = np.mean(r >= r[n]) > alpha
    return acc


def set_length(intervals: np.ndarray) -> float:
    if len(intervals) == 0:
        return 0.0
    return float((intervals[:, 1] - intervals[:, 0]).sum())


def set_contains(intervals: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if len(intervals) == 0:
        return np.zeros(len(ys), dtype=bool)
    return ((ys[:, None] >= intervals[None, :, 0]) & (ys[:, None] <= intervals[None, :, 1])).any(axis=1)
