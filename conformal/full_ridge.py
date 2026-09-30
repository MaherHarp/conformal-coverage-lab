"""Exact full (transductive) conformal prediction for ridge regression.

Full conformal accepts a candidate label y for a test point x if, after
refitting on the n training points plus (x, y), the test point's residual is
not unusually large:

    p(y) = #{i <= n+1 : |r_i(y)| >= |r_{n+1}(y)|} / (n + 1),    C(x) = {y : p(y) > alpha}.

Naively this needs one refit per candidate y. Ridge fitted values are linear in
the response vector, so with H the hat matrix of the augmented design,

    r(y) = (I - H) [y_1, ..., y_n, 0]^T + y (I - H) e_{n+1} = a + b y.

Each comparison 1{|a_i + b_i y| >= |a_{n+1} + b_{n+1} y|} is the sign of a
quadratic q_i(y), with at most 2 roots. So the count N(y) = sum_i 1{q_i(y) >= 0}
is piecewise constant. It is known at y = -inf (from each leading coefficient)
and changes by +-1 exactly at the roots. One sort of <= 2n root events and a
cumulative sum give N on every region: O(n d^2 + n log n) per test point,
with no hat matrix, no grid, and no refit. (This follows the ridge-regression
confidence machine of Vovk, Gammerman & Shafer.)
"""
from __future__ import annotations

import math

import numpy as np


def _affine_residuals(X: np.ndarray, y: np.ndarray, x_new: np.ndarray, lam: float):
    """a = (I - H) [y; 0] and b = (I - H) e_{n+1}, computed in O(n d^2) without forming H."""
    n, d = X.shape
    Xa = np.vstack([X, x_new[None, :]])
    G = Xa.T @ Xa + lam * np.eye(d)
    ya = np.append(y, 0.0)
    sol = np.linalg.solve(G, np.column_stack([Xa.T @ ya, x_new]))
    a = ya - Xa @ sol[:, 0]
    b = -(Xa @ sol[:, 1])
    b[n] += 1.0
    return a, b


def _sign_events(q2: np.ndarray, q1: np.ndarray, q0: np.ndarray):
    """For each quadratic, its indicator 1{q >= 0} at -inf, plus (root, +-1) toggle events.

    A double root is emitted twice with opposite toggles, so it cancels, which is
    correct because the sign does not change there.
    """
    scale = np.maximum(np.maximum(np.abs(q2), np.abs(q1)), np.abs(q0)) + 1e-300
    quad = np.abs(q2) > 1e-12 * scale
    lin = ~quad & (np.abs(q1) > 1e-12 * scale)
    init = np.where(quad, q2 > 0, np.where(lin, q1 < 0, q0 >= 0))
    flip = 1 - 2 * init.astype(np.int64)  # change applied at the first root

    roots, deltas = [-q0[lin] / q1[lin]], [flip[lin]]
    A, B, C = q2[quad], q1[quad], q0[quad]
    disc = B * B - 4.0 * A * C
    ok = disc >= 0.0
    A, B, C, sq, f = A[ok], B[ok], C[ok], np.sqrt(disc[ok]), flip[quad][ok]
    t = -0.5 * (B + np.where(B >= 0, sq, -sq))  # stable quadratic formula
    safe_t = np.where(t != 0.0, t, 1.0)
    r1 = np.where(t != 0.0, t / A, 0.0)
    r2 = np.where(t != 0.0, C / safe_t, 0.0)  # t = 0 only if B = C = 0 -> double root at 0
    roots += [np.minimum(r1, r2), np.maximum(r1, r2)]
    deltas += [f, -f]
    return int(init.sum()), np.concatenate(roots), np.concatenate(deltas)


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
    n_init, roots, deltas = _sign_events(q2, q1, q0)
    order = np.argsort(roots, kind="stable")
    roots = roots[order]
    counts = n_init + np.concatenate([[0], np.cumsum(deltas[order])])  # count on each region
    acc = counts + 1 > alpha * (n + 1)  # +1: the test point always conforms to itself
    edges = np.concatenate([[-math.inf], roots, [math.inf]])  # region j spans edges[j], edges[j+1]
    # Merge runs of accepted regions: a run starts where acc goes 0->1 and ends where it goes 1->0.
    padded = np.concatenate([[False], acc, [False]]).astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    out = np.column_stack([edges[starts], edges[ends]]) if len(starts) else np.empty((0, 2))
    return out[out[:, 1] > out[:, 0]] if len(out) else out


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
