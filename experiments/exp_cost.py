"""E7: what exactness costs. Exact full-conformal ridge vs grid refits vs jackknife+ vs split."""
from __future__ import annotations

import time

import numpy as np

from conformal import Ridge, full_conformal_bruteforce, full_conformal_ridge, jackknife_plus, split_abs
from conformal.data import rbf_features, sample_hetero

from .common import fmt, md_table


def _time(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def e7_cost(ns=(50, 100, 200, 400, 800, 1600), grid=1000, alpha=0.1, lam=1e-2, seed=700):
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        x, y = sample_hetero(n + 1000, rng)
        X = rbf_features(x)
        Xtr, ytr, Xt = X[:n], y[:n], X[n:]
        reps = max(3, 400 // n)
        exact = _time(lambda: full_conformal_ridge(Xtr, ytr, Xt[0], lam, alpha), reps)
        brute = _time(lambda: full_conformal_bruteforce(Xtr, ytr, Xt[0], lam, alpha, np.linspace(-6, 6, grid)), 1)
        jp = _time(lambda: jackknife_plus(Xtr, ytr, Xt, lam, alpha), 3) / len(Xt)
        h = n // 2
        sp = _time(lambda: split_abs(Ridge(lam), Xtr[:h], ytr[:h], Xtr[h:], ytr[h:], Xt, alpha), 5) / len(Xt)
        rows.append([n, fmt(1e3 * exact, 2), fmt(1e3 * brute, 1), fmt(brute / exact, 0), fmt(1e6 * jp, 1), fmt(1e6 * sp, 2)])
    return (
        "Python/NumPy wall-clock on one laptop CPU, d = 13. These numbers are for relative comparison only. "
        f"The grid method reuses the factorized Gram matrix but refits at {grid} candidate labels (O(grid · n d)), "
        "and it only approximates the exact set. Exact full conformal is O(n d² + n log n) per test point. "
        "Jackknife+ and split amortize one fit over all 1000 test points.\n\n"
        + md_table(["n", "exact full (ms / pt)", f"grid×{grid} refits (ms / pt)", "speedup", "jackknife+ (µs / pt)",
                    "split (µs / pt)"], rows)
    )
