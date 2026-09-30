"""E3: conditional coverage under heteroscedastic noise (marginal validity is not conditional validity).
E4: jackknife vs jackknife+ as the base model becomes unstable (d -> n).
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from conformal import (
    QuantileLinear,
    Ridge,
    cqr,
    cv_plus,
    jackknife,
    jackknife_minmax,
    jackknife_plus,
    naive_insample,
    split_abs,
    split_normalized,
)
from conformal.data import hetero_coverage, linear_coverage, rbf_features, sample_hetero, sample_linear, sigma_true

from .common import COLORS, MARKERS, MUTED, fmt, md_table, save

XQ = (np.arange(4000) + 0.5) / 4000
Z95 = 1.6448536269514722


def e3_conditional(trials=40, n_tr=1000, n_cal=1000, alpha=0.1, bins=10, seed=300):
    rng = np.random.default_rng(seed)
    XQf = rbf_features(XQ)
    methods = {
        "split (|residual|)": lambda a: split_abs(Ridge(1e-2), *a, alpha),
        "split-normalized": lambda a: split_normalized(Ridge(1e-2), Ridge(0.1), *a, alpha),
        "CQR": lambda a: cqr(QuantileLinear(alpha / 2), QuantileLinear(1 - alpha / 2), *a, alpha),
    }
    b = np.minimum((XQ * bins).astype(int), bins - 1)
    cov = {k: np.zeros(bins) for k in methods}
    wid = {k: np.zeros(bins) for k in methods}
    for _ in range(trials):
        x, y = sample_hetero(n_tr + n_cal, rng)
        X = rbf_features(x)
        args = (X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], XQf)
        for k, fn in methods.items():
            lo, hi = fn(args)
            c = hetero_coverage(XQ, lo, hi)
            cov[k] += np.bincount(b, c, bins) / np.bincount(b, None, bins) / trials
            wid[k] += np.bincount(b, hi - lo, bins) / np.bincount(b, None, bins) / trials
    oracle = 2 * Z95 * sigma_true(XQ)
    oracle_bin = np.bincount(b, oracle, bins) / np.bincount(b, None, bins)

    rows = [[k, fmt(cov[k].mean()), fmt(cov[k].min()), fmt(cov[k].max()), fmt(wid[k].mean()),
             fmt(wid[k].mean() / oracle.mean(), 2)] for k in methods]
    rows.append(["oracle (true 5%/95% quantiles)", "0.900", "0.900", "0.900", fmt(oracle.mean()), "1.00"])

    centers = (np.arange(bins) + 0.5) / bins
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, k in enumerate(methods):
        axes[0].plot(centers, cov[k], marker=MARKERS[i], color=COLORS[i], label=k)
        axes[1].plot(centers, wid[k], marker=MARKERS[i], color=COLORS[i], label=k)
    axes[1].plot(centers, oracle_bin, color=MUTED, ls="--", label="oracle width")
    axes[0].axhline(1 - alpha, color=MUTED, ls="--", lw=1.2)
    axes[0].set_xlabel("x (noise sd grows from 0.1 to 1.0)")
    axes[0].set_ylabel("coverage in bin")
    axes[0].set_title("Conditional coverage, 10 bins of x", loc="left")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("mean width in bin")
    axes[1].set_title("Width tracks noise only if the score adapts", loc="left")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    save(fig, "e3_conditional.png")

    per_bin = md_table(["x bin"] + list(methods),
                       [[f"{i / bins:.1f}–{(i + 1) / bins:.1f}"] + [fmt(cov[k][i]) for k in methods] for i in range(bins)])
    text = (
        f"{trials} trials, {n_tr} train / {n_cal} calibration points, exact per-x coverage averaged within 10 "
        "equal-width bins of x. All three methods are marginally valid. Only adaptive scores are close to "
        "conditionally valid.\n\n"
        + md_table(["method", "marginal", "worst bin", "best bin", "mean width", "width / oracle"], rows)
        + "\n\nPer-bin coverage:\n\n" + per_bin
    )
    return text, {k: (cov[k].min(), cov[k].max()) for k in methods}


def e4_jackknife(trials=50, n=100, alpha=0.1, lam=1e-3, dims=(10, 50, 80, 90, 95, 100, 105, 110, 125, 150, 200), m=200, seed=400):
    rng = np.random.default_rng(seed)
    names = ["naive (in-sample)", "jackknife", "jackknife+", "jackknife-minmax", "CV+ (K=10)"]
    cov = {k: [] for k in names}
    wid = {k: [] for k in names}
    for d in dims:
        beta = rng.standard_normal(d)
        beta *= math.sqrt(10.0) / np.linalg.norm(beta)  # signal variance 10, noise variance 1
        cs = {k: [] for k in names}
        ws = {k: [] for k in names}
        for _ in range(trials):
            X, y = sample_linear(n, beta, rng)
            Xt = rng.standard_normal((m, d))
            outs = {
                "naive (in-sample)": naive_insample(X, y, Xt, lam, alpha),
                "jackknife": jackknife(X, y, Xt, lam, alpha),
                "jackknife+": jackknife_plus(X, y, Xt, lam, alpha),
                "jackknife-minmax": jackknife_minmax(X, y, Xt, lam, alpha),
                "CV+ (K=10)": cv_plus(lambda: Ridge(lam), X, y, Xt, alpha, 10, rng),
            }
            for k, (lo, hi) in outs.items():
                cs[k].append(linear_coverage(Xt, beta, lo, hi).mean())
                ws[k].append(np.median(hi - lo))
        for k in names:
            cov[k].append(np.mean(cs[k]))
            wid[k].append(np.median(ws[k]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, k in enumerate(names):
        axes[0].plot(dims, cov[k], marker=MARKERS[i], color=COLORS[i], label=k)
        axes[1].plot(dims, wid[k], marker=MARKERS[i], color=COLORS[i], label=k)
    axes[0].axhline(1 - alpha, color=MUTED, ls="--", lw=1.2)
    axes[0].axhline(1 - 2 * alpha, color=MUTED, ls=":", lw=1.2)
    axes[0].text(dims[0], 1 - 2 * alpha + 0.01, "jackknife+ floor 1-2α", color=MUTED, fontsize=8)
    for ax in axes:
        ax.axvline(n, color=MUTED, lw=0.8)
        ax.set_xlabel("dimension d (n = 100)")
    axes[0].set_ylabel("coverage")
    axes[0].set_title("Ridge with λ=1e-3: coverage vs dimension", loc="left")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("median width (log)")
    axes[1].set_title("Width", loc="left")
    axes[0].legend(fontsize=8, loc="lower left")
    save(fig, "e4_jackknife.png")

    rows = [[d] + [f"{fmt(cov[k][j])} / {fmt(wid[k][j], 1)}" for k in names] for j, d in enumerate(dims)]
    text = (
        f"n = {n}, X ~ N(0, I_d), ||beta||² = 10, unit noise, ridge lambda = {lam}, {trials} trials per d, "
        f"exact coverage over {m} test points. Cells show coverage / median width.\n\n"
        + md_table(["d"] + names, rows)
    )
    return text, {k: (min(cov[k]), dims[int(np.argmin(cov[k]))]) for k in names}
