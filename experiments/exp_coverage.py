"""E1: marginal coverage, width, and cost for every method.
E2: the calibration-conditional coverage law and the PAC correction.
"""
from __future__ import annotations

import math
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest

from conformal import (
    QuantileLinear,
    Ridge,
    conformal_rank,
    coverage_law,
    cqr,
    cv_plus,
    full_conformal_ridge,
    jackknife,
    jackknife_minmax,
    jackknife_plus,
    pac_rank,
    set_length,
    split_abs,
    split_normalized,
)
from conformal.data import hetero_coverage, rbf_features, sample_hetero

from .common import COLORS, INK, MARKERS, MUTED, fmt, md_table, save, union_coverage

LAM = 1e-2
XQ = (np.arange(4000) + 0.5) / 4000


def e1_marginal(trials=200, n=200, alpha=0.1, m=300, m_full=30, seed=100):
    rng = np.random.default_rng(seed)
    h = n // 2
    names = ["split", "split-normalized", "CQR", "full (exact ridge)", "jackknife",
             "jackknife+", "jackknife-minmax", "CV+ (K=10)"]
    cov = {k: [] for k in names}
    wid = {k: [] for k in names}
    secs = {k: 0.0 for k in names}
    npts = {k: 0 for k in names}

    def record(name, xt, lo, hi, dt):
        cov[name].append(hetero_coverage(xt, lo, hi).mean())
        wid[name].append(np.mean(hi - lo))
        secs[name] += dt
        npts[name] += len(xt)

    for _ in range(trials):
        x, y = sample_hetero(n, rng)
        X = rbf_features(x)
        xt = rng.uniform(0, 1, m)
        Xt = rbf_features(xt)
        split_args = (X[:h], y[:h], X[h:], y[h:], Xt)

        t0 = time.perf_counter(); lo, hi = split_abs(Ridge(LAM), *split_args, alpha)
        record("split", xt, lo, hi, time.perf_counter() - t0)
        t0 = time.perf_counter(); lo, hi = split_normalized(Ridge(LAM), Ridge(0.1), *split_args, alpha)
        record("split-normalized", xt, lo, hi, time.perf_counter() - t0)
        t0 = time.perf_counter(); lo, hi = cqr(QuantileLinear(alpha / 2), QuantileLinear(1 - alpha / 2), *split_args, alpha)
        record("CQR", xt, lo, hi, time.perf_counter() - t0)
        for name, fn in [("jackknife", jackknife), ("jackknife+", jackknife_plus), ("jackknife-minmax", jackknife_minmax)]:
            t0 = time.perf_counter(); lo, hi = fn(X, y, Xt, LAM, alpha)
            record(name, xt, lo, hi, time.perf_counter() - t0)
        t0 = time.perf_counter(); lo, hi = cv_plus(lambda: Ridge(LAM), X, y, Xt, alpha, 10, rng)
        record("CV+ (K=10)", xt, lo, hi, time.perf_counter() - t0)

        c, w = [], []
        t0 = time.perf_counter()
        sets = [full_conformal_ridge(X, y, Xt[j], LAM, alpha) for j in range(m_full)]
        dt = time.perf_counter() - t0
        for j, iv in enumerate(sets):
            c.append(union_coverage(xt[j], iv))
            w.append(set_length(iv))
        cov["full (exact ridge)"].append(np.mean(c))
        wid["full (exact ridge)"].append(np.mean(w))
        secs["full (exact ridge)"] += dt
        npts["full (exact ridge)"] += m_full

    rows, summary = [], {}
    for k in names:
        c = np.array(cov[k])
        summary[k] = (c.mean(), c.std(ddof=1) / math.sqrt(trials), np.mean(wid[k]))
        rows.append([k, fmt(c.mean()), fmt(c.std(ddof=1) / math.sqrt(trials), 4), fmt(c.std(ddof=1)),
                     fmt(float(np.mean(wid[k]))), fmt(1e6 * secs[k] / npts[k], 1)])
    kk = conformal_rank(h, alpha)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    ypos = np.arange(len(names))[::-1]
    for i, k in enumerate(names):
        mcov, se, w = summary[k]
        axes[0].errorbar(mcov, ypos[i], xerr=2 * se, fmt="o", color=COLORS[0], ms=7, capsize=3)
        axes[1].plot(w, ypos[i], "o", color=COLORS[0], ms=7)
    axes[0].axvline(1 - alpha, color=MUTED, ls="--", lw=1.2)
    axes[0].axvline(kk / (h + 1), color=COLORS[1], ls=":", lw=1.5)
    axes[0].text(kk / (h + 1), ypos[-1] - 0.45, f" split theory {kk}/{h + 1}", color=INK, fontsize=8, va="center")
    axes[0].set_ylim(ypos[-1] - 0.8, ypos[0] + 0.5)
    axes[0].set_yticks(ypos, names)
    axes[0].set_xlabel("marginal coverage (mean ± 2 SE)")
    axes[0].set_title("Coverage, target 0.90", loc="left")
    axes[1].set_xlabel("mean interval width")
    axes[1].set_title("Width (smaller is better)", loc="left")
    save(fig, "e1_marginal.png")

    text = (
        f"n = {n} (split methods: {h} train / {h} calibration), alpha = {alpha}, {trials} trials; "
        f"heteroscedastic sine data, 13 RBF features, ridge lambda = {LAM}. Coverage is exact per test point "
        f"(Gaussian CDF), averaged over {m} uniform test x per trial ({m_full} for full conformal). "
        f"Split theory: E[coverage] = k/(n_cal+1) = {kk}/{h + 1} = {kk / (h + 1):.4f}. "
        "Time is Python wall-clock per test point, including fitting, on one machine.\n\n"
        + md_table(["method", "coverage", "SE", "sd across draws", "mean width", "µs / test point"], rows)
    )
    return text, summary


def _split_cov(rng, n_tr, n_cal, alpha, k):
    x, y = sample_hetero(n_tr + n_cal, rng)
    X = rbf_features(x)
    lo, hi = split_abs(Ridge(LAM), X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], rbf_features(XQ), alpha, k=k)
    return float(hetero_coverage(XQ, lo, hi).mean()), float(np.mean(hi - lo))


def e2_beta_law(trials=2000, alpha=0.1, delta=0.1, seed=200):
    rng = np.random.default_rng(seed)
    rows, hist = [], {}
    for n_cal in (20, 100, 1000):
        k, kp = conformal_rank(n_cal, alpha), pac_rank(n_cal, alpha, delta)
        res = {}
        for label, kk in (("standard", k), ("PAC", kp)):
            if kk > n_cal:
                res[label] = None
                continue
            cw = np.array([_split_cov(rng, 100, n_cal, alpha, kk) for _ in range(trials)])
            res[label] = cw
            law = coverage_law(n_cal, alpha, kk)
            rows.append([n_cal, label, kk, fmt(law.mean(), 4), fmt(cw[:, 0].mean(), 4), fmt(law.std(), 4),
                         fmt(cw[:, 0].std(ddof=1), 4), fmt(kstest(cw[:, 0], law.cdf).pvalue, 3),
                         fmt(law.cdf(1 - alpha)), fmt(float(np.mean(cw[:, 0] < 1 - alpha))), fmt(cw[:, 1].mean())])
        hist[n_cal] = (k, kp, res)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    xr = {20: (0.6, 1.0), 100: (0.8, 1.0), 1000: (0.86, 0.95)}
    for ax, n_cal in zip(axes, (20, 100, 1000)):
        k, kp, res = hist[n_cal]
        grid = np.linspace(*xr[n_cal], 400)
        for i, (label, kk) in enumerate((("standard", k), ("PAC", kp))):
            cw = res[label]
            if cw is None:
                continue
            ax.hist(cw[:, 0], bins=40, range=xr[n_cal], density=True, color=COLORS[i], alpha=0.35,
                    edgecolor="#fcfcfb", linewidth=0.5)
            ax.plot(grid, coverage_law(n_cal, alpha, kk).pdf(grid), color=COLORS[i], label=f"{label} k={kk}")
        ax.axvline(1 - alpha, color=MUTED, ls="--", lw=1.2)
        ax.set_title(f"n_cal = {n_cal}", loc="left")
        ax.set_xlabel("coverage given the calibration set")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density (bars: simulation, lines: Beta law)")
    save(fig, "e2_beta_law.png")

    text = (
        f"{trials} independent train/calibration draws per row (100 training points). Coverage for each draw "
        "is computed exactly by quadrature over x. The theory columns are Beta(k, n+1-k). "
        f"PAC rows use the smallest k with P(coverage < {1 - alpha}) <= {delta}. With n_cal = 20 no k <= n "
        "achieves this, so the PAC interval would be infinite.\n\n"
        + md_table(["n_cal", "rank rule", "k", "theory mean", "sim mean", "theory sd", "sim sd", "KS p",
                    "theory P(cov<0.9)", "sim P(cov<0.9)", "mean width"], rows)
    )
    return text
