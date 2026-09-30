"""E5: known covariate shift, comparing weighted and unweighted split conformal.
E6: online conformal on a drifting stream (ACI, quantile tracking, and baselines).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from conformal import (
    Ridge,
    aci,
    aci_bound,
    conformal_quantile,
    effective_sample_size,
    quantile_tracker,
    rls_one_step,
    rolling_coverage,
    split_abs,
    split_normalized,
    split_weighted,
)
from conformal.data import drift_stream, hetero_coverage, rbf_features, sample_hetero, tilt_density, tilt_ratio

from .common import COLORS, MARKERS, MUTED, fmt, md_table, save

XQ = (np.arange(4000) + 0.5) / 4000


def e5_covariate_shift(trials=100, n_tr=500, n_cal=500, alpha=0.1, thetas=(0, 1, 2, 3, 4, 6, 8, 12), seed=500):
    rng = np.random.default_rng(seed)
    XQf = rbf_features(XQ)
    names = ["split (unweighted)", "split-normalized (unweighted)", "weighted split"]
    res = {k: [] for k in names}
    wres, ess_l, inf_l = [], [], []
    for th in thetas:
        p = tilt_density(XQ, th) / tilt_density(XQ, th).mean()  # test density on the quadrature grid
        cs = {k: [] for k in names}
        ws, ess, infm = [], [], []
        for _ in range(trials):
            x, y = sample_hetero(n_tr + n_cal, rng)
            X = rbf_features(x)
            args = (X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], XQf)
            lo, hi = split_abs(Ridge(1e-2), *args, alpha)
            cs[names[0]].append(np.mean(hetero_coverage(XQ, lo, hi) * p))
            lo, hi = split_normalized(Ridge(1e-2), Ridge(0.1), *args, alpha)
            cs[names[1]].append(np.mean(hetero_coverage(XQ, lo, hi) * p))
            w_cal = tilt_ratio(x[n_tr:], th)
            lo, hi = split_weighted(Ridge(1e-2), *args, w_cal, tilt_ratio(XQ, th), alpha)
            cs[names[2]].append(np.mean(hetero_coverage(XQ, lo, hi) * p))
            fin = np.isfinite(hi)
            infm.append(np.mean((~fin) * p))
            ws.append(np.sum(((hi - lo) * p)[fin]) / max(np.sum(p[fin]), 1e-12))
            ess.append(effective_sample_size(w_cal))
        for k in names:
            res[k].append(np.mean(cs[k]))
        wres.append(np.mean(ws))
        ess_l.append(np.mean(ess))
        inf_l.append(np.mean(infm))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, k in enumerate(names):
        ax.plot(thetas, res[k], marker=MARKERS[i], color=COLORS[i], label=k)
    ax.axhline(1 - alpha, color=MUTED, ls="--", lw=1.2)
    ax.set_xlabel("tilt θ (test density ∝ exp(θx), toward the high-noise end)")
    ax.set_ylabel("test coverage")
    ax.set_title("Covariate shift with a known likelihood ratio", loc="left")
    ax.legend(fontsize=8)
    save(fig, "e5_covariate_shift.png")

    rows = [[th] + [fmt(res[k][j]) for k in names] + [fmt(wres[j]), fmt(ess_l[j], 0), fmt(inf_l[j], 4)]
            for j, th in enumerate(thetas)]
    text = (
        f"Train/calibration x ~ U[0,1]; test x has density ∝ exp(θx). {trials} trials, {n_tr}/{n_cal} split. "
        "Coverage is exact under the test distribution. The weighted method uses the true likelihood ratio. "
        "'ESS' is the Kish effective size of the calibration weights, and 'P(inf)' is the test mass that "
        "receives an unbounded interval because the weighted calibration mass is not enough.\n\n"
        + md_table(["θ"] + names + ["weighted: finite width", "ESS", "weighted: P(inf)"], rows)
    )
    return text, {k: res[k] for k in names}


def e6_online(T=6000, d=5, alpha=0.1, n_cal=300, burn=200, window=300, seed=600):
    rng = np.random.default_rng(seed)
    X, y, bounds = drift_stream(T, d, rng)
    pred = rls_one_step(X, y, forget=0.99)
    s = np.abs(y - pred)
    t0 = burn + n_cal  # evaluation starts after calibration

    runs = {}
    q_static = np.full(T, conformal_quantile(s[burn:t0], alpha))
    runs["static split"] = (q_static, s > q_static)
    q, e, _ = aci(s, alpha, 0.0, window)
    runs[f"rolling window {window}"] = (q, e)
    for g in (0.005, 0.05):
        q, e, _ = aci(s, alpha, g, window)
        runs[f"ACI γ={g}"] = (q, e)
    q, e = quantile_tracker(s, alpha, eta=0.1, q_1=float(np.median(s[burn:t0])))
    runs["quantile tracker η=0.1"] = (q, e)

    reg_names = ["stationary", "after β jump", "noise ramp 1→4", "t(2) noise"]
    rows = []
    for name, (q, e) in runs.items():
        e_ = e[t0:]
        rc = rolling_coverage(e_, window)
        per = []
        for a, b in zip([t0] + bounds, bounds + [T]):
            a = max(a, t0)
            per.append(fmt(1 - e[a:b].mean()))
        qq = q[t0:]
        rows.append([name, fmt(1 - e_.mean())] + per +
                     [fmt(float(np.nanmin(rc))), fmt(float(np.median(2 * qq[np.isfinite(qq)])), 2), fmt(float(np.mean(~np.isfinite(qq))), 4)])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    tt = np.arange(t0, T)
    for i, (name, (q, e)) in enumerate(runs.items()):
        axes[0].plot(tt, rolling_coverage(e[t0:], window), color=COLORS[i], lw=1.4, label=name)
        qq = np.where(np.isfinite(q[t0:]), q[t0:], np.nan)
        axes[1].plot(tt, qq, color=COLORS[i], lw=1.2, label=name)
    for ax in axes:
        for b in bounds:
            ax.axvline(b, color=MUTED, lw=0.8, ls=":")
    axes[0].axhline(1 - alpha, color=MUTED, ls="--", lw=1.2)
    axes[0].set_ylabel(f"rolling coverage ({window})")
    axes[0].set_title("Online conformal on a drifting stream (dotted lines: regime changes)", loc="left")
    axes[0].legend(fontsize=8, ncol=3, loc="lower left")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("interval half-width q_t (log)")
    axes[1].set_xlabel("time t")
    save(fig, "e6_online.png")

    checks = []
    for g in (0.005, 0.05):
        _, e_full, _ = aci(s, alpha, g, window)
        checks.append(f"γ={g}: |err − α| = {abs(e_full.mean() - alpha):.5f} ≤ bound {aci_bound(alpha, g, T):.5f}")
    text = (
        f"T = {T}, d = {d}. The predictor is one-step-ahead RLS (forgetting 0.99), and the score is |y − ŷ|. "
        f"The static split is calibrated once on t ∈ [{burn}, {t0}). All metrics are over t ≥ {t0}. "
        f"Regimes: β flips at t={bounds[0]}, noise sd ramps 1→4 over [{bounds[1]}, {bounds[2]}), then "
        "Student-t(2) noise.\n\n"
        + md_table(["method", "overall"] + reg_names + [f"worst rolling-{window}", "median width", "P(inf)"], rows)
        + "\n\nACI long-run guarantee over the whole stream: " + "; ".join(checks) + "."
    )
    return text
