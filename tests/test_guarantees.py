"""Statistical tests of the coverage theorems, run on data whose coverage is known exactly.

Seeds are fixed, so every test is deterministic. Tolerances are several
standard errors wide.
"""
import math
import unittest

import numpy as np
from scipy.stats import kstest

from conformal import (
    Ridge,
    aci,
    aci_bound,
    conformal_rank,
    coverage_law,
    full_conformal_ridge,
    jackknife_minmax,
    jackknife_plus,
    pac_rank,
    quantile_tracker,
    rls_one_step,
    rolling_coverage,
    set_contains,
    split_abs,
    split_weighted,
)
from conformal.data import (
    drift_stream,
    hetero_coverage,
    linear_coverage,
    rbf_features,
    sample_hetero,
    sample_linear,
    sample_tilted_x,
    tilt_density,
    tilt_ratio,
)

XQ = (np.arange(4000) + 0.5) / 4000  # midpoint quadrature grid on [0, 1]


def split_cov(rng, n_tr, n_cal, alpha, k=None, theta=0.0, weighted=False):
    x, y = sample_hetero(n_tr + n_cal, rng)
    X = rbf_features(x)
    args = (X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], rbf_features(XQ))
    if weighted:
        lo, hi = split_weighted(Ridge(1e-2), *args, tilt_ratio(x[n_tr:], theta), tilt_ratio(XQ, theta), alpha)
    else:
        lo, hi = split_abs(Ridge(1e-2), *args, alpha, k=k)
    return float(np.mean(hetero_coverage(XQ, lo, hi) * tilt_density(XQ, theta)))


class TestSplit(unittest.TestCase):
    def test_calibration_conditional_coverage_is_beta(self):
        rng = np.random.default_rng(10)
        n, alpha = 30, 0.1
        covs = [split_cov(rng, 40, n, alpha) for _ in range(400)]
        law = coverage_law(n, alpha)
        self.assertGreater(kstest(covs, law.cdf).pvalue, 0.01)
        se = law.std() / math.sqrt(400)
        self.assertAlmostEqual(np.mean(covs), conformal_rank(n, alpha) / (n + 1), delta=4 * se)

    def test_pac_rank_controls_bad_calibration_draws(self):
        n, alpha, delta = 50, 0.1, 0.1
        k = pac_rank(n, alpha, delta)
        self.assertLessEqual(coverage_law(n, alpha, k).cdf(1 - alpha), delta)
        self.assertGreater(coverage_law(n, alpha, k - 1).cdf(1 - alpha), delta)
        rng = np.random.default_rng(11)
        covs = np.array([split_cov(rng, 40, n, alpha, k=k) for _ in range(300)])
        self.assertLess(np.mean(covs < 1 - alpha), delta + 3 * math.sqrt(delta * (1 - delta) / 300))

    def test_weighted_conformal_restores_coverage_under_shift(self):
        rng = np.random.default_rng(12)
        theta, alpha = 4.0, 0.1
        naive = np.mean([split_cov(rng, 200, 200, alpha, theta=theta) for _ in range(60)])
        weighted = np.mean([split_cov(rng, 200, 200, alpha, theta=theta, weighted=True) for _ in range(60)])
        self.assertLess(naive, 0.86)
        self.assertGreater(weighted, 1 - alpha - 0.01)

    def test_tilted_sampler_matches_density(self):
        rng = np.random.default_rng(13)
        x = sample_tilted_x(200_000, 3.0, rng)
        hist, edges = np.histogram(x, bins=10, range=(0, 1), density=True)
        mids = 0.5 * (edges[1:] + edges[:-1])
        np.testing.assert_allclose(hist, tilt_density(mids, 3.0), rtol=0.03)


class TestFullAndJackknife(unittest.TestCase):
    def test_full_conformal_marginal_coverage_window(self):
        # Exchangeable, continuous: 1 - alpha <= coverage <= 1 - alpha + 1/(n + 1).
        rng = np.random.default_rng(14)
        n, d, alpha, trials = 19, 3, 0.1, 3000
        hits = 0
        for _ in range(trials):
            X = rng.standard_normal((n + 1, d))
            y = X @ np.ones(d) + rng.standard_normal(n + 1)
            iv = full_conformal_ridge(X[:n], y[:n], X[n], 0.5, alpha)
            hits += bool(set_contains(iv, y[n : n + 1])[0])
        cov, se = hits / trials, math.sqrt(0.9 * 0.1 / trials)
        self.assertGreater(cov, 1 - alpha - 3 * se)
        self.assertLess(cov, 1 - alpha + 1 / (n + 1) + 3 * se)

    def test_jackknife_plus_and_minmax_guarantees_when_unstable(self):
        # d close to n with almost no regularization is the regime where plain jackknife can fail.
        rng = np.random.default_rng(15)
        n, d, alpha = 60, 60, 0.1
        beta = rng.standard_normal(d) * math.sqrt(10 / d)
        jp, mm = [], []
        for _ in range(60):
            X, y = sample_linear(n, beta, rng)
            Xt = rng.standard_normal((200, d))
            jp.append(linear_coverage(Xt, beta, *jackknife_plus(X, y, Xt, 1e-3, alpha)).mean())
            mm.append(linear_coverage(Xt, beta, *jackknife_minmax(X, y, Xt, 1e-3, alpha)).mean())
        self.assertGreater(np.mean(jp), 1 - 2 * alpha)
        self.assertGreater(np.mean(mm), 1 - alpha - 0.01)


class TestOnline(unittest.TestCase):
    def sequences(self, rng):
        T = 3000
        t = np.arange(T)
        return {
            "iid": rng.exponential(size=T),
            "trend": np.abs(rng.standard_normal(T)) * (1 + t / 100.0),
            "alternating": np.where(t % 2 == 0, 100.0, 0.01),
            "collapse": np.where(t < T // 2, 50.0 + rng.random(T), rng.random(T) * 1e-3),
            "adversarial_blocks": np.where((t // 37) % 2 == 0, rng.random(T) * 1e3, rng.random(T)),
        }

    def test_aci_long_run_bound_holds_for_any_sequence(self):
        rng = np.random.default_rng(16)
        for name, s in self.sequences(rng).items():
            for gamma in (0.001, 0.01, 0.1):
                for window in (None, 50):
                    _, err, _ = aci(s, 0.1, gamma, window)
                    gap = abs(err.mean() - 0.1)
                    self.assertLessEqual(gap, aci_bound(0.1, gamma, len(s)) + 1e-12, f"{name} {gamma} {window}")

    def test_quantile_tracker_bound_for_bounded_scores(self):
        rng = np.random.default_rng(17)
        for name, s in self.sequences(rng).items():
            B = float(s.max())
            for eta in (0.01 * B, 0.1 * B):
                _, err = quantile_tracker(s, 0.1, eta, q_1=0.0)
                self.assertLessEqual(abs(err.mean() - 0.1), (B + eta) / (eta * len(s)) + 1e-12, name)

    def test_rls_is_one_step_ahead(self):
        rng = np.random.default_rng(18)
        X, y = rng.standard_normal((100, 3)), rng.standard_normal(100)
        p1 = rls_one_step(X, y)
        y2 = y.copy()
        y2[50] += 1e3
        p2 = rls_one_step(X, y2)
        np.testing.assert_array_equal(p1[:51], p2[:51])
        self.assertFalse(np.allclose(p1[51:], p2[51:]))

    def test_rls_stays_stable_on_drift_stream(self):
        # Regression test: without re-symmetrizing P, forget=0.99 diverged in the t(2) regime.
        X, y, bounds = drift_stream(6000, 5, np.random.default_rng(600))
        s = np.abs(y - rls_one_step(X, y, forget=0.99))
        self.assertLess(np.quantile(s[bounds[2]:], 0.9), 10.0)

    def test_rolling_coverage(self):
        err = np.array([0, 1, 0, 0, 1, 1, 0], dtype=bool)
        rc = rolling_coverage(err, 3)
        self.assertTrue(np.all(np.isnan(rc[:2])))
        np.testing.assert_allclose(rc[2:], [2 / 3, 2 / 3, 2 / 3, 1 / 3, 1 / 3])


if __name__ == "__main__":
    unittest.main()
