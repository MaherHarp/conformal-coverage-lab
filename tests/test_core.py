"""Exactness tests: closed forms vs brute force, ranks, and equivalences between methods."""
import math
import unittest

import numpy as np

from conformal import (
    QuantileLinear,
    Ridge,
    conformal_quantile,
    conformal_rank,
    cv_plus,
    full_conformal_bruteforce,
    full_conformal_ridge,
    jackknife_plus,
    ridge_loo,
    set_contains,
    weighted_quantile,
)
from conformal.data import rbf_features, sample_hetero
from conformal.jackknife import q_minus, q_plus


class TestRanks(unittest.TestCase):
    def test_conformal_rank_float_guard(self):
        self.assertEqual(conformal_rank(999, 0.1), 900)  # 1000 * 0.9 is 900.0000000000001 in floating point
        self.assertEqual(conformal_rank(9, 0.1), 9)
        self.assertEqual(conformal_rank(19, 0.05), 19)
        self.assertEqual(conformal_rank(8, 0.1), 9)

    def test_quantile_out_of_range_is_infinite(self):
        self.assertEqual(conformal_quantile(np.arange(8.0), 0.1), math.inf)
        self.assertEqual(conformal_quantile(np.arange(9.0), 0.1), 8.0)

    def test_q_plus_q_minus_ranks(self):
        rng = np.random.default_rng(1)
        for n in (5, 19, 50, 101):
            for alpha in (0.05, 0.1, 0.2, 0.33):
                v = rng.standard_normal((3, n))
                s = np.sort(v, axis=1)
                k = math.ceil((1 - alpha) * (n + 1) - 1e-9)
                j = math.floor(alpha * (n + 1) + 1e-9)
                exp_hi = s[:, k - 1] if k <= n else np.full(3, np.inf)
                exp_lo = s[:, j - 1] if j >= 1 else np.full(3, -np.inf)
                np.testing.assert_array_equal(q_plus(v, alpha), exp_hi)
                np.testing.assert_array_equal(q_minus(v, alpha), exp_lo)

    def test_weighted_quantile_equal_weights_is_standard(self):
        rng = np.random.default_rng(2)
        for n in (9, 50, 200):
            s = rng.exponential(size=n)
            for alpha in (0.05, 0.1, 0.25):
                q = weighted_quantile(s, np.full(n, 2.5), np.full(4, 2.5), alpha)
                self.assertTrue(np.all(q == conformal_quantile(s, alpha)))


class TestModels(unittest.TestCase):
    def test_ridge_loo_matches_refits(self):
        rng = np.random.default_rng(3)
        for n, d, lam in [(30, 5, 0.1), (20, 40, 1e-3), (50, 10, 10.0)]:
            X, y, Xt = rng.standard_normal((n, d)), rng.standard_normal(n), rng.standard_normal((7, d))
            o = ridge_loo(X, y, lam, Xt)
            for i in range(n):
                keep = np.arange(n) != i
                m = Ridge(lam).fit(X[keep], y[keep])
                self.assertAlmostEqual(o["loo_resid"][i], y[i] - m.predict(X[i : i + 1])[0], places=6)
                np.testing.assert_allclose(o["mu_loo_test"][:, i], m.predict(Xt), atol=1e-6)

    def test_quantile_regression_optimality_counts(self):
        # With an intercept, an LP optimum satisfies #{y < fit} <= tau n <= #{y <= fit}.
        rng = np.random.default_rng(4)
        x, y = sample_hetero(300, rng)
        X = rbf_features(x, 6)
        for tau in (0.05, 0.5, 0.9):
            fit = QuantileLinear(tau).fit(X, y).predict(X)
            below, at_or_below = np.sum(y < fit - 1e-7), np.sum(y <= fit + 1e-7)
            self.assertLessEqual(below, tau * 300 + 1e-6)
            self.assertGreaterEqual(at_or_below, tau * 300 - 1e-6)


class TestFullConformal(unittest.TestCase):
    def test_exact_equals_bruteforce(self):
        rng = np.random.default_rng(5)
        cases = [(15, 3, 0.1, 0.2), (40, 13, 1e-2, 0.1), (12, 20, 1e-3, 0.1), (60, 4, 1.0, 0.05), (25, 2, 0.5, 0.3), (300, 13, 1e-2, 0.1)]
        for n, d, lam, alpha in cases:
            for _ in range(4):
                X = rng.standard_normal((n, d))
                y = X @ rng.standard_normal(d) + rng.standard_t(3, n)
                x_new = rng.standard_normal(d)
                iv = full_conformal_ridge(X, y, x_new, lam, alpha)
                span = 4 * (np.abs(y).max() + 5)
                grid = np.linspace(-span, span, 3001)
                brute = full_conformal_bruteforce(X, y, x_new, lam, alpha, grid)
                ends = iv[np.isfinite(iv)] if len(iv) else np.array([])
                far = np.ones(len(grid), dtype=bool)
                if len(ends):
                    far = np.min(np.abs(grid[:, None] - ends[None, :]), axis=1) > 1e-7 * span
                np.testing.assert_array_equal(set_contains(iv, grid)[far], brute[far], err_msg=f"{n},{d},{lam},{alpha}")

    def test_small_n_gives_whole_line(self):
        # With n + 1 = 5 and alpha = 0.1, p(y) >= 1/5 > 0.1 for every y.
        rng = np.random.default_rng(6)
        X, y = rng.standard_normal((4, 2)), rng.standard_normal(4)
        iv = full_conformal_ridge(X, y, rng.standard_normal(2), 0.1, 0.1)
        self.assertEqual(iv.tolist(), [[-math.inf, math.inf]])


class TestEquivalences(unittest.TestCase):
    def test_cv_plus_with_n_folds_is_jackknife_plus(self):
        rng = np.random.default_rng(7)
        X, y, Xt = rng.standard_normal((40, 6)), rng.standard_normal(40), rng.standard_normal((10, 6))
        lo1, hi1 = jackknife_plus(X, y, Xt, 0.3, 0.1)
        lo2, hi2 = cv_plus(lambda: Ridge(0.3), X, y, Xt, 0.1, K=40, rng=rng)
        np.testing.assert_allclose(lo1, lo2, atol=1e-8)
        np.testing.assert_allclose(hi1, hi2, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
