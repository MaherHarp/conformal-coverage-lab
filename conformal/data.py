"""Synthetic data with a known conditional law, so coverage can be computed exactly.

Every generator here has Gaussian noise with a known mean and scale. That means
the coverage of any interval [lo(x), hi(x)] can be computed in closed form,
P(lo <= Y <= hi | X = x) = Phi((hi - f(x)) / s(x)) - Phi((lo - f(x)) / s(x)),
so no Monte Carlo test set is needed. Experiments average this quantity over x
(by quadrature or over sampled x). The only remaining randomness is the
train/calibration draw, which is exactly what conformal guarantees are about.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


# ---------------------------------------------------------------- 1-D heteroscedastic


def f_true(x: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * x) + 0.5 * x


def sigma_true(x: np.ndarray) -> np.ndarray:
    """Noise scale grows 10x across [0, 1]: 0.1 at x=0 and 1.0 at x=1."""
    return 0.1 + 0.9 * x**2


def sample_hetero(n: int, rng: np.random.Generator, x: np.ndarray | None = None):
    if x is None:
        x = rng.uniform(0.0, 1.0, n)
    y = f_true(x) + sigma_true(x) * rng.standard_normal(len(x))
    return x, y


def rbf_features(x: np.ndarray, n_centers: int = 12) -> np.ndarray:
    """Constant column plus Gaussian bumps on [0, 1]. Width = 1.5 x center spacing."""
    centers = np.linspace(0.0, 1.0, n_centers)
    width = 1.5 / (n_centers - 1)
    phi = np.exp(-0.5 * ((x[:, None] - centers[None, :]) / width) ** 2)
    return np.hstack([np.ones((len(x), 1)), phi])


def hetero_coverage(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Exact P(lo(x) <= Y <= hi(x) | X = x) for the heteroscedastic model."""
    m, s = f_true(x), sigma_true(x)
    cov = ndtr((hi - m) / s) - ndtr((lo - m) / s)
    return np.clip(cov, 0.0, 1.0)


# ---------------------------------------------------------------- covariate shift


def sample_tilted_x(n: int, theta: float, rng: np.random.Generator) -> np.ndarray:
    """Draw x on [0, 1] with density proportional to exp(theta * x) (inverse CDF)."""
    u = rng.uniform(0.0, 1.0, n)
    if abs(theta) < 1e-12:
        return u
    return np.log1p(u * np.expm1(theta)) / theta


def tilt_density(x: np.ndarray, theta: float) -> np.ndarray:
    if abs(theta) < 1e-12:
        return np.ones_like(x)
    return theta * np.exp(theta * x) / np.expm1(theta)


def tilt_ratio(x: np.ndarray, theta: float) -> np.ndarray:
    """Likelihood ratio dP_test/dP_train, with P_train = Uniform[0, 1]."""
    return tilt_density(x, theta)


# ---------------------------------------------------------------- linear Gaussian


def sample_linear(n: int, beta: np.ndarray, rng: np.random.Generator):
    """X ~ N(0, I_d), Y = X beta + N(0, 1)."""
    X = rng.standard_normal((n, len(beta)))
    return X, X @ beta + rng.standard_normal(n)


def linear_coverage(X: np.ndarray, beta: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    m = X @ beta
    return np.clip(ndtr(hi - m) - ndtr(lo - m), 0.0, 1.0)


# ---------------------------------------------------------------- non-stationary stream


def drift_stream(T: int, d: int, rng: np.random.Generator):
    """A stream with three kinds of non-stationarity, for online conformal.

    * t in [0, T/4): stationary, noise sd 1
    * t = T/4: abrupt coefficient change (beta -> -beta rotated)
    * t in [T/2, 3T/4): noise sd ramps linearly from 1 to 4
    * t in [3T/4, T): noise becomes Student-t(2), with sd-scale 1 (heavy tails)

    Returns X, y, and the list of regime boundaries.
    """
    X = rng.standard_normal((T, d))
    beta1 = rng.standard_normal(d)
    beta2 = -np.roll(beta1, 1)
    t = np.arange(T)
    beta = np.where((t < T // 4)[:, None], beta1[None, :], beta2[None, :])
    scale = np.ones(T)
    ramp = (t >= T // 2) & (t < 3 * T // 4)
    scale[ramp] = 1.0 + 3.0 * (t[ramp] - T // 2) / (T // 4)
    scale[t >= 3 * T // 4] = 1.0
    noise = rng.standard_normal(T)
    heavy = t >= 3 * T // 4
    noise[heavy] = rng.standard_t(2.0, heavy.sum())
    y = np.einsum("td,td->t", X, beta) + scale * noise
    return X, y, [T // 4, T // 2, 3 * T // 4]
