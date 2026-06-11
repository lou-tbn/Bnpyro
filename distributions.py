"""
distributions.py — Lightweight distribution classes for Bnpyro.

Replaces pyro/torch distributions. Each class wraps scipy.stats
and exposes a minimal API: constructor, .cdf(x), .sample(n).

Supported:
    Bernoulli(p)
    Categorical(probs)
    Normal(loc, scale)
    Beta(concentration1, concentration0)
    Gamma(concentration, rate)
    Uniform(low, high)
    Exponential(rate)
    LogNormal(loc, scale)
"""

import numpy as np
from scipy.stats import (
    norm as _norm, beta as _beta, gamma as _gamma,
    uniform as _uniform, expon as _expon, lognorm as _lognorm,
    poisson as _poisson, binom as _binom,
)


class Bernoulli:
    """Bernoulli(p): P(X=1) = p."""
    def __init__(self, p):
        self.probs = float(p)

    def sample(self, n: int) -> np.ndarray:
        return np.random.binomial(1, self.probs, size=n).astype(float)


class Categorical:
    """Categorical(probs): discrete over {0, ..., K-1}."""
    def __init__(self, probs):
        self.probs = [float(x) for x in probs]

    def sample(self, n: int) -> np.ndarray:
        return np.random.choice(len(self.probs), size=n, p=self.probs)


class Normal:
    """Normal(loc, scale): Gaussian distribution."""
    def __init__(self, loc, scale):
        self.loc = float(loc)
        self.scale = float(scale)

    def cdf(self, x: float) -> float:
        return float(_norm.cdf(x, self.loc, self.scale))

    def sample(self, n: int) -> np.ndarray:
        return _norm.rvs(self.loc, self.scale, size=n)


class Beta:
    """Beta(concentration1, concentration0): Beta(alpha, beta) distribution."""
    def __init__(self, concentration1, concentration0):
        self.concentration1 = float(concentration1)
        self.concentration0 = float(concentration0)

    def cdf(self, x: float) -> float:
        return float(_beta.cdf(x, self.concentration1, self.concentration0))

    def sample(self, n: int) -> np.ndarray:
        return _beta.rvs(self.concentration1, self.concentration0, size=n)


class Gamma:
    """Gamma(concentration, rate): shape=concentration, scale=1/rate."""
    def __init__(self, concentration, rate):
        self.concentration = float(concentration)
        self.rate = float(rate)

    def cdf(self, x: float) -> float:
        return float(_gamma.cdf(x, self.concentration, scale=1.0 / self.rate))

    def sample(self, n: int) -> np.ndarray:
        return _gamma.rvs(self.concentration, scale=1.0 / self.rate, size=n)


class Uniform:
    """Uniform(low, high): uniform distribution on [low, high]."""
    def __init__(self, low, high):
        self.low = float(low)
        self.high = float(high)

    def cdf(self, x: float) -> float:
        return float(_uniform.cdf(x, self.low, self.high - self.low))

    def sample(self, n: int) -> np.ndarray:
        return _uniform.rvs(self.low, self.high - self.low, size=n)


class Exponential:
    """Exponential(rate): scale=1/rate."""
    def __init__(self, rate):
        self.rate = float(rate)

    def cdf(self, x: float) -> float:
        return float(_expon.cdf(x, scale=1.0 / self.rate))

    def sample(self, n: int) -> np.ndarray:
        return _expon.rvs(scale=1.0 / self.rate, size=n)


class Poisson:
    """Poisson(rate): discrete distribution over {0, 1, 2, ...}."""
    def __init__(self, rate):
        self.rate = float(rate)

    def pmf(self, k: int) -> float:
        return float(_poisson.pmf(k, self.rate))

    def sample(self, n: int) -> np.ndarray:
        return _poisson.rvs(self.rate, size=n)


class Binomial:
    """Binomial(total_count, probs): discrete distribution over {0, 1, ..., total_count}."""
    def __init__(self, total_count, probs):
        self.total_count = int(total_count)
        self.probs = float(probs)

    def pmf(self, k: int) -> float:
        return float(_binom.pmf(k, self.total_count, self.probs))

    def sample(self, n: int) -> np.ndarray:
        return _binom.rvs(self.total_count, self.probs, size=n)


class LogNormal:
    """LogNormal(loc, scale): log-normal with underlying Normal(loc, scale)."""
    def __init__(self, loc, scale):
        self.loc = float(loc)
        self.scale = float(scale)

    def cdf(self, x: float) -> float:
        return float(_lognorm.cdf(x, s=self.scale, scale=np.exp(self.loc)))

    def sample(self, n: int) -> np.ndarray:
        return _lognorm.rvs(s=self.scale, scale=np.exp(self.loc), size=n)
