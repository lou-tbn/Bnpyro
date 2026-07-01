"""
distributions.py — Lightweight distribution classes for Bnpyro.

Replaces pyro/torch distributions. Each class wraps scipy.stats
and exposes a minimal API: constructor, .cdf(x), .sample(n).

Supported
---------
Bernoulli(p)
Categorical(probs)
Normal(loc, scale)
Beta(concentration1, concentration0)
Gamma(concentration, rate)
Uniform(low, high)
Exponential(rate)
LogNormal(loc, scale)
Poisson(rate)
Binomial(total_count, probs)
"""

import numpy as np
from scipy.stats import (
    norm as _norm, beta as _beta, gamma as _gamma,
    uniform as _uniform, expon as _expon, lognorm as _lognorm,
    poisson as _poisson, binom as _binom,
)


class Bernoulli:
    """Bernoulli distribution — ``P(X=1) = p``.

    Parameters
    ----------
    p : float
        Probability of the outcome ``True`` (success), in ``[0, 1]``.
    """

    def __init__(self, p):
        """
        Parameters
        ----------
        p : float
            Probability of success, in ``[0, 1]``.
        """
        self.probs = float(p)

    def sample(self, n):
        """Draw *n* i.i.d. samples.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        numpy.ndarray of float, shape (n,)
            Array of 0.0 (False) and 1.0 (True).
        """
        return np.random.binomial(1, self.probs, size=n).astype(float)


class Categorical:
    """Categorical distribution — discrete over ``{0, …, K-1}``.

    Parameters
    ----------
    probs : array-like of float, length K
        Probability vector; must sum to 1.
    """

    def __init__(self, probs):
        """
        Parameters
        ----------
        probs : array-like of float, length K
            Probability vector. Does not need to be normalised
            (Bnpyro normalises internally when building the CPT).
        """
        self.probs = [float(x) for x in probs]

    def sample(self, n):
        """Draw *n* i.i.d. samples.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        numpy.ndarray of int, shape (n,)
            Indices in ``{0, …, K-1}``.
        """
        return np.random.choice(len(self.probs), size=n, p=self.probs)


class Normal:
    """Normal (Gaussian) distribution — ``X ~ N(loc, scale²)``.

    Parameters
    ----------
    loc : float
        Mean μ.
    scale : float
        Standard deviation σ > 0.
    """

    def __init__(self, loc, scale):
        """
        Parameters
        ----------
        loc : float
            Mean μ of the distribution.
        scale : float
            Standard deviation σ > 0.
        """
        self.loc = float(loc)
        self.scale = float(scale)

    def cdf(self, x):
        """Cumulative distribution function ``F(x) = P(X ≤ x)``.

        Parameters
        ----------
        x : float
            Evaluation point.

        Returns
        -------
        float
            Probability mass in ``(-∞, x]``.
        """
        return float(_norm.cdf(x, self.loc, self.scale))

    def sample(self, n):
        """Draw *n* i.i.d. samples.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        numpy.ndarray of float, shape (n,)
        """
        return _norm.rvs(self.loc, self.scale, size=n)


class Beta:
    """Beta distribution — ``X ~ Beta(α, β)``, supported on ``(0, 1)``.

    Parameters
    ----------
    concentration1 : float
        Shape parameter α > 0.
    concentration0 : float
        Shape parameter β > 0.
    """

    def __init__(self, concentration1, concentration0):
        """
        Parameters
        ----------
        concentration1 : float
            Shape parameter α > 0 (matches Pyro/PyTorch convention).
        concentration0 : float
            Shape parameter β > 0.
        """
        self.concentration1 = float(concentration1)
        self.concentration0 = float(concentration0)

    def cdf(self, x):
        """Cumulative distribution function ``F(x) = P(X ≤ x)``.

        Parameters
        ----------
        x : float
            Evaluation point in ``[0, 1]``.

        Returns
        -------
        float
        """
        return float(_beta.cdf(x, self.concentration1, self.concentration0))

    def sample(self, n):
        """Draw *n* i.i.d. samples.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        numpy.ndarray of float, shape (n,)
            Values in ``(0, 1)``.
        """
        return _beta.rvs(self.concentration1, self.concentration0, size=n)


class Gamma:
    """Gamma distribution — shape/rate parameterisation.

    ``X ~ Gamma(concentration, rate)`` with mean ``concentration / rate``.

    Parameters
    ----------
    concentration : float
        Shape parameter k > 0.
    rate : float
        Rate parameter λ > 0 (inverse of scale).
    """

    def __init__(self, concentration, rate):
        """
        Parameters
        ----------
        concentration : float
            Shape parameter k > 0.
        rate : float
            Rate λ > 0; scale = 1 / rate.
        """
        self.concentration = float(concentration)
        self.rate = float(rate)

    def cdf(self, x):
        """Cumulative distribution function ``F(x) = P(X ≤ x)``.

        Parameters
        ----------
        x : float
            Evaluation point ≥ 0.

        Returns
        -------
        float
        """
        return float(_gamma.cdf(x, self.concentration, scale=1.0 / self.rate))

    def sample(self, n):
        """Draw *n* i.i.d. samples.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        numpy.ndarray of float, shape (n,)
            Non-negative values.
        """
        return _gamma.rvs(self.concentration, scale=1.0 / self.rate, size=n)


class Uniform:
    """Continuous uniform distribution on ``[low, high]``.

    Parameters
    ----------
    low : float
        Lower bound (inclusive).
    high : float
        Upper bound (inclusive).
    """

    def __init__(self, low, high):
        """
        Parameters
        ----------
        low : float
            Lower bound.
        high : float
            Upper bound; must satisfy ``high > low``.
        """
        self.low = float(low)
        self.high = float(high)

    def cdf(self, x):
        """Cumulative distribution function ``F(x) = P(X ≤ x)``.

        Parameters
        ----------
        x : float
            Evaluation point.

        Returns
        -------
        float
            0 for ``x < low``, 1 for ``x > high``, linear in between.
        """
        return float(_uniform.cdf(x, self.low, self.high - self.low))

    def sample(self, n):
        """Draw *n* i.i.d. samples.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        numpy.ndarray of float, shape (n,)
            Values in ``[low, high]``.
        """
        return _uniform.rvs(self.low, self.high - self.low, size=n)


class Exponential:
    """Exponential distribution — ``X ~ Exp(rate)``, mean = ``1 / rate``.

    Parameters
    ----------
    rate : float
        Rate parameter λ > 0 (inverse of the scale/mean).
    """

    def __init__(self, rate):
        """
        Parameters
        ----------
        rate : float
            Rate λ > 0; scale = 1 / rate.
        """
        self.rate = float(rate)

    def cdf(self, x):
        """Cumulative distribution function ``F(x) = 1 - exp(-λx)``.

        Parameters
        ----------
        x : float
            Evaluation point ≥ 0.

        Returns
        -------
        float
        """
        return float(_expon.cdf(x, scale=1.0 / self.rate))

    def sample(self, n):
        """Draw *n* i.i.d. samples.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        numpy.ndarray of float, shape (n,)
            Non-negative values.
        """
        return _expon.rvs(scale=1.0 / self.rate, size=n)


class Poisson:
    """Poisson distribution — discrete over ``{0, 1, 2, …}``.

    Represented by a ``RangeVariable`` in pyAgrum (no discretization).

    Parameters
    ----------
    rate : float
        Rate parameter λ > 0; mean = variance = λ.
    """

    def __init__(self, rate):
        """
        Parameters
        ----------
        rate : float
            Mean arrival rate λ > 0.
        """
        self.rate = float(rate)

    def pmf(self, k):
        """Probability mass function ``P(X = k)``.

        Parameters
        ----------
        k : int
            Non-negative integer.

        Returns
        -------
        float
        """
        return float(_poisson.pmf(k, self.rate))

    def sample(self, n):
        """Draw *n* i.i.d. samples.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        numpy.ndarray of int, shape (n,)
        """
        return _poisson.rvs(self.rate, size=n)


class Binomial:
    """Binomial distribution — discrete over ``{0, 1, …, n}``.

    Represented by a ``RangeVariable`` in pyAgrum (no discretization).

    Parameters
    ----------
    total_count : int
        Number of trials n ≥ 1.
    probs : float
        Success probability p in ``[0, 1]``.
    """

    def __init__(self, total_count, probs):
        """
        Parameters
        ----------
        total_count : int
            Number of Bernoulli trials n ≥ 1.
        probs : float
            Probability of success p in ``[0, 1]``.
        """
        self.total_count = int(total_count)
        self.probs = float(probs)

    def pmf(self, k):
        """Probability mass function ``P(X = k)``.

        Parameters
        ----------
        k : int
            Integer in ``{0, …, total_count}``.

        Returns
        -------
        float
        """
        return float(_binom.pmf(k, self.total_count, self.probs))

    def sample(self, n):
        """Draw *n* i.i.d. samples.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        numpy.ndarray of int, shape (n,)
            Values in ``{0, …, total_count}``.
        """
        return _binom.rvs(self.total_count, self.probs, size=n)


class LogNormal:
    """Log-normal distribution — ``X = exp(Y)`` where ``Y ~ N(loc, scale²)``.

    Parameters
    ----------
    loc : float
        Mean of the underlying normal distribution (log-space).
    scale : float
        Standard deviation of the underlying normal distribution (log-space).
    """

    def __init__(self, loc, scale):
        """
        Parameters
        ----------
        loc : float
            Mean μ of the underlying ``N(μ, σ²)`` (in log-space).
        scale : float
            Standard deviation σ > 0 of the underlying normal (in log-space).
        """
        self.loc = float(loc)
        self.scale = float(scale)

    def cdf(self, x):
        """Cumulative distribution function ``F(x) = P(X ≤ x)``.

        Parameters
        ----------
        x : float
            Evaluation point > 0.

        Returns
        -------
        float
        """
        return float(_lognorm.cdf(x, s=self.scale, scale=np.exp(self.loc)))

    def sample(self, n):
        """Draw *n* i.i.d. samples.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        numpy.ndarray of float, shape (n,)
            Positive values.
        """
        return _lognorm.rvs(s=self.scale, scale=np.exp(self.loc), size=n)
