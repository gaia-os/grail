"""scipy-backed PDF/PMF helpers for plotting GRAIL distributions.

Kept separate from `grail.stats.distributions` (which builds `pyro.distributions`
objects) so scipy stays a viz-only dependency.
"""

from typing import Any

import numpy as np
from scipy import stats

# Distribution codes whose mass is discrete; plotted as a PMF instead of a PDF.
DISCRETE_CODES = {"bernoulli", "binomial"}


def pdf_values(code: str, params: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Evaluate the PDF (or PMF, for discrete distributions) at points `x`."""
    if code == "beta":
        return stats.beta.pdf(x, params.get("alpha", 1.0), params.get("beta", 1.0))
    if code == "normal":
        return stats.norm.pdf(x, params.get("loc", 0.0), params.get("scale", 1.0))
    if code == "uniform":
        low = params.get("low", 0.0)
        high = params.get("high", 1.0)
        return stats.uniform.pdf(x, loc=low, scale=high - low)
    if code == "exponential":
        return stats.expon.pdf(x, scale=1.0 / params.get("rate", 1.0))
    if code == "gamma":
        return stats.gamma.pdf(x, params.get("concentration", 1.0), scale=1.0 / params.get("rate", 1.0))
    if code == "lognormal":
        return stats.lognorm.pdf(x, params.get("scale", 1.0), scale=np.exp(params.get("loc", 0.0)))
    if code == "bernoulli":
        return stats.bernoulli.pmf(x, params.get("theta", 0.5))
    if code == "binomial":
        return stats.binom.pmf(x, params.get("n", 1), params.get("theta", 0.5))

    known = ", ".join(sorted(DISCRETE_CODES | {"beta", "normal", "uniform", "exponential", "gamma", "lognormal"}))
    raise ValueError(f"unsupported distribution code '{code}'; known codes: {known}")


def default_support(code: str, params: dict[str, Any], num_points: int = 200) -> np.ndarray:
    """Return a reasonable set of x values to plot a distribution's PDF/PMF over."""
    if code == "beta" or code == "uniform":
        return np.linspace(0.0, 1.0, num_points)
    if code == "bernoulli":
        return np.array([0, 1])
    if code == "binomial":
        return np.arange(0, int(params.get("n", 1)) + 1)
    if code in {"normal", "lognormal"}:
        loc = params.get("loc", 0.0)
        scale = params.get("scale", 1.0)
        spread = 4 * scale if code == "normal" else 4 * scale * np.exp(loc)
        low = loc - spread if code == "normal" else max(1e-6, np.exp(loc) - spread)
        high = loc + spread if code == "normal" else np.exp(loc) + spread
        return np.linspace(low, high, num_points)
    if code in {"exponential", "gamma"}:
        rate = params.get("rate", 1.0)
        return np.linspace(0.0, 8.0 / rate, num_points)

    raise ValueError(f"unsupported distribution code '{code}'")
