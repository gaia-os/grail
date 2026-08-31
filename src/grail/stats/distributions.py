from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pyro.distributions as dist
import torch

from grail.logger import logger


class Distribution(ABC):
    """
    Base class for runtime distribution adapters.

    Subclasses define `name`, `code`, and `allowed_param_names`, then implement
    `_create()` to build the concrete `pyro.distributions` object from validated
    params.

    Callers should not instantiate distribution classes directly; use
    `DistributionFactory.create(code, params)` so lookup and validation stay
    consistent across the engine.
    """

    name: ClassVar[str]
    code: ClassVar[str]
    allowed_param_names: ClassVar[set[str]] = set()

    @staticmethod
    def to_tensor(value: Any) -> Any:
        if isinstance(value, (list, float, int)):
            return torch.tensor(value, dtype=torch.float32)
        return value

    def normalize_and_validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        unknown = set(params) - self.allowed_param_names
        if unknown:
            accepted = ", ".join(sorted(self.allowed_param_names))
            invalid = ", ".join(sorted(unknown))
            raise ValueError(
                f"Distribution '{self.code}' does not accept params: {invalid}. "
                f"Accepted params: {accepted}"
            )
        return params

    def create(self, params: dict[str, Any]):
        """Create a concrete Pyro distribution from validated params."""
        normalized = self.normalize_and_validate_params(params)
        return self._create(normalized)

    @abstractmethod
    def _create(self, params: dict[str, Any]):
        """Create a concrete Pyro distribution from canonicalized params."""


class NormalDistribution(Distribution):
    name = "Normal"
    code = "normal"
    allowed_param_names: ClassVar[set[str]] = {"loc", "scale"}

    def _create(self, params: dict[str, Any]):
        loc = self.to_tensor(params.get("loc", 0.0))
        scale = self.to_tensor(params.get("scale", 1.0))
        return dist.Normal(loc, scale)


class BernoulliDistribution(Distribution):
    name = "Bernoulli"
    code = "bernoulli"
    allowed_param_names: ClassVar[set[str]] = {"theta"}

    def _create(self, params: dict[str, Any]):
        theta = params.get("theta", 0.5)
        return dist.Bernoulli(self.to_tensor(theta))


class UniformDistribution(Distribution):
    name = "Uniform"
    code = "uniform"
    allowed_param_names: ClassVar[set[str]] = {"low", "high"}

    def _create(self, params: dict[str, Any]):
        low = self.to_tensor(params.get("low", 0.0))
        high = self.to_tensor(params.get("high", 1.0))
        return dist.Uniform(low, high)


class ExponentialDistribution(Distribution):
    name = "Exponential"
    code = "exponential"
    allowed_param_names: ClassVar[set[str]] = {"rate"}

    def _create(self, params: dict[str, Any]):
        rate = self.to_tensor(params.get("rate", 1.0))
        return dist.Exponential(rate)


class GammaDistribution(Distribution):
    name = "Gamma"
    code = "gamma"
    allowed_param_names: ClassVar[set[str]] = {"concentration", "rate"}

    def _create(self, params: dict[str, Any]):
        concentration = self.to_tensor(params.get("concentration", 1.0))
        rate = self.to_tensor(params.get("rate", 1.0))
        return dist.Gamma(concentration, rate)


class LogNormalDistribution(Distribution):
    name = "LogNormal"
    code = "lognormal"
    allowed_param_names: ClassVar[set[str]] = {"loc", "scale"}

    def _create(self, params: dict[str, Any]):
        loc = self.to_tensor(params.get("loc", 0.0))
        scale = self.to_tensor(params.get("scale", 1.0))
        return dist.LogNormal(loc, scale)


class BinomialDistribution(Distribution):
    name = "Binomial"
    code = "binomial"
    allowed_param_names: ClassVar[set[str]] = {"n", "theta"}

    def _create(self, params: dict[str, Any]):
        total_count = self.to_tensor(params.get("n", 1))
        theta = self.to_tensor(params.get("theta", 0.5))
        return dist.Binomial(total_count=total_count, probs=theta)


class BetaDistribution(Distribution):
    name = "Beta"
    code = "beta"
    allowed_param_names: ClassVar[set[str]] = {"alpha", "beta"}

    def _create(self, params: dict[str, Any]):
        alpha = self.to_tensor(params.get("alpha", 1.0))
        beta = self.to_tensor(params.get("beta", 1.0))
        return dist.Beta(alpha, beta)


class ConstantDistribution(Distribution):
    name = "Constant"
    code = "constant"
    allowed_param_names: ClassVar[set[str]] = {"value"}

    def _create(self, params: dict[str, Any]):
        if "value" not in params:
            raise ValueError("Distribution 'constant' requires param: value")
        value = self.to_tensor(params["value"])
        return dist.Delta(value)

#===============================================================================

AVAILABLE_DISTRIBUTIONS = {
    NormalDistribution.code: NormalDistribution,
    BernoulliDistribution.code: BernoulliDistribution,
    UniformDistribution.code: UniformDistribution,
    ExponentialDistribution.code: ExponentialDistribution,
    GammaDistribution.code: GammaDistribution,
    LogNormalDistribution.code: LogNormalDistribution,
    BinomialDistribution.code: BinomialDistribution,
    BetaDistribution.code: BetaDistribution,
    ConstantDistribution.code: ConstantDistribution,
}


class DistributionFactory:
    """Resolves a distribution code to a Distribution constructor."""

    @classmethod
    def create(cls, code: str, params: dict[str, Any]):
        logger.debug(f"Creating distribution: {code} with params: {params.keys()}")
        dist_cls = AVAILABLE_DISTRIBUTIONS.get(code)
        if dist_cls is None:
            known = ", ".join(sorted(AVAILABLE_DISTRIBUTIONS.keys()))
            raise ValueError(f"Unknown distribution code: {code}. Known codes: {known}")
        return dist_cls().create(params)

    @classmethod
    def get_distribution(cls, name: str, params: dict[str, Any]):
        """Backward-compatible alias; resolves display name or code."""
        return cls.create(name, params)

