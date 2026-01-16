from typing import Any, Dict

import pyro.distributions as dist
import torch

from grail.logger import logger


class DistributionFactory:
    """
    Maps string names and parameters to Pyro distributions.
    Acts as a translation layer between GRAIL definitions and the PPL.
    """

    @staticmethod
    def get_distribution(name: str, params: Dict[str, Any]):
        """
        Returns a Pyro distribution instance based on the name and parameters.
        Adjusts parameter types (e.g., lists to tensors) as necessary.
        """
        name = name.lower()
        logger.debug(f"Creating distribution: {name} with params: {params.keys()}")

        # Helper to ensure tensors
        def to_tensor(val):
            if isinstance(val, (list, float, int)):
                return torch.tensor(val, dtype=torch.float32)
            return val

        if name == "normal":
            loc = to_tensor(params.get("loc", 0.0))
            scale = to_tensor(params.get("scale", 1.0))
            return dist.Normal(loc, scale)

        elif name == "bernoulli":
            probs = to_tensor(params.get("probs", 0.5))
            return dist.Bernoulli(probs) # noqa; for some reason ide doesn't detect bernoulli

        elif name == "uniform":
            low = to_tensor(params.get("low", 0.0))
            high = to_tensor(params.get("high", 1.0))
            return dist.Uniform(low, high)

        elif name == "exponential":
            rate = to_tensor(params.get("rate", 1.0))
            return dist.Exponential(rate) # noqa; for some reason ide doesn't detect exponential

        elif name == "gamma":
            concentration = to_tensor(params.get("concentration", 1.0))
            rate = to_tensor(params.get("rate", 1.0))
            return dist.Gamma(concentration, rate)

        else:
            raise ValueError(f"Unknown distribution: {name}")
