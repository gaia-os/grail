"""Runtime statistical variables owned by a :class:`grail.frame.Frame`."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from grail.logger import logger


@dataclass
class DistributionPrior:
    """Typed distribution configuration for a variable."""
    # Should probably expand this to official types of distribution names...
    distribution: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Variable:
    """A named statistical variable and its distribution specification."""
    name: str
    prior: DistributionPrior = field(default_factory=DistributionPrior)
    description: str | None = None
    code: str | None = None
    observations: Any | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    node_id: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Handle Prior construction/normalization
        prior_valid = False
        if isinstance(self.prior, DistributionPrior):
            self.prior.params = dict(self.prior.params)
            prior_valid = True

        if not prior_valid:
            prior_data = self.prior
            if not isinstance(prior_data, Mapping):
                raise TypeError("prior must be a DistributionPrior or mapping")

            distribution = prior_data.get("distribution")
            if distribution is not None and not isinstance(distribution, str):
                raise TypeError("prior['distribution'] must be a string when provided")

            params = prior_data.get("params", {})
            if not isinstance(params, Mapping):
                raise TypeError("prior['params'] must be a mapping")

            # Okay, create object
            self.prior = DistributionPrior(distribution=distribution, params=dict(params))

    def get_distribution_name(self) -> str | None:
        """Return the distribution identifier, when configured."""
        return self.prior.distribution

    def get_distribution_params(self) -> dict[str, Any]:
        """Return a defensive copy of configured distribution parameters."""
        return dict(self.prior.params)

    def set_distribution(self, name: str, params: dict[str, Any]) -> None:
        """Set the distribution identifier and its parameter mapping."""
        self.prior.distribution = name
        self.prior.params = dict(params)

    def get_observations(self) -> Any | None:
        """Return attached observations, if present."""
        return self.observations

    def set_observations(self, values: Any) -> None:
        """Attach observational data to this variable."""
        self.observations = values

    def clear_observations(self) -> None:
        """Remove attached observations."""
        self.observations = None

    def is_observed(self) -> bool:
        """Return whether observational data is attached."""
        return self.observations is not None

    def get_distribution_spec(self) -> dict[str, Any]:
        """Return the serializable distribution configuration."""
        return {
            "name": self.get_distribution_name(),
            "params": self.get_distribution_params(),
        }

    def bind_node_id(self, node_id: str) -> None:
        """Bind this variable to a runtime graph node ID exactly once."""
        if self.node_id is None:
            self.node_id = node_id
            logger.debug(f"Bound Variable '{self.name}' to node id '{node_id}'")
            return
        if self.node_id != node_id:
            logger.warning(
                f"Variable '{self.name}' rebind attempted from '{self.node_id}' to '{node_id}'"
            )
            raise ValueError(
                f"Variable '{self.name}' is already bound to node id '{self.node_id}'"
            )
