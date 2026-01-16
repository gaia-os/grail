# Builtin imports
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Variable:
    """
    Represents a statistical variable tracked inside a graph.

    :param name:           Human-readable variable name
    :param prior:          Prior specification containing distribution metadata
    :param description:    Optional narrative describing the variable
    :param code:           Optional machine-readable code/identifier
    :param observations:   Optional raw observation data attached to the variable
    """

    name: str
    prior: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    code: Optional[str] = None
    observations: Optional[Any] = None

    def get_distribution_name(self) -> Optional[str]:
        """
        Return the configured distribution name, if available.

        :returns:   Distribution identifier or ``None`` when unspecified
        """

        return self.prior.get("distribution")

    def get_distribution_params(self) -> Dict[str, Any]:
        """
        Return a defensive copy of the distribution parameters.

        :returns:   Dictionary of parameters used to configure the prior
        """

        params = self.prior.get("params", {})
        return dict(params)

    def set_distribution(self, name: str, params: Dict[str, Any]) -> None:
        """
        Update the prior specification in-place.

        :param name:    Distribution identifier understood by the engine
        :param params:  Parameter mapping to feed to the distribution factory
        """

        self.prior["distribution"] = name
        self.prior["params"] = dict(params)

    def get_observations(self) -> Optional[Any]:
        """
        Return any attached observational data.

        :returns:   Raw observation payload or ``None`` when absent
        """

        return self.observations

    def set_observations(self, values: Any) -> None:
        """
        Attach observational data to the variable.

        :param values:  Observation payload (list, tensor, etc.)
        """

        self.observations = values

    def clear_observations(self) -> None:
        """
        Remove any observation payload currently stored on the variable.
        """

        self.observations = None

    def is_observed(self) -> bool:
        """
        Determine whether observation data is present.

        :returns:   ``True`` when observations were attached, otherwise ``False``
        """

        return self.observations is not None

    def get_distribution_spec(self) -> Dict[str, Any]:
        """
        Return the combination of distribution name and parameters.

        :returns:   Mapping with ``name`` and ``params`` entries
        """

        return {
            "name": self.get_distribution_name(),
            "params": self.get_distribution_params()
        }
