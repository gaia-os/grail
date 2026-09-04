"""Compilation of declarative Frames into executable Pyro models."""

from collections.abc import Callable, Mapping
from typing import Any

import pyro
import torch

from grail.frame import Frame
from grail.graph.base import VariableNode
from grail.logger import logger
from grail.stats.distributions import DistributionFactory


class Engine:
    """
    The Engine converts a GRAIL Frame into an executable Pyro model.

    Compilation walks the Frame graph in topological order so that every
    parameter reference resolves to a value already sampled in the same trace.
    A reference is any string parameter naming another variable, either by its
    runtime node ID or by its stable name.
    """

    def __init__(self, frame: Frame):
        self.frame = frame
        logger.info(f"Engine initialized for Frame: {frame.name}")

    def get_model(self) -> Callable:
        """
        Return a callable Pyro model function based on the Frame.

        The returned callable accepts an optional ``observations`` mapping of variable
        name (or node ID) to observed values, which takes precedence over any
        observations attached to the Frame's variables. It carries a
        ``variable_names`` attribute listing the Pyro sample sites it creates.
        """
        graph = self.frame.graph
        topo_order = graph.topological_sort()
        logger.debug(f"Topological order for execution: {topo_order}")

        variable_nodes = [
            node
            for node_id in topo_order
            if isinstance(node := graph.get_node(node_id), VariableNode)
        ]
        if not variable_nodes:
            raise ValueError(f"Frame '{self.frame.name}' has no variables to compile")

        variable_names = frozenset(node.name for node in variable_nodes)
        # Observations may be keyed by either stable name or runtime node ID.
        observable_keys = variable_names | {node.id for node in variable_nodes}

        def model(observations: Mapping[str, Any] | None = None) -> dict[str, Any]:
            observations = {} if observations is None else dict(observations)
            unknown_keys = sorted(set(observations) - observable_keys)
            if unknown_keys:
                raise KeyError(
                    f"Frame '{self.frame.name}' has no variables named {unknown_keys}. "
                    f"Known variables: {sorted(variable_names)}"
                )

            # Sampled values are returned by node ID, but are resolvable by either
            # key so that parameter references can use whichever the caller wrote.
            runtime_values: dict[str, Any] = {}
            scope: dict[str, Any] = {}

            for node in variable_nodes:
                distribution_name = node.get_distribution_name()
                if distribution_name is None:
                    raise ValueError(
                        f"Variable '{node.name}' is missing a distribution specification"
                    )

                params = self._resolve_params(
                    node.get_distribution_params(), scope, node.name
                )
                distribution = DistributionFactory.create(distribution_name, params)

                observed = self._resolve_observation(node, observations)
                value = pyro.sample(node.name, distribution, obs=observed)

                runtime_values[node.id] = value
                scope[node.id] = value
                scope[node.name] = value

            return runtime_values

        model.variable_names = variable_names
        return model

    def _resolve_params(
        self, params: Mapping[str, Any], scope: Mapping[str, Any], variable_name: str
    ) -> dict[str, Any]:
        """Replace parameter references with values sampled earlier in the trace."""
        return {
            name: self._resolve_value(value, scope, variable_name, name)
            for name, value in params.items()
        }

    def _resolve_value(
        self, value: Any, scope: Mapping[str, Any], variable_name: str, param_name: str
    ) -> Any:
        """Recursively resolve references nested inside a parameter value."""
        if isinstance(value, str):
            if value in scope:
                return scope[value]
            raise self._reference_error(variable_name, param_name, value)
        if isinstance(value, Mapping):
            if set(value) == {"$ref"}:
                raise ValueError(
                    f"Variable '{variable_name}' parameter '{param_name}' still holds an "
                    f"unresolved {{'$ref': {value['$ref']!r}}}. Compile the Frame with "
                    f"Frame.from_spec() or FrameRepository.load() to resolve declarative "
                    f"references."
                )
            return {
                key: self._resolve_value(item, scope, variable_name, param_name)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                self._resolve_value(item, scope, variable_name, param_name)
                for item in value
            ]
        return value

    def _reference_error(
        self, variable_name: str, param_name: str, reference: str
    ) -> ValueError:
        """Explain an unresolved reference in terms of the Frame's own structure."""
        node = self.frame.graph.get_node(reference)
        if isinstance(node, VariableNode):
            return ValueError(
                f"Variable '{variable_name}' parameter '{param_name}' references "
                f"'{node.name}', which has not been sampled yet. Add a dependency from "
                f"'{node.name}' to '{variable_name}' so the graph orders them correctly."
            )
        return ValueError(
            f"Variable '{variable_name}' parameter '{param_name}' references unknown "
            f"variable '{reference}'."
        )

    @staticmethod
    def _resolve_observation(
        node: VariableNode, observations: Mapping[str, Any]
    ) -> Any | None:
        """Select runtime observations over Frame-attached observations for one variable."""
        if node.name in observations:
            observed = observations[node.name]
        elif node.id in observations:
            observed = observations[node.id]
        elif node.is_observed:
            observed = node.get_observations()
        else:
            return None

        if observed is None or isinstance(observed, torch.Tensor):
            return observed
        return torch.tensor(observed, dtype=torch.float32)
