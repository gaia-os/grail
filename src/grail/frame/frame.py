"""Compiled runtime Frame objects consumed by GRAIL's execution layer."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from grail.graph.base import VariableNode
from grail.graph.causal import CausalGraph
from grail.logger import logger
from grail.settings import FRAME_VERSION

from grail.frame.spec import DependencySpec, FrameMetadata, FrameSpec, VariableSpec
from grail.frame.variable import Variable


# Variable names must start with a letter, use only letters/digits/underscores,
# and be at most 50 characters long.
VARIABLE_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,49}$")


@dataclass(eq=False, repr=False)
class Frame:
    """
    A runtime world-model container with an explicit dependency graph.

    YAML is represented by :class:`FrameSpec`; this class is its compiled form.  The
    graph deliberately stores generated node IDs internally while public Frame APIs
    accept stable variable names as well as node IDs.
    """

    name: str
    graph: CausalGraph = field(default_factory=CausalGraph)
    metadata: FrameMetadata | dict[str, Any] = field(default_factory=FrameMetadata)
    version: int = FRAME_VERSION

    def add_variable(
        self,
        name: str,
        dist: str,
        params: dict[str, Any],
        observations: Any | None = None,
        *,
        description: str | None = None,
        code: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """Add a variable and return its runtime node ID.

        Parameter references use already-resolved runtime node IDs.  YAML callers
        should use ``{"$ref": "variable_name"}``, which is resolved by
        :meth:`from_spec`.
        """
        self._validate_variable_name(name)
        if self._find_node_id_by_name(name) is not None:
            raise ValueError(f"Frame '{self.name}' already contains variable '{name}'")

        logger.debug(f"Frame '{self.name}': adding variable '{name}' with distribution '{dist}'")
        variable = Variable(
            name=name,
            description=description,
            code=code,
            attributes=dict(attributes or {}),
        )
        variable.set_distribution(dist, params)
        if observations is not None:
            variable.set_observations(observations)

        node = VariableNode(name=name, variable=variable)
        self.graph.add_node(node)
        # Bind node ID to Variable for easy referencing
        variable.bind_node_id(node.id)
        return node.id

    def add_dependency(
        self,
        source: Variable | str,
        target: Variable | str,
        **attributes: Any,
    ) -> None:
        """Add a directed dependency between existing variables."""
        source_id = self._resolve_dependency_node_id(source)
        target_id = self._resolve_dependency_node_id(target)
        self.graph.add_edge(source_id, target_id, **attributes)

    def get_variable_id(self, name_or_variable: str | Variable) -> str:
        """Return a variable's runtime node ID by name or Variable instance."""
        if isinstance(name_or_variable, Variable):
            return self._resolve_dependency_node_id(name_or_variable)

        node_id = self._find_node_id_by_name(name_or_variable)
        if node_id is None:
            raise KeyError(f"Frame '{self.name}' has no variable named '{name_or_variable}'")
        return node_id

    def get_variable(self, name_or_id: str) -> Variable:
        """Return a Frame Variable by stable name or runtime node ID."""
        node = self.graph.get_node(name_or_id)
        if isinstance(node, VariableNode):
            return node.variable

        node_id = self._find_node_id_by_name(name_or_id)
        if node_id is not None:
            named_node = self.graph.get_node(node_id)
            if isinstance(named_node, VariableNode):
                return named_node.variable

        raise KeyError(f"Frame '{self.name}' has no variable named or id '{name_or_id}'")

    def get_variables(self) -> list[Variable]:
        """Return all registered Frame variables."""
        return [node.variable for node in self.graph.get_variables()]

    @classmethod
    def from_spec(cls, spec: FrameSpec) -> Frame:
        """Compile a validated declarative specification into a runtime Frame."""
        frame = cls(name=spec.name, metadata=spec.metadata, version=spec.version)
        logger.info(
            f"Frame '{spec.name}': compiling from spec and loading {len(spec.variables)} variables"
        )
        variable_ids_by_name: dict[str, str] = {}
        for variable in spec.variables:
            logger.debug(
                f"Frame '{spec.name}': loading variable '{variable.name}' ({variable.distribution})"
            )
            variable_ids_by_name[variable.name] = frame.add_variable(
                name=variable.name,
                dist=variable.distribution,
                params=variable.params,
                observations=variable.observations,
                description=variable.description,
                code=variable.code,
                attributes=variable.attributes,
            )
        for variable in spec.variables:
            logger.debug(f"Frame '{spec.name}': resolving references for '{variable.name}'")
            frame.get_variable(variable_ids_by_name[variable.name]).set_distribution(
                variable.distribution,
                _resolve_references(variable.params, variable_ids_by_name.__getitem__),
            )
        logger.info(f"Frame '{spec.name}': loading {len(spec.dependencies)} dependencies")
        for dependency in spec.dependencies:
            frame.add_dependency(
                variable_ids_by_name[dependency.source],
                variable_ids_by_name[dependency.target],
                **dependency.attributes,
            )
        logger.info(f"Frame '{spec.name}': compile complete")
        return frame

    def to_spec(self) -> FrameSpec:
        """Convert this runtime graph into a portable declarative Frame spec."""
        variables = []
        frame_variables = self.get_variables()
        logger.debug(f"Frame '{self.name}': serializing {len(frame_variables)} variables")
        for variable in frame_variables:
            distribution = variable.get_distribution_name()
            if distribution is None:
                raise ValueError(
                    f"Variable '{variable.name}' is missing a distribution specification"
                )
            variables.append(
                VariableSpec(
                    name=variable.name,
                    distribution=distribution,
                    params=_restore_references(
                        variable.get_distribution_params(), self._name_for_node_id
                    ),
                    observations=variable.get_observations(),
                    description=variable.description,
                    code=variable.code,
                    attributes=variable.attributes,
                )
            )

        dependencies = []
        for source_id, target_id, attributes in self.graph.graph.edges(data=True):
            dependencies.append(
                DependencySpec(
                    source=self._name_for_node_id(source_id),
                    target=self._name_for_node_id(target_id),
                    attributes=dict(attributes),
                )
            )
        logger.debug(f"Frame '{self.name}': serialized {len(dependencies)} dependencies")
        return FrameSpec(
            version=self.version,
            name=self.name,
            metadata=self.metadata,
            variables=variables,
            dependencies=dependencies,
        )

    def _validate_variable_name(self, name: str) -> None:
        """Validate runtime variable names used by the imperative API."""
        if not isinstance(name, str):
            raise TypeError("variable name must be a string")
        if not name.strip():
            raise ValueError("variable name must be non-empty")
        if not VARIABLE_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                "variable name must start with a letter, be <= 50 chars, and contain only letters, digits, and underscores"
            )

    def _name_for_node_id(self, node_id: str) -> str:
        """Resolve a runtime node ID to its stable variable name."""
        node = self.graph.get_node(node_id)
        if not isinstance(node, VariableNode):
            raise KeyError(f"Frame '{self.name}' has no node '{node_id}'")
        return node.name

    def _find_node_id_by_name(self, name: str) -> str | None:
        """Find a variable node ID by stable name, if present."""
        for variable in self.get_variables():
            if variable.name == name:
                return variable.node_id
        return None

    def _resolve_dependency_node_id(self, variable_or_id: Variable | str) -> str:
        """Resolve a dependency endpoint given a Variable instance or node ID."""
        if isinstance(variable_or_id, Variable):
            if variable_or_id.node_id is None:
                raise KeyError(
                    f"Frame '{self.name}' Variable '{variable_or_id.name}' is not bound to a node"
                )
            node = self.graph.get_node(variable_or_id.node_id)
            # Check that the retrieved VariableNode's attribute also matches the passed-in variable
            if not isinstance(node, VariableNode) or node.variable is not variable_or_id:
                raise KeyError(
                    f"Frame '{self.name}' Variable '{variable_or_id.name}' does not match runtime node"
                )
            return variable_or_id.node_id

        # Should be a VariableNode graph ID then
        node = self.graph.get_node(variable_or_id)
        if isinstance(node, VariableNode):
            return variable_or_id
        raise KeyError(f"Frame '{self.name}' has no variable with id '{variable_or_id}'")

def _resolve_references(value: Any, resolve_name: Any) -> Any:
    """Replace recursive declarative ``$ref`` mappings with runtime node IDs."""
    if isinstance(value, dict):
        if set(value) == {"$ref"}:
            return resolve_name(value["$ref"])
        return {key: _resolve_references(item, resolve_name) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_references(item, resolve_name) for item in value]
    return value


def _restore_references(value: Any, resolve_id: Any) -> Any:
    """Replace known runtime node IDs with portable declarative ``$ref`` mappings."""
    if isinstance(value, str):
        try:
            return {"$ref": resolve_id(value)}
        except KeyError:
            return value
    if isinstance(value, dict):
        return {key: _restore_references(item, resolve_id) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_references(item, resolve_id) for item in value]
    return value
