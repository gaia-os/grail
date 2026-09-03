"""Versioned, declarative schemas for Frame YAML specifications.

This module defines the portable, human-authored schema used in YAML files.
Within variable parameter maps, GRAIL supports a project-specific reference form:
``{"$ref": "VariableName"}``. A `$ref` maps a variable name to that
variable's runtime value (its sampled content) when evaluating params, and it
must be backed by a matching declared dependency edge.
"""

from collections.abc import Mapping
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from grail.logger import logger
from grail.settings import FRAME_SPEC_VERSION


class FrameMetadata(BaseModel):
    """Human and machine metadata attached to a Frame specification."""

    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class VariableSpec(BaseModel):
    """Declarative definition of one random or observed variable."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=50, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    distribution: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    observations: Any | None = None
    description: str | None = None
    code: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "distribution", mode="before")
    @classmethod
    def strip_required_strings(cls, value: Any) -> Any:
        """Normalize simple YAML strings while rejecting blank identifiers."""
        if isinstance(value, str):
            return value.strip()
        return value

    def referenced_variables(self) -> set[str]:
        """Return names referenced in params via explicit ``{"$ref": "name"}`` mappings."""
        return _find_references(self.params)


class DependencySpec(BaseModel):
    """A directed dependency from a parent variable to a child variable."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, max_length=50, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    target: str = Field(min_length=1, max_length=50, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_self_dependency(self) -> "DependencySpec":
        """Reject self-links before graph construction."""
        if self.source == self.target:
            raise ValueError("a dependency cannot target the same variable")
        return self


class FrameSpec(BaseModel):
    """The complete YAML source-of-truth for a world-model Frame."""

    model_config = ConfigDict(extra="forbid")

    version: int = FRAME_SPEC_VERSION
    name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    metadata: FrameMetadata = Field(default_factory=FrameMetadata)
    variables: list[VariableSpec] = Field(min_length=1)
    dependencies: list[DependencySpec] = Field(default_factory=list)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: Any) -> Any:
        """Normalize the Frame name from YAML."""
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_graph(self) -> "FrameSpec":
        """Validate names, references, and DAG constraints before runtime compilation."""
        logger.debug(
            f"Validating Frame spec '{self.name}' with {len(self.variables)} variables"
        )
        if self.version != FRAME_SPEC_VERSION:
            raise ValueError(
                f"unsupported Frame spec version {self.version}; "
                f"expected {FRAME_SPEC_VERSION}"
            )

        names = [variable.name for variable in self.variables]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"variable names must be unique; duplicates: {duplicates}")

        known_names = set(names)
        edges = {(dependency.source, dependency.target) for dependency in self.dependencies}
        unknown_dependencies = sorted(
            f"{source} -> {target}"
            for source, target in edges
            if source not in known_names or target not in known_names
        )
        if unknown_dependencies:
            raise ValueError(
                "dependencies must reference declared variables: "
                f"{', '.join(unknown_dependencies)}"
            )

        if len(edges) != len(self.dependencies):
            raise ValueError("dependencies must be unique")

        for variable in self.variables:
            # Every `$ref` must target an existing variable.
            missing_refs = sorted(variable.referenced_variables() - known_names)
            if missing_refs:
                raise ValueError(
                    f"variable '{variable.name}' references unknown variables: {missing_refs}"
                )
            # Every `$ref` in params must correspond to a declared edge ref -> variable.
            undeclared_refs = sorted(
                reference
                for reference in variable.referenced_variables()
                if (reference, variable.name) not in edges
            )
            if undeclared_refs:
                raise ValueError(
                    f"variable '{variable.name}' has $ref parameters without matching "
                    f"dependencies: {undeclared_refs}"
                )

        graph = nx.DiGraph()
        graph.add_nodes_from(names)
        graph.add_edges_from(edges)
        if not nx.is_directed_acyclic_graph(graph):
            cycle = nx.find_cycle(graph, orientation="original")
            path = " -> ".join(source for source, _, _ in cycle)
            raise ValueError(f"dependencies must form a DAG; cycle detected: {path}")
        logger.debug(
            f"Frame spec '{self.name}' validation complete: {len(edges)} dependencies"
        )
        return self


def _find_references(value: Any) -> set[str]:
    """Recursively extract GRAIL `$ref` values from nested param structures.

    Only the exact mapping shape ``{"$ref": "variable_name"}`` is treated as a
    reference. Other mappings are traversed recursively.
    """
    if isinstance(value, Mapping):
        if set(value) == {"$ref"}:
            reference = value["$ref"]
            if not isinstance(reference, str) or not reference:
                raise ValueError("$ref values must be non-empty variable names")
            return {reference}
        return set().union(*(_find_references(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_find_references(item) for item in value), set())
    return set()
