"""Compiled runtime Frame objects consumed by GRAIL's execution layer."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from grail.frame.spec import DependencySpec, FrameMetadata, FrameSpec, VariableSpec
from grail.frame.state import FrameState, FrameStateStore, PosteriorState, VariableState
from grail.frame.variable import Variable
from grail.graph.base import VariableNode
from grail.graph.causal import CausalGraph
from grail.logger import logger
from grail.settings import FRAME_VERSION

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
    _state_store: FrameStateStore | None = field(default=None, init=False, repr=False, compare=False)

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

    @property
    def definition_hash(self) -> str:
        """Return a stable hash of this Frame's yaml definition."""
        payload = self.to_spec().model_dump_json(exclude_none=False)
        return sha256(payload.encode("utf-8")).hexdigest()

    def attach_state_store(self, store: FrameStateStore) -> None:
        """
        Attach persistent runtime state and register any YAML baseline evidence.

        Existing ``Variable.observations`` values originate from a Frame spec. They
        are preserved for model compatibility and copied once into the append-only
        ledger using a definition-scoped deterministic batch ID.
        """
        self._state_store = store
        for variable in self.get_variables():
            if variable.observations is not None:
                store.append_observations(
                    self.name,
                    self.definition_hash,
                    variable.name,
                    variable.observations,
                    batch_id=f"spec-{self.definition_hash}-{variable.name}",
                    source="frame-spec",
                )

    def record_observations(
        self,
        variable_name_or_id: str,
        values: Any,
        *,
        batch_id: str | None = None,
        source: str = "runtime",
    ) -> str:
        """
        Append evidence for a variable and return its durable batch ID.

        Runtime evidence is not written into the Frame YAML and does not overwrite
        previous batches. Reusing a batch ID with identical contents is safe for
        retrying an upload; different contents are rejected.
        """
        variable = self.get_variable(variable_name_or_id)
        store = self._require_state_store()
        batch = store.append_observations(
            self.name,
            self.definition_hash,
            variable.name,
            values,
            batch_id=batch_id,
            source=source,
        )
        return batch.id

    def load_observations(self, path: Path | str) -> list[str]:
        """
        Load and append the batches declared by a JSON observation file.

        The portable file format is ``{"frame": "...", "batches": [{"id":
        "...", "variable": "...", "values": [...], "source": "..."}]}``.
        A file is merely a convenient input transport; values are retained in the
        SQLite graph-variable observation ledger after this method succeeds.
        """
        observation_path = Path(path).expanduser().resolve()
        try:
            payload = json.loads(observation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid observation JSON in '{observation_path}'") from error
        if not isinstance(payload, Mapping):
            raise TypeError("observation JSON must contain an object at its root")
        declared_frame = payload.get("frame")
        if declared_frame is not None and declared_frame != self.name:
            raise ValueError(
                f"observation file is for Frame '{declared_frame}', not '{self.name}'"
            )
        batches = payload.get("batches")
        if not isinstance(batches, list) or not batches:
            raise ValueError("observation JSON must contain a non-empty 'batches' list")

        batch_ids = []
        for entry in batches:
            if not isinstance(entry, Mapping):
                raise TypeError("every observation batch must be an object")
            unknown_fields = set(entry) - {"id", "variable", "values", "source"}
            if unknown_fields:
                raise ValueError(
                    f"observation batch has unsupported fields: {sorted(unknown_fields)}"
                )
            if "variable" not in entry or "values" not in entry:
                raise ValueError("every observation batch requires 'variable' and 'values'")
            batch_ids.append(
                self.record_observations(
                    entry["variable"],
                    entry["values"],
                    batch_id=entry.get("id"),
                    source=entry.get("source", f"file:{observation_path.name}"),
                )
            )
        return batch_ids

    def get_observation_batches(self, variable_name_or_id: str | None = None) -> list[Any]:
        """Return append-only evidence history, optionally for one variable."""
        variable_name = (
            self.get_variable(variable_name_or_id).name if variable_name_or_id is not None else None
        )
        return self._require_state_store().get_observation_batches(
            self.name, self.definition_hash, variable_name=variable_name
        )

    def get_posterior(
        self, variable_name_or_id: str, *, strategy: str | None = None
    ) -> PosteriorState | None:
        """Return a retained posterior for a variable, if inference has produced one.

        Args:
            strategy: Selects the inference method that produced the snapshot.
                ``None`` (the default) returns the most recently updated posterior
                across all strategies. ``"beta-bernoulli-exact"`` selects the
                bundled exact conjugate updater. Custom strategies use the stable
                value of their :attr:`InferenceStrategy.name` attribute.
        """
        variable = self.get_variable(variable_name_or_id)
        return self._require_state_store().get_posterior(
            self.name,
            self.definition_hash,
            variable_name=variable.name,
            strategy=strategy,
        )

    def save_posterior(
        self,
        variable_name_or_id: str,
        *,
        strategy: str,
        distribution: str,
        prior: Mapping[str, Any],
        params: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> PosteriorState:
        """Persist an inference result without altering this Frame's declared prior."""
        variable = self.get_variable(variable_name_or_id)
        return self._require_state_store().save_posterior(
            self.name,
            self.definition_hash,
            variable_name=variable.name,
            strategy=strategy,
            distribution=distribution,
            prior=prior,
            params=params,
            metadata=metadata,
        )

    def inspect_state(self) -> FrameState:
        """Return the current prior, evidence history, and latest posterior by variable."""
        store = self._require_state_store()
        posterior_by_variable = store.get_posteriors(self.name, self.definition_hash)
        variables = {}
        for variable in self.get_variables():
            distribution = variable.get_distribution_name()
            if distribution is None:
                raise ValueError(f"Variable '{variable.name}' is missing a distribution specification")
            variables[variable.name] = VariableState(
                name=variable.name,
                prior={
                    "distribution": distribution,
                    "params": _restore_references(
                        variable.get_distribution_params(), self._name_for_node_id
                    ),
                },
                observation_batches=store.get_observation_batches(
                    self.name, self.definition_hash, variable_name=variable.name
                ),
                posterior=posterior_by_variable.get(variable.name),
            )
        return FrameState(
            frame_name=self.name,
            definition_hash=self.definition_hash,
            variables=variables,
        )

    @classmethod
    def from_spec(cls, spec: FrameSpec) -> "Frame":
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
                "variable name must start with a letter, be <= 50 chars, and contain only "
                "letters, digits, and underscores"
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
        """Resolve a dependency endpoint given a Variable instance, name, or node ID."""
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

        # A runtime node ID, or the stable variable name accepted elsewhere on Frame.
        node = self.graph.get_node(variable_or_id)
        if isinstance(node, VariableNode):
            return variable_or_id
        node_id = self._find_node_id_by_name(variable_or_id)
        if node_id is not None:
            return node_id
        raise KeyError(f"Frame '{self.name}' has no variable named or id '{variable_or_id}'")

    def _require_state_store(self) -> FrameStateStore:
        if self._state_store is None:
            raise RuntimeError(
                "Frame has no state store. Load it through FrameRepository or call "
                "attach_state_store(FrameStateStore(...)) first."
            )
        return self._state_store

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
