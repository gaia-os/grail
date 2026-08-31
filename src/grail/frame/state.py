"""
Persistent runtime state for Frame observations and inference results.

Frame YAML describes a model's initial priors.  This module deliberately keeps
runtime evidence and derived posterior state in SQLite instead of mutating that
declarative source.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import desc
from sqlmodel import Field, Session, SQLModel, col, create_engine, select

from grail.settings import FRAME_REGISTRY_DB_PATH


def _utcnow() -> datetime:
    return datetime.now(UTC)


class GraphFrameStateRecord(SQLModel, table=True):
    """One persisted runtime-state namespace for a version of a Frame."""

    __tablename__ = "graph_frame_states"

    id: int | None = Field(default=None, primary_key=True)
    frame_name: str = Field(index=True)
    definition_hash: str = Field(index=True)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class GraphObservationBatchRecord(SQLModel, table=True):
    """Append-only metadata for one submitted observation batch."""

    __tablename__ = "graph_observation_batches"

    id: str = Field(primary_key=True)
    frame_state_id: int = Field(foreign_key="graph_frame_states.id", index=True)
    variable_name: str = Field(index=True)
    source: str
    value_count: int
    payload_hash: str
    recorded_at: datetime = Field(default_factory=_utcnow, index=True)


class GraphVariableObservationValueRecord(SQLModel, table=True):
    """An individual JSON observation belonging to an append-only batch."""

    __tablename__ = "graph_variable_observation_values"

    id: int | None = Field(default=None, primary_key=True)
    batch_id: str = Field(foreign_key="graph_observation_batches.id", index=True)
    ordinal: int
    value_json: str


class GraphPosteriorRecord(SQLModel, table=True):
    """The latest posterior snapshot for one variable and inference strategy."""

    __tablename__ = "graph_posterior_records"

    id: str = Field(primary_key=True)
    frame_state_id: int = Field(foreign_key="graph_frame_states.id", index=True)
    variable_name: str = Field(index=True)
    strategy: str = Field(index=True)
    distribution: str
    prior_json: str
    posterior_json: str
    metadata_json: str
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow, index=True)


@dataclass(frozen=True)
class ObservationBatch:
    """A public, fully materialized observation batch."""

    id: str
    variable_name: str
    values: list[Any]
    source: str
    recorded_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "variable": self.variable_name,
            "values": self.values,
            "source": self.source,
            "recorded_at": self.recorded_at.isoformat(),
        }


@dataclass(frozen=True)
class PosteriorState:
    """A persisted posterior result suitable for inspection or export."""

    variable_name: str
    strategy: str
    distribution: str
    prior: dict[str, Any]
    params: dict[str, Any]
    metadata: dict[str, Any]
    updated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "distribution": self.distribution,
            "prior": self.prior,
            "params": self.params,
            "metadata": self.metadata,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True)
class VariableState:
    """Inspectable prior, evidence history, and latest posterior for a variable."""

    name: str
    prior: dict[str, Any]
    observation_batches: list[ObservationBatch]
    posterior: PosteriorState | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prior": self.prior,
            "observations": [batch.to_dict() for batch in self.observation_batches],
            "posterior": self.posterior.to_dict() if self.posterior else None,
        }


@dataclass(frozen=True)
class FrameState:
    """A serializable diagnostic snapshot of a Frame's current runtime state."""

    frame_name: str
    definition_hash: str
    variables: dict[str, VariableState]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame_name,
            "definition_hash": self.definition_hash,
            "variables": {name: state.to_dict() for name, state in self.variables.items()},
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Return a JSON diagnostic artifact without writing to disk."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    def format(self) -> str:
        """Return a compact human-readable summary for ``print()``."""
        lines = [f"Frame state: {self.frame_name}", f"Definition: {self.definition_hash}"]
        for name, state in self.variables.items():
            observation_count = sum(len(batch.values) for batch in state.observation_batches)
            lines.append(f"- {name}: prior={state.prior['distribution']}, observations={observation_count}")
            if state.posterior is None:
                lines.append("  posterior: unavailable")
            else:
                params = ", ".join(
                    f"{key}={value}" for key, value in state.posterior.params.items()
                )
                lines.append(
                    f"  posterior: {state.posterior.distribution}({params}) "
                    f"via {state.posterior.strategy}"
                )
        return "\n".join(lines)


class FrameStateStore:
    """
    SQLite repository for append-only observation and posterior state.

    A state namespace is keyed by the Frame name and a hash of its declarative
    definition.  Changing the YAML prior or graph therefore cannot accidentally
    reuse a posterior calculated for a different model.
    """

    def __init__(self, database_path: Path | str = FRAME_REGISTRY_DB_PATH) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.database_path}")
        SQLModel.metadata.create_all(self.engine)

    def append_observations(
        self,
        frame_name: str,
        definition_hash: str,
        variable_name: str,
        values: Any,
        *,
        batch_id: str | None = None,
        source: str = "runtime",
    ) -> ObservationBatch:
        """Persist a batch once, returning the existing batch on an idempotent retry."""
        if not isinstance(variable_name, str) or not variable_name:
            raise ValueError("variable_name must be a non-empty string")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")

        normalized_values = _normalize_values(values)
        payload_hash = _payload_hash(normalized_values)
        resolved_batch_id = batch_id or str(uuid4())
        if not isinstance(resolved_batch_id, str) or not resolved_batch_id.strip():
            raise ValueError("batch_id must be a non-empty string when provided")

        with Session(self.engine) as session:
            frame_state_id = self._ensure_frame_state(session, frame_name, definition_hash)
            existing = session.get(GraphObservationBatchRecord, resolved_batch_id)
            if existing is not None:
                if (
                    existing.frame_state_id != frame_state_id
                    or existing.variable_name != variable_name
                    or existing.payload_hash != payload_hash
                ):
                    raise ValueError(
                        f"observation batch '{resolved_batch_id}' already exists with different content"
                    )
                return self._materialize_batch(session, existing)

            record = GraphObservationBatchRecord(
                id=resolved_batch_id,
                frame_state_id=frame_state_id,
                variable_name=variable_name,
                source=source,
                value_count=len(normalized_values),
                payload_hash=payload_hash,
            )
            session.add(record)
            for ordinal, value in enumerate(normalized_values):
                session.add(
                    GraphVariableObservationValueRecord(
                        batch_id=resolved_batch_id,
                        ordinal=ordinal,
                        value_json=json.dumps(value, sort_keys=True, allow_nan=False),
                    )
                )
            session.commit()
            session.refresh(record)
            return self._materialize_batch(session, record)

    def get_observation_batches(
        self,
        frame_name: str,
        definition_hash: str,
        *,
        variable_name: str | None = None,
    ) -> list[ObservationBatch]:
        """Return evidence ordered by submission time and batch ID."""
        with Session(self.engine) as session:
            frame_state_id = self._ensure_frame_state(session, frame_name, definition_hash)
            statement = select(GraphObservationBatchRecord).where(
                GraphObservationBatchRecord.frame_state_id == frame_state_id
            )
            if variable_name is not None:
                statement = statement.where(
                    GraphObservationBatchRecord.variable_name == variable_name
                )
            records = session.exec(
                statement.order_by(
                    col(GraphObservationBatchRecord.recorded_at),
                    col(GraphObservationBatchRecord.id),
                )
            ).all()
            return [self._materialize_batch(session, record) for record in records]

    def save_posterior(
        self,
        frame_name: str,
        definition_hash: str,
        *,
        variable_name: str,
        strategy: str,
        distribution: str,
        prior: Mapping[str, Any],
        params: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> PosteriorState:
        """Upsert a posterior snapshot for a variable and inference strategy."""
        with Session(self.engine) as session:
            frame_state_id = self._ensure_frame_state(session, frame_name, definition_hash)
            record = session.exec(
                select(GraphPosteriorRecord).where(
                    GraphPosteriorRecord.frame_state_id == frame_state_id,
                    GraphPosteriorRecord.variable_name == variable_name,
                    GraphPosteriorRecord.strategy == strategy,
                )
            ).first()
            serialized_prior = _serialize_mapping(prior)
            serialized_params = _serialize_mapping(params)
            serialized_metadata = _serialize_mapping(metadata)
            if record is None:
                record = GraphPosteriorRecord(
                    id=str(uuid4()),
                    frame_state_id=frame_state_id,
                    variable_name=variable_name,
                    strategy=strategy,
                    distribution=distribution,
                    prior_json=serialized_prior,
                    posterior_json=serialized_params,
                    metadata_json=serialized_metadata,
                )
            else:
                record.distribution = distribution
                record.prior_json = serialized_prior
                record.posterior_json = serialized_params
                record.metadata_json = serialized_metadata
                record.updated_at = _utcnow()
            session.add(record)
            session.commit()
            session.refresh(record)
            return _materialize_posterior(record)

    def get_posterior(
        self,
        frame_name: str,
        definition_hash: str,
        *,
        variable_name: str,
        strategy: str | None = None,
    ) -> PosteriorState | None:
        """Return one saved posterior, optionally for a specific strategy.

        ``strategy=None`` returns the most recently updated snapshot. The bundled
        exact updater is named ``"beta-bernoulli-exact"``. Other values are valid
        when they match a custom :class:`InferenceStrategy`'s stable ``name``.
        """
        with Session(self.engine) as session:
            frame_state_id = self._ensure_frame_state(session, frame_name, definition_hash)
            statement = select(GraphPosteriorRecord).where(
                GraphPosteriorRecord.frame_state_id == frame_state_id,
                GraphPosteriorRecord.variable_name == variable_name,
            )
            if strategy is not None:
                statement = statement.where(GraphPosteriorRecord.strategy == strategy)
            record = session.exec(
                statement.order_by(desc(col(GraphPosteriorRecord.updated_at)))
            ).first()
            return _materialize_posterior(record) if record else None

    def get_posteriors(
        self, frame_name: str, definition_hash: str
    ) -> dict[str, PosteriorState]:
        """Return the most recent posterior for every variable in a Frame."""
        with Session(self.engine) as session:
            frame_state_id = self._ensure_frame_state(session, frame_name, definition_hash)
            records = session.exec(
                select(GraphPosteriorRecord)
                .where(GraphPosteriorRecord.frame_state_id == frame_state_id)
                .order_by(desc(col(GraphPosteriorRecord.updated_at)))
            ).all()
            posteriors: dict[str, PosteriorState] = {}
            for record in records:
                posteriors.setdefault(record.variable_name, _materialize_posterior(record))
            return posteriors

    def _ensure_frame_state(
        self, session: Session, frame_name: str, definition_hash: str
    ) -> int:
        record = session.exec(
            select(GraphFrameStateRecord).where(
                GraphFrameStateRecord.frame_name == frame_name,
                GraphFrameStateRecord.definition_hash == definition_hash,
            )
        ).first()
        if record is None:
            # Create
            record = GraphFrameStateRecord(
                frame_name=frame_name,
                definition_hash=definition_hash,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
        assert record.id is not None
        return record.id

    @staticmethod
    def _materialize_batch(session: Session, record: GraphObservationBatchRecord) -> ObservationBatch:
        """Hydrate a public batch, preserving ordinal order of its JSON values."""
        values = session.exec(
            select(GraphVariableObservationValueRecord)
            .where(GraphVariableObservationValueRecord.batch_id == record.id)
            .order_by(col(GraphVariableObservationValueRecord.ordinal))
        ).all()
        return ObservationBatch(
            id=record.id,
            variable_name=record.variable_name,
            values=[json.loads(value.value_json) for value in values],
            source=record.source,
            recorded_at=record.recorded_at,
        )


def _materialize_posterior(record: GraphPosteriorRecord) -> PosteriorState:
    """Deserialize a stored posterior ORM record into its public state value."""
    return PosteriorState(
        variable_name=record.variable_name,
        strategy=record.strategy,
        distribution=record.distribution,
        prior=json.loads(record.prior_json),
        params=json.loads(record.posterior_json),
        metadata=json.loads(record.metadata_json),
        updated_at=record.updated_at,
    )


def _normalize_values(values: Any) -> list[Any]:
    normalized = _json_safe(values)
    return normalized if isinstance(normalized, list) else [normalized]


def _json_safe(value: Any) -> Any:
    """Convert common tensor/array values to strict JSON-compatible data."""
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError(f"observation value is not JSON serializable: {value!r}") from error
    return value


def _payload_hash(values: list[Any]) -> str:
    return sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _serialize_mapping(value: Mapping[str, Any]) -> str:
    normalized = _json_safe(dict(value))
    return json.dumps(normalized, sort_keys=True, allow_nan=False)
