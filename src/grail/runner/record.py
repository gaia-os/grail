"""Immutable execution records for durable inference provenance."""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
import json
from typing import Any


class RunStatus(str, Enum):
    """Lifecycle states for a persisted run."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class RunRecord:
    """Immutable metadata for one execution event."""

    id: str
    frame_name: str
    definition_hash: str
    strategy_id: str
    operation_kind: str
    status: RunStatus
    observation_batch_ids: list[str]
    parameters: dict[str, Any]
    versions: dict[str, Any]
    diagnostics: dict[str, Any]
    artifact_paths: dict[str, str]
    error_type: str | None
    error_message: str | None
    error_traceback: str | None
    created_at: datetime
    started_at: datetime
    completed_at: datetime | None
    updated_at: datetime

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunRecord":
        """Build a record from serialized storage data."""

        return cls(
            id=payload["id"],
            frame_name=payload["frame_name"],
            definition_hash=payload["definition_hash"],
            strategy_id=payload["strategy_id"],
            operation_kind=payload["operation_kind"],
            status=RunStatus(payload["status"]),
            observation_batch_ids=list(payload.get("observation_batch_ids", [])),
            parameters=dict(payload.get("parameters", {})),
            versions=dict(payload.get("versions", {})),
            diagnostics=dict(payload.get("diagnostics", {})),
            artifact_paths=dict(payload.get("artifact_paths", {})),
            error_type=payload.get("error_type"),
            error_message=payload.get("error_message"),
            error_traceback=payload.get("error_traceback"),
            created_at=_parse_datetime(payload["created_at"]),
            started_at=_parse_datetime(payload["started_at"]),
            completed_at=_parse_optional_datetime(payload.get("completed_at")),
            updated_at=_parse_datetime(payload["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize into JSON-safe metadata."""

        return {
            "id": self.id,
            "frame_name": self.frame_name,
            "definition_hash": self.definition_hash,
            "strategy_id": self.strategy_id,
            "operation_kind": self.operation_kind,
            "status": self.status.value,
            "observation_batch_ids": self.observation_batch_ids,
            "parameters": self.parameters,
            "versions": self.versions,
            "diagnostics": self.diagnostics,
            "artifact_paths": self.artifact_paths,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "updated_at": self.updated_at.isoformat(),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)


def _parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _parse_optional_datetime(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    return _parse_datetime(value)
