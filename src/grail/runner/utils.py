"""Utility helpers for querying persisted run records."""

import json
from typing import TYPE_CHECKING

from grail.runner.record import RunRecord
from grail.settings import RUNS_DIR

if TYPE_CHECKING:
    from grail.frame import Frame


def list_all_runs() -> list[RunRecord]:
    """List every persisted run record on disk, across all Frames.

    Unlike :func:`list_runs`, this does not require a compiled `Frame` and
    reads directly from each run directory's `metadata.json` artifact.
    """
    records = []
    for metadata_path in sorted(RUNS_DIR.glob("*/metadata.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        records.append(RunRecord.from_dict(payload))
    return records


def list_runs(
    frame: "Frame", *, strategy_id: str | None = None, status: str | None = None
) -> list[RunRecord]:
    """List persisted runs for one Frame definition hash."""
    records = frame._require_state_store().list_run_records(
        frame.name,
        frame.definition_hash,
        strategy_id=strategy_id,
        status=status,
    )
    return [RunRecord.from_dict(record) for record in records]


def get_run(frame: "Frame", run_id: str) -> RunRecord | None:
    """Return one persisted run when it belongs to this Frame definition hash."""
    record = frame._require_state_store().get_run_record(
        frame.name,
        frame.definition_hash,
        run_id,
    )
    if record is None:
        return None
    return RunRecord.from_dict(record)