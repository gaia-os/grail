"""SQLite-backed index for canonical Frame YAML artifacts."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from sqlmodel import Field, Session, SQLModel, create_engine, select

from grail.logger import logger
from grail.settings import FRAME_REGISTRY_DB_PATH

from grail.frame.spec import FrameSpec


def _utcnow() -> datetime:
    """Produce timezone-aware timestamps for metadata records."""
    return datetime.now(timezone.utc)


class FrameRecord(SQLModel, table=True):
    """An index record for a canonical Frame YAML specification."""

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    spec_path: str = Field(index=True)
    spec_hash: str
    spec_version: int
    description: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class FrameRegistry:
    """
    Small Python-first registry for Frame locations and revisions.

    The registry deliberately stores no model definition: its YAML file remains the
    canonical source of truth. SQLite only supports discovery, change detection, and
    a future link to generated operations and execution artifacts.
    """

    def __init__(self, database_path: Path | str = FRAME_REGISTRY_DB_PATH) -> None:
        path = Path(database_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{path}")
        SQLModel.metadata.create_all(self.engine)
        logger.debug(f"Frame registry initialized at '{path}'")

    def register(self, spec: FrameSpec, spec_path: Path | str) -> FrameRecord:
        """Create or update the registry record for one canonical Frame spec."""
        path = str(Path(spec_path).expanduser().resolve())
        spec_hash = sha256(
            spec.model_dump_json(exclude_none=True).encode("utf-8")
        ).hexdigest()
        with Session(self.engine) as session:
            record = session.exec(
                select(FrameRecord).where(FrameRecord.spec_path == path)
            ).first()
            if record is None:
                logger.info(f"Registering new Frame '{spec.name}' at '{path}'")
                record = FrameRecord(
                    name=spec.name,
                    spec_path=path,
                    spec_hash=spec_hash,
                    spec_version=spec.version,
                    description=spec.metadata.description,
                )
            else:
                if record.spec_hash != spec_hash:
                    logger.info(f"Updating Frame '{spec.name}' due to spec changes")
                else:
                    logger.debug(f"Refreshing Frame registry metadata for '{spec.name}'")
                record.name = spec.name
                record.spec_hash = spec_hash
                record.spec_version = spec.version
                record.description = spec.metadata.description
                record.updated_at = _utcnow()
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def get_by_name(self, name: str) -> FrameRecord | None:
        """Return the most recently updated record for a Frame name, if indexed."""
        with Session(self.engine) as session:
            record = session.exec(
                select(FrameRecord)
                .where(FrameRecord.name == name)
                .order_by(FrameRecord.updated_at.desc())
            ).first()
            if record is None:
                logger.debug(f"Frame '{name}' not found in registry")
            return record

    def list_frames(self) -> list[FrameRecord]:
        """List registered Frames, newest update first."""
        with Session(self.engine) as session:
            records = list(session.exec(select(FrameRecord).order_by(FrameRecord.updated_at.desc())))
            logger.debug(f"Listed {len(records)} Frame records")
            return records
