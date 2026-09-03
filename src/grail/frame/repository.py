"""YAML persistence for canonical Frame specifications."""

from pathlib import Path

import yaml

from grail.frame import Frame
from grail.frame.spec import FrameSpec
from grail.frame.state import FrameStateStore
from grail.logger import logger
from grail.settings import FRAME_SPECS_DIR

FrameSource = Frame | FrameSpec


class FrameRepository:
    """Read and write canonical Frame YAML files below one trusted root directory."""

    def __init__(
        self,
        root: Path | str = FRAME_SPECS_DIR,
        *,
        state_database_path: Path | str | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        default_state_root = self.root.parent if self.root.name == "frames" else self.root
        self.state_store = FrameStateStore(
            state_database_path or default_state_root / "grail.sqlite3"
        )

    def save(self, source: FrameSource, filename: str | None = None) -> Path:
        """Atomically write a Frame or FrameSpec and return its YAML path."""
        spec = source.to_spec() if isinstance(source, Frame) else source
        if isinstance(source, Frame):
            source.attach_state_store(self.state_store)
        path = self._resolve_path(filename or f"{spec.name}.yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving Frame spec '{spec.name}' to '{path}'")
        payload = yaml.safe_dump(
            spec.model_dump(mode="json", exclude_none=True),
            allow_unicode=True,
            sort_keys=False,
        )
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(payload, encoding="utf-8")
        temporary_path.replace(path)
        logger.info(f"Saved Frame spec '{spec.name}' to '{path}'")
        return path

    def load_spec(self, filename: str | Path) -> FrameSpec:
        """Load, parse, and validate a YAML Frame specification."""
        path = self._resolve_path(filename)
        logger.debug(f"Loading Frame spec from '{path}'")
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            logger.warning(f"Invalid Frame YAML in '{path}': {error}")
            raise ValueError(f"invalid Frame YAML in '{path}'") from error
        if payload is None:
            raise ValueError(f"Frame YAML '{path}' is empty")
        if not isinstance(payload, dict):
            raise TypeError(f"Frame YAML '{path}' was not loaded as dict object")
        spec = FrameSpec.model_validate(payload)
        logger.info(
            f"Loaded Frame spec '{spec.name}' ({len(spec.variables)} variables, "
            f"{len(spec.dependencies)} dependencies)"
        )
        return spec

    def load(self, filename: str | Path) -> Frame:
        """Load a validated YAML specification and compile it into a runtime Frame."""
        logger.info(f"Compiling runtime Frame from '{filename}'")
        frame = Frame.from_spec(self.load_spec(filename))
        frame.attach_state_store(self.state_store)
        return frame

    def path_for(self, name: str) -> Path:
        """Return the canonical path for a valid Frame name."""
        return self._resolve_path(f"{name}.yaml")

    def _resolve_path(self, filename: str | Path) -> Path:
        """Resolve a YAML file beneath the repository root, rejecting traversal."""
        candidate = Path(filename)
        if candidate.suffix not in {".yaml", ".yml"}:
            candidate = candidate.with_suffix(".yaml")
        path = (self.root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            logger.warning(f"Rejected Frame path outside repository root: '{filename}'")
            raise ValueError(f"Frame path must be within '{self.root}'") from error
        return path
