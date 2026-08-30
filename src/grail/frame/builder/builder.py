import pickle
from pathlib import Path
from typing import Optional

from grail.frame import Frame
from grail.logger import logger


class Builder:
    """
    The Builder constructs and manages Frames.
    It handles initialization, composition, and persistence.
    """

    def __init__(self, workspace_path: Optional[str] = None):
        self.workspace_path = Path(workspace_path) if workspace_path else Path("./grail_workspace")
        self.workspace_path.mkdir(exist_ok=True, parents=True)
        self.current_frame: Optional[Frame] = None
        logger.info(f"Builder: workspace ready at '{self.workspace_path}'")

    def new_frame(self, name: str) -> Frame:
        """Initializes a new Frame."""
        logger.info(f"Builder: creating Frame '{name}'")
        self.current_frame = Frame(name=name)
        return self.current_frame

    def save_frame(self, frame: Frame, filename: Optional[str] = None):
        """Saves a frame to disk."""
        if not filename:
            filename = f"{frame.name}.grail"

        file_path = self.workspace_path / filename
        logger.info(f"Builder: saving Frame '{frame.name}' to '{file_path}'")
        with open(file_path, "wb") as f:
            pickle.dump(frame, f)
        logger.debug(f"Builder: saved Frame '{frame.name}'")

    def load_frame(self, filename: str) -> Frame:
        """Loads a frame from disk."""
        file_path = self.workspace_path / filename
        logger.info(f"Builder: loading Frame from '{file_path}'")
        with open(file_path, "rb") as f:
            frame = pickle.load(f)

        self.current_frame = frame
        logger.debug(f"Builder: loaded Frame '{frame.name}'")
        return frame
