import pickle
from typing import Dict, Optional
from pathlib import Path

from grail.logger import logger
from .frame import Frame


class Builder:
    """
    The Builder constructs and manages Frames.
    It handles initialization, composition, and persistence.
    """
    def __init__(self, workspace_path: Optional[str] = None):
        self.workspace_path = Path(workspace_path) if workspace_path else Path("./grail_workspace")
        self.workspace_path.mkdir(exist_ok=True, parents=True)
        self.current_frame: Optional[Frame] = None
        logger.info(f"Builder initialized with workspace: {self.workspace_path}")

    def new_frame(self, name: str) -> Frame:
        """Initializes a new Frame."""
        logger.info(f"Creating new Frame: {name}")
        self.current_frame = Frame(name=name)
        return self.current_frame

    def save_frame(self, frame: Frame, filename: Optional[str] = None):
        """Saves a frame to disk."""
        if not filename:
            filename = f"{frame.name}.grail"
        
        file_path = self.workspace_path / filename
        logger.info(f"Saving Frame '{frame.name}' to {file_path}")
        with open(file_path, "wb") as f:
            pickle.dump(frame, f)
        print(f"Frame saved to {file_path}")

    def load_frame(self, filename: str) -> Frame:
        """Loads a frame from disk."""
        file_path = self.workspace_path / filename
        logger.info(f"Loading Frame from {file_path}")
        with open(file_path, "rb") as f:
            frame = pickle.load(f)
        
        self.current_frame = frame
        return frame
