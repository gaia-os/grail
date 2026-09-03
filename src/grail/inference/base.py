"""Inference strategy interfaces independent of the model execution engine."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from grail.frame.state import PosteriorState

if TYPE_CHECKING:
    from grail.frame import Frame


class InferenceStrategy(ABC):
    """Calculates and persists posterior state for a compatible Frame."""

    name: str

    @abstractmethod
    def infer(self, frame: "Frame") -> dict[str, PosteriorState]:
        """Update compatible latent variables from unprocessed persistent evidence."""
