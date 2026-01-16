"""
Consolidate LLM models, and LLM utils etc.
"""
from abc import ABC

import instructor

from grail.llm import roles


def init_llm_prompt_logger(log_path: str):
    """Initialize a logger specifically for LLM prompts"""
    import logging
    # From the Instructor github, it looks like they use 'instructor' as logger name
    logger = logging.getLogger("instructor")
    logger.setLevel(logging.DEBUG)

    # Create a file handler for logging prompts
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Add formatter for better output
    formatter = logging.Formatter('[ {name}:{levelname} ]\t{message}', style='{')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger


class LLMBaseClass(ABC):
    """
    Base class for all LLM models
    """
    name = None
    code = None
    path_code = None  # Certain characters--looking at you colon (wait what?)--are bad for making file paths
    context_window = None
    aliases = []

    system_roles = {
        "elixir": roles.ELIXIR,
        "elixir_critic": roles.ELIXIR_CRITIC,
    }

    def __init__(
        self,
        temperature: float = 0.0,
        seed: int = 101010,
    ):
        if not self.path_code:
            raise ValueError("Undefined path_code for LLM Class")

        self.temperature = temperature
        self.seed = seed
        self._client = None
        # Separate LLM prompt logger
        self.logger = None

    def to_json(self):
        return {
            "name": self.name,
            "code": self.code,
            "path_code": self.path_code,
            "seed": self.seed,
            "context_window": self.context_window,
            "temperature": self.temperature,
        }

    def log_to_file(self, file_path: str):
        """
        Initialize a log file to record LLM-specific events (namely, prompts).
        Instructor automatically logs at the DEBUG level:
            https://python.useinstructor.com/concepts/logging/
        """
        if not self.logger:
            self.logger = init_llm_prompt_logger(file_path)

    def init_client(self) -> None:
        raise NotImplementedError()

    def get_client(self, cached: bool = True) -> instructor.Instructor | instructor.AsyncInstructor:
        """Return the client, initializing if needed. Use cached=False to re-init"""
        if not cached or not self._client:
            self.init_client()
        return self._client

    def __str__(self):
        """Return name and class dot notation"""
        return f"{self.name} ({self.__class__.__module__}.{self.__class__.__name__})"

    def __repr__(self):
        """Return name and class dot notation"""
        return self.__str__()
