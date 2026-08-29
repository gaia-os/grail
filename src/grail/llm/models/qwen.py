import instructor
from openai import OpenAI

from grail.llm.models.base import LLMBaseClass


class Qwen3(LLMBaseClass):
    """
    Qwen3.8 27b
    """
    name = "Qwen3.8-27b"
    code = "qwen3.8:27b"
    path_code = "qwen_3p8_27b"
    aliases = [
        "qwen",
        "qwen3",
    ]

    def __init__(self, temperature: float = 0.0, seed: int = 101010):
        super().__init__(
            temperature=temperature,
            seed=seed
        )

    def process_asset(self, asset):
        pass

    def init_client(self) -> None:
        self._client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )
