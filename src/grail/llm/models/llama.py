import instructor
from openai import OpenAI

from grail.llm.models.base import LLMBaseClass


class LlamaSmall(LLMBaseClass):
    """
    Small Llama model
    """
    name = "Llama 3.1 8B"
    code = "llama3.1:8b"
    path_code = "llama3_1_8b"
    context_window = 128_000
    aliases = [
        "llama-small",
        "llama-8b",
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


class LlamaSmallAlt(LlamaSmall):
    """Another 8B model, that works with older versions of Ollama : / """
    name = "Llama 3 8B"
    code = "llama3:8b"
    path_code = "llama3_8b"
    context_window = 8192
    aliases = []


class LlamaTiny(LlamaSmall):
    name = "Llama 3.2 1B"
    code = "llama3.2:1b"
    path_code = "llama3_2_1b"
    context_window = 128_000
    aliases = [
        "llama-tiny",
        "llama-1b",
    ]


class LlamaLarge(LlamaSmall):
    name = "Llama 3 70B"
    code = "llama3:70b"
    path_code = "llama3_70b"
    context_window = 128_000
    aliases = [
        "llama-70b",
        "llama-large",
    ]
