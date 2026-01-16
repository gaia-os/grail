import instructor
from openai import OpenAI

from grail.llm.models.base import LLMBaseClass


class DeepseekSmall(LLMBaseClass):
    """
    Small Deepseek model.
    """
    name = "Deepseek R1 14B"
    code = "deepseek-r1:14b"
    path_code = "deepseek-r1_14b"
    aliases = [
        "deepseek-small",
        "deepseek-r1-14b",
        "deepseek-r1 14b",
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


class DeepseekMedium(DeepseekSmall):
    """
    Medium Deepseek model.
    """
    name = "Deepseek R1 70B"
    code = "deepseek-r1:70b"
    path_code = "deepseek-r1_70b"
    aliases = [
        "deepseek-medium",
        "deepseek-r1-70b",
        "deepseek-r1 70b",
    ]


class DeepseekLarge(DeepseekSmall):
    """
    Large Deepseek model.
    """
    name = "Deepseek R1 671B"
    code = "deepseek-r1:671b"
    path_code = "deepseek-r1_671b"
    aliases = [
        "deepseek-large",
        "deepseek-r1-671b",
        "deepseek-r1 671b",
    ]


class Deepseek7b(DeepseekSmall):
    """
    Mini Deepseek model.
    """
    name = "Deepseek R1 7B"
    code = "deepseek-r1:7b"
    path_code = "deepseek-r1_7b"
    aliases = [
        "deepseek-r1-7b",
        "deepseek-r1 7b",
    ]


class DeepseekMini(DeepseekSmall):
    """
    Mini Deepseek model.
    """
    name = "Deepseek R1 1.5B"
    code = "deepseek-r1:1.5b"
    path_code = "deepseek-r1_1.5b"
    aliases = [
        "deepseek-mini",
        "deepseek-r1-1.5b",
        "deepseek-r1 1.5b",
    ]
