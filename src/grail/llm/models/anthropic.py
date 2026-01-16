import json
import os

import instructor
from grail.settings import PROJECT_ROOT

from anthropic import Anthropic
from grail.llm.models.base import LLMBaseClass


def get_api_key() -> str:
    """
    Retrieves the Anthropic API key from a .secrets.json file located in the project root directory.
    The function constructs the path to the .secrets.json file, reads its contents, and extracts the value
    associated with the "ANTHROPIC_API_KEY" key.
    Returns:
        str: The Anthropic API key if found, otherwise None.
    """
    secrets_path = os.path.join(PROJECT_ROOT, ".secrets.json")
    with open(secrets_path, "r") as f:
        secrets = json.load(f)
    api_key = secrets.get("ANTHROPIC_API_KEY")
    return api_key


class AnthropicModel(LLMBaseClass):
    """
    Anthropic class for interacting with the Claude-3-5-Sonnet model.
    """
    name = "claude-3-5-sonnet"
    code = "claude-3-5-sonnet-20240620"
    path_code = "claude3_5_sonnet_20240620"
    api_key = get_api_key()
    aliases = [
        "claude35",
        "claude3",
    ]

    def __init__(self, temperature: float = 0.0, seed: int = 101010):
        super().__init__(
            temperature=temperature,
            seed=seed
        )

    def init_client(self):
        """Init the client and cache it"""
        self._client = instructor.from_anthropic(Anthropic(api_key=self.api_key))

    def process_asset(self, asset):
        pass
