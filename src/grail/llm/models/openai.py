import json
import os

import instructor
from openai import OpenAI

from grail.llm.models.base import LLMBaseClass
from grail.settings import PROJECT_ROOT


def get_api_key():
    secrets_path = os.path.join(PROJECT_ROOT, ".secrets.json")
    with open(secrets_path, "r") as f:
        secrets = json.load(f)
    api_key = secrets.get("OPENAI_API_KEY")
    return api_key


class GPT(LLMBaseClass):
    """
    GPT class for interacting with the GPT-4o-mini model.
    """

    name = "gpt-4o-mini"
    code = "gpt-4o-mini"
    path_code = "gpt-4o-mini"
    api_key = get_api_key()
    aliases = [
        "gpt-4o-mini",
        "gpt4omini",
    ]

    def __init__(self, temperature: float = 0.0, seed: int = 101010):
        super().__init__(
            temperature=temperature,
            seed=seed
        )
        # set the api key and configure it for the gemini model

    def get_client(self, cached: bool = True):
        return instructor.from_openai(OpenAI(api_key=self.api_key), mode=instructor.Mode.TOOLS_STRICT)

    def process_asset(self, asset):
        pass


class GPT4o(GPT):
    """
    GPT4o is a subclass of the GPT class representing a specific version of the GPT model.
    Attributes:
        name (str): The name of the model.
        code (str): The code identifier for the model version.
        api_key (str): The API key used for authentication.
        aliases (list): A list of alternative names for the model.
    """

    name = "gpt-4o"
    code = "gpt-4o-2024-08-06"
    path_code = "gpt-4o"
    api_key = get_api_key()
    aliases = [
        "gpt-4o",
        "gpt4o",
    ]
