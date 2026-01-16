import json
import os

import instructor
from grail.settings import PROJECT_ROOT

from grail.llm.models.base import LLMBaseClass


def get_api_key():
    secrets_path = os.path.join(PROJECT_ROOT, ".secrets.json")
    with open(secrets_path, "r") as f:
        secrets = json.load(f)
    api_key = secrets.get("GEMINI_API_KEY")
    return api_key


class GeminiBase(LLMBaseClass):
    """
    GeminiBase is a class that extends LLMBaseClass and provides functionality for interacting with the Gemini
    """
    api_key = get_api_key()

    def __init__(self, temperature: float = 0.0, seed: int = 101010):
        super().__init__(
            temperature=temperature,
            seed=seed
        )
        # # set the api key and configure it for the gemini model
        # genai.configure(api_key=self.api_key)

    def init_client(self) -> None:
        # TODO -- GEMINI seems to be treated differently.
        # TODO -- From these deviations, we are not currently passing in the temp and seed anywhere
        # It also seems that it prefers having "models/" prepended
        self._client = instructor.from_provider(
            f"models/{self.code}",
            mode=instructor.Mode.GEMINI_JSON,
            api_key=self.api_key,
        )


class Gemini2(GeminiBase):
    """
    Gemini2 class inherits from Gemini15 and represents the Gemini 2.0 model.
    Attributes:
        name (str): The name of the model.
        code (str): The code identifier for the model.
        api_key (str): The API key for accessing the model.
        aliases (list): A list of alternative names for the model.
    """

    name = "gemini_2.0"
    code = "gemini-2.0-flash-exp"
    path_code = "gemini-2_0-flash-exp"
    api_key = get_api_key()
    aliases = [
        "gemini2",
        "gemini-2",
    ]


class Gemini25(GeminiBase):
    """
    Gemini2 class inherits from Gemini15 and represents the Gemini 2.0 model.
    Attributes:
        name (str): The name of the model.
        code (str): The code identifier for the model.
        api_key (str): The API key for accessing the model.
        aliases (list): A list of alternative names for the model.
    """

    name = "gemini_2.5"
    code = "models/gemini-2.5-pro-exp-03-25"
    path_code = "gemini-2_5-pro-exp-03-25"
    api_key = get_api_key()
    aliases = [
        "gemini25",
        "gemini-25",
    ]
