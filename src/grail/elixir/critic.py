"""
Elixir Critic module
"""
from typing import TypedDict

from instructor.core import InstructorRetryException
from pydantic import BaseModel, Field

from grail.elixir.validator import ElixirException
from grail.llm.models.base import LLMBaseClass
from grail.llm.utils import get_retry_control
from grail.logger import logger


class CriticResponse(BaseModel):
    """
    Critic response model
    """
    evaluation: str = Field(
        ..., title="Evaluation",
        description="The Critic's evaluation of the proposed function solution to the Task."
    )
    approved: bool = Field(
        title="Approved",
        description="Approval boolean of the proposed function solution."
    )


class CriticEvalItem(TypedDict):
    code: str
    evaluation: str
    approved: bool
    # From elixir attempt
    validated: bool
    validation_error: str | None


class DirectiveResponse(BaseModel):
    directive: str


class ElixirCritic:
    """
    Elixir Critic

    The critic analyzes proposed code responses from the Elixir model, and provides tailored
    guidance from its own (potentially superior) LLM.
    Both the Critic and the Agent are given the objective/task.
    """

    def __init__(self, llm: LLMBaseClass, task: str):
        self.llm: LLMBaseClass = llm
        self.task: str = task
        # Store a history of the agent solutions and the critic responses
        # Keys of dict items will be "code", "evaluation", "approved"
        self.history: list[CriticEvalItem] = []

    def reset(self, task: str):
        """
        Reset the critic with a new task.

        :param task:    The new task to set.
        """
        if not task:
            raise ValueError("Task cannot be empty to reset ElixirCritic.")

        self.task = task
        self.history = []

    def initial_directive_prompt(self) -> str:
        """
        Generate an initial directive that will help the actor generate the Python code solution

        :return:    Directive.
        """
        client = self.llm.get_client()
        if "gemini" in self.llm.code.lower():
            additional_kwargs = {}
        else:
            additional_kwargs = {
                "model": self.llm.code,
                "temperature": self.llm.temperature,
                "seed": self.llm.seed,
            }

        prompt = f"""
        Before we begin evaluating code solutions to the task from the actor model, 
        please generate a prompt that will help the LLM actor model in thinking about its solution to the provided task.
        The task is given now, enclosed in html tags:
        
        <TASK>
        {self.task}
        </TASK>

        NOTE: The point here is not to create pseudocode for it to fill in the blanks, but more high-level 
        considerations specifically about the task at hand. You should reference specific elements of the task. 
        
        Try to keep the directive length at most a page of text.
        Please return your response in JSON format with the single key 'directive'.
        """

        # Custom system prompt for the directive
        system_prompt = """
        You are a masterful Python developer and excel at writing precise, efficient code for any task.
        You have an exceptional ability to critique code, identifying potential issues and suggesting improvements.
        
        Your role will be as a Critic, evaluating code solutions from an LLM Actor model regarding the provided Task.
        
        """

        try:
            response, completion = client.chat.completions.create_with_completion(
                **additional_kwargs,
                messages=[
                    {
                        "role": "developer",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                max_retries=3,
                response_model=DirectiveResponse,
            )
        except InstructorRetryException as e:
            raise ElixirException("Failed to construct directive", e)

        return response.model_dump()["directive"]

    def evaluate_attempt(
        self,
        code: str,
        validated: bool,
        validation_error: str | None,
        max_retries: int = 3,
        budget: float = 30,  # seconds
    ) -> CriticEvalItem:
        """
        Evaluate the proposed function code and return a text response, and an approval boolean.

        :param code:            The proposed function code.
        :param validated:       Boolean indicating if the code passed validation.
        :param validation_error: Error message if the code was not validated.
        :param max_retries:     Maximum number of retries to attempt.
        :param budget:          Budget to perform the evaluation within.
        :return:                Text response and approval boolean.
        """
        client = self.llm.get_client()
        if "gemini" in self.llm.code.lower():
            additional_kwargs = {}
        else:
            additional_kwargs = {
                "model": self.llm.code,
                "temperature": self.llm.temperature,
                "seed": self.llm.seed,
            }

        retry_ctrl = get_retry_control(max_retries, budget, wait_range=(0.1, 0.3))
        prompt = self._get_eval_prompt(code, validated, validation_error)
        logger.debug(f"\nCRITIC EVAL PROMPT:\n{prompt}")

        try:
            response, completion = client.chat.completions.create_with_completion(
                **additional_kwargs,
                messages=[
                    {
                        "role": "developer",  # OpenAI docs say this over 'system' for newer models. Hopefully is
                        # back-compatible
                        "content": self.llm.system_roles["elixir_critic"],
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                max_retries=retry_ctrl,
                response_model=CriticResponse,
            )
        except InstructorRetryException as e:
            raise ElixirException("Critic failed to evaluate attempt", e)

        result = response.model_dump()
        # Update the history
        logger.debug(f"Critic Eval result:\nAPPROVED: {result['approved']}\nEVALUATION: {result['evaluation']}")
        self.history.append(
            {
                "code": code,
                "evaluation": result["evaluation"],
                "approved": bool(result["approved"]),
                "validated": validated,
                "validation_error": validation_error,
            }
        )
        # Return the evaluation
        return self.history[-1]

    def _get_eval_prompt(self, code: str, validated: bool, validation_error: str | None) -> str:
        """
        Get the prompt for the critic.

        :param code:                The code to be evaluated.
        :param validated:           Boolean indicating if the code passed validation.
        :param validation_error:    Error message if the code was not validated.
        :return:                    The prompt string.
        """
        prompt: str = f"""
        Given the task described, please evaluate the proposed code.
        Respond with feedback on correctness, issues, and improvements.
        The task is given now, enclosed in <TASK> html tags.
        
        <TASK>
        {self.task}
        </TASK>
        
        """

        if self.history:
            latest = self.history[-1]
            prompt += f"""
            ---
            There exists an entry from your last evaluation for you to consider.
            The details will be provided now, with the earlier code attempt within <PREV_SOLN> tags, 
            and your previous evaluation in <PREV_EVAL> tags.

            <PREV_SOLN>            
            {latest['code']}
            </PREV_SOLN>
            
            <PREV_EVAL>
            {latest['evaluation']}
            </PREV_EVAL>
            ---
            """

        if not validated or validation_error:
            prompt += f"""
            ---
            NOTE: THE PREVIOUS SOLUTION FAILED VALIDATION WITH THE FOLLOWING ERROR:
            
            <VALIDATION_ERROR>
            {validation_error}
            </VALIDATION_ERROR>
            
            Your evaluation should take this into account when reviewing the proposed solution.
            Note that validation failure is not always from logic errors, but could be formatting issues.
            ---
            """

        prompt += f"""
        The current proposed solution is given now, enclosed in <PROPOSED_SOLN> html tags.

        <PROPOSED_SOLN>
        {code}
        </PROPOSED_SOLN>

        Please provide your evaluation.
        Format your response in JSON format with the following keys:
        - 'evaluation' (str): Your evaluation of the code.
        - 'approved' (bool): Boolean approval of the code.
        """

        return prompt
