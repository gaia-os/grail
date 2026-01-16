"""
Elixir model
"""
import json
import time
from typing import Type

from instructor.core import InstructorRetryException
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from grail.elixir.critic import CriticEvalItem, ElixirCritic
from grail.elixir.validator import ElixirException, ElixirValidator
from grail.llm.models.base import LLMBaseClass
from grail.llm.utils import get_retry_control
from grail.logger import logger


class Elixir:
    """
    Elixir is the utility within GRAIL that performs the LLM code generation functionality
    Initialization requires an initialized LLMBaseClass object.
    The critic model is initialized with the given task

    TODO -- Implement a "stage" where code development is held
    """

    def __init__(
        self,
        llm: LLMBaseClass,
        critic: ElixirCritic | None = None
    ):
        self.llm: LLMBaseClass = llm
        self.critic: ElixirCritic | None = critic
        self.stage = None
        # Note that validated does not me Critic approved, just that it is "clean"
        self.validated = False

    def clear(self) -> None:
        """Reset the staging and validated status, to prepare for another usage"""
        self.stage = None
        self.validated = False
        return

    def _call_llm(
        self,
        prompt: str,
        validator: Type[ElixirValidator],
        max_retries: int | None = None,
        budget: float | None = None
    ) -> tuple[str, bool, str | None]:
        """
        Generate an elixir function

        Note, both max_retries and budget can be passed in as valid terminating conditions.

        :param prompt:          The prompt to use for the code generation.
        :param validator:       The validator to use against the proposed code string.
        :param max_retries:     The maximum number of retries to attempt.
        :param budget:          Budget to perform the evaluation within.
        :return:                The attempt, and a boolean indicating success, and an error if present.
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
        try:
            response, completion = client.chat.completions.create_with_completion(
                **additional_kwargs,
                messages=[
                    {
                        "role": "developer",  # OpenAI docs say this over 'system' for newer models. Hopefully is
                        # back-compatible
                        "content": self.llm.system_roles["elixir"],
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                response_model=validator,
                max_retries=retry_ctrl,
            )
        except InstructorRetryException as e:
            # Could not manage to produce a response
            last = e.last_completion
            if not last:
                # Failed to get response
                return "", False, "Failed to get LLM response."

            returned_message = last.to_dict()['choices'][-1]['message']
            content = returned_message['content']
            # See if we can get the attempt string
            try:
                py_content = json.loads(content)
                code = py_content['code']
            except Exception:
                # Sometimes the 'content' key is empty. Not sure why this happens.
                # Or 'code' is not a key
                try:
                    # From gpt observations, it looks like the code can be located/parsed here
                    content = returned_message['tool_calls'][0]['function']['arguments']
                    code = json.loads(content)['code']
                except Exception:
                    # Okay, just convert the message object into a string. Somewhere it should contain the function.
                    # information
                    info = json.dumps(returned_message)
                    return info, False, str(e)

            # Return the attempted code
            return str(code), False, str(e)

        return str(response.model_dump()["code"]), True, None

    def _get_function_gen_prompt(self, validator: Type[ElixirValidator]) -> str:
        args = ""
        for arg in validator.required_args:
            args += f"\n\t{arg}"
        if not args:
            args = "\n\tNo arguments"

        returns = ""
        for ret in validator.returned_data:
            returns += f"\n\t{ret}"
        if not returns:
            returns = "\n\tDoes not return data"

        prompt = f"""Please write a python function according to the following spec:
            Function name: {validator.function_name}
            Purpose: {validator.prompt_description}
            Args: {args}
            Returns: {returns}

            Return your response in JSON format as:
            {{
                "code": <str>
            }}
            where the "code" string is the entire function code.
            """
        return prompt

    def generate_function(
        self,
        validator: Type[ElixirValidator],
        max_retries: int | None = None,
        budget: float | None = None,
        prompt_override: str | None = None,
    ) -> str:
        """
        Given a validator, query the model and attempt to generate its code.
        Optionally control LLM retries with max_retries and budget.
        Note that this is for a one-off gen, and not for Critic loops.
        Validated code is returned as string

        :param validator:       ElixirValidator object
        :param max_retries:     Maximum number of retries
        :param budget:          Budget for the LLM
        :param prompt_override: Optional prompt override
        :return:                Code string
        """
        if prompt_override:
            prompt = prompt_override
        else:
            prompt = self._get_function_gen_prompt(validator)

        logger.debug(f"Elixir prompt:\n{prompt}")
        # Submit this to the LLM
        code, success, error = self._call_llm(prompt, validator, max_retries, budget)
        if not success:
            self.validated = False
            self.stage = code
            raise ElixirException(
                f"Failed to generate code for {validator.function_name}.\nError: {error}\n"
                f"Last attempt:\n{code}",
                code=code
            )

        # Successful code string
        self.validated = True
        self.stage = code
        logger.debug(f"Generated validated code for {validator.function_name}:\n{code}")
        return code

    def _get_gen_prompt_with_eval(
        self, validator: Type[ElixirValidator], last_attempt: str, evaluation: str
    ) -> str:
        """
        Generate the prompt for the elixir actor call, including the evaluation result
        """
        prompt = self._get_function_gen_prompt(validator)
        prompt += f"""
        ---
        You have previously attempted a solution to this function. However, a Critic Evaluator disapproved
        of your last submission. Please consider your previous solution, enclosed in <PREV_SOLN> tags, and the
        critic's evaluation, enclosed in <EVALUATION> tags, both of which are provided now.
        
        <PREV_SOLN>
        {last_attempt}
        </PREV_SOLN>
        
        <EVALUATION>
        {evaluation}
        </EVALUATION>
        """
        return prompt

    def critic_loop(
        self,
        validator: Type[ElixirValidator],
        max_iters: int = 10,
        loop_budget: float = 120,  # two minutes
        actor_retries: int | None = None,
        actor_budget: float | None = None,
    ) -> dict:
        """
        Similar to the generate_function method, but incorporates a critic loop.
        The critic's task has already been provided to it.

        :param validator:       ElixirValidator object. Defines the output format for the actor
        :param max_iters:       Maximum number of iterations for the critic loop
        :param loop_budget:     Budget for the critic loop
        :param actor_retries:   Maximum number of retries for the actor
        :param actor_budget:    Budget for the actor
        :return:                Callable function
        """
        if not self.critic:
            raise ElixirException("Critic model not initialized")

        # Get initial code attempt
        prompt = self._get_function_gen_prompt(validator)
        directive = self.critic.initial_directive_prompt()
        logger.debug(
            f"DIRECTIVE: "
            f"\n{directive}"
        )
        prompt += f"""
        ---
        To aid in thinking about the solution, consider the following directive.
        
        <DIRECTIVE>
        {directive}
        </DIRECTIVE>
        """

        # We'll start the clock here
        start = iter_start = actor_start = time.time()
        remaining_budget = loop_budget

        # We circumnavigate the self.generate_function method, because we want to capture and continue with whatever
        # the produce code attempt is.
        attempt, validated, error = self._call_llm(
            prompt, validator, actor_retries, actor_budget,
        )
        actor_delta = time.time() - actor_start
        remaining_budget = loop_budget - (time.time() - start)
        self.stage = attempt
        self.validated = validated

        # Main loop
        self.stage = attempt
        approved = False
        max_critic_eval_retries = 3  # After three, it's probably not returning JSON correct.
        approved_solution = None

        # Gather some data about the loop
        results = []

        with logging_redirect_tqdm(loggers=[logger]):
            for i in tqdm(range(1, max_iters + 1), desc="Evals"):
                # Validation and validation errors are taken into account
                eval_start = time.time()
                eval_item: CriticEvalItem = self.critic.evaluate_attempt(
                    attempt, validated, error, budget=remaining_budget, max_retries=max_critic_eval_retries
                )
                eval_delta = time.time() - eval_start
                approved = eval_item['approved']

                iter_delta = time.time() - iter_start
                iter_start = time.time()
                results.append(
                    [
                        {
                            "code": attempt,
                            "evaluation": eval_item['evaluation'],
                            "approved": eval_item['approved'],
                            "validated": validated,
                            "validation_error": error,
                            "actor_delta": actor_delta,
                            "eval_delta": eval_delta,
                            "iter_delta": iter_delta,
                        }
                    ]
                )

                if approved:
                    if not validated:
                        # Trouble. The LLM approves, but it failed our validation.
                        # TODO -- Maybe record these instances
                        logger.warning(
                            "Elixir function approved by critic, but failed internal validation. "
                            "Rejecting proposed solution."
                        )
                        # Set a custom message for passing to the critic that indicates the validation failed,
                        # and count this as a failure.
                        approved = False
                        self.critic.history[-1]['approved'] = False
                        new_eval = f"""
                        {eval_item['evaluation']}
                        ---
                        NOTE: THIS EVALUATION APPROVED THE PROPOSED SOLUTION. HOWEVER, IT FAILED INTERNAL CODE QUALITY 
                        VALIDATION, AND IS THUS REJECTED AS A SOLUTION. 
                        """
                        eval_item['evaluation'] = new_eval
                        self.critic.history[-1]['evaluation'] = new_eval
                    else:
                        # Great
                        logger.info(f"Elixir function approved by critic after {i} loops")
                        approved_solution = attempt
                        break

                # Check if exhausted
                if i >= max_iters:
                    logger.warning(
                        f"Max iterations ({max_iters}) of critic loop reached without approved function. Latest "
                        f"attempt:\n\n{attempt}"
                    )
                    break

                remaining_budget = loop_budget - (time.time() - start)
                if remaining_budget < 0:
                    logger.warning(
                        f"Critic loop budget exhausted without approved function. Latest attempt:\n\n{attempt}"
                    )
                    break

                # We have enough resources to have another attempt
                prompt = self._get_gen_prompt_with_eval(validator, attempt, eval_item['evaluation'])
                attempt_budget = min(actor_budget, remaining_budget)
                actor_start = time.time()
                attempt, validated, error = self._call_llm(
                    prompt, validator, actor_retries, attempt_budget,
                )
                self.stage = attempt
                self.validated = validated

        return {
            "all_results": results,
            "last_result": results[-1],
            "approved": approved,
            "evaluation": self.critic.history[-1]['evaluation'],
            "code": approved_solution,
            "function_name": validator.function_name,
            "iters_used": i,
            "total_time": time.time() - start,
        }
