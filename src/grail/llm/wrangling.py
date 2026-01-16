"""
Tools and objects for information wrangling and densification

Below, the SummaryInit class description and its field descriptions are actually used
to inform langauge models.

See: https://python.useinstructor.com/tutorials/6-chain-of-density/#data-classes
for more info on the Chain-of-Density used here.
"""
import os
import time
from typing import TYPE_CHECKING

import instructor
import nltk
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from grail.llm.utils import get_retry_control
from grail.logger import logger

if TYPE_CHECKING:
    from grail.llm.models.base import LLMBaseClass

SUMMARY_TARGET_LEN = 256
MIN_SUMMARY_TOKENS = 64
MAX_SUMMARY_TOKENS = 1024

forbidden_terms = [
    "JSON",
    "json",
    "I am",
    "I will",
    "Entities",
    "RewrittenSummary",
    "SummaryInit",
    "My",
    " my",
    "The task",
    "the task",
    f"{SUMMARY_TARGET_LEN} words",
    f"{SUMMARY_TARGET_LEN} tokens",
    # Special wrappers
    "TEXT START",
    "TEXT END",
]
# Replace each entry with two entries that either have a space, period, or bracket appended to them
forbidden_terms = [term + " " for term in forbidden_terms] + [term + "." for term in forbidden_terms] + [term + ")" for
    term in forbidden_terms]

"""
TODO -- Use the llm_validator to ensure the model isn't doing the annoying meta self-referential stuff
https://python.useinstructor.com/concepts/reask_validation/#llm-based-validation-example
"""


class SummaryInit(BaseModel):
    """
    This is an initial summary about a source document which should be long (several sentences)
    and try to cover a breadth of the content within.
    """

    summary: str = Field(
        ...,
        description="This is an initial summary about a source document which should be long "
                    "(several sentences) and try to cover a breadth of the content within.",
    )

    @field_validator("summary")
    @classmethod
    def check_size(cls, v: str):
        tokens = nltk.word_tokenize(v)
        num_tokens = len(tokens)
        if not MIN_SUMMARY_TOKENS <= num_tokens <= MAX_SUMMARY_TOKENS:
            raise ValueError(
                f"Summary size is outside required parameters: [{MIN_SUMMARY_TOKENS}, {MAX_SUMMARY_TOKENS}]. "
                f"Received {num_tokens}."
            )
        return v

    @field_validator("summary")
    @classmethod
    def proper_focus(cls, v: str):
        """
        Sometimes the LLM gets very self-referential and meta about the task, and goes on about
        how we are "creating summaries with intricate Entities and JSON" and so on.
        """
        if any([term in v for term in forbidden_terms]):
            raise ValueError(
                f"Summary is too self-referential and likely veering off focus: {v[:80]}..."
            )
        return v


class RewrittenSummary(BaseModel):
    """
    This is a new, denser summary of identical length which covers every entity and detail 
    from the previous summary plus new Desired Entities.
    An Entity is a real-world object that's assigned a name. For example, a person, country, a product, etc.
    Guidelines
        - Make every word count: Rewrite the previous summary to improve flow and make space for additional entities 
        - Never drop entities from the previous summary. If space cannot be made, add fewer new entities. 
        - The new summary should be highly dense and concise yet self-contained, 
            eg., easily understood without the Document.
        - Make space with fusion, compression, and removal of uninformative phrases like "the document discusses"
        - Desired entities can appear anywhere in the new summary
    """
    summary: str = Field(
        ...,
        description="This is a new, denser summary which covers every entity and detail from the previous "
                    f"summary plus the new Desired Entities. It should be between {MIN_SUMMARY_TOKENS}-{MAX_SUMMARY_TOKENS} "
                    f"words (tokens), and should be easily understood without the source Document",
    )
    absent: list[str] = Field(
        ...,
        # default_factory=list,
        description="This is a list of Entities absent from the new summary, but present in the previous summary. "
                    "This should be minimized to prevent information loss."
    )
    desired_entities: list[str] = Field(
        # default_factory=list,
        description="This is a list of some Entities from the main Document that could not be fit into "
                    "the new summary due to space, flow, or because sufficient new entities have already been found. "
                    "These will help guide the next iteration of summarizing. Remove entities from this "
                    "list if they appear in the new summary.",
    )

    @field_validator("summary")
    @classmethod
    def length(cls, v: str):
        tokens = nltk.word_tokenize(v)
        num_tokens = len(tokens)
        if not MIN_SUMMARY_TOKENS <= num_tokens <= MAX_SUMMARY_TOKENS:
            raise ValueError(
                f"Summary size is outside required parameters: [{MIN_SUMMARY_TOKENS}, {MAX_SUMMARY_TOKENS}]. "
                f"Received {num_tokens}."
            )
        return v

    # @field_validator("desired_entities")
    # def has_missing_entities(cls, missing_entities: list[str]):
    #     if len(missing_entities) == 0:
    #         raise ValueError(
    #             "You must identify 1-3 informative Entities from the Document which are missing from "
    #             "the previously generated summary to be used in a new summary"
    #         )
    #     return missing_entities
    #
    # @field_validator("absent")
    # def has_no_absent_entities(cls, absent_entities: list[str]):
    #     absent_entity_string = ",".join(absent_entities)
    #     if len(absent_entities) > 0:
    #         print(f"Detected absent entities of {absent_entity_string}")
    #         raise ValueError(
    #             f"Do not omit the following Entities {absent_entity_string} from the new summary"
    #         )
    #     return absent_entities

    @field_validator("summary")
    @classmethod
    def proper_focus(cls, v: str):
        """
        Sometimes the LLM gets very self-referential and meta about the task, and goes on about
        how we are "creating summaries with intricate Entities and JSON" and so on.
        """
        if any([term in v for term in forbidden_terms]):
            raise ValueError(
                f"Summary is too self-referential and likely veering off focus: {v[:80]}..."
            )
        return v


def summarize_document(
    llm: "LLMBaseClass", document: str, summary_steps: int = 3, max_retries: int = 3, budget: int | None = 120,
) -> list[str]:
    """
    Generate a chain of summaries for a given document using the provided language model.

    :param llm:             The language model to use for generating summaries
    :param document:        The document to summarize
    :param summary_steps:   The number of summary steps to generate
    :param max_retries:     The maximum number of retries to attempt for each completion
    :param budget:          The budget to use for each completion
    :return:                A list of summaries generated for the document
    """

    client = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        ),
        mode=instructor.Mode.JSON,
    )
    # client = instructor.patch(
    #     OpenAI(
    #         base_url="http://localhost:11434/v1",
    #         api_key="ollama",  # required, but unused
    #     ),
    #     mode=instructor.Mode.JSON,
    # )

    # Give it half the budget
    summary_retries = get_retry_control(max_retries=1, budget=budget // 2)
    s1 = time.time()
    summary_chain = []  # We first generate an initial summary
    if "gemini" in llm.code.lower():
        additional_kwargs = {}
    else:
        additional_kwargs = {
            "model": llm.code,
        }

    summary_init: SummaryInit = client.chat.completions.create(
        **additional_kwargs,
        response_model=SummaryInit,
        max_retries=summary_retries,
        messages=[
            {
                "role": "developer",
                "content": "You are a financial analyst, trying to summarize some text."
            },
            {
                "role": "developer",
                "content": f"Write an initial summary about the document provided. It must be between "
                           f"between {MIN_SUMMARY_TOKENS}-{MAX_SUMMARY_TOKENS} words (tokens), and should try "
                           "to capture the breadth of the content within.",
            },
            # Jesus
            {
                "role": "developer", "content": "Remember, your summary is about the document content. "
                                                "Avoid self-referential language about yourself and this task, etc."
            },
            {
                "role": "user",
                "content": f"Please summarize the following document: \n<TEXT START>\n{document}\n<TEXT END>"
                           f"Please summarize the above document in {MIN_SUMMARY_TOKENS}"
                           f"-{MAX_SUMMARY_TOKENS} words (tokens), and return your result in JSON "
                           f"format with the key 'summary'."
            },
        ],
    )

    summary_chain.append(summary_init.summary)
    logger.debug(f"\n<><><  INIT SUMMARY  ><><>\n{summary_chain[-1]}\n")

    prev_summary = summary_chain[-1]
    # desired_entity_messages = ([])
    desired_entity_messages = [
        {
            "role": "user",
            "content": "Desired Entities: None as yet."
        },
    ]

    for _i in range(1, summary_steps + 1):
        remaining_budget = budget - (time.time() - s1)
        if remaining_budget < 1:
            logger.debug(f"Summary Budget exhausted at step {_i}")
            return summary_chain

        retries = get_retry_control(max_retries=max_retries, budget=remaining_budget)

        new_summary: RewrittenSummary = client.chat.completions.create(
            **additional_kwargs,
            max_retries=retries,
            response_model=RewrittenSummary,
            messages=[
                {
                    "role": "developer",
                    "content": "You are a helpful analyst that carefully summarizes content."
                               " Return your responses in JSON format with keys 'summary', 'absent', and 'desired_entities'."
                },
                {
                    "role": "user",
                    "content": f"""
                    Please generate an increasingly concise, entity-dense summary of some text.
                    The subject text will be enclosed in the special wrappers <TEXT START> and <TEXT END>.
                    You will also be provided a previous summary of the text for use.
                    To create the current summary you will perform the following tasks:
                        - Identify 1-3 informative entities from the subject text which are missing from 
                        the previous summary, but desirable to include.
                        - Write a new denser summary between {MIN_SUMMARY_TOKENS}-{MAX_SUMMARY_TOKENS} words (tokens) which 
                        covers every entity and detail from the previous summary plus the Desired Entities Guidelines. 
                        - Make every word count: re-write the previous summary to improve flow and make space for 
                        additional entities. 
                        - Make space with fusion, compression, and removal of uninformative phrases 
                        like "the document discusses". 
                        - The summaries should become highly dense and concise yet self-contained, 
                        e.g., easily understood on their own. 
                        - Desired entities can appear anywhere in the new summary. 
                        - Never drop entities from the previous summary. If space cannot be made, add fewer new entities. 
                    """,
                },
                # Jesus
                # {
                #     "role": "developer", "content": "Remember, your summary is about the document content, "
                #                                  "avoid self-referential language about yourself and this task, etc."
                # },
                {
                    "role": "user", "content": f"Here is the main subject text: \n<TEXT START>\n{document}\n<TEXT END>"
                },
                {
                    "role": "user", "content": f"Here is the previous summary of the subject text: {prev_summary}",
                },
                desired_entity_messages[-1],
                {
                    "role": "user", "content": "Please format your response in JSON with keys 'summary', 'absent', "
                                               "and 'desired_entities'."
                }
            ],
        )
        summary_chain.append(new_summary.summary)
        logger.debug(f"\n<><><  SUMMARY {_i}  ><><>\n{summary_chain[-1]}\n")
        logger.debug(f"Desired Entities for Summary {_i + 1}: {','.join(new_summary.desired_entities)}\n")
        # For next iter
        desired_entity_messages.append(
            {
                "role": "user",
                "content": f"Please include these Desired Entities in your summary: "
                           f"{','.join(new_summary.desired_entities)}",
            },
        )
        prev_summary = new_summary

    return summary_chain


def read_reports(
    report_dir: str,
    include: list | None = None,
    exclude: list | None = None,
):
    if not os.path.exists(report_dir):
        raise FileNotFoundError(f"Report directory '{report_dir}' not found")

    # Read the documents for the asset, with recursion if no reports list included
    reports = ""

    for filename in os.listdir(report_dir):
        if exclude is not None and filename in exclude:
            continue
        if include is not None and filename not in include:
            continue

        file_path = os.path.join(report_dir, filename)
        logger.debug(f"Reading file {file_path}")
        with open(file_path, 'r') as file:
            reports += "\n<START REPORT>\n" + file.read() + "\n<END REPORT>\n"

    return reports
