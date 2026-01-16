"""
LLM Utils. In particular, those for retrieving and constructing prompts
"""
import time

from tenacity import RetryCallState, Retrying, stop_after_attempt, stop_after_delay, wait_none, wait_random

from grail.logger import logger


def retry_error_callback(state: RetryCallState) -> None:
    exception = state.outcome.exception()
    result = f"{exception.__class__.__name__} {exception})"
    start = state.start_time
    elapsed = float(round(time.monotonic() - start, 2))
    logger.debug(
        f"Retry ({state.attempt_number}) Error. Elapsed time {elapsed}. Reason: {result}"
    )


def before_retry_callback(state: RetryCallState) -> None:
    if state.attempt_number > 1:
        start = state.start_time
        elapsed = float(round(time.monotonic() - start, 2))
        slept = float(round(state.idle_for, 2))
        logger.debug(
            f"Retry ({state.attempt_number}) Start. Slept {slept}. Elapsed time {elapsed}..."
        )


def get_retry_control(
    max_retries: int | None = None, budget: float | None = None, wait_range: tuple[float, float] | None = None
) -> Retrying:
    # Wait time between requests
    wait = wait_random(wait_range[0], wait_range[1]) if wait_range else wait_none()

    if max_retries and budget:
        stop = (stop_after_attempt(max_retries) | stop_after_delay(budget))
    elif max_retries:
        stop = stop_after_attempt(max_retries)
    elif budget:
        stop = stop_after_delay(budget)
    else:
        logger.warning("No retry control specified. Defaulting to 3 attempts.")
        stop = stop_after_attempt(3)

    return Retrying(
        stop=stop,
        wait=wait,
        before=before_retry_callback,
        # TODO -- Seems to cause an error
        # retry_error_callback=retry_error_callback,
    )
