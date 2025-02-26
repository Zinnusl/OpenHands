"""Module providing retry functionality for LLM API calls with rate limit handling."""
import time
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from litellm.exceptions import RateLimitError, InternalServerError
from litellm import completion as litellm_completion

from openhands.core.logger import openhands_logger as logger
from openhands.utils.tenacity_stop import stop_if_should_exit


class RetryMixin:
    """Mixin class for retry logic."""

    def handle_rate_limit(self, exception: Exception, retry_min_wait: float) -> None:
        """
        Handle rate limit errors by polling until the rate limit is over.
        Works with both RateLimitError and InternalServerError containing rate limit information.

        Args:
            exception: The exception that triggered the rate limit handling
            retry_min_wait: Minimum wait time between polls in seconds
        """
        # Extract reset time if available
        reset_time = None
        is_rate_limit = False

        if isinstance(exception, RateLimitError):
            reset_time = getattr(exception, 'reset_time', None)
            is_rate_limit = True
        elif isinstance(exception, InternalServerError) and '429' in str(exception):
            logger.info('Detected rate limit in InternalServerError')
            is_rate_limit = True

        if not is_rate_limit:
            logger.warning('Unexpected error type in handle_rate_limit: %s', type(exception))
            return

        if reset_time:
            wait_time = max(0, reset_time - time.time())
            logger.info('Rate limit hit. Waiting for %.2f seconds until reset...', wait_time)
            time.sleep(wait_time)
            return

        # If no reset time is available, use polling
        logger.info('Rate limit hit. Polling every %s seconds...', retry_min_wait)
        while True:
            try:
                # Try a minimal request to check if rate limit is over
                test_msg = [{'role': 'user', 'content': 'test'}]
                test_kwargs = {
                    k: v for k, v in self._completion.keywords.items()
                    if k not in ['messages', 'max_tokens']
                }
                litellm_completion(
                    messages=test_msg,
                    max_tokens=1,
                    **test_kwargs
                )
                logger.info('Rate limit has been lifted. Proceeding with the request.')
                return
            except (RateLimitError, InternalServerError) as e:
                if isinstance(e, InternalServerError) and '429' not in str(e):
                    raise
                logger.info('Still rate limited, waiting before next check...')
                time.sleep(retry_min_wait)
            except (ValueError, KeyError, AttributeError) as e:
                # Handle common errors that might occur during the test request
                logger.warning('Error while polling rate limit status: %s', e)
                logger.info('Waiting %s seconds before next attempt...', retry_min_wait)
                time.sleep(retry_min_wait)

    def retry_decorator(self, **kwargs):
        """
        Create a LLM retry decorator with customizable parameters.
        This is used for 429 errors, and a few other exceptions in LLM classes.

        For rate limit errors (429), this implementation will:
        1. Check for a reset_time in the exception and wait until that time if available
        2. If no reset_time is available, actively poll the API with minimal requests
           until the rate limit is lifted
        3. Continue with the original request once the rate limit is over

        Args:
            **kwargs: Keyword arguments to override default retry behavior.
                   Keys: num_retries, retry_exceptions, retry_min_wait, retry_max_wait,
                   retry_multiplier

        Returns:
            A retry decorator with the parameters customizable in configuration.
        """
        num_retries = kwargs.get('num_retries')
        retry_exceptions: tuple = kwargs.get('retry_exceptions', ())
        retry_min_wait = kwargs.get('retry_min_wait')
        retry_max_wait = kwargs.get('retry_max_wait')
        retry_multiplier = kwargs.get('retry_multiplier')
        retry_listener = kwargs.get('retry_listener')

        def before_sleep(retry_state):
            exception = retry_state.outcome.exception()

            # Handle rate limits differently
            if isinstance(exception, (RateLimitError, InternalServerError)):
                # Check if this is a rate limit error
                if isinstance(exception, InternalServerError):
                    if '429' in str(exception) and 'rate limit' in str(exception).lower():
                        self.handle_rate_limit(exception, retry_min_wait)
                        return
                else:  # RateLimitError
                    self.handle_rate_limit(exception, retry_min_wait)
                    return

            # For other exceptions, use normal retry behavior
            self.log_retry_attempt(retry_state)
            if retry_listener:
                retry_listener(retry_state.attempt_number, num_retries)

        return retry(
            before_sleep=before_sleep,
            stop=stop_after_attempt(num_retries) | stop_if_should_exit(),
            reraise=True,
            retry=(
                retry_if_exception_type(retry_exceptions)
            ),  # retry only for these types
            wait=wait_exponential(
                multiplier=retry_multiplier,
                min=retry_min_wait,
                max=retry_max_wait,
            ),
        )

    def log_retry_attempt(self, retry_state):
        """Log retry attempts."""
        exception = retry_state.outcome.exception()
        logger.error(
            '%s. Attempt #%d | You can customize retry values in the configuration.',
            exception, retry_state.attempt_number
        )
