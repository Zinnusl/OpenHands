import asyncio
import time
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from litellm.exceptions import RateLimitError
from litellm import completion as litellm_completion

from openhands.core.logger import openhands_logger as logger
from openhands.utils.tenacity_stop import stop_if_should_exit


class RetryMixin:
    """Mixin class for retry logic."""

    def handle_rate_limit(self, exception: RateLimitError, retry_min_wait: float) -> None:
        """
        Handle rate limit by polling until the rate limit is over.
        
        Args:
            exception: The rate limit exception containing reset time information
            retry_min_wait: Minimum wait time between polls in seconds
        """
        # Get the reset time from the exception if available
        reset_time = getattr(exception, 'reset_time', None)
        
        if reset_time:
            wait_time = max(0, reset_time - time.time())
            logger.info(f"Rate limit hit. Waiting for {wait_time:.2f} seconds until reset...")
            time.sleep(wait_time)
        else:
            # If no reset time is available, use exponential backoff with polling
            logger.info(f"Rate limit hit. Polling every {retry_min_wait} seconds...")
            while True:
                try:
                    # Try a minimal request to check if rate limit is over
                    test_msg = [{"role": "user", "content": "test"}]
                    # Get the original request parameters but modify for minimal usage
                    test_kwargs = {
                        k: v for k, v in self._completion.keywords.items() 
                        if k not in ['messages', 'max_tokens']
                    }
                    litellm_completion(
                        messages=test_msg,
                        max_tokens=1,
                        **test_kwargs
                    )
                    logger.info("Rate limit has been lifted. Proceeding with the request.")
                    return
                except RateLimitError:
                    logger.info("Still rate limited, waiting before next check...")
                    time.sleep(retry_min_wait)
                    continue
                except Exception as e:
                    logger.warning(f"Error while polling rate limit status: {e}")
                    logger.info(f"Waiting {retry_min_wait} seconds before next attempt...")
                    time.sleep(retry_min_wait)

    def retry_decorator(self, **kwargs):
        """
        Create a LLM retry decorator with customizable parameters. This is used for 429 errors, and a few other exceptions in LLM classes.
        
        For rate limit errors (429), this implementation will:
        1. Check for a reset_time in the exception and wait until that time if available
        2. If no reset_time is available, actively poll the API with minimal requests until the rate limit is lifted
        3. Continue with the original request once the rate limit is over

        Args:
            **kwargs: Keyword arguments to override default retry behavior.
                      Keys: num_retries, retry_exceptions, retry_min_wait, retry_max_wait, retry_multiplier

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
            if isinstance(exception, RateLimitError):
                self.handle_rate_limit(exception, retry_min_wait)
                return
                
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
            f'{exception}. Attempt #{retry_state.attempt_number} | You can customize retry values in the configuration.',
        )
