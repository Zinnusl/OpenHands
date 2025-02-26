"""Module providing retry functionality for LLM API calls with rate limit handling.

This module implements a robust retry mechanism for handling API rate limits and other
transient errors in LLM API calls. It provides sophisticated polling strategies and
fallback mechanisms to ensure optimal handling of rate limits while maintaining
efficiency and reliability.
"""
import time
from typing import Optional, Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    RetryCallState,
)
from litellm.exceptions import RateLimitError, InternalServerError
from litellm import completion as litellm_completion

from openhands.core.logger import openhands_logger as logger
from openhands.utils.tenacity_stop import stop_if_should_exit


class RetryMixin:
    """A mixin class that provides retry functionality for LLM API calls.
    
    This mixin implements sophisticated retry logic for handling API rate limits and other
    transient errors. For rate limits, it implements an active polling strategy that:
    1. Checks for explicit reset times in rate limit responses
    2. Falls back to periodic polling if no reset time is provided
    3. Uses exponential backoff as a last resort if polling fails
    
    The mixin is designed to work with litellm's completion function and handles both
    explicit RateLimitError exceptions and 429 status codes wrapped in InternalServerError.
    
    Attributes:
        config: Expected to be provided by the parent class, containing retry configuration
               such as num_retries, retry_min_wait, retry_max_wait, etc.
    """

    def handle_rate_limit(self, exception: Exception, retry_min_wait: float) -> None:
        """Handle rate limit errors by actively polling until the limit is lifted.

        This method implements a sophisticated polling strategy:
        1. First checks for reset_time in response headers (x-ratelimit-reset)
        2. Then checks for reset_time in the exception object
        3. If no reset time is found, implements periodic polling with minimal requests
        4. Uses exponential backoff if polling fails

        The polling mechanism:
        - Uses minimal test requests (1 token) to check rate limit status
        - Respects configuration parameters from parent class
        - Has a configurable maximum polling time (default: 1 hour)
        - Provides detailed logging of all attempts and status

        Args:
            exception: The rate limit exception (RateLimitError or InternalServerError)
            retry_min_wait: Minimum time in seconds to wait between polling attempts

        Raises:
            ValueError: If configuration is missing required fields
            AttributeError: If required attributes are not found
            Exception: If polling exceeds maximum time or other unexpected errors

        Configuration:
            The following can be set in the parent class's config:
            - max_rate_limit_poll_time: Maximum time to poll for (default: 3600s)
            - api_key: API key for test requests
            - model: Model name for test requests
            - base_url/api_base: API endpoint for test requests
        """
        # Check for reset time in headers or error message
        reset_time = None
        if hasattr(exception, 'headers'):
            reset_time = exception.headers.get('x-ratelimit-reset')
        elif hasattr(exception, 'response'):
            resp = getattr(exception, 'response', {})
            if isinstance(resp, dict):
                reset_time = resp.get('headers', {}).get('x-ratelimit-reset')

        # If we have a reset time, wait until then
        if reset_time:
            try:
                reset_time = float(reset_time)
                wait_time = max(0, reset_time - time.time())
                if wait_time > 0:
                    logger.info('Rate limit will reset in %.1f seconds', wait_time)
                    time.sleep(wait_time)
                    return
            except (ValueError, TypeError) as e:
                logger.warning('Failed to parse rate limit reset time: %s', str(e))

        # Also check for reset_time in RateLimitError
        if isinstance(exception, RateLimitError):
            reset_time = getattr(exception, 'reset_time', None)
            if reset_time:
                try:
                    wait_time = max(0, float(reset_time) - time.time())
                    if wait_time > 0:
                        logger.info('Rate limit will reset in %.1f seconds (from exception)', wait_time)
                        time.sleep(wait_time)
                        return
                except (ValueError, TypeError) as e:
                    logger.warning('Failed to parse rate limit reset time from exception: %s', str(e))

        # If we get here, no valid reset time was found - implement polling with timeout
        test_msg = [{"role": "user", "content": "test"}]
        start_time = time.time()
        
        # Get max poll time from config or use default
        max_poll_time = 3600  # Default 1 hour maximum polling time
        if hasattr(self, 'config'):
            max_poll_time = getattr(self.config, 'max_rate_limit_poll_time', max_poll_time)
        
        poll_attempt = 0
        
        while True:
            # Check if we've exceeded maximum polling time
            if time.time() - start_time > max_poll_time:
                logger.error('Rate limit polling exceeded maximum time of %d seconds', max_poll_time)
                raise Exception(f'Rate limit polling timeout after {max_poll_time} seconds')

            poll_attempt += 1
            try:
                # Try to get config parameters for test request
                test_kwargs = {}
                if hasattr(self, 'config'):
                    for key in ['api_key', 'model', 'base_url', 'api_base']:
                        value = getattr(self.config, key, None)
                        if value is not None:
                            test_kwargs[key] = value
                else:
                    logger.warning('No config found for rate limit test request')

                logger.debug('Poll attempt #%d: Checking rate limit status with kwargs: %s', 
                           poll_attempt,
                           {k: v for k, v in test_kwargs.items() if k != 'api_key'})
                
                # Make minimal test request
                litellm_completion(
                    messages=test_msg,
                    max_tokens=1,
                    **test_kwargs
                )
                
                logger.info('Rate limit has been lifted after %d attempts. Proceeding with the request.',
                          poll_attempt)
                return

            except (RateLimitError, InternalServerError) as e:
                # Only handle 429 errors
                if isinstance(e, InternalServerError) and '429' not in str(e):
                    logger.error('Unexpected server error during rate limit check: %s', str(e))
                    raise
                
                elapsed = time.time() - start_time
                logger.info('Still rate limited after %.1f seconds (%s). Waiting %s seconds before next check...', 
                          elapsed, str(e), retry_min_wait)
                time.sleep(retry_min_wait)
                
            except Exception as e:
                logger.error('Unexpected error during rate limit check: %s', str(e))
                raise

    def retry_if_exception_type(self, exceptions: tuple) -> callable:
        """Create a custom retry condition for specific exception types.

        This method creates a custom retry condition that retries only for specific
        exception types. It is used in the retry_decorator method to handle rate limits
        and other transient errors separately.

        Args:
            exceptions: Tuple of exception types to retry on

        Returns:
            callable: A retry condition that retries only for the specified exceptions
        """
        def retry_if_exception(exception):
            return isinstance(exception, exceptions)
        return retry_if_exception


    def retry_decorator(
        self,
        *,
        num_retries: Optional[int] = None,
        retry_exceptions: tuple = (),
        retry_min_wait: Optional[float] = None,
        retry_max_wait: Optional[float] = None,
        retry_multiplier: Optional[float] = None,
        retry_listener: Optional[callable] = None,
        **kwargs: Any
    ) -> callable:
        """Create a sophisticated retry decorator for LLM API calls.

        This decorator implements a multi-layered retry strategy:
        1. For rate limits (429 errors):
           - First attempts active polling with minimal test requests
           - Falls back to exponential backoff if polling fails
        2. For other errors:
           - Uses exponential backoff with configurable parameters
        3. Provides detailed logging of retry attempts and errors

        Args:
            num_retries: Maximum number of retry attempts (None for infinite)
            retry_exceptions: Tuple of exception types to retry on
            retry_min_wait: Minimum wait time between retries in seconds
            retry_max_wait: Maximum wait time between retries in seconds
            retry_multiplier: Multiplier for exponential backoff
            retry_listener: Optional callback for retry events
            **kwargs: Additional arguments passed to tenacity.retry

        Returns:
            callable: A retry decorator with the configured parameters

        Example:
            @retry_decorator(
                num_retries=3,
                retry_min_wait=1,
                retry_max_wait=60,
                retry_multiplier=2
            )
            def my_api_call():
                pass
        """
        # Use config values as defaults if parameters are not provided
        if hasattr(self, 'config'):
            num_retries = num_retries or getattr(self.config, 'num_retries', 3)
            retry_min_wait = retry_min_wait or getattr(self.config, 'retry_min_wait', 1)
            retry_max_wait = retry_max_wait or getattr(self.config, 'retry_max_wait', 60)
            retry_multiplier = retry_multiplier or getattr(self.config, 'retry_multiplier', 2)

        # Ensure we have reasonable defaults even without config
        num_retries = num_retries or 3
        retry_min_wait = retry_min_wait or 1
        retry_max_wait = retry_max_wait or 60
        retry_multiplier = retry_multiplier or 2

        # Include rate limit errors in retry exceptions
        retry_exceptions = tuple(set(retry_exceptions + (RateLimitError, InternalServerError)))

        def before_sleep(retry_state):
            """Handle retry attempts before sleeping.
            
            For rate limits:
            1. Attempts to handle via polling if it's a rate limit error
            2. Falls back to exponential backoff if polling fails
            
            For other errors:
            - Uses standard exponential backoff with logging
            """
            exception = retry_state.outcome.exception()
            
            # Handle rate limits
            if isinstance(exception, (RateLimitError, InternalServerError)):
                is_rate_limit = (
                    isinstance(exception, RateLimitError) or
                    ('429' in str(exception) and 'rate limit' in str(exception).lower())
                )
                if is_rate_limit:
                    try:
                        logger.info('Attempting to handle rate limit via polling...')
                        self.handle_rate_limit(exception, retry_min_wait)
                        return
                    except Exception as e:
                        logger.warning('Rate limit polling failed, falling back to exponential backoff: %s', str(e))

            # For other exceptions or if rate limit handling failed
            self.log_retry_attempt(retry_state)
            if retry_listener:
                retry_listener(retry_state.attempt_number, num_retries)

        return retry(
            before_sleep=before_sleep,
            stop=stop_after_attempt(num_retries) | stop_if_should_exit(),
            reraise=True,
            retry=(
                self.retry_if_exception_type(retry_exceptions)
            ),  # retry only for these types
            wait=wait_exponential(
                multiplier=retry_multiplier,
                min=retry_min_wait,
                max=retry_max_wait,
            ),
        )

    def log_retry_attempt(self, retry_state: RetryCallState) -> None:
        """Log retry attempts with detailed information.
        
        Args:
            retry_state: The current retry state containing attempt number and exception
                        information from tenacity.
        """
        if not retry_state or not retry_state.outcome:
            logger.error("Invalid retry state received")
            return

        try:
            exception = retry_state.outcome.exception()
            error_type = type(exception).__name__
            attempt = retry_state.attempt_number
            sleep_time = getattr(retry_state.next_action, 'sleep', None)

            # Build log message
            msg_parts = [
                f'{error_type}: {str(exception)}',
                f'Attempt #{attempt}',
                'You can customize retry values in the configuration'
            ]
            
            if sleep_time is not None:
                msg_parts.append(f'Next retry in: {sleep_time:.1f} seconds')

            logger.error(' | '.join(msg_parts))

            # Log additional debug information
            if hasattr(exception, 'response'):
                resp = getattr(exception, 'response', '')
                if resp:
                    logger.debug('Response details: %s', str(resp))
            
            if hasattr(exception, 'headers'):
                headers = getattr(exception, 'headers', '')
                if headers:
                    # Mask sensitive information in headers
                    safe_headers = {
                        k: v if 'key' not in k.lower() else '[MASKED]'
                        for k, v in headers.items()
                    }
                    logger.debug('Response headers: %s', safe_headers)

        except Exception as e:
            logger.error("Error while logging retry attempt: %s", str(e))
