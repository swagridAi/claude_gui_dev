#!/usr/bin/env python3
"""
Retry Utilities

This module provides standardized retry functionality with exponential backoff
for handling transient failures in a consistent way across the application.
"""

import time
import random
import logging
import functools
from typing import Callable, Type, Tuple, Optional, Any, Union, TypeVar

# Type variables for better type hinting
F = TypeVar('F', bound=Callable[..., Any])
E = TypeVar('E', bound=Type[Exception])

def calculate_retry_delay(
    attempt: int,
    initial_delay: float,
    backoff_factor: float,
    jitter: float,
    max_delay: Optional[float] = None
) -> float:
    """
    Calculate delay for a retry attempt with exponential backoff and jitter.
    
    Args:
        attempt: Current attempt number (0-based)
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each attempt
        jitter: Random factor to add to delay (0-1)
        max_delay: Maximum delay in seconds
        
    Returns:
        Delay in seconds with jitter applied
    """
    # Calculate base delay with exponential backoff
    delay = initial_delay * (backoff_factor ** attempt)
    
    # Apply maximum delay if specified
    if max_delay is not None:
        delay = min(delay, max_delay)
    
    # Add jitter to avoid thundering herd problem
    jitter_amount = delay * jitter
    return delay + random.uniform(-jitter_amount, jitter_amount)


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.2,
    max_delay: Optional[float] = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
    logger: Optional[logging.Logger] = None
) -> Callable[[F], F]:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts (including first try)
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each failure
        jitter: Random factor to add to delay (0-1)
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function run before each retry with args (exception, attempt, delay)
        logger: Logger to use (defaults to root logger)
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @with_retry(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
        def fetch_data():
            # Function that might fail temporarily
    """
    log = logger or logging.getLogger()
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            operation_name = func.__name__
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Skip delay calculation on the last attempt
                    if attempt >= max_attempts - 1:
                        break
                    
                    # Calculate delay with jitter
                    delay = calculate_retry_delay(
                        attempt, initial_delay, backoff_factor, jitter, max_delay
                    )
                    
                    # Log retry information
                    log.warning(
                        f"Attempt {attempt + 1}/{max_attempts} for '{operation_name}' failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    
                    # Call on_retry callback if provided
                    if on_retry:
                        try:
                            on_retry(e, attempt + 1, delay)
                        except Exception as callback_error:
                            log.warning(f"Error in retry callback: {callback_error}")
                    
                    # Wait before next attempt
                    time.sleep(delay)
            
            # If we get here, all attempts failed
            log.error(f"All {max_attempts} attempts for '{operation_name}' failed")
            if last_exception:
                raise last_exception
            
            # This should never happen if at least one attempt was made
            raise RuntimeError(f"No attempts made for '{operation_name}'")
            
        return wrapper
    
    return decorator


class RetryContext:
    """
    Context manager for retryable operations.
    
    Allows code blocks to be retried with configurable backoff strategy.
    
    Example:
        with RetryContext(max_attempts=3, exceptions=(ConnectionError,)) as retry:
            while retry.attempts():
                try:
                    # Code that might fail temporarily
                    break  # Success, exit the retry loop
                except retry.exceptions as e:
                    retry.handle_exception(e)
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        jitter: float = 0.2,
        max_delay: Optional[float] = None,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        on_retry: Optional[Callable[[Exception, int, float], None]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize retry context with specified parameters.
        
        Args:
            max_attempts: Maximum number of attempts (including first try)
            initial_delay: Initial delay between retries in seconds
            backoff_factor: Multiplier for delay after each failure
            jitter: Random factor to add to delay (0-1)
            max_delay: Maximum delay in seconds
            exceptions: Tuple of exception types to catch and retry
            on_retry: Optional callback function run before each retry
            logger: Logger to use (defaults to root logger)
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.max_delay = max_delay
        self.exceptions = exceptions
        self.on_retry = on_retry
        self.logger = logger or logging.getLogger()
        
        # Internal state
        self.attempt = 0
        self.last_exception = None
    
    def __enter__(self):
        self.attempt = 0
        self.last_exception = None
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # If there was an exception and it's in our handled exceptions
        if exc_type and issubclass(exc_type, self.exceptions):
            self.last_exception = exc_val
            
            # If we haven't exhausted all attempts, suppress the exception
            if self.attempt < self.max_attempts:
                return True
                
        # Don't suppress other exceptions or if we've exhausted retries
        return False
    
    def attempts(self) -> bool:
        """
        Check if another attempt should be made.
        
        Returns:
            True if another attempt should be made, False otherwise
        """
        # If we're starting or just had an exception
        if self.attempt < self.max_attempts:
            if self.attempt > 0:
                # Calculate delay based on the previous attempt
                delay = calculate_retry_delay(
                    self.attempt - 1,  # Previous attempt
                    self.initial_delay,
                    self.backoff_factor,
                    self.jitter,
                    self.max_delay
                )
                
                # Log retry information
                self.logger.warning(
                    f"Attempt {self.attempt}/{self.max_attempts} failed. "
                    f"Retrying in {delay:.2f}s"
                )
                
                # Call on_retry callback if provided
                if self.on_retry and self.last_exception:
                    try:
                        self.on_retry(self.last_exception, self.attempt, delay)
                    except Exception as callback_error:
                        self.logger.warning(f"Error in retry callback: {callback_error}")
                
                # Wait before next attempt
                time.sleep(delay)
            
            # Increment attempt counter and return True to continue
            self.attempt += 1
            return True
        
        return False
    
    def handle_exception(self, exception: Exception) -> None:
        """
        Handle an exception during retry.
        
        Args:
            exception: The exception that occurred
        """
        self.last_exception = exception
        
        # Log the exception
        if self.attempt < self.max_attempts:
            self.logger.debug(f"Caught retriable exception (attempt {self.attempt}/{self.max_attempts}): {exception}")
        else:
            self.logger.error(f"Failed after {self.max_attempts} attempts: {exception}")


# Simple utility function for one-off retries
def retry_operation(
    operation: Callable[[], Any],
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.2,
    max_delay: Optional[float] = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
    logger: Optional[logging.Logger] = None
) -> Any:
    """
    Execute an operation with retries.
    
    Args:
        operation: Function to execute
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each failure
        jitter: Random factor to add to delay (0-1)
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function run before each retry
        logger: Logger to use (defaults to root logger)
        
    Returns:
        Result of the operation
        
    Raises:
        Exception: If all retry attempts fail
    """
    # Use the decorator internally
    @with_retry(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
        jitter=jitter,
        max_delay=max_delay,
        exceptions=exceptions,
        on_retry=on_retry,
        logger=logger
    )
    def wrapped_operation():
        return operation()
    
    return wrapped_operation()