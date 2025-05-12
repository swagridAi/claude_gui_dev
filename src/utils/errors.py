#!/usr/bin/env python3
"""
Error handling utilities for Claude GUI Automation.

This module provides a standardized exception hierarchy, error handling decorators,
and utility functions for consistent error management across the application.
"""

import logging
import functools
import traceback
import os
import time
from enum import Enum, auto
from typing import Callable, Optional, Any, Type, Dict, List, Tuple, Union, Set

# Try to import screenshots from main logging utils if available
try:
    from src.utils.logging_util import log_with_screenshot
    ADVANCED_LOGGING = True
except ImportError:
    ADVANCED_LOGGING = False
    def log_with_screenshot(message, level=logging.INFO, region=None, stage_name=None):
        logging.log(level, message)


# ---- Exception Hierarchy ----

class ApplicationError(Exception):
    """Base exception for all application errors."""
    pass


# -- Domain-specific base exceptions --

class ConfigError(ApplicationError):
    """Base exception for configuration-related errors."""
    pass


class BrowserError(ApplicationError):
    """Base exception for browser-related errors."""
    pass


class InteractionError(ApplicationError):
    """Base exception for UI interaction errors."""
    pass


class SessionError(ApplicationError):
    """Base exception for session management errors."""
    pass


class RecognitionError(ApplicationError):
    """Base exception for UI element recognition errors."""
    pass


# -- Configuration specific exceptions --

class ConfigLoadError(ConfigError):
    """Exception raised when configuration cannot be loaded."""
    pass


class ValidationError(ConfigError):
    """Exception raised when configuration validation fails."""
    def __init__(self, errors):
        self.errors = errors
        message = f"Validation failed: {'; '.join(errors)}"
        super().__init__(message)


class ConfigSaveError(ConfigError):
    """Exception raised when configuration cannot be saved."""
    pass


# -- Browser specific exceptions --

class BrowserLaunchError(BrowserError):
    """Exception raised when browser fails to launch."""
    pass


class BrowserNotReadyError(BrowserError):
    """Exception raised when browser is not ready within timeout."""
    pass


class BrowserNavigationError(BrowserError):
    """Exception raised when browser navigation fails."""
    pass


class BrowserClosingError(BrowserError):
    """Exception raised when browser cannot be closed properly."""
    pass


class BrowserStateError(BrowserError):
    """Exception raised when browser is in an unexpected state."""
    pass


# -- Interaction specific exceptions --

class InteractionTimeout(InteractionError):
    """Exception raised when an interaction times out."""
    pass


class VerificationError(InteractionError):
    """Exception raised when verification of an interaction fails."""
    pass


class TypeInputError(InteractionError):
    """Exception raised when typing input fails."""
    pass


class ClickError(InteractionError):
    """Exception raised when a click operation fails."""
    pass


# -- Session specific exceptions --

class SessionConfigError(SessionError):
    """Exception raised for session configuration errors."""
    pass


class SessionExecutionError(SessionError):
    """Exception raised when session execution fails."""
    pass


class PromptExecutionError(SessionError):
    """Exception raised when prompt execution fails."""
    pass


class CheckpointError(SessionError):
    """Exception raised for checkpoint-related errors."""
    pass


# -- Recognition specific exceptions --

class ElementNotFoundError(RecognitionError):
    """Exception raised when a UI element cannot be found."""
    pass


class TemplateMatchError(RecognitionError):
    """Exception raised when template matching fails."""
    pass


class ReferenceImageError(RecognitionError):
    """Exception raised when there's an issue with reference images."""
    pass


# ---- Error Classification ----

class ErrorType(Enum):
    """Enumeration of error types for classification."""
    UNKNOWN = auto()
    UI_NOT_FOUND = auto()      # UI element not found
    NETWORK_ERROR = auto()     # Network or connectivity issue
    BROWSER_ERROR = auto()     # Browser crashed or not responding  
    TIMEOUT = auto()           # Operation timed out
    CONFIG_ERROR = auto()      # Configuration issue
    PERMISSION_ERROR = auto()  # Permissions issue
    INPUT_ERROR = auto()       # Input formatting or validation
    SYSTEM_ERROR = auto()      # OS or system-level error


def classify_error(error: Exception) -> ErrorType:
    """
    Classify an exception into a standardized error type.
    
    Args:
        error: The exception to classify
        
    Returns:
        ErrorType indicating the category of error
    """
    error_str = str(error).lower()
    error_type = type(error)
    
    # Check exception type first
    if isinstance(error, ElementNotFoundError) or "not found" in error_str:
        return ErrorType.UI_NOT_FOUND
    elif isinstance(error, InteractionTimeout) or "timeout" in error_str or "timed out" in error_str:
        return ErrorType.TIMEOUT
    elif isinstance(error, BrowserError) or "browser" in error_str:
        return ErrorType.BROWSER_ERROR
    elif isinstance(error, ConfigError):
        return ErrorType.CONFIG_ERROR
    elif isinstance(error, (PermissionError, OSError)) or "permission" in error_str:
        return ErrorType.PERMISSION_ERROR
    elif isinstance(error, (ValueError, TypeError)) or "input" in error_str:
        return ErrorType.INPUT_ERROR
    elif "network" in error_str or "connection" in error_str or "internet" in error_str:
        return ErrorType.NETWORK_ERROR
    elif any(system_term in error_str for system_term in ["system", "os ", "memory", "disk"]):
        return ErrorType.SYSTEM_ERROR
        
    # Default case
    return ErrorType.UNKNOWN


def is_retriable(error: Exception) -> bool:
    """
    Determine if an error should be retried based on its type.
    
    Args:
        error: The exception to check
        
    Returns:
        True if the error is suitable for retry, False otherwise
    """
    error_type = classify_error(error)
    
    # Define which error types can be retried
    RETRIABLE_TYPES = {
        ErrorType.NETWORK_ERROR,
        ErrorType.TIMEOUT,
        ErrorType.UI_NOT_FOUND,
        ErrorType.BROWSER_ERROR
    }
    
    return error_type in RETRIABLE_TYPES


def extract_error_details(error: Exception) -> Dict[str, Any]:
    """
    Extract useful details from an error for logging and diagnosis.
    
    Args:
        error: The exception to extract details from
        
    Returns:
        Dictionary with error details
    """
    error_type = classify_error(error)
    
    details = {
        "type": type(error).__name__,
        "message": str(error),
        "classification": error_type.name,
        "traceback": traceback.format_exc(),
        "retriable": is_retriable(error),
        "timestamp": time.time()
    }
    
    # Extract additional details for specific error types
    if hasattr(error, "errors"):
        details["sub_errors"] = error.errors
        
    return details


# ---- Error Handling Decorators ----

def with_error_handling(
    screenshot: bool = True,
    rethrow: bool = True,
    convert_exception: bool = True,
    error_map: Optional[Dict[Type[Exception], Type[Exception]]] = None
):
    """
    Decorator to add consistent error handling to functions.
    
    Args:
        screenshot: Whether to take a screenshot on error
        rethrow: Whether to re-raise exceptions after handling
        convert_exception: Whether to convert generic exceptions to application-specific ones
        error_map: Optional mapping of exception types to convert to
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                # Log the error
                error_message = f"Error in {func_name}: {str(e)}"
                
                # Take screenshot if enabled and available
                if screenshot and ADVANCED_LOGGING:
                    log_with_screenshot(
                        error_message,
                        level=logging.ERROR,
                        stage_name=f"ERROR_{func_name.upper()}"
                    )
                else:
                    logging.error(error_message)
                
                # Extract and log detailed error information
                details = extract_error_details(e)
                logging.debug(f"Error details: {details}")
                
                # Convert exception if requested
                if convert_exception:
                    e = _convert_exception(e, error_map)
                
                # Re-raise if requested
                if rethrow:
                    raise e
                
                # Return False as a default failure indicator
                return False
                
        return wrapper
    return decorator


def _convert_exception(error: Exception, error_map: Optional[Dict[Type[Exception], Type[Exception]]]) -> Exception:
    """
    Convert an exception to a more specific application exception.
    
    Args:
        error: The original exception
        error_map: Optional mapping of exception types to convert to
        
    Returns:
        Converted exception
    """
    # If a mapping is provided, use it first
    if error_map:
        for source_type, target_type in error_map.items():
            if isinstance(error, source_type):
                return target_type(str(error)) from error
    
    # Otherwise use default conversion based on error message and type
    error_str = str(error).lower()
    
    # Convert based on keywords in error message
    if "timeout" in error_str or "timed out" in error_str:
        return InteractionTimeout(f"Operation timed out: {error}") from error
    elif "browser" in error_str:
        return BrowserError(f"Browser error: {error}") from error
    elif "verification" in error_str:
        return VerificationError(f"Verification failed: {error}") from error
    elif "typing" in error_str or "input" in error_str:
        return TypeInputError(f"Input error: {error}") from error
    elif "element" in error_str and "not found" in error_str:
        return ElementNotFoundError(f"Element not found: {error}") from error
    elif "config" in error_str:
        return ConfigError(f"Configuration error: {error}") from error
    
    # Return the original if no conversion rule matches
    return error


def retry_on_error(
    max_attempts: int = 3, 
    delay: float = 1.0, 
    backoff: float = 2.0, 
    jitter: float = 0.2,
    retriable_errors: Optional[List[Type[Exception]]] = None
):
    """
    Decorator to retry functions when they raise specific exceptions.
    This is a simplified version - use the dedicated retry module for more options.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        jitter: Random factor to add to delay (0-1)
        retriable_errors: List of exception types to retry on, or None for auto-detection
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import random  # Import here to avoid global import
            
            last_exception = None
            current_delay = delay
            
            # Set default retriable errors if none provided
            nonlocal retriable_errors
            if retriable_errors is None:
                retriable_errors = [ApplicationError, Exception]
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(retriable_errors) as e:
                    # Check if this specific error should be retried
                    if not is_retriable(e) and Exception in retriable_errors:
                        # Don't retry errors that are explicitly non-retriable
                        # unless they're in the provided retriable_errors list
                        raise
                    
                    last_exception = e
                    
                    if attempt < max_attempts:
                        # Calculate jittered delay
                        jitter_amount = current_delay * jitter
                        actual_delay = current_delay + random.uniform(-jitter_amount, jitter_amount)
                        actual_delay = max(0.1, actual_delay)  # Ensure positive delay
                        
                        # Log retry attempt
                        logging.warning(
                            f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {actual_delay:.2f} seconds..."
                        )
                        
                        # Wait before retrying
                        time.sleep(actual_delay)
                        
                        # Increase delay for next attempt using backoff
                        current_delay *= backoff
                    else:
                        # Log final failure
                        logging.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            # If we get here, all retries failed
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


# ---- Utils for Screenshot on Error ----

def save_error_screenshot(error: Exception, context: str = "error") -> Optional[str]:
    """
    Save a screenshot when an error occurs.
    
    Args:
        error: The exception that occurred
        context: Context description for the filename
        
    Returns:
        Path to the saved screenshot or None if failed
    """
    try:
        import pyautogui  # Import here to avoid global import
        
        # Create directory for error screenshots
        screenshot_dir = os.path.join("logs", "error_screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)
        
        # Generate filename
        timestamp = int(time.time())
        error_type = type(error).__name__
        filename = f"{context}_{error_type}_{timestamp}.png"
        filepath = os.path.join(screenshot_dir, filename)
        
        # Take and save screenshot
        screenshot = pyautogui.screenshot()
        screenshot.save(filepath)
        
        logging.info(f"Error screenshot saved to {filepath}")
        return filepath
    
    except Exception as e:
        logging.warning(f"Failed to save error screenshot: {e}")
        return None


# ---- Error Reporter ----

class ErrorReporter:
    """
    Handles aggregation and reporting of errors for analysis.
    """
    
    def __init__(self, report_file: str = "logs/error_report.json"):
        """
        Initialize error reporter.
        
        Args:
            report_file: Path to the error report file
        """
        self.report_file = report_file
        self.errors = []
        self.error_counts = {}
        
        # Create log directory if needed
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    def record_error(self, error: Exception, context: Optional[str] = None):
        """
        Record an error for later reporting.
        
        Args:
            error: The exception to record
            context: Optional context information
        """
        error_type = type(error).__name__
        
        # Update error counts
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Get detailed error information
        details = extract_error_details(error)
        if context:
            details["context"] = context
            
        self.errors.append(details)
    
    def save_report(self):
        """
        Save the current error report to a file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import json  # Import here to avoid global import
            
            report = {
                "errors": self.errors,
                "counts": self.error_counts,
                "total": sum(self.error_counts.values()),
                "timestamp": time.time()
            }
            
            with open(self.report_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            return True
            
        except Exception as e:
            logging.error(f"Failed to save error report: {e}")
            return False
    
    def get_summary(self) -> str:
        """
        Get a text summary of recorded errors.
        
        Returns:
            Summary string
        """
        if not self.errors:
            return "No errors recorded."
            
        total = sum(self.error_counts.values())
        summary = f"Recorded {total} errors:\n"
        
        for error_type, count in sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            summary += f"  - {error_type}: {count} ({percentage:.1f}%)\n"
            
        return summary


# Initialize a global error reporter
error_reporter = ErrorReporter()