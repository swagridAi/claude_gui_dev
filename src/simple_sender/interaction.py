#!/usr/bin/env python3
"""
Interaction module for Simple Claude Sender.

This module provides simplified functions for interacting with Claude's web UI,
focusing on essential actions needed for prompt submission while maintaining
a lightweight approach compared to the full automation system.
"""

import pyautogui
import time
import random
import logging
import functools
import os
from typing import Callable, Any, Optional, Union

# Try to import logging utilities from main project, fall back to basic logging
try:
    from src.utils.logging_util import log_with_screenshot
    ADVANCED_LOGGING = True
except ImportError:
    ADVANCED_LOGGING = False
    def log_with_screenshot(message, level=logging.INFO, region=None, stage_name=None):
        logging.log(level, message)

# Configurable constants
DEFAULT_TYPING_DELAY = 0.03
DEFAULT_ACTION_DELAY = 0.5
DEFAULT_WAIT_TIMEOUT = 300
DEFAULT_CHECK_INTERVAL = 10
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1
DEFAULT_RETRY_BACKOFF = 2


# ---- Core Functions ----

def add_humanized_delay(min_delay: float = 0.1, max_delay: float = 0.3) -> None:
    """Add a random delay between actions to simulate human behavior."""
    delay = random.uniform(min_delay, max_delay)
    time.sleep(delay)


def take_verification_screenshot(name: str) -> Optional[str]:
    """Take a screenshot for verification purposes."""
    try:
        timestamp = int(time.time())
        screenshot_dir = os.path.join("logs", "screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)
        filename = os.path.join(screenshot_dir, f"{name}_{timestamp}.png")
        
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        logging.debug(f"Saved verification screenshot to {filename}")
        return filename
    except Exception as e:
        logging.error(f"Failed to take verification screenshot: {e}")
        return None


def retry_decorator(max_attempts: int = DEFAULT_RETRY_ATTEMPTS, 
                    delay: float = DEFAULT_RETRY_DELAY, 
                    backoff: float = DEFAULT_RETRY_BACKOFF):
    """Decorator to retry functions with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    retry_delay = delay * (backoff ** (attempt - 1))
                    jitter = random.uniform(0.8, 1.2)  # Add ±20% jitter
                    wait_time = retry_delay * jitter
                    
                    logging.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {wait_time:.2f} seconds..."
                    )
                    time.sleep(wait_time)
            
            # If we get here, all retries failed
            logging.error(f"All {max_attempts} attempts failed for {func.__name__}")
            if last_exception:
                raise last_exception
        return wrapper
    return decorator


def with_error_handling(func: Callable) -> Callable:
    """Decorator to add consistent error handling to interaction functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        try:
            logging.debug(f"Starting {func_name}")
            result = func(*args, **kwargs)
            logging.debug(f"Completed {func_name} successfully")
            return result
        except Exception as e:
            logging.error(f"Error in {func_name}: {e}")
            if ADVANCED_LOGGING:
                log_with_screenshot(
                    f"Error during {func_name}: {e}", 
                    level=logging.ERROR,
                    stage_name=f"ERROR_{func_name.upper()}"
                )
            raise
    return wrapper


@with_error_handling
def type_text(text: str, human_like: bool = True, clear_existing: bool = True) -> bool:
    """
    Type text into the currently focused element with optional human-like timing.
    
    Args:
        text: The text to type
        human_like: Whether to add random delays between keystrokes
        clear_existing: Whether to clear existing text before typing
        
    Returns:
        True if successful, False otherwise
    """
    if clear_existing:
        # Clear any existing text with Ctrl+A and Delete
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(DEFAULT_ACTION_DELAY)
        pyautogui.press('delete')
        time.sleep(DEFAULT_ACTION_DELAY)
    
    if ADVANCED_LOGGING:
        log_with_screenshot("Before typing text", stage_name="BEFORE_TYPE")
    
    # Type the text character by character with random delays if human-like is enabled
    if human_like:
        for char in text:
            pyautogui.write(char)
            
            # Random delay between keystrokes for more human-like typing
            typing_delay = DEFAULT_TYPING_DELAY * random.uniform(0.8, 1.2)
            time.sleep(typing_delay)
            
            # Occasionally add a slightly longer pause (simulating thinking)
            if random.random() < 0.05:  # 5% chance
                time.sleep(random.uniform(0.2, 0.5))
    else:
        # Type all at once for speed
        pyautogui.write(text)
    
    if ADVANCED_LOGGING:
        log_with_screenshot("After typing text", stage_name="AFTER_TYPE")
    
    return True


@with_error_handling
def press_submit() -> bool:
    """
    Press Enter to submit the current prompt.
    
    Returns:
        True if successful, False otherwise
    """
    if ADVANCED_LOGGING:
        log_with_screenshot("Before pressing Enter", stage_name="BEFORE_SUBMIT")
    
    # Press Enter to submit
    pyautogui.press('enter')
    logging.info("Pressed Enter to submit prompt")
    
    if ADVANCED_LOGGING:
        log_with_screenshot("After pressing Enter", stage_name="AFTER_SUBMIT")
    
    # Basic verification that submission occurred
    time.sleep(1)  # Short delay to let UI update
    
    return True


@with_error_handling
def wait_for_response(timeout: int = DEFAULT_WAIT_TIMEOUT, 
                     check_interval: int = DEFAULT_CHECK_INTERVAL,
                     verify_visually: bool = False) -> bool:
    """
    Wait for Claude to respond, with optional visual verification.
    
    Args:
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
        verify_visually: Whether to use visual cues to detect completion
        
    Returns:
        True if response completed within timeout, False otherwise
    """
    logging.info(f"Waiting up to {timeout} seconds for response...")
    
    # Start wait timer
    start_time = time.time()
    elapsed = 0
    
    while elapsed < timeout:
        # Take periodic screenshots if advanced logging is enabled
        if ADVANCED_LOGGING and elapsed % 30 == 0:  # Every 30 seconds
            log_with_screenshot(
                f"Still waiting for response ({elapsed}/{timeout} seconds elapsed)", 
                stage_name=f"WAITING_{elapsed}"
            )
        
        # If visual verification is enabled, attempt to detect response completion
        if verify_visually and verify_response_complete():
            logging.info(f"Response detected as complete after {elapsed} seconds")
            if ADVANCED_LOGGING:
                log_with_screenshot("Response complete", stage_name="RESPONSE_COMPLETE")
            return True
        
        # Sleep for the check interval
        time.sleep(check_interval)
        
        # Update elapsed time
        elapsed = int(time.time() - start_time)
        
        # Log progress periodically
        if elapsed % 30 == 0:  # Log every 30 seconds
            remaining = timeout - elapsed
            logging.info(f"Still waiting: {remaining} seconds remaining...")
    
    logging.warning(f"Wait timeout of {timeout} seconds reached")
    if ADVANCED_LOGGING:
        log_with_screenshot("Wait timeout reached", stage_name="WAIT_TIMEOUT")
    
    return False


# ---- Verification Utilities ----

@with_error_handling
def verify_text_entered(expected_text: str, timeout: int = 5) -> bool:
    """
    Verify text was entered correctly in prompt box.
    
    This is a simplified verification that doesn't perform actual OCR,
    but can be extended to use OCR libraries if available.
    
    Args:
        expected_text: The text that should have been entered
        timeout: Maximum time to verify in seconds
        
    Returns:
        True if verified, False otherwise
    """
    # This is a placeholder for actual verification
    # In a full implementation, this would use OCR or other techniques
    
    # For now, we just take a verification screenshot and assume success
    if ADVANCED_LOGGING:
        log_with_screenshot("Text entry verification", stage_name="VERIFY_TEXT")
    
    return True


@with_error_handling
def verify_claude_responding() -> bool:
    """
    Check if Claude is currently responding.
    
    This is a simplified check that can be enhanced with visual recognition.
    
    Returns:
        True if Claude appears to be responding, False otherwise
    """
    # This is a placeholder for actual verification
    # In a full implementation, this would check for "thinking" indicators
    
    # For now, take a verification screenshot and assume Claude is responding
    if ADVANCED_LOGGING:
        log_with_screenshot("Claude response verification", stage_name="VERIFY_RESPONDING")
    
    return True


@with_error_handling
def verify_response_complete(timeout: int = 10) -> bool:
    """
    Verify Claude has completed its response.
    
    Args:
        timeout: Maximum time to verify in seconds
        
    Returns:
        True if response appears complete, False otherwise
    """
    # This is a placeholder for actual verification
    # In a full implementation, this would check for completion indicators
    
    # For now, we just take a verification screenshot and assume completion
    if ADVANCED_LOGGING:
        log_with_screenshot("Response completion verification", stage_name="VERIFY_COMPLETE")
    
    return True


# ---- Compound Actions ----

@retry_decorator(max_attempts=DEFAULT_RETRY_ATTEMPTS)
@with_error_handling
def send_prompt(prompt: str, verify_sent: bool = True, 
               wait_time: int = DEFAULT_WAIT_TIMEOUT,
               human_like: bool = True) -> bool:
    """
    Send a prompt and optionally wait for response.
    
    Args:
        prompt: The text prompt to send
        verify_sent: Whether to verify the prompt was sent successfully
        wait_time: How long to wait for a response in seconds
        human_like: Whether to use human-like typing behavior
        
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Sending prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Sending prompt: {prompt}")
    
    # Type the prompt text
    success = type_text(prompt, human_like=human_like)
    if not success:
        logging.error("Failed to type prompt text")
        return False
    
    # Small delay before submitting
    time.sleep(DEFAULT_ACTION_DELAY)
    
    # Press Enter to submit
    success = press_submit()
    if not success:
        logging.error("Failed to submit prompt")
        return False
    
    # Verify the prompt was sent if requested
    if verify_sent:
        if ADVANCED_LOGGING:
            log_with_screenshot("Verifying prompt was sent", stage_name="VERIFY_SENT")
    
    # Wait for response if wait_time > 0
    if wait_time > 0:
        wait_for_response(timeout=wait_time, verify_visually=True)
    
    return True


@with_error_handling
def clear_conversation_if_needed(max_messages: int = 10) -> bool:
    """
    Clear conversation if too many messages exist.
    
    Args:
        max_messages: Maximum number of messages before clearing
        
    Returns:
        True if cleared or no clearing needed, False if clearing failed
    """
    # This is a placeholder for actual implementation
    # In a full implementation, this would check message count and clear if needed
    logging.info("Checking if conversation needs clearing")
    
    # For now, just log and continue
    if ADVANCED_LOGGING:
        log_with_screenshot("Conversation clearing check", stage_name="CHECK_CLEAR")
    
    return True


# ---- Focus and Window Management ----

@with_error_handling
def ensure_browser_focused() -> bool:
    """
    Ensure the browser window is in focus for interaction.
    
    Returns:
        True if browser is focused, False otherwise
    """
    # This function depends on browser.py implementation
    # For now, it's a placeholder that assumes focus is correct
    
    logging.debug("Checking browser focus")
    # In a real implementation, this would check and correct focus
    
    return True


@with_error_handling
def click_prompt_area(coordinates: Optional[tuple] = None) -> bool:
    """
    Click in the prompt input area.
    
    Args:
        coordinates: Optional tuple of (x, y) coordinates
        
    Returns:
        True if clicked successfully, False otherwise
    """
    if coordinates:
        x, y = coordinates
        logging.debug(f"Clicking prompt area at coordinates ({x}, {y})")
        
        # Add a slight humanized movement
        pyautogui.moveTo(x, y, duration=random.uniform(0.1, 0.3))
        pyautogui.click()
        
        if ADVANCED_LOGGING:
            log_with_screenshot("After clicking prompt area", stage_name="AFTER_CLICK_PROMPT")
        
        return True
    else:
        logging.warning("No coordinates provided for prompt area")
        return False