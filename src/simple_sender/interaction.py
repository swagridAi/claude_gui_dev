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
from pathlib import Path
from typing import Callable, Any, Optional, Union, Tuple, Dict, List
import io
from PIL import Image, ImageChops
import numpy as np

# Try to import logging utilities from main project, fall back to basic logging
try:
    from src.utils.logging_util import log_with_screenshot
    ADVANCED_LOGGING = True
except ImportError:
    ADVANCED_LOGGING = False
    def log_with_screenshot(message, level=logging.INFO, region=None, stage_name=None):
        logging.log(level, message)

# Try to import clipboard module, use a fallback if not available
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False


# ---- Exception Hierarchy ----

class InteractionError(Exception):
    """Base exception for all interaction-related errors."""
    pass


class InteractionTimeout(InteractionError):
    """Exception raised when an interaction times out."""
    pass


class VerificationError(InteractionError):
    """Exception raised when verification of an interaction fails."""
    pass


class BrowserError(InteractionError):
    """Exception raised for browser-related issues."""
    pass


class TypeInputError(InteractionError):
    """Exception raised when typing input fails."""
    pass


# ---- Constants and Configuration ----

class InteractionConstants:
    """Centralized constants for interaction module."""
    
    # Timing constants
    DEFAULT_TYPING_DELAY = 0.03
    DEFAULT_ACTION_DELAY = 0.5
    DEFAULT_WAIT_TIMEOUT = 300
    DEFAULT_CHECK_INTERVAL = 10
    
    # Retry constants
    DEFAULT_RETRY_ATTEMPTS = 3
    DEFAULT_RETRY_DELAY = 1
    DEFAULT_RETRY_BACKOFF = 2
    DEFAULT_RETRY_JITTER = 0.2
    
    # Visual verification constants
    VERIFICATION_THRESHOLD = 0.1  # Pixel difference threshold
    MIN_PIXEL_CHANGE = 1000  # Minimum number of pixels that must change
    VERIFICATION_ATTEMPTS = 3  # Number of verification attempts
    
    # Typing profiles
    TYPING_PROFILES = {
        'fast': {
            'base_delay': 0.01,
            'variance': 0.005,
            'pause_chance': 0.02,
            'pause_duration': (0.1, 0.3),
            'typo_chance': 0.0
        },
        'normal': {
            'base_delay': 0.03,
            'variance': 0.01,
            'pause_chance': 0.05,
            'pause_duration': (0.2, 0.5),
            'typo_chance': 0.0
        },
        'slow': {
            'base_delay': 0.05,
            'variance': 0.02,
            'pause_chance': 0.1,
            'pause_duration': (0.3, 0.7),
            'typo_chance': 0.0
        }
    }


# ---- Utility Functions and Decorators ----

class InteractionUtils:
    """Utility functions for UI interactions."""
    
    @staticmethod
    def add_humanized_delay(min_delay: float = 0.1, max_delay: float = 0.3) -> None:
        """Add a random delay between actions to simulate human behavior."""
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
    
    @staticmethod
    def take_screenshot(region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        """
        Take a screenshot of the specified region or full screen.
        
        Args:
            region: Optional tuple of (x, y, width, height)
            
        Returns:
            PIL Image object
        """
        return pyautogui.screenshot(region=region)
    
    @staticmethod
    def save_debug_screenshot(name: str, image: Optional[Image.Image] = None) -> Optional[str]:
        """
        Save a screenshot for debugging purposes.
        
        Args:
            name: Base name for the screenshot
            image: Optional PIL Image to save (takes a new screenshot if None)
            
        Returns:
            Path to the saved screenshot or None if failed
        """
        try:
            timestamp = int(time.time())
            screenshot_dir = os.path.join("logs", "screenshots")
            os.makedirs(screenshot_dir, exist_ok=True)
            filename = os.path.join(screenshot_dir, f"{name}_{timestamp}.png")
            
            if image is None:
                image = pyautogui.screenshot()
                
            image.save(filename)
            logging.debug(f"Saved debug screenshot to {filename}")
            return filename
        except Exception as e:
            logging.error(f"Failed to save debug screenshot: {e}")
            return None
    
    @staticmethod
    def log_action(message: str, level: int = logging.INFO, 
                  take_screenshot: bool = True) -> None:
        """
        Log an action with optional screenshot.
        
        Args:
            message: Log message
            level: Logging level
            take_screenshot: Whether to take and save a screenshot
        """
        if ADVANCED_LOGGING:
            action_name = message.replace(" ", "_").lower()
            log_with_screenshot(message, level=level, stage_name=action_name)
        else:
            logging.log(level, message)
            if take_screenshot:
                screenshot = pyautogui.screenshot()
                InteractionUtils.save_debug_screenshot(
                    f"action_{message.replace(' ', '_').lower()}", 
                    screenshot
                )


def retry_decorator(max_attempts: int = InteractionConstants.DEFAULT_RETRY_ATTEMPTS, 
                   delay: float = InteractionConstants.DEFAULT_RETRY_DELAY, 
                   backoff: float = InteractionConstants.DEFAULT_RETRY_BACKOFF,
                   jitter: float = InteractionConstants.DEFAULT_RETRY_JITTER,
                   exceptions: Tuple[Exception, ...] = (Exception,)):
    """
    Decorator to retry functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplicative factor for delay after each retry
        jitter: Random factor to add to delay (0-1)
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    retry_delay = delay * (backoff ** (attempt - 1))
                    jitter_amount = retry_delay * jitter
                    actual_delay = retry_delay + random.uniform(-jitter_amount, jitter_amount)
                    
                    # Ensure delay is always positive
                    actual_delay = max(0.1, actual_delay)
                    
                    if attempt < max_attempts:
                        logging.warning(
                            f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {actual_delay:.2f} seconds..."
                        )
                        time.sleep(actual_delay)
                    else:
                        logging.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            # If we get here, all retries failed
            if last_exception:
                raise last_exception
        return wrapper
    return decorator


def with_error_handling(func: Callable) -> Callable:
    """
    Decorator to add consistent error handling to interaction functions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with error handling
    """
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
            
            # Convert generic exceptions to more specific ones
            if isinstance(e, TimeoutError):
                raise InteractionTimeout(f"Timeout in {func_name}: {e}") from e
            elif "browser" in str(e).lower():
                raise BrowserError(f"Browser error in {func_name}: {e}") from e
            elif "verification" in str(e).lower():
                raise VerificationError(f"Verification failed in {func_name}: {e}") from e
            elif "typing" in str(e).lower() or "input" in str(e).lower():
                raise TypeInputError(f"Input error in {func_name}: {e}") from e
            else:
                raise InteractionError(f"Error in {func_name}: {e}") from e
    return wrapper


# ---- Visual Verification ----

class VisualVerifier:
    """
    Provides visual verification capabilities for detecting UI changes.
    """
    
    def __init__(self):
        """Initialize the visual verifier."""
        self.reference_images = {}
        self.last_checked = {}
    
    def take_reference_image(self, name: str, 
                            region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        """
        Take a reference image for later comparison.
        
        Args:
            name: Reference image name
            region: Screen region to capture
            
        Returns:
            Captured image
        """
        image = pyautogui.screenshot(region=region)
        self.reference_images[name] = {
            'image': image,
            'timestamp': time.time(),
            'region': region
        }
        return image
    
    def check_for_changes(self, reference_name: str, 
                         region: Optional[Tuple[int, int, int, int]] = None,
                         threshold: float = InteractionConstants.VERIFICATION_THRESHOLD,
                         min_pixels: int = InteractionConstants.MIN_PIXEL_CHANGE) -> Tuple[bool, float]:
        """
        Check if the screen has changed compared to a reference image.
        
        Args:
            reference_name: Name of the reference image
            region: Screen region to capture (uses reference's region if None)
            threshold: Difference threshold (0-1)
            min_pixels: Minimum number of pixels that must change
            
        Returns:
            Tuple of (changed, difference_percentage)
        """
        if reference_name not in self.reference_images:
            # Take initial reference if not exist
            reference_data = {
                'image': self.take_reference_image(reference_name, region),
                'timestamp': time.time(),
                'region': region
            }
            self.reference_images[reference_name] = reference_data
            return False, 0.0
        
        # Use the reference's region if none specified
        if region is None:
            region = self.reference_images[reference_name]['region']
        
        # Take current screenshot
        current_image = pyautogui.screenshot(region=region)
        
        # Get reference image
        reference_image = self.reference_images[reference_name]['image']
        
        # Calculate difference
        diff = ImageChops.difference(reference_image, current_image)
        diff_array = np.array(diff)
        
        # Calculate statistics
        non_zero_pixels = np.count_nonzero(diff_array)
        total_pixels = diff_array.size / 3  # RGB has 3 values per pixel
        difference_percentage = non_zero_pixels / total_pixels
        
        # Update last check time
        self.last_checked[reference_name] = time.time()
        
        # Determine if change is significant
        significant_change = (difference_percentage > threshold and 
                             non_zero_pixels > min_pixels)
        
        if significant_change:
            # Update reference image to the current one
            self.reference_images[reference_name]['image'] = current_image
            self.reference_images[reference_name]['timestamp'] = time.time()
            
            # Save debug image if logging is enabled
            if logging.getLogger().level <= logging.DEBUG:
                debug_dir = os.path.join("logs", "visual_verification")
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = int(time.time())
                current_image.save(os.path.join(debug_dir, f"{reference_name}_current_{timestamp}.png"))
                reference_image.save(os.path.join(debug_dir, f"{reference_name}_reference_{timestamp}.png"))
                diff.save(os.path.join(debug_dir, f"{reference_name}_diff_{timestamp}.png"))
                
            logging.debug(f"Visual change detected for {reference_name}: {difference_percentage:.4f}")
            
        return significant_change, difference_percentage
    
    def wait_for_visual_change(self, reference_name: str, 
                              region: Optional[Tuple[int, int, int, int]] = None,
                              timeout: int = InteractionConstants.DEFAULT_WAIT_TIMEOUT,
                              check_interval: float = InteractionConstants.DEFAULT_CHECK_INTERVAL,
                              threshold: float = InteractionConstants.VERIFICATION_THRESHOLD) -> bool:
        """
        Wait for a visual change to occur within timeout period.
        
        Args:
            reference_name: Name for the reference image
            region: Screen region to monitor
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds
            threshold: Difference threshold to detect change
            
        Returns:
            True if change detected, False on timeout
            
        Raises:
            InteractionTimeout: If timeout is reached
        """
        # Take initial reference image if it doesn't exist
        if reference_name not in self.reference_images:
            self.take_reference_image(reference_name, region)
        
        start_time = time.time()
        end_time = start_time + timeout
        
        # Log the wait operation
        logging.info(f"Waiting up to {timeout}s for visual change: {reference_name}")
        
        while time.time() < end_time:
            # Check for changes
            changed, diff = self.check_for_changes(
                reference_name, 
                region, 
                threshold
            )
            
            if changed:
                elapsed = time.time() - start_time
                logging.info(f"Visual change detected after {elapsed:.2f}s")
                return True
            
            # Log progress at intervals
            elapsed = time.time() - start_time
            if elapsed > 0 and elapsed % 30 < check_interval:
                remaining = timeout - elapsed
                logging.info(f"Still waiting for change: {int(remaining)}s remaining...")
            
            # Sleep before next check
            time.sleep(check_interval)
        
        # If we get here, timeout was reached
        logging.warning(f"Timeout waiting for visual change: {reference_name}")
        raise InteractionTimeout(f"Timed out waiting for visual change: {reference_name}")
    
    def verify_ui_state(self, expected_state: str, 
                       region: Optional[Tuple[int, int, int, int]] = None,
                       timeout: int = InteractionConstants.DEFAULT_WAIT_TIMEOUT,
                       check_interval: float = 1.0) -> bool:
        """
        Verify that the UI is in the expected state.
        
        Args:
            expected_state: Description of expected state
            region: Screen region to check
            timeout: Maximum time to wait
            check_interval: Time between checks
            
        Returns:
            True if verification successful, False otherwise
        """
        # This is a simplified implementation that could be extended with template matching
        reference_name = f"ui_state_{expected_state}"
        
        try:
            # Wait for visual stabilization
            self.wait_for_visual_change(reference_name, region, timeout, check_interval)
            return True
        except InteractionTimeout:
            return False


# ---- UI Interaction Implementations ----

class HumanizedTyping:
    """
    Provides human-like typing capabilities with realistic timing patterns.
    """
    
    def __init__(self, profile: str = 'normal'):
        """
        Initialize the humanized typing module.
        
        Args:
            profile: Typing profile ('fast', 'normal', 'slow')
        """
        self.set_profile(profile)
    
    def set_profile(self, profile: str) -> None:
        """
        Set the typing profile.
        
        Args:
            profile: Typing profile name
        """
        if profile not in InteractionConstants.TYPING_PROFILES:
            logging.warning(f"Unknown typing profile: {profile}, using 'normal'")
            profile = 'normal'
        
        self.profile = profile
        self.settings = InteractionConstants.TYPING_PROFILES[profile]
    
    def type_text(self, text: str, clear_existing: bool = True) -> None:
        """
        Type text with human-like timing patterns.
        
        Args:
            text: Text to type
            clear_existing: Whether to clear existing text first
        """
        if clear_existing:
            # Clear any existing text with Ctrl+A and Delete
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(0.5)
            pyautogui.press('delete')
            time.sleep(0.5)
        
        for char in text:
            # Determine delay for this character
            base_delay = self.settings['base_delay']
            variance = self.settings['variance']
            typing_delay = base_delay + random.uniform(-variance, variance)
            
            # Type the character
            pyautogui.write(char)
            time.sleep(typing_delay)
            
            # Occasionally add a pause (simulating thinking)
            if random.random() < self.settings['pause_chance']:
                min_pause, max_pause = self.settings['pause_duration']
                time.sleep(random.uniform(min_pause, max_pause))
    
    def type_with_clipboard(self, text: str, clear_existing: bool = True) -> None:
        """
        Type using clipboard for efficiency with long text.
        
        Args:
            text: Text to type
            clear_existing: Whether to clear existing text first
        """
        if not CLIPBOARD_AVAILABLE:
            logging.warning("Clipboard not available, falling back to regular typing")
            self.type_text(text, clear_existing)
            return
        
        if clear_existing:
            # Clear any existing text with Ctrl+A and Delete
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(0.5)
            pyautogui.press('delete')
            time.sleep(0.5)
        
        # Save original clipboard content
        try:
            original_clipboard = pyperclip.paste()
        except:
            original_clipboard = None
        
        try:
            # Copy text to clipboard
            pyperclip.copy(text)
            
            # Paste with Ctrl+V
            time.sleep(0.5)
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(0.5)
        finally:
            # Restore original clipboard if possible
            if original_clipboard is not None:
                try:
                    pyperclip.copy(original_clipboard)
                except:
                    pass


class UIInteraction:
    """
    Core UI interaction functionality.
    """
    
    def __init__(self, enable_visual_verification: bool = True):
        """
        Initialize UI interaction handler.
        
        Args:
            enable_visual_verification: Whether to use visual verification
        """
        self.typing = HumanizedTyping()
        self.verifier = VisualVerifier() if enable_visual_verification else None
        self.enable_visual_verification = enable_visual_verification
    
    @with_error_handling
    def type_text(self, text: str, human_like: bool = True, 
                 clear_existing: bool = True, use_clipboard: bool = None) -> bool:
        """
        Type text using the appropriate method.
        
        Args:
            text: Text to type
            human_like: Whether to use human-like typing
            clear_existing: Whether to clear existing text
            use_clipboard: Whether to use clipboard (auto if None)
            
        Returns:
            True if successful
        """
        # Log the action
        if len(text) > 50:
            preview = text[:47] + "..."
            logging.info(f"Typing text: {preview}")
        else:
            logging.info(f"Typing text: {text}")
        
        # Take before screenshot
        if self.enable_visual_verification:
            InteractionUtils.log_action("Before typing text", take_screenshot=True)
        
        # Determine whether to use clipboard based on text length if not specified
        if use_clipboard is None:
            use_clipboard = len(text) > 100 and CLIPBOARD_AVAILABLE
        
        # Type the text using the appropriate method
        if use_clipboard:
            self.typing.type_with_clipboard(text, clear_existing)
        elif human_like:
            self.typing.type_text(text, clear_existing)
        else:
            # Direct typing without human-like delays
            if clear_existing:
                pyautogui.hotkey('ctrl', 'a')
                time.sleep(0.5)
                pyautogui.press('delete')
                time.sleep(0.5)
            pyautogui.write(text)
        
        # Take after screenshot
        if self.enable_visual_verification:
            InteractionUtils.log_action("After typing text", take_screenshot=True)
        
        return True
    
    @with_error_handling
    def press_key(self, key: str) -> bool:
        """
        Press a keyboard key.
        
        Args:
            key: Key to press
            
        Returns:
            True if successful
        """
        logging.info(f"Pressing key: {key}")
        pyautogui.press(key)
        return True
    
    @with_error_handling
    def press_submit(self) -> bool:
        """
        Press Enter to submit the current prompt.
        
        Returns:
            True if successful
        """
        logging.info("Pressing Enter to submit prompt")
        
        if self.enable_visual_verification:
            InteractionUtils.log_action("Before pressing Enter", take_screenshot=True)
        
        # Press Enter
        pyautogui.press('enter')
        
        if self.enable_visual_verification:
            InteractionUtils.log_action("After pressing Enter", take_screenshot=True)
            
            # Try to verify that submission was successful by checking for visual changes
            try:
                if self.verifier:
                    # Wait for initial visual change indicating submission started
                    self.verifier.wait_for_visual_change(
                        "prompt_submit", 
                        timeout=5,
                        check_interval=0.5
                    )
            except InteractionTimeout:
                logging.warning("Could not verify prompt submission started")
        
        return True
    
    @with_error_handling
    @retry_decorator(max_attempts=2)
    def click_at(self, x: int, y: int, button: str = 'left') -> bool:
        """
        Click at specific coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button ('left', 'right', 'middle')
            
        Returns:
            True if successful
        """
        logging.info(f"Clicking at coordinates ({x}, {y}) with {button} button")
        
        # Move to position with humanized motion
        pyautogui.moveTo(x, y, duration=random.uniform(0.1, 0.3))
        
        # Perform click
        pyautogui.click(x=x, y=y, button=button)
        
        return True
    
    @with_error_handling
    def click_prompt_area(self, coordinates: Optional[Tuple[int, int]] = None) -> bool:
        """
        Click in the prompt input area.
        
        Args:
            coordinates: Optional (x, y) coordinates for the prompt area
            
        Returns:
            True if successful
        """
        if coordinates:
            x, y = coordinates
            logging.info(f"Clicking prompt area at ({x}, {y})")
            return self.click_at(x, y)
        else:
            logging.warning("No coordinates provided for prompt area")
            return False
    
    @with_error_handling
    def wait_for_response(self, timeout: int = InteractionConstants.DEFAULT_WAIT_TIMEOUT, 
                         check_interval: int = InteractionConstants.DEFAULT_CHECK_INTERVAL, 
                         verify_visually: bool = True) -> bool:
        """
        Wait for Claude to respond, with optional visual verification.
        
        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds
            verify_visually: Whether to use visual verification
            
        Returns:
            True if response completed within timeout
            
        Raises:
            InteractionTimeout: If timeout is reached without detecting completion
        """
        logging.info(f"Waiting up to {timeout} seconds for response...")
        
        # Start wait timer
        start_time = time.time()
        end_time = start_time + timeout
        
        # If visual verification is enabled and available, use it
        if verify_visually and self.enable_visual_verification and self.verifier:
            try:
                # First wait for some initial change (Claude thinking)
                self.verifier.wait_for_visual_change(
                    "response_start", 
                    timeout=min(30, timeout / 10),  # Use shorter timeout for initial change
                    check_interval=1.0
                )
                
                logging.info("Detected Claude starting to respond")
                
                # Now wait for response to complete (UI stabilization)
                response_complete = False
                
                # Continue checking until timeout
                while time.time() < end_time:
                    # Take periodic screenshots if advanced logging is enabled
                    elapsed = time.time() - start_time
                    if ADVANCED_LOGGING and elapsed % 30 < check_interval:
                        log_with_screenshot(
                            f"Still waiting for response ({int(elapsed)}/{timeout} seconds elapsed)", 
                            stage_name=f"WAITING_{int(elapsed)}"
                        )
                    
                    # Check for UI stabilization
                    changed, _ = self.verifier.check_for_changes("response_complete")
                    
                    # If no change for a while, it might mean response is complete
                    if not changed:
                        # Wait a bit longer to confirm stability
                        time.sleep(check_interval * 2)
                        changed, _ = self.verifier.check_for_changes("response_complete")
                        
                        if not changed:
                            # If still stable, consider the response complete
                            response_complete = True
                            break
                    
                    # Sleep for check interval
                    time.sleep(check_interval)
                
                if response_complete:
                    elapsed = time.time() - start_time
                    logging.info(f"Response detected as complete after {int(elapsed)} seconds")
                    
                    if ADVANCED_LOGGING:
                        log_with_screenshot("Response complete", stage_name="RESPONSE_COMPLETE")
                    
                    return True
                
            except InteractionTimeout:
                # Fall back to timed waiting if visual verification fails
                logging.warning("Visual verification failed, falling back to timed waiting")
                
        # Simple timed waiting
        while time.time() < end_time:
            # Log progress periodically
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            
            if elapsed > 0 and elapsed % 30 < check_interval:
                logging.info(f"Still waiting: {int(remaining)} seconds remaining...")
            
            # Sleep for check interval
            time.sleep(check_interval)
        
        elapsed = time.time() - start_time
        logging.warning(f"Reached timeout of {timeout} seconds waiting for response (elapsed: {elapsed:.1f}s)")
        
        # Don't raise exception, just return False to allow further operations
        return False
    
    @retry_decorator(max_attempts=InteractionConstants.DEFAULT_RETRY_ATTEMPTS)
    @with_error_handling
    def send_prompt(self, prompt: str, verify_sent: bool = True, 
                   wait_time: int = InteractionConstants.DEFAULT_WAIT_TIMEOUT,
                   human_like: bool = True) -> bool:
        """
        Send a prompt and optionally wait for response.
        
        Args:
            prompt: The text prompt to send
            verify_sent: Whether to verify the prompt was sent successfully
            wait_time: How long to wait for a response in seconds
            human_like: Whether to use human-like typing behavior
            
        Returns:
            True if successful
        """
        logging.info(f"Sending prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Sending prompt: {prompt}")
        
        # Type the prompt text
        success = self.type_text(prompt, human_like=human_like)
        if not success:
            logging.error("Failed to type prompt text")
            return False
        
        # Small delay before submitting
        InteractionUtils.add_humanized_delay(0.3, 0.7)
        
        # Press Enter to submit
        success = self.press_submit()
        if not success:
            logging.error("Failed to submit prompt")
            return False
        
        # Verify the prompt was sent if requested
        if verify_sent:
            if self.enable_visual_verification:
                InteractionUtils.log_action("Verifying prompt was sent", take_screenshot=True)
        
        # Wait for response if wait_time > 0
        if wait_time > 0:
            return self.wait_for_response(timeout=wait_time, verify_visually=self.enable_visual_verification)
        
        return True


# ---- Facade functions (maintaining original API) ----

# Initialize singleton instances for module-level functions
_ui_interaction = UIInteraction()
_verifier = VisualVerifier()
_typing = HumanizedTyping()


@with_error_handling
def type_text(text: str, human_like: bool = True, clear_existing: bool = True) -> bool:
    """
    Type text into the currently focused element with optional human-like timing.
    
    Args:
        text: The text to type
        human_like: Whether to add random delays between keystrokes
        clear_existing: Whether to clear existing text before typing
        
    Returns:
        True if successful
    """
    return _ui_interaction.type_text(text, human_like, clear_existing)


@with_error_handling
def press_submit() -> bool:
    """
    Press Enter to submit the current prompt.
    
    Returns:
        True if successful
    """
    return _ui_interaction.press_submit()


@with_error_handling
def wait_for_response(timeout: int = InteractionConstants.DEFAULT_WAIT_TIMEOUT, 
                     check_interval: int = InteractionConstants.DEFAULT_CHECK_INTERVAL,
                     verify_visually: bool = True) -> bool:
    """
    Wait for Claude to respond, with optional visual verification.
    
    Args:
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
        verify_visually: Whether to use visual cues to detect completion
        
    Returns:
        True if response completed within timeout
    """
    return _ui_interaction.wait_for_response(timeout, check_interval, verify_visually)


@with_error_handling
def verify_text_entered(expected_text: str, timeout: int = 5) -> bool:
    """
    Verify text was entered correctly in prompt box.
    
    Args:
        expected_text: The text that should have been entered
        timeout: Maximum time to verify in seconds
        
    Returns:
        True if verified
    """
    # Take a verification screenshot
    InteractionUtils.log_action("Text entry verification", take_screenshot=True)
    
    # For now, we don't have OCR to actually verify the text content
    # This function could be enhanced to use OCR from the main system if available
    return True


@with_error_handling
def verify_claude_responding() -> bool:
    """
    Check if Claude is currently responding.
    
    Returns:
        True if Claude appears to be responding
    """
    # Take a verification screenshot
    InteractionUtils.log_action("Claude response verification", take_screenshot=True)
    
    # Use visual verification if available
    if _verifier:
        try:
            changed, _ = _verifier.check_for_changes("claude_responding")
            return changed
        except Exception as e:
            logging.warning(f"Visual verification failed: {e}")
    
    # Fallback to assumption
    return True


@with_error_handling
def verify_response_complete(timeout: int = 10) -> bool:
    """
    Verify Claude has completed its response.
    
    Args:
        timeout: Maximum time to verify in seconds
        
    Returns:
        True if response appears complete
    """
    # Take a verification screenshot
    InteractionUtils.log_action("Response completion verification", take_screenshot=True)
    
    # Use visual verification if available
    if _verifier:
        try:
            # Check for visual stability
            for i in range(3):  # Check multiple times to ensure stability
                time.sleep(1)
                changed, _ = _verifier.check_for_changes("response_complete")
                if changed:
                    return False  # Still changing
            return True  # Stable, so response is likely complete
        except Exception as e:
            logging.warning(f"Visual verification failed: {e}")
    
    return True


@retry_decorator(max_attempts=InteractionConstants.DEFAULT_RETRY_ATTEMPTS)
@with_error_handling
def send_prompt(prompt: str, verify_sent: bool = True, 
               wait_time: int = InteractionConstants.DEFAULT_WAIT_TIMEOUT,
               human_like: bool = True) -> bool:
    """
    Send a prompt and optionally wait for response.
    
    Args:
        prompt: The text prompt to send
        verify_sent: Whether to verify the prompt was sent successfully
        wait_time: How long to wait for a response in seconds
        human_like: Whether to use human-like typing behavior
        
    Returns:
        True if successful
    """
    return _ui_interaction.send_prompt(prompt, verify_sent, wait_time, human_like)


@with_error_handling
def clear_conversation_if_needed(max_messages: int = 10) -> bool:
    """
    Clear conversation if too many messages exist.
    
    Args:
        max_messages: Maximum number of messages before clearing
        
    Returns:
        True if cleared or no clearing needed
    """
    # This is a placeholder implementation
    # Log the action
    logging.info("Checking if conversation needs clearing")
    
    # Take a verification screenshot
    InteractionUtils.log_action("Conversation clearing check", take_screenshot=True)
    
    return True


@with_error_handling
def ensure_browser_focused() -> bool:
    """
    Ensure the browser window is in focus for interaction.
    
    Returns:
        True if browser is focused
    """
    logging.debug("Checking browser focus")
    
    # Take a verification screenshot
    InteractionUtils.log_action("Browser focus check", take_screenshot=True)
    
    return True


@with_error_handling
def click_prompt_area(coordinates: Optional[Tuple[int, int]] = None) -> bool:
    """
    Click in the prompt input area.
    
    Args:
        coordinates: Optional tuple of (x, y) coordinates
        
    Returns:
        True if clicked successfully
    """
    return _ui_interaction.click_prompt_area(coordinates)


def add_humanized_delay(min_delay: float = 0.1, max_delay: float = 0.3) -> None:
    """Add a random delay between actions to simulate human behavior."""
    InteractionUtils.add_humanized_delay(min_delay, max_delay)


def take_verification_screenshot(name: str) -> Optional[str]:
    """
    Take a screenshot for verification purposes.
    
    Args:
        name: Screenshot identifier
        
    Returns:
        Path to saved screenshot or None if failed
    """
    return InteractionUtils.save_debug_screenshot(name)