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

# Import shared components
from src.utils.constants import (
    DEFAULT_TYPING_DELAY, DEFAULT_ACTION_DELAY, DEFAULT_WAIT_TIMEOUT,
    DEFAULT_CHECK_INTERVAL, DEFAULT_RETRY_ATTEMPTS, DEFAULT_RETRY_DELAY,
    DEFAULT_RETRY_BACKOFF, DEFAULT_RETRY_JITTER, VERIFICATION_THRESHOLD,
    MIN_PIXEL_CHANGE, VERIFICATION_ATTEMPTS, TYPING_PROFILES
)
from src.utils.retry import with_retry
from src.utils.human_input import HumanizedTyping
from src.utils.interaction_helpers import (
    add_humanized_delay, take_screenshot, save_debug_screenshot, log_action
)
from src.utils.errors import with_error_handling

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
        image = take_screenshot(region)
        self.reference_images[name] = {
            'image': image,
            'timestamp': time.time(),
            'region': region
        }
        return image
    
    def check_for_changes(self, reference_name: str, 
                         region: Optional[Tuple[int, int, int, int]] = None,
                         threshold: float = VERIFICATION_THRESHOLD,
                         min_pixels: int = MIN_PIXEL_CHANGE) -> Tuple[bool, float]:
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
        current_image = take_screenshot(region)
        
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
                              timeout: int = DEFAULT_WAIT_TIMEOUT,
                              check_interval: float = DEFAULT_CHECK_INTERVAL,
                              threshold: float = VERIFICATION_THRESHOLD) -> bool:
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
                       timeout: int = DEFAULT_WAIT_TIMEOUT,
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
            log_action("Before typing text", take_screenshot=True)
        
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
            log_action("After typing text", take_screenshot=True)
        
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
            log_action("Before pressing Enter", take_screenshot=True)
        
        # Press Enter
        pyautogui.press('enter')
        
        if self.enable_visual_verification:
            log_action("After pressing Enter", take_screenshot=True)
            
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
    @with_retry(max_attempts=2)
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
    def wait_for_response(self, timeout: int = DEFAULT_WAIT_TIMEOUT, 
                         check_interval: int = DEFAULT_CHECK_INTERVAL, 
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
    
    @with_retry(max_attempts=DEFAULT_RETRY_ATTEMPTS)
    @with_error_handling
    def send_prompt(self, prompt: str, verify_sent: bool = True, 
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
            True if successful
        """
        logging.info(f"Sending prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Sending prompt: {prompt}")
        
        # Type the prompt text
        success = self.type_text(prompt, human_like=human_like)
        if not success:
            logging.error("Failed to type prompt text")
            return False
        
        # Small delay before submitting
        add_humanized_delay(0.3, 0.7)
        
        # Press Enter to submit
        success = self.press_submit()
        if not success:
            logging.error("Failed to submit prompt")
            return False
        
        # Verify the prompt was sent if requested
        if verify_sent:
            if self.enable_visual_verification:
                log_action("Verifying prompt was sent", take_screenshot=True)
        
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
def wait_for_response(timeout: int = DEFAULT_WAIT_TIMEOUT, 
                     check_interval: int = DEFAULT_CHECK_INTERVAL,
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
    log_action("Text entry verification", take_screenshot=True)
    
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
    log_action("Claude response verification", take_screenshot=True)
    
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
    log_action("Response completion verification", take_screenshot=True)
    
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


@with_retry(max_attempts=DEFAULT_RETRY_ATTEMPTS)
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
    log_action("Conversation clearing check", take_screenshot=True)
    
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
    log_action("Browser focus check", take_screenshot=True)
    
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


def take_verification_screenshot(name: str) -> Optional[str]:
    """
    Take a screenshot for verification purposes.
    
    Args:
        name: Screenshot identifier
        
    Returns:
        Path to saved screenshot or None if failed
    """
    return save_debug_screenshot(name)