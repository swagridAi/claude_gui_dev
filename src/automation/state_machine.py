from enum import Enum, auto
import logging
import time
import random
import functools
from dataclasses import dataclass
from typing import Optional, Dict, Any
from src.automation.browser import launch_browser, close_browser, refresh_page
from src.automation.recognition import find_element
from src.automation.interaction import click_element, send_text
from src.models.ui_element import UIElement
from src.utils.logging_util import log_with_screenshot
from src.utils.reference_manager import ReferenceImageManager
from src.utils.region_manager import RegionManager


# Inline utility classes for refactoring
class AutomationConstants:
    """Centralized timing constants for automation."""
    BROWSER_INIT_WAIT = 5
    BROWSER_STARTUP_WAIT = 10
    PAGE_RELOAD_WAIT = 5
    BROWSER_CLOSE_RETRY_WAIT = 2
    PROMPT_WAIT_TIME = 300  # Fixed wait time for prompts (5 minutes)
    
    # Default retry settings
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 2
    DEFAULT_MAX_RETRY_DELAY = 30
    DEFAULT_RETRY_JITTER = 0.5
    DEFAULT_RETRY_BACKOFF = 1.5


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = AutomationConstants.DEFAULT_MAX_RETRIES
    retry_delay: float = AutomationConstants.DEFAULT_RETRY_DELAY
    max_retry_delay: float = AutomationConstants.DEFAULT_MAX_RETRY_DELAY
    retry_jitter: float = AutomationConstants.DEFAULT_RETRY_JITTER
    retry_backoff: float = AutomationConstants.DEFAULT_RETRY_BACKOFF


class RetryHandler:
    """Handles retry logic with exponential backoff and jitter."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        current_delay = min(
            self.config.retry_delay * (self.config.retry_backoff ** (attempt - 1)),
            self.config.max_retry_delay
        )
        
        # Add jitter to avoid thundering herd problem
        jitter_factor = 1 + random.uniform(-self.config.retry_jitter, self.config.retry_jitter)
        return current_delay * jitter_factor
    
    def wait(self, attempt: int, failure_type: 'FailureType') -> None:
        """Wait for the calculated delay before retry."""
        delay = self.calculate_delay(attempt)
        logging.info(f"Waiting {delay:.2f} seconds before retry (attempt {attempt}, type: {failure_type})")
        time.sleep(delay)


def with_error_handling(func):
    """Decorator for standardized error handling with screenshots."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            # Log error with screenshot
            log_with_screenshot(
                f"Error in state {self.state.name}: {str(e)}", 
                level=logging.ERROR,
                stage_name=f"{self.state.name}_ERROR"
            )
            self._handle_error(e)
            return None
    return wrapper


class AutomationState(Enum):
    INITIALIZE = auto()
    BROWSER_LAUNCH = auto()
    WAIT_FOR_LOGIN = auto()
    SEND_PROMPTS = auto()
    COMPLETE = auto()
    ERROR = auto()
    RETRY = auto()


class FailureType(Enum):
    """Types of failures that can occur during automation"""
    UI_NOT_FOUND = auto()
    NETWORK_ERROR = auto()
    BROWSER_ERROR = auto()
    UNKNOWN = auto()


class SimpleAutomationMachine:
    def __init__(self, config):
        self.config = config
        self.state = AutomationState.INITIALIZE
        self.prompts = config.get("prompts", [])
        self.current_prompt_index = 0
        self.retry_count = 0
        self.ui_elements = {}
        self.delay_between_prompts = config.get("delay_between_prompts", 3)
        self.failure_type = None
        self.last_error = None
        self.reference_manager = ReferenceImageManager()
        self.region_manager = RegionManager()
        self.detected_window = False
        
        # Initialize retry configuration and handler
        self.retry_config = RetryConfig(
            max_retries=config.get("max_retries", AutomationConstants.DEFAULT_MAX_RETRIES),
            retry_delay=config.get("retry_delay", AutomationConstants.DEFAULT_RETRY_DELAY),
            max_retry_delay=config.get("max_retry_delay", AutomationConstants.DEFAULT_MAX_RETRY_DELAY),
            retry_jitter=config.get("retry_jitter", AutomationConstants.DEFAULT_RETRY_JITTER),
            retry_backoff=config.get("retry_backoff", AutomationConstants.DEFAULT_RETRY_BACKOFF)
        )
        self.retry_handler = RetryHandler(self.retry_config)
    
    def run(self):
        """Run the automation state machine until completion or error."""
        while self.state not in [AutomationState.COMPLETE, AutomationState.ERROR]:
            self._execute_current_state()
        
        # Final log message
        if self.state == AutomationState.COMPLETE:
            logging.info(f"Automation completed successfully. Sent {self.current_prompt_index} prompts.")
        else:
            logging.error(f"Automation stopped with errors after sending {self.current_prompt_index} prompts.")
    
    @with_error_handling
    def _execute_current_state(self):
        """Execute the current state and transition to the next state."""
        original_state = self.state
        
        # Log the state transition with screenshot
        log_with_screenshot(
            f"Entering state: {original_state.name}", 
            level=logging.INFO,
            stage_name=original_state.name
        )
        
        if self.state == AutomationState.INITIALIZE:
            self._handle_initialize()
        elif self.state == AutomationState.BROWSER_LAUNCH:
            self._handle_browser_launch()
        elif self.state == AutomationState.WAIT_FOR_LOGIN:
            self._handle_wait_for_login()
        elif self.state == AutomationState.SEND_PROMPTS:
            self._handle_send_prompts()
        elif self.state == AutomationState.RETRY:
            self._handle_retry()
        
        # Log successful state completion with screenshot
        log_with_screenshot(
            f"Successfully completed state: {original_state.name}", 
            level=logging.INFO,
            stage_name=f"{original_state.name}_COMPLETE"
        )
        
        # Reset retry count on successful state execution (only for non-retry states)
        if original_state != AutomationState.RETRY:
            self.retry_count = 0

    def _handle_initialize(self):
        """Initialize the automation process."""
        # Load UI elements from config
        for element_name, element_config in self.config.get("ui_elements", {}).items():
            self._create_ui_element(element_name, element_config)
        self.state = AutomationState.BROWSER_LAUNCH
    
    def _create_ui_element(self, element_name: str, element_config: Dict[str, Any]) -> None:
        """Create and store a UI element from configuration."""
        self.ui_elements[element_name] = UIElement(
            name=element_name,
            reference_paths=element_config.get("reference_paths", []),
            region=element_config.get("region"),
            relative_region=element_config.get("relative_region"),
            parent=element_config.get("parent"),
            confidence=element_config.get("confidence", 0.8),
            click_coordinates=element_config.get("click_coordinates"),
            use_coordinates_first=element_config.get("use_coordinates_first", True)
        )

    def _handle_browser_launch(self):
        """Launch the browser and navigate to Claude."""
        logging.info("Launching browser")
        
        # Use the configuration object as expected by the browser module
        launch_browser(self.config.get("claude_url"), self.config)
        
        # Wait for browser to launch
        time.sleep(AutomationConstants.BROWSER_INIT_WAIT)
        self._detect_window_position()
        self.state = AutomationState.WAIT_FOR_LOGIN
    
    def _detect_window_position(self) -> None:
        """Try to detect window position and set anchor point."""
        window_rect = self.region_manager.detect_window_position("Claude")
        if window_rect:
            logging.info(f"Detected Claude window at {window_rect}")
            self.region_manager.set_anchor_point("window_top_left", (window_rect[0], window_rect[1]))
            self.detected_window = True
        else:
            logging.warning("Could not detect Claude window position, using full screen")
    
    def _handle_wait_for_login(self):
        """Wait for the user to complete login."""
        logging.info("Waiting for login completion")
        # Skip any login checks and proceed directly to sending prompts
        self.state = AutomationState.SEND_PROMPTS
    
    def _handle_send_prompts(self):
        """Send all prompts in sequence."""
        if self.current_prompt_index >= len(self.prompts):
            logging.info("All prompts have been sent")
            log_with_screenshot("All prompts completed", stage_name="PROMPTS_COMPLETED")
            self.state = AutomationState.COMPLETE
            self.close_browser()
            return
        
        current_prompt = self.prompts[self.current_prompt_index]
        logging.info(f"Sending prompt {self.current_prompt_index + 1}/{len(self.prompts)}: {current_prompt[:50]}...")
        log_with_screenshot(f"Before sending prompt {self.current_prompt_index + 1}", stage_name="BEFORE_PROMPT")
        
        # Determine which prompt box to use (initial or final)
        prompt_element = self._get_prompt_element()
        
        if not prompt_element:
            log_with_screenshot("No prompt element defined", level=logging.ERROR, 
                            stage_name="NO_PROMPT_ELEMENT")
            self.failure_type = FailureType.UI_NOT_FOUND
            raise Exception("No prompt element defined")
        
        # Use coordinates or visual recognition
        if not self._click_prompt_element(prompt_element):
            log_with_screenshot(f"Failed to click {prompt_element.name}", level=logging.ERROR, 
                            stage_name=f"{prompt_element.name.upper()}_CLICK_FAILED")
            self.failure_type = FailureType.UI_NOT_FOUND
            raise Exception(f"Failed to click {prompt_element.name}")
        
        log_with_screenshot(f"After clicking {prompt_element.name}", 
                        stage_name=f"AFTER_{prompt_element.name.upper()}_CLICK")
        
        # Send the prompt text
        send_text(current_prompt)
        log_with_screenshot("After typing prompt", stage_name="AFTER_TYPE_PROMPT")
        
        # Press Enter to send
        from pyautogui import press
        press("enter")
        logging.info("Pressed Enter to send prompt")
        log_with_screenshot("Prompt sent using Enter key", stage_name="PROMPT_SENT_ENTER")
        
        # Wait fixed time after sending prompt
        self._wait_for_response()
        
        # Check for message limit
        if self._check_message_limit():
            self.state = AutomationState.COMPLETE
            return
        
        # Move to next prompt
        self.current_prompt_index += 1
    
    def _get_prompt_element(self) -> Optional[UIElement]:
        """Get the appropriate prompt element based on current prompt index."""
        if self.current_prompt_index == 0:
            logging.info("First prompt - using initial_prompt_box")
            return self.ui_elements.get("Initial_prompt_box")
        else:
            logging.info("Subsequent prompt - using final_prompt_box")
            return self.ui_elements.get("final_prompt_box")
    
    def _click_prompt_element(self, prompt_element: UIElement) -> bool:
        """Click the prompt element using coordinates or visual recognition."""
        if prompt_element.click_coordinates and (prompt_element.use_coordinates_first or 
                                            self.config.get("automation_settings", {}).get("prefer_coordinates", True)):
            from src.automation.interaction import click_at_coordinates
            x, y = prompt_element.click_coordinates
            logging.info(f"Using direct coordinates for {prompt_element.name}: ({x}, {y})")
            return click_at_coordinates(x, y, element_name=prompt_element.name)
        else:
            logging.info(f"Using visual recognition to find {prompt_element.name}")
            return click_element(prompt_element)
    
    def _wait_for_response(self) -> None:
        """Wait for Claude to respond with the configured delay."""
        wait_time = AutomationConstants.PROMPT_WAIT_TIME
        logging.info(f"Waiting {wait_time} seconds (5 minutes) after sending prompt...")
        
        # Log progress during the wait time
        start_time = time.time()
        while time.time() - start_time < wait_time:
            time.sleep(30)  # Check every 30 seconds
            waited_so_far = time.time() - start_time
            remaining = wait_time - waited_so_far
            if remaining > 0:
                logging.info(f"Still waiting: {int(remaining)} seconds remaining in 5-minute wait period...")
        
        logging.info("Completed mandatory 5-minute wait period after sending prompt")
        log_with_screenshot("Completed 5-minute wait period", stage_name="WAIT_COMPLETED")
    
    def _check_message_limit(self) -> bool:
        """Check if message limit has been reached."""
        limit_element = self.ui_elements.get("limit_reached")
        if limit_element:
            # Use visual recognition for limit checking
            limit_reached = find_element(limit_element)
            if limit_reached:
                logging.warning("Message limit reached, cannot send more prompts")
                log_with_screenshot("Message limit reached", stage_name="LIMIT_REACHED", region=limit_reached)
                return True
        return False

    def _handle_error(self, error):
        """Handle errors and decide whether to retry."""
        logging.error(f"Error in state {self.state}: {error}")
        
        # Analyze error and set failure type if not already set
        if not self.failure_type:
            self.failure_type = self._classify_error(error)
        
        self.last_error = str(error)
        self.retry_count += 1
        
        # Try refreshing references for UI not found errors
        if self.failure_type == FailureType.UI_NOT_FOUND:
            self._handle_ui_not_found_error(error)
        
        if self.retry_count <= self.retry_config.max_retries:
            logging.info(f"Retry {self.retry_count}/{self.retry_config.max_retries}")
            self.state = AutomationState.RETRY
        else:
            logging.error("Max retries reached, stopping automation")
            self.state = AutomationState.ERROR
    
    def _handle_ui_not_found_error(self, error: Exception) -> None:
        """Handle UI not found errors by refreshing references."""
        element_name = self._extract_element_name_from_error(str(error))
        if element_name and element_name in self.ui_elements:
            logging.info(f"Attempting to refresh reference for {element_name}")
            self.reference_manager.update_stale_references(
                self.ui_elements[element_name], 
                self.config
            )
    
    def set_preserve_config(self, preserve):
        """Set whether to preserve configuration when saving."""
        self.preserve_config = preserve
        logging.debug(f"Set preserve_config to {preserve}")

    def _handle_retry(self):
        """Handle the retry logic based on the failure type."""
        logging.info(f"Handling retry attempt {self.retry_count} for {self.failure_type}")
        
        # Use the retry handler for standardized delay calculation
        self.retry_handler.wait(self.retry_count, self.failure_type)
        
        # Apply recovery strategy based on failure type
        self._apply_recovery_strategy()
        
        # Reset failure type
        self.failure_type = None
        
        # Go back to sending prompts
        self.state = AutomationState.SEND_PROMPTS
    
    def _apply_recovery_strategy(self) -> None:
        """Apply the appropriate recovery strategy based on failure type."""
        if self.failure_type == FailureType.UI_NOT_FOUND:
            self._recover_from_ui_not_found()
        elif self.failure_type == FailureType.NETWORK_ERROR:
            self._recover_from_network_error()
        elif self.failure_type == FailureType.BROWSER_ERROR:
            self._recover_from_browser_error()
        else:
            self._recover_from_unknown_error()
    
    def _recover_from_ui_not_found(self):
        """Recovery strategy for UI element not found."""
        logging.info("Recovering from UI element not found...")
        
        # Try refreshing the page
        logging.info("Refreshing page...")
        refresh_page()
        time.sleep(AutomationConstants.PAGE_RELOAD_WAIT)
        
        # Check if we're still logged in
        logged_in_element = find_element(self.ui_elements["prompt_box"])
        if not logged_in_element:
            logging.warning("Not logged in after refresh, waiting for login...")
            self.state = AutomationState.WAIT_FOR_LOGIN
    
    def _recover_from_network_error(self):
        """Recovery strategy for network errors."""
        logging.info("Recovering from network error...")
        
        # Try refreshing the page with longer wait
        logging.info("Refreshing page...")
        refresh_page()
        time.sleep(7)  # Wait longer for network issues
    
    def _recover_from_browser_error(self):
        """Recovery strategy for browser errors."""
        logging.info("Recovering from browser error...")
        
        # Close and relaunch browser
        self.close_browser()
        time.sleep(AutomationConstants.BROWSER_CLOSE_RETRY_WAIT)
        logging.info("Relaunching browser...")
        launch_browser(self.config.get("claude_url"))
        time.sleep(AutomationConstants.BROWSER_INIT_WAIT)
        
        # Wait for login
        self.state = AutomationState.WAIT_FOR_LOGIN
    
    def _recover_from_unknown_error(self):
        """Recovery strategy for unknown errors."""
        logging.info("Recovering from unknown error...")
        
        # Try a page refresh first
        refresh_page()
        time.sleep(AutomationConstants.PAGE_RELOAD_WAIT)
        
        # If multiple failures, restart the browser
        if self.retry_count > 2:
            logging.info("Multiple failures, restarting browser...")
            self.close_browser()
            time.sleep(AutomationConstants.BROWSER_CLOSE_RETRY_WAIT)
            launch_browser(self.config.get("claude_url"))
            time.sleep(AutomationConstants.BROWSER_INIT_WAIT)
            self.state = AutomationState.WAIT_FOR_LOGIN
    
    def _classify_error(self, error):
        """Classify the type of error based on the exception."""
        error_str = str(error).lower()
        
        if "not found" in error_str or "element" in error_str:
            return FailureType.UI_NOT_FOUND
        elif "network" in error_str or "connection" in error_str or "timeout" in error_str:
            return FailureType.NETWORK_ERROR
        elif "browser" in error_str or "chrome" in error_str or "crashed" in error_str:
            return FailureType.BROWSER_ERROR
        else:
            return FailureType.UNKNOWN
        
    def _extract_element_name_from_error(self, error_str):
        """Extract element name from error message."""
        if "not found" in error_str.lower():
            # Common error patterns
            patterns = [
                r"(\w+) not found",
                r"Could not find element (\w+)",
                r"Element (\w+) not found"
            ]
            
            import re
            for pattern in patterns:
                match = re.search(pattern, error_str)
                if match:
                    return match.group(1)
        
        return None
    
    def cleanup(self):
        """Clean up resources before exit."""
        logging.info("Cleaning up resources")
        self.close_browser()
        
    def close_browser(self):
        """Close browser with verification and retry."""
        logging.info("Closing browser...")
        
        # First attempt
        if close_browser():
            logging.info("Browser closed successfully")
            return True
            
        # Retry if first attempt fails
        logging.warning("First attempt to close browser failed, retrying...")
        time.sleep(AutomationConstants.BROWSER_CLOSE_RETRY_WAIT)
        
        # Second attempt with force
        from pyautogui import hotkey
        try:
            # Try to use Alt+F4 to force close
            hotkey('alt', 'f4')
            time.sleep(1)
            
            # One more try with browser close function
            if close_browser():
                logging.info("Browser closed successfully on second attempt")
                return True
        except Exception as e:
            logging.error(f"Error during forced browser close: {e}")
        
        logging.warning("Could not verify browser closure")