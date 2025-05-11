from enum import Enum, auto
import logging
import time
import random
from src.automation.browser import launch_browser, close_browser, refresh_page
from src.automation.recognition import find_element
from src.automation.interaction import click_element, send_text
from src.models.ui_element import UIElement
from src.utils.logging_util import log_with_screenshot
from src.utils.reference_manager import ReferenceImageManager
from src.utils.region_manager import RegionManager

class AutomationState(Enum):
    INITIALIZE = auto()
    BROWSER_LAUNCH = auto()
    WAIT_FOR_LOGIN = auto()
    SEND_PROMPTS = auto()
    COMPLETE = auto()
    ERROR = auto()
    RETRY = auto()  # New state for handling retries

class FailureType(Enum):
    """Types of failures that can occur during automation"""
    UI_NOT_FOUND = auto()     # UI element not found
    NETWORK_ERROR = auto()    # Network or connectivity issue
    BROWSER_ERROR = auto()    # Browser crashed or not responding
    UNKNOWN = auto()          # Unknown error type

class SimpleAutomationMachine:
    def __init__(self, config):
        self.config = config
        self.state = AutomationState.INITIALIZE
        self.prompts = config.get("prompts", [])
        self.current_prompt_index = 0
        self.max_retries = config.get("max_retries", 3)
        self.retry_count = 0
        self.ui_elements = {}
        self.delay_between_prompts = config.get("delay_between_prompts", 3)
        self.failure_type = None
        self.last_error = None
        self.retry_delay = config.get("retry_delay", 2)
        self.max_retry_delay = config.get("max_retry_delay", 30)
        self.retry_jitter = config.get("retry_jitter", 0.5)  # Random jitter percentage
        self.retry_backoff = config.get("retry_backoff", 1.5)  # Exponential backoff multiplier
        self.reference_manager = ReferenceImageManager()
        # Initialize region manager
        self.region_manager = RegionManager()
        
        # Add a new stage for detecting window positioning
        self.detected_window = False
    
    def run(self):
        """Run the automation state machine until completion or error."""
        while self.state not in [AutomationState.COMPLETE, AutomationState.ERROR]:
            self._execute_current_state()
        
        # Final log message
        if self.state == AutomationState.COMPLETE:
            logging.info(f"Automation completed successfully. Sent {self.current_prompt_index} prompts.")
        else:
            logging.error(f"Automation stopped with errors after sending {self.current_prompt_index} prompts.")
    
    def _execute_current_state(self):
        """Execute the current state and transition to the next state."""
        try:
            # Store the original state for logging
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
            
            # Log successful state completion with screenshot using the ORIGINAL state
            log_with_screenshot(
                f"Successfully completed state: {original_state.name}", 
                level=logging.INFO,
                stage_name=f"{original_state.name}_COMPLETE"
            )
            
            # Reset retry count on successful state execution (only for non-retry states)
            if original_state != AutomationState.RETRY:
                self.retry_count = 0
                
        except Exception as e:
            # Log error with screenshot
            log_with_screenshot(
                f"Error in state {self.state.name}: {str(e)}", 
                level=logging.ERROR,
                stage_name=f"{self.state.name}_ERROR"
            )
            self._handle_error(e)

    def _handle_initialize(self):
        """Initialize the automation process."""
        # Load UI elements from config
        for element_name, element_config in self.config.get("ui_elements", {}).items():
            # Support both relative and absolute regions
            region = element_config.get("region")
            relative_region = element_config.get("relative_region")
            parent = element_config.get("parent")
            
            self.ui_elements[element_name] = UIElement(
                name=element_name,
                reference_paths=element_config.get("reference_paths", []),
                region=region,
                relative_region=relative_region,
                parent=parent,
                confidence=element_config.get("confidence", 0.8),
                click_coordinates=element_config.get("click_coordinates"),  # MISSING
                use_coordinates_first=element_config.get("use_coordinates_first", True)  # MISSING
            )
        self.state = AutomationState.BROWSER_LAUNCH

    def _handle_browser_launch(self):
        """Launch the browser and navigate to Claude."""
        logging.info("Launching browser")
        
        # Fix: Pass the entire config object as the second parameter
        launch_browser(self.config.get("claude_url"), self.config)
        
        # Wait for browser to launch and detect window position
        time.sleep(5)
        
        # Try to detect window position
        window_rect = self.region_manager.detect_window_position("Claude")
        if window_rect:
            logging.info(f"Detected Claude window at {window_rect}")
            # Set an anchor point at the top-left corner of the window
            self.region_manager.set_anchor_point("window_top_left", (window_rect[0], window_rect[1]))
            self.detected_window = True
        else:
            logging.warning("Could not detect Claude window position, using full screen")
                
        self.state = AutomationState.WAIT_FOR_LOGIN
    
    def _handle_wait_for_login(self):
        """Wait for the user to complete login."""
        logging.info("Waiting for login completion")
        # Skip any login checks and proceed directly to sending prompts
        self.state = AutomationState.SEND_PROMPTS
        """
        # Check if already logged in
        logged_in_element = find_element(self.ui_elements["prompt_box"])
        
        if logged_in_element:
            logging.info("Already logged in")
            self.state = AutomationState.SEND_PROMPTS
        else:
            input("Please complete login/CAPTCHA and press Enter to continue...")
            self.state = AutomationState.SEND_PROMPTS
        """
        return
    
    def _handle_send_prompts(self):
        """Send all prompts in sequence."""
        if self.current_prompt_index >= len(self.prompts):
            logging.info("All prompts have been sent")
            log_with_screenshot("All prompts completed", stage_name="PROMPTS_COMPLETED")
            self.state = AutomationState.COMPLETE
            # Close browser when all prompts are sent
            self.close_browser()
            return
        
        current_prompt = self.prompts[self.current_prompt_index]
        logging.info(f"Sending prompt {self.current_prompt_index + 1}/{len(self.prompts)}: {current_prompt[:50]}...")
        log_with_screenshot(f"Before sending prompt {self.current_prompt_index + 1}", stage_name="BEFORE_PROMPT")
        
        try:
            # Determine which prompt box to use (initial or final)
            if self.current_prompt_index == 0:
                logging.info("First prompt - using initial_prompt_box")
                prompt_element = self.ui_elements.get("Initial_prompt_box")
                prompt_element_name = "Initial_prompt_box"
            else:
                logging.info(f"Subsequent prompt - using final_prompt_box")
                prompt_element = self.ui_elements.get("final_prompt_box")
                prompt_element_name = "final_prompt_box"
            
            # Check if the required element exists
            if not prompt_element:
                log_with_screenshot(f"{prompt_element_name} element not defined", level=logging.ERROR, 
                                stage_name=f"{prompt_element_name.upper()}_NOT_FOUND")
                self.failure_type = FailureType.UI_NOT_FOUND
                raise Exception(f"{prompt_element_name} element not defined")
            
            # DIRECT COORDINATES APPROACH: Use coordinates directly if available
            if prompt_element.click_coordinates and (prompt_element.use_coordinates_first or 
                                                self.config.get("automation_settings", {}).get("prefer_coordinates", True)):
                from src.automation.interaction import click_at_coordinates
                x, y = prompt_element.click_coordinates
                logging.info(f"Using direct coordinates for {prompt_element_name}: ({x}, {y})")
                success = click_at_coordinates(x, y, element_name=prompt_element_name)
                
                if not success:
                    logging.warning(f"Direct coordinate click failed, falling back to visual recognition")
                    # Fall back to visual recognition if direct click fails
                    success = click_element(prompt_element)
            else:
                # VISUAL RECOGNITION APPROACH: Use traditional element finding if no coordinates
                logging.info(f"Using visual recognition to find {prompt_element_name}")
                success = click_element(prompt_element)
            
            # Check if the click was successful
            if not success:
                log_with_screenshot(f"Failed to click {prompt_element_name}", level=logging.ERROR, 
                                stage_name=f"{prompt_element_name.upper()}_CLICK_FAILED")
                self.failure_type = FailureType.UI_NOT_FOUND
                raise Exception(f"Failed to click {prompt_element_name}")
            
            log_with_screenshot(f"After clicking {prompt_element_name}", 
                            stage_name=f"AFTER_{prompt_element_name.upper()}_CLICK")
            
            # Send the prompt text
            send_text(current_prompt)
            log_with_screenshot("After typing prompt", stage_name="AFTER_TYPE_PROMPT")
            
            # Press Enter instead of clicking send button
            from pyautogui import press
            press("enter")
            logging.info("Pressed Enter to send prompt")
            log_with_screenshot("Prompt sent using Enter key", stage_name="PROMPT_SENT_ENTER")
            
            # Wait fixed 5 minutes (300 seconds) after sending prompt
            fixed_wait_time = 300  # 5 minutes in seconds
            logging.info(f"Waiting {fixed_wait_time} seconds (5 minutes) after sending prompt...")
            
            # Log progress during the fixed wait time at 30-second intervals
            start_time = time.time()
            while time.time() - start_time < fixed_wait_time:
                time.sleep(30)  # Check every 30 seconds
                waited_so_far = time.time() - start_time
                remaining = fixed_wait_time - waited_so_far
                if remaining > 0:
                    logging.info(f"Still waiting: {int(remaining)} seconds remaining in 5-minute wait period...")
            
            logging.info("Completed mandatory 5-minute wait period after sending prompt")
            log_with_screenshot("Completed 5-minute wait period", stage_name="WAIT_COMPLETED")
            
            # Check for message limit reached notification
            limit_element = self.ui_elements.get("limit_reached")
            if limit_element:
                # For limit checking, prefer coordinates first is less important than accuracy
                # So we'll use visual recognition here
                limit_reached = find_element(limit_element)
                if limit_reached:
                    logging.warning("Message limit reached, cannot send more prompts")
                    log_with_screenshot("Message limit reached", stage_name="LIMIT_REACHED", region=limit_reached)
                    self.state = AutomationState.COMPLETE
                    return
            
            # Move to next prompt
            self.current_prompt_index += 1
            
        except Exception as e:
            logging.error(f"Error sending prompt: {e}")
            # Set failure type if not already set
            if not self.failure_type:
                self.failure_type = FailureType.UNKNOWN
            self.last_error = str(e)
            raise

    def _handle_error(self, error):
        """Handle errors and decide whether to retry."""
        logging.error(f"Error in state {self.state}: {error}")
        log_with_screenshot(f"Error: {error}")
        
        # Analyze error and set failure type if not already set
        if not self.failure_type:
            self.failure_type = self._classify_error(error)
        
        self.last_error = str(error)
        self.retry_count += 1
        
        # If the failure is due to UI element not found, try refreshing references
        if self.failure_type == FailureType.UI_NOT_FOUND:
            element_name = self._extract_element_name_from_error(str(error))
            if element_name and element_name in self.ui_elements:
                logging.info(f"Attempting to refresh reference for {element_name}")
                self.reference_manager.update_stale_references(
                    self.ui_elements[element_name], 
                    self.config
                )
        
        if self.retry_count <= self.max_retries:
            logging.info(f"Retry {self.retry_count}/{self.max_retries}")
            self.state = AutomationState.RETRY
        else:
            logging.error(f"Max retries reached, stopping automation")
            self.state = AutomationState.ERROR
    
    def set_preserve_config(self, preserve):
        """
        Set whether to preserve configuration when saving.
        
        Args:
            preserve: Boolean flag indicating whether to preserve config
        """
        self.preserve_config = preserve
        logging.debug(f"Set preserve_config to {preserve}")

    def _handle_retry(self):
        """Handle the retry logic based on the failure type."""
        logging.info(f"Handling retry attempt {self.retry_count} for {self.failure_type}")
        
        # Calculate exponential backoff with jitter
        current_delay = min(
            self.retry_delay * (self.retry_backoff ** (self.retry_count - 1)),
            self.max_retry_delay
        )
        
        # Add some random jitter to avoid thundering herd problem
        jitter_factor = 1 + random.uniform(-self.retry_jitter, self.retry_jitter)
        delay = current_delay * jitter_factor
        
        logging.info(f"Waiting {delay:.2f} seconds before retry...")
        time.sleep(delay)
        
        # Apply recovery strategy based on failure type
        if self.failure_type == FailureType.UI_NOT_FOUND:
            self._recover_from_ui_not_found()
        elif self.failure_type == FailureType.NETWORK_ERROR:
            self._recover_from_network_error()
        elif self.failure_type == FailureType.BROWSER_ERROR:
            self._recover_from_browser_error()
        else:
            self._recover_from_unknown_error()
        
        # Reset failure type
        self.failure_type = None
        
        # Go back to sending prompts
        self.state = AutomationState.SEND_PROMPTS
    
    def _recover_from_ui_not_found(self):
        """Recovery strategy for UI element not found."""
        logging.info("Recovering from UI element not found...")
        
        # Try refreshing the page
        logging.info("Refreshing page...")
        refresh_page()
        time.sleep(5)  # Wait for page to reload
        
        # Check if we're still logged in
        logged_in_element = find_element(self.ui_elements["prompt_box"])
        if not logged_in_element:
            logging.warning("Not logged in after refresh, waiting for login...")
            self.state = AutomationState.WAIT_FOR_LOGIN
    
    def _recover_from_network_error(self):
        """Recovery strategy for network errors."""
        logging.info("Recovering from network error...")
        
        # Try refreshing the page
        logging.info("Refreshing page...")
        refresh_page()
        time.sleep(7)  # Wait longer for network issues
    
    def _recover_from_browser_error(self):
        """Recovery strategy for browser errors."""
        logging.info("Recovering from browser error...")
        
        # Close and relaunch browser
        self.close_browser()
        time.sleep(2)
        logging.info("Relaunching browser...")
        launch_browser(self.config.get("claude_url"))
        time.sleep(5)
        
        # Wait for login
        self.state = AutomationState.WAIT_FOR_LOGIN
    
    def _recover_from_unknown_error(self):
        """Recovery strategy for unknown errors."""
        logging.info("Recovering from unknown error...")
        
        # Try a page refresh first
        refresh_page()
        time.sleep(5)
        
        # If that doesn't work after a couple of tries, restart the browser
        if self.retry_count > 2:
            logging.info("Multiple failures, restarting browser...")
            self.close_browser()
            time.sleep(2)
            launch_browser(self.config.get("claude_url"))
            time.sleep(5)
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
        time.sleep(2)
        
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