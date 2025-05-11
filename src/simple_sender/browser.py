#!/usr/bin/env python3
"""
Browser management module for Claude automation.

Provides browser operations needed for Claude automation with support for
browser pooling, adaptive timing, and standardized error recovery.
"""

import os
import time
import logging
import subprocess
import platform
from pathlib import Path
import random
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import uuid
from functools import wraps

# Import pyautogui for screenshots and basic interaction
import pyautogui

# Try to import from main automation, use simplified versions if not available
try:
    from src.automation.browser import get_chrome_path as main_get_chrome_path
    from src.automation.browser import close_browser as main_close_browser
    MAIN_IMPORTS_AVAILABLE = True
except ImportError:
    MAIN_IMPORTS_AVAILABLE = False


# Define exception hierarchy
class BrowserError(Exception):
    """Base exception for browser-related errors."""
    pass


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


# Utility decorator for retrying operations
def retry(max_attempts=3, initial_delay=1, backoff_factor=2, jitter=0.1,
         error_types=(BrowserError,)):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each failure
        jitter: Random factor to add to delay (0-1)
        error_types: Tuple of exception types to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            operation_name = func.__name__
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except error_types as e:
                    last_exception = e
                    if attempt < max_attempts:
                        # Add jitter to delay
                        jitter_amount = delay * jitter
                        actual_delay = delay + random.uniform(-jitter_amount, jitter_amount)
                        logging.warning(f"Attempt {attempt}/{max_attempts} for '{operation_name}' failed, "
                                        f"retrying in {actual_delay:.2f}s: {e}")
                        time.sleep(actual_delay)
                        # Increase delay for next attempt
                        delay *= backoff_factor
            
            # If we get here, all attempts failed
            logging.error(f"All {max_attempts} attempts for '{operation_name}' failed: {last_exception}")
            raise last_exception
        return wrapper
    return decorator


# Abstract interface for browser implementations
class BrowserInterface(ABC):
    """Abstract interface defining operations for browser automation."""
    
    @abstractmethod
    def launch(self, url: str, profile_dir: Optional[str] = None) -> bool:
        """Launch browser with specified URL and profile."""
        pass
    
    @abstractmethod
    def close(self) -> bool:
        """Close the browser and clean up resources."""
        pass
    
    @abstractmethod
    def is_closed(self) -> bool:
        """Check if the browser is closed."""
        pass
    
    @abstractmethod
    def is_page_loaded(self, timeout: float = 5) -> bool:
        """Check if the current page is loaded."""
        pass
    
    @abstractmethod
    def wait_for_page_load(self, timeout: float = 15) -> bool:
        """Wait for the page to load, with timeout."""
        pass
    
    @abstractmethod
    def refresh(self) -> bool:
        """Refresh the current page."""
        pass
    
    @abstractmethod
    def navigate(self, url: str) -> bool:
        """Navigate to a new URL."""
        pass
    
    @abstractmethod
    def get_screenshot(self) -> Optional[Any]:
        """Take a screenshot of the current browser window."""
        pass
    
    @abstractmethod
    def clear_input(self) -> bool:
        """Clear any active input field."""
        pass
    
    @abstractmethod
    def type_character(self, char: str) -> bool:
        """Type a single character."""
        pass
    
    @abstractmethod
    def send_prompt(self) -> bool:
        """Submit the current prompt (e.g., press Enter)."""
        pass


class ChromeBrowser(BrowserInterface):
    """Chrome browser implementation for Claude automation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Chrome browser with optional configuration.
        
        Args:
            config: Dictionary containing browser configuration or None
        """
        self.config = config or {}
        self.process = None
        self._chrome_path = None
        self.profile_dir = None
        self.last_known_url = None
        self.startup_delay = self.config.get("browser_launch_wait", 10)
        
        # Browser state tracking
        self.id = str(uuid.uuid4())[:8]  # Unique ID for this instance
        self.state = "initialized"  # initialized, running, closed, error
        self.launch_time = None
        self.error = None
        
        # Screenshot comparison
        self._last_screenshot = None
        self._screenshot_history = []  # For debugging
    
    @property
    def chrome_path(self) -> str:
        """
        Get Chrome executable path, finding it if needed.
        
        Returns:
            Path to Chrome executable
        """
        if not self._chrome_path:
            self._chrome_path = self._find_chrome_path()
            if not self._chrome_path:
                raise BrowserLaunchError("Could not find Chrome browser")
        return self._chrome_path
    
    @chrome_path.setter
    def chrome_path(self, path: str):
        """Set Chrome executable path explicitly."""
        if not os.path.exists(path):
            raise BrowserLaunchError(f"Chrome executable not found at: {path}")
        self._chrome_path = path
    
    def _find_chrome_path(self) -> Optional[str]:
        """
        Find Chrome executable path.
        
        Returns:
            Path to Chrome or None if not found
        """
        # If config has explicit path, use it
        if self.config.get("chrome_path"):
            if os.path.exists(self.config["chrome_path"]):
                return self.config["chrome_path"]
        
        # Try to use main automation's function if available
        if MAIN_IMPORTS_AVAILABLE:
            try:
                chrome_path = main_get_chrome_path()
                if chrome_path and os.path.exists(chrome_path):
                    return chrome_path
            except Exception as e:
                logging.warning(f"Error using main automation's get_chrome_path: {e}")
        
        # Fall back to platform-specific detection
        return self._detect_chrome_path_by_platform()
    
    def _detect_chrome_path_by_platform(self) -> Optional[str]:
        """Detect Chrome path based on operating system."""
        system = platform.system()
        
        if system == "Windows":
            paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Google\Chrome\Application\chrome.exe")
            ]
        elif system == "Darwin":  # macOS
            paths = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                os.path.expanduser("~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
            ]
        else:  # Linux and others
            paths = [
                "/usr/bin/google-chrome",
                "/usr/bin/chromium-browser",
                "/usr/bin/chromium"
            ]
        
        # Return the first path that exists
        for path in paths:
            if os.path.exists(path):
                return path
        
        return None
    
    @retry(max_attempts=3, error_types=(BrowserLaunchError,))
    def launch(self, url: str, profile_dir: Optional[str] = None) -> bool:
        """
        Launch Chrome browser with the specified URL and profile.
        
        Args:
            url: URL to navigate to
            profile_dir: Browser profile directory or None for default
            
        Returns:
            True if browser launched successfully
            
        Raises:
            BrowserLaunchError: If browser fails to launch
        """
        if self.state == "running":
            logging.warning("Browser is already running. Use navigate() for a new URL.")
            return True
        
        # Set profile directory
        self.profile_dir = profile_dir or self.config.get("browser_profile")
        if not self.profile_dir:
            self.profile_dir = os.path.join(os.path.expanduser("~"), "ClaudeProfile")
        
        # Ensure profile directory exists
        os.makedirs(self.profile_dir, exist_ok=True)
        
        # Prepare launch command
        cmd = [
            self.chrome_path,
            f"--user-data-dir={self.profile_dir}",
            "--start-maximized",
            "--disable-extensions",
            url
        ]
        
        try:
            logging.info(f"Launching Chrome browser (ID: {self.id})")
            logging.debug(f"Using Chrome: {self.chrome_path}")
            logging.debug(f"Using profile: {self.profile_dir}")
            logging.debug(f"Navigating to: {url}")
            
            # Launch browser process
            self.process = subprocess.Popen(cmd)
            
            # Store the URL
            self.last_known_url = url
            
            # Check if process started successfully
            if self.process.poll() is not None:
                error_msg = f"Browser process exited with code {self.process.returncode}"
                self.state = "error"
                self.error = error_msg
                logging.error(error_msg)
                raise BrowserLaunchError(error_msg)
            
            # Update state
            self.state = "running"
            self.launch_time = time.time()
            
            # Wait for browser to initialize
            time.sleep(self.startup_delay)
            
            # Take screenshot after launch
            self._update_screenshot()
            
            logging.info(f"Chrome browser launched successfully (ID: {self.id})")
            return True
            
        except Exception as e:
            self.state = "error"
            self.error = str(e)
            logging.error(f"Failed to launch browser: {e}")
            raise BrowserLaunchError(f"Failed to launch browser: {e}")
    
    def close(self) -> bool:
        """
        Close the browser and clean up resources.
        
        Returns:
            True if browser closed successfully
        
        Raises:
            BrowserClosingError: If browser cannot be closed properly
        """
        # Skip if already closed
        if self.state == "closed":
            return True
        
        logging.info(f"Closing browser (ID: {self.id})")
        
        try:
            success = False
            
            if MAIN_IMPORTS_AVAILABLE:
                # Try main automation's close_browser function
                try:
                    success = main_close_browser()
                except Exception as e:
                    logging.warning(f"Error using main automation's close_browser: {e}")
            
            if not success:
                # Use platform-specific browser closing
                system = platform.system()
                
                if system == "Windows":
                    subprocess.run(["taskkill", "/f", "/im", "chrome.exe"], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
                else:
                    subprocess.run(["pkill", "-f", "chrome"], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
                
                # Allow time for browsers to close
                time.sleep(2)
                
                # Verify browser is closed
                success = self._verify_browser_closed()
            
            if success:
                logging.info(f"Browser closed successfully (ID: {self.id})")
                self.state = "closed"
            else:
                logging.warning(f"Browser may not have closed properly (ID: {self.id})")
                self.state = "error"
                self.error = "Failed to close browser"
                raise BrowserClosingError("Failed to close browser process")
                
            return success
            
        except Exception as e:
            self.state = "error"
            self.error = str(e)
            logging.error(f"Error closing browser: {e}")
            raise BrowserClosingError(f"Error closing browser: {e}")
    
    def is_closed(self) -> bool:
        """
        Check if the browser is closed.
        
        Returns:
            True if browser is closed, False if still running
        """
        if self.state == "closed":
            return True
            
        # Check process if we have it
        if self.process and self.process.poll() is None:
            return False
            
        return self._verify_browser_closed()
    
    def _verify_browser_closed(self) -> bool:
        """Check if Chrome is no longer running."""
        try:
            system = platform.system()
            
            if system == "Windows":
                # Check for chrome.exe process on Windows
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq chrome.exe"], 
                    capture_output=True, 
                    text=True
                )
                return "chrome.exe" not in result.stdout
            else:
                # Check for chrome process on Unix systems
                result = subprocess.run(
                    ["pgrep", "-f", "chrome"], 
                    capture_output=True, 
                    text=True
                )
                return result.stdout.strip() == ""
        
        except Exception as e:
            logging.error(f"Error verifying browser closure: {e}")
            # If verification fails, assume browser may still be running
            return False
    
    def _update_screenshot(self) -> bool:
        """
        Take a new screenshot and update the internal state.
        
        Returns:
            True if screenshot was taken successfully
        """
        try:
            # Take screenshot
            screenshot = pyautogui.screenshot()
            
            # Keep history (limited to last 5)
            if self._last_screenshot:
                self._screenshot_history.append(self._last_screenshot)
                if len(self._screenshot_history) > 5:
                    self._screenshot_history.pop(0)
            
            # Update current screenshot
            self._last_screenshot = screenshot
            return True
            
        except Exception as e:
            logging.error(f"Error taking screenshot: {e}")
            return False
    
    def get_screenshot(self) -> Optional[Any]:
        """
        Take a screenshot of the current browser window.
        
        Returns:
            PIL Image object or None if failed
        """
        try:
            screenshot = pyautogui.screenshot()
            return screenshot
        except Exception as e:
            logging.error(f"Error taking screenshot: {e}")
            return None
    
    def is_page_loaded(self, timeout: float = 5) -> bool:
        """
        Check if the current page is loaded by detecting visual changes.
        
        Args:
            timeout: Maximum time to wait for check
            
        Returns:
            True if page appears to be loaded
        """
        if not self._last_screenshot:
            self._update_screenshot()
            return False
            
        # Take current screenshot
        current_screenshot = self.get_screenshot()
        if not current_screenshot:
            return False
            
        try:
            # Compare with last screenshot
            import numpy as np
            
            # Convert to numpy arrays
            last_data = np.array(self._last_screenshot)
            current_data = np.array(current_screenshot)
            
            # Get screen dimensions
            height, width = current_data.shape[:2]
            
            # Define region to check (center of screen)
            center_region = (
                width // 4,      # x start (1/4 from left)
                height // 4,     # y start (1/4 from top)
                width // 2,      # width (half of screen width)
                height // 2      # height (half of screen height)
            )
            
            # Create cropped versions for center region
            x1, y1, w, h = center_region
            x2, y2 = x1 + w, y1 + h
            
            last_crop = last_data[y1:y2, x1:x2]
            current_crop = current_data[y1:y2, x1:x2]
            
            # Calculate difference
            if last_crop.shape == current_crop.shape:
                difference = np.sum(np.abs(current_crop - last_crop)) / (last_crop.size * 255)
                
                # Update last screenshot
                self._last_screenshot = current_screenshot
                
                # If difference is significant, page has likely changed
                PAGE_CHANGE_THRESHOLD = 0.05  # 5% change threshold
                return difference > PAGE_CHANGE_THRESHOLD
                
        except Exception as e:
            logging.warning(f"Error comparing screenshots: {e}")
            
        # Update last screenshot even if comparison failed
        self._last_screenshot = current_screenshot
        return False
    
    def wait_for_page_load(self, timeout: float = 15) -> bool:
        """
        Wait for the page to load by periodically checking for visual changes.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if page loaded within timeout
            
        Raises:
            BrowserNotReadyError: If page not loaded within timeout
        """
        logging.info(f"Waiting up to {timeout} seconds for page to load")
        
        # Initialize with current screenshot
        self._update_screenshot()
        
        start_time = time.time()
        check_interval = 0.5  # Start with frequent checks
        
        # Use adaptive check interval: start frequent, then slower
        while time.time() - start_time < timeout:
            if self.is_page_loaded():
                # Wait a bit more for additional content to load
                time.sleep(1)
                self._update_screenshot()  # Update reference for future checks
                logging.info(f"Page loaded after {time.time() - start_time:.1f} seconds")
                return True
                
            # Adaptive interval: longer as we wait longer
            elapsed = time.time() - start_time
            check_interval = min(0.5 + (elapsed / 10), 2.0)  # Cap at 2 seconds
            
            # Wait before checking again
            time.sleep(check_interval)
            
            # Log progress periodically
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                logging.debug(f"Still waiting for page to load ({int(elapsed)}s elapsed)")
        
        logging.warning(f"Page not loaded within {timeout} seconds")
        raise BrowserNotReadyError(f"Page not loaded within {timeout} seconds")
    
    @retry(max_attempts=2, error_types=(BrowserError,))
    def refresh(self) -> bool:
        """
        Refresh the current page.
        
        Returns:
            True if refresh succeeded
        
        Raises:
            BrowserError: If refresh fails
        """
        if self.state != "running":
            raise BrowserStateError(f"Cannot refresh page: browser is {self.state}")
            
        try:
            logging.info("Refreshing page")
            pyautogui.hotkey('f5')
            
            # Wait for page to start loading
            time.sleep(1)
            
            # Wait for page to finish loading
            self.wait_for_page_load()
            
            return True
            
        except Exception as e:
            logging.error(f"Error refreshing page: {e}")
            raise BrowserError(f"Error refreshing page: {e}")
    
    @retry(max_attempts=2, error_types=(BrowserNavigationError,))
    def navigate(self, url: str) -> bool:
        """
        Navigate to a new URL, using a new tab if needed.
        
        Args:
            url: URL to navigate to
            
        Returns:
            True if navigation succeeded
        
        Raises:
            BrowserNavigationError: If navigation fails
        """
        if self.state != "running":
            raise BrowserStateError(f"Cannot navigate: browser is {self.state}")
            
        try:
            logging.info(f"Navigating to: {url}")
            
            # Try using clipboard for reliability
            try:
                import pyperclip
                pyperclip.copy(url)
                has_clipboard = True
            except ImportError:
                has_clipboard = False
            
            # Open new tab
            pyautogui.hotkey('ctrl', 't')
            time.sleep(1)
            
            if has_clipboard:
                # Paste URL from clipboard
                pyautogui.hotkey('ctrl', 'v')
            else:
                # Type URL manually
                pyautogui.write(url)
            
            time.sleep(0.5)
            
            # Press Enter to navigate
            pyautogui.press('enter')
            
            # Update the last known URL
            self.last_known_url = url
            
            # Wait for page to load
            self.wait_for_page_load()
            
            return True
            
        except BrowserNotReadyError as e:
            # Convert to navigation error
            raise BrowserNavigationError(f"Navigation timeout: {e}")
        except Exception as e:
            logging.error(f"Error navigating to URL: {e}")
            raise BrowserNavigationError(f"Error navigating to URL: {e}")
    
    def clear_input(self) -> bool:
        """
        Clear any active input field using keyboard shortcuts.
        
        Returns:
            True if operation completed (success not guaranteed)
        """
        try:
            # Select all text and delete
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(0.5)
            pyautogui.press('delete')
            return True
        except Exception as e:
            logging.error(f"Error clearing input: {e}")
            return False
    
    def type_character(self, char: str) -> bool:
        """
        Type a single character.
        
        Args:
            char: Character to type
            
        Returns:
            True if typing succeeded
        """
        try:
            pyautogui.write(char)
            return True
        except Exception as e:
            logging.error(f"Error typing character: {e}")
            return False
    
    def send_prompt(self) -> bool:
        """
        Submit the current prompt (press Enter).
        
        Returns:
            True if operation completed
        """
        try:
            pyautogui.press('enter')
            logging.debug("Pressed Enter to send prompt")
            return True
        except Exception as e:
            logging.error(f"Error sending prompt: {e}")
            return False


class BrowserPool:
    """
    Manages a pool of browser instances for efficient reuse.
    """
    
    def __init__(self, max_size: int = 3, config: Optional[Dict[str, Any]] = None):
        """
        Initialize browser pool.
        
        Args:
            max_size: Maximum number of browser instances to maintain
            config: Browser configuration dictionary
        """
        self.max_size = max_size
        self.config = config or {}
        self.browsers: Dict[str, ChromeBrowser] = {}
        self.active_browser: Optional[str] = None
    
    def get_browser(self) -> ChromeBrowser:
        """
        Get a browser instance, reusing an existing one or creating new.
        
        Returns:
            ChromeBrowser instance
        """
        # Check for available browsers
        available_browsers = [bid for bid, browser in self.browsers.items() 
                             if browser.state == "running" and not browser.is_closed()]
        
        if available_browsers:
            # Use an existing browser
            browser_id = available_browsers[0]
            self.active_browser = browser_id
            logging.debug(f"Reusing browser instance {browser_id}")
            return self.browsers[browser_id]
        
        # Create a new browser
        browser = ChromeBrowser(self.config)
        self.browsers[browser.id] = browser
        self.active_browser = browser.id
        
        # Enforce pool size limit
        if len(self.browsers) > self.max_size:
            self._cleanup_excess_browsers()
            
        return browser
    
    def _cleanup_excess_browsers(self):
        """Close oldest browsers when pool exceeds max size."""
        # Sort by launch time (oldest first)
        browser_ids = sorted(
            [bid for bid in self.browsers.keys()],
            key=lambda bid: self.browsers[bid].launch_time or 0
        )
        
        # Keep active and newest browsers, close others
        browsers_to_keep = set([self.active_browser] + browser_ids[-self.max_size:])
        
        for browser_id in list(self.browsers.keys()):
            if browser_id not in browsers_to_keep:
                try:
                    logging.info(f"Closing excess browser {browser_id}")
                    self.browsers[browser_id].close()
                    del self.browsers[browser_id]
                except Exception as e:
                    logging.warning(f"Error closing excess browser {browser_id}: {e}")
    
    def close_all(self):
        """Close all browser instances in the pool."""
        logging.info(f"Closing all browsers in pool ({len(self.browsers)} instances)")
        
        for browser_id, browser in list(self.browsers.items()):
            try:
                browser.close()
            except Exception as e:
                logging.warning(f"Error closing browser {browser_id}: {e}")
        
        # Clear the pool
        self.browsers = {}
        self.active_browser = None


class BrowserSession:
    """
    High-level browser session manager for Claude automation.
    Combines browser instance management with common operations.
    """
    
    def __init__(self, reuse_browser: bool = True, config: Optional[Dict[str, Any]] = None):
        """
        Initialize browser session manager.
        
        Args:
            reuse_browser: Whether to reuse browser instances between sessions
            config: Browser configuration dictionary
        """
        self.config = config or {}
        self.reuse_browser = reuse_browser
        
        # Create browser pool if reusing browsers
        self.pool = BrowserPool(max_size=3, config=self.config) if reuse_browser else None
        self.browser: Optional[ChromeBrowser] = None
    
    def start(self, url: str, profile_dir: Optional[str] = None) -> bool:
        """
        Start a browser session, launching browser if needed.
        
        Args:
            url: URL to navigate to
            profile_dir: Browser profile directory
            
        Returns:
            True if session started successfully
        """
        try:
            if self.reuse_browser and self.pool:
                # Get browser from pool
                self.browser = self.pool.get_browser()
                
                # If browser is already running, navigate to URL
                if self.browser.state == "running":
                    try:
                        return self.browser.navigate(url)
                    except BrowserNavigationError:
                        # If navigation fails, try launching fresh
                        logging.warning("Navigation failed, will try launching fresh browser")
                        pass
            else:
                # Create a new browser instance
                self.browser = ChromeBrowser(self.config)
            
            # Launch the browser
            return self.browser.launch(url, profile_dir)
            
        except Exception as e:
            logging.error(f"Error starting browser session: {e}")
            return False
    
    def end(self) -> bool:
        """
        End the current browser session.
        
        Returns:
            True if session ended successfully
        """
        if not self.browser:
            return True
            
        try:
            # For pooled browsers, we don't close automatically
            if self.reuse_browser and self.pool:
                return True
                
            # Close browser directly
            return self.browser.close()
            
        except Exception as e:
            logging.error(f"Error ending browser session: {e}")
            return False
    
    def close(self) -> bool:
        """
        Close all browser instances and clean up resources.
        
        Returns:
            True if closed successfully
        """
        success = True
        
        try:
            # Close browser pool if using one
            if self.reuse_browser and self.pool:
                self.pool.close_all()
            elif self.browser:
                # Close individual browser
                success = self.browser.close()
                
            self.browser = None
            return success
            
        except Exception as e:
            logging.error(f"Error closing browser completely: {e}")
            return False
    
    # Convenience methods that delegate to the current browser
    
    def refresh_page(self) -> bool:
        """Refresh the current page."""
        if not self.browser:
            return False
        return self.browser.refresh()
    
    def clear_input(self) -> bool:
        """Clear the current input field."""
        if not self.browser:
            return False
        return self.browser.clear_input()
    
    def type_character(self, char: str) -> bool:
        """Type a single character."""
        if not self.browser:
            return False
        return self.browser.type_character(char)
    
    def send_prompt(self) -> bool:
        """Submit the current prompt."""
        if not self.browser:
            return False
        return self.browser.send_prompt()
    
    def wait_for_page_load(self, timeout: float = 15) -> bool:
        """Wait for the page to load."""
        if not self.browser:
            return False
        try:
            return self.browser.wait_for_page_load(timeout)
        except BrowserNotReadyError:
            return False
    
    def is_browser_ready(self) -> bool:
        """Check if browser is ready for interaction."""
        return (self.browser is not None and 
                self.browser.state == "running" and 
                not self.browser.is_closed())