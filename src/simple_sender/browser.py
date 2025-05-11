#!/usr/bin/env python3
"""
Simple browser management for Claude automation.

Provides a lightweight wrapper around browser operations with minimal
dependencies on the main automation system.
"""

import os
import time
import logging
import subprocess
import platform
from pathlib import Path
import random
import sys

# Try to import from main automation, use simplified versions if not available
try:
    from src.automation.browser import get_chrome_path, close_browser
    MAIN_IMPORTS_AVAILABLE = True
except ImportError:
    MAIN_IMPORTS_AVAILABLE = False

# Try to import enhanced logging if available
try:
    from src.utils.logging_util import log_with_screenshot
    ENHANCED_LOGGING = True
except ImportError:
    ENHANCED_LOGGING = False

# Import pyautogui for screenshots and basic interaction
import pyautogui

class BrowserError(Exception):
    """Base exception for browser-related errors."""
    pass

class BrowserLaunchError(BrowserError):
    """Exception raised when browser fails to launch."""
    pass

class BrowserNotReadyError(BrowserError):
    """Exception raised when browser is not ready within timeout."""
    pass

class SimpleBrowser:
    """Simplified browser manager for Claude automation."""
    
    def __init__(self, config=None):
        """
        Initialize the SimpleBrowser with optional configuration.
        
        Args:
            config: Dictionary containing browser configuration or None
        """
        self.config = config or {}
        self.process = None
        self.chrome_path = None
        self.profile_dir = None
        self.last_known_url = None
        self.startup_delay = self.config.get("browser_launch_wait", 10)
        self.retry_count = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 2)
        
        # Take screenshot of initial desktop for comparison
        self._initial_screen = None
        
    def launch(self, url, profile_dir=None):
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
        # Find Chrome path if not explicitly set
        if not self.chrome_path:
            self.chrome_path = self.get_chrome_path()
            if not self.chrome_path:
                logging.error("Could not find Chrome browser")
                raise BrowserLaunchError("Chrome browser not found")
        
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
            logging.info(f"Launching browser: {self.chrome_path}")
            logging.info(f"Using profile: {self.profile_dir}")
            logging.info(f"Navigating to: {url}")
            
            # Launch browser process
            self.process = subprocess.Popen(cmd)
            
            # Store the URL
            self.last_known_url = url
            
            # Check if process started successfully
            if self.process.poll() is not None:
                error_msg = f"Browser process exited with code {self.process.returncode}"
                logging.error(error_msg)
                raise BrowserLaunchError(error_msg)
            
            # Wait for browser to initialize
            time.sleep(self.startup_delay)
            
            # Take screenshot after launch for comparison
            self._initial_screen = pyautogui.screenshot()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to launch browser: {e}")
            raise BrowserLaunchError(f"Failed to launch browser: {e}")
    
    def close(self):
        """
        Close the browser and clean up resources.
        
        Returns:
            True if browser closed successfully
        """
        logging.info("Closing browser...")
        
        if MAIN_IMPORTS_AVAILABLE:
            # Use the main automation's close_browser function
            success = close_browser()
        else:
            # Simplified implementation
            try:
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/f", "/im", "chrome.exe"], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
                else:
                    subprocess.run(["pkill", "-f", "chrome"], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
                
                # Allow time for browsers to close
                time.sleep(2)
                success = self.is_closed()
            except Exception as e:
                logging.error(f"Error closing browser: {e}")
                success = False
        
        if success:
            logging.info("Browser closed successfully")
        else:
            logging.warning("Browser may not have closed properly")
            
        # Reset process tracking
        self.process = None
        return success
    
    def is_closed(self):
        """
        Check if the browser is closed.
        
        Returns:
            True if browser is closed, False if still running
        """
        try:
            if self.process and self.process.poll() is None:
                return False
                
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
    
    def is_ready(self, timeout=5):
        """
        Check if the browser is ready by detecting visual changes.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if browser appears ready, False otherwise
        """
        if not self._initial_screen:
            # No initial screen to compare, take one now
            self._initial_screen = pyautogui.screenshot()
            return False
            
        try:
            # Take current screenshot
            current_screen = pyautogui.screenshot()
            
            # Compare with initial screenshot
            # For simplicity, we'll just check if there's any visible difference
            # in the center portion of the screen which should be the main content area
            
            # Get screen dimensions
            width, height = current_screen.size
            
            # Define region to check (center of screen)
            center_region = (
                width // 4,  # x start (1/4 from left)
                height // 4,  # y start (1/4 from top)
                width // 2,   # width (half of screen)
                height // 2   # height (half of screen)
            )
            
            # Take cropped screenshots of both initial and current screens
            initial_crop = self._initial_screen.crop((
                center_region[0],
                center_region[1],
                center_region[0] + center_region[2],
                center_region[1] + center_region[3]
            ))
            
            current_crop = current_screen.crop((
                center_region[0],
                center_region[1],
                center_region[0] + center_region[2],
                center_region[1] + center_region[3]
            ))
            
            # Convert to data for comparison
            import numpy as np
            initial_data = np.array(initial_crop)
            current_data = np.array(current_crop)
            
            # Calculate difference
            difference = np.sum(np.abs(current_data - initial_data)) / (initial_data.size * 255)
            
            # Update initial screen for next comparison
            self._initial_screen = current_screen
            
            # If difference is significant, browser has probably loaded
            ready = difference > 0.1  # 10% change threshold
            
            if ready:
                logging.info("Browser appears ready based on visual changes")
            
            return ready
            
        except Exception as e:
            logging.warning(f"Error checking browser readiness: {e}")
            return False
    
    def refresh(self):
        """
        Refresh the current page.
        
        Returns:
            True if refresh succeeded
        """
        try:
            logging.info("Refreshing page")
            pyautogui.hotkey('f5')
            time.sleep(5)  # Wait for page to reload
            
            # Update initial screen
            self._initial_screen = pyautogui.screenshot()
            return True
        except Exception as e:
            logging.error(f"Error refreshing page: {e}")
            return False
    
    def wait_for_ready(self, timeout=15):
        """
        Wait for the browser to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if browser is ready
            
        Raises:
            BrowserNotReadyError: If browser not ready within timeout
        """
        logging.info(f"Waiting up to {timeout} seconds for browser to be ready...")
        
        start_time = time.time()
        check_interval = 1  # Check every second
        
        while time.time() - start_time < timeout:
            if self.is_ready():
                # Wait a bit more for additional content to load
                time.sleep(2)
                logging.info("Browser is ready")
                return True
                
            # Wait before checking again
            time.sleep(check_interval)
            
            # Log progress periodically
            elapsed = time.time() - start_time
            if elapsed > 5 and elapsed % 5 < 1:
                logging.info(f"Still waiting for browser to be ready ({int(elapsed)}s elapsed)...")
        
        logging.error(f"Browser not ready within {timeout} seconds")
        raise BrowserNotReadyError(f"Browser not ready within {timeout} seconds")
    
    def reuse_existing(self, url):
        """
        Navigate to a new URL in an existing browser instance.
        
        Args:
            url: URL to navigate to
            
        Returns:
            True if navigation succeeded
            
        Raises:
            BrowserError: If navigation failed
        """
        if self.is_closed():
            logging.warning("Browser is not running, launching new instance")
            return self.launch(url, self.profile_dir)
            
        try:
            logging.info(f"Navigating to: {url}")
            
            # Copy URL to clipboard
            import pyperclip
            pyperclip.copy(url)
            
            # Open new tab
            pyautogui.hotkey('ctrl', 't')
            time.sleep(1)
            
            # Paste URL
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(0.5)
            
            # Press Enter to navigate
            pyautogui.press('enter')
            
            # Update the last known URL
            self.last_known_url = url
            
            # Wait for page to load
            time.sleep(5)
            
            # Update initial screen
            self._initial_screen = pyautogui.screenshot()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to navigate to URL: {e}")
            raise BrowserError(f"Failed to navigate to URL: {e}")
    
    @staticmethod
    def get_chrome_path():
        """
        Get the path to Chrome executable.
        
        Returns:
            String path to Chrome or None if not found
        """
        if MAIN_IMPORTS_AVAILABLE:
            # Use the main automation's get_chrome_path function
            return get_chrome_path()
        else:
            # Simplified implementation
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

def retry(max_attempts=3, initial_delay=1, backoff_factor=2, jitter=0.1):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each failure
        jitter: Random factor to add to delay (0-1)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        # Add jitter to delay
                        jitter_amount = delay * jitter
                        actual_delay = delay + random.uniform(-jitter_amount, jitter_amount)
                        logging.warning(f"Attempt {attempt} failed, retrying in {actual_delay:.2f}s: {e}")
                        time.sleep(actual_delay)
                        # Increase delay for next attempt
                        delay *= backoff_factor
            
            # If we get here, all attempts failed
            raise last_exception
        return wrapper
    return decorator