import subprocess
import time
import logging
import os
import platform
from pathlib import Path
from src.models.ui_element import UIElement
from src.automation.recognition import find_element, wait_for_visual_change

# Import shared utilities
from src.utils.browser_core import (
    get_chrome_path as core_get_chrome_path,
    close_browser_process,
    verify_browser_closed as core_verify_browser_closed
)
from src.utils.constants import (
    DEFAULT_BROWSER_LAUNCH_WAIT,
    DEFAULT_BROWSER_READY_TIMEOUT,
    BROWSER_CLOSE_WAIT_TIME,
    PAGE_REFRESH_DELAY
)
from src.utils.config_base import get_browser_config

def launch_browser(url, config=None):
    """
    Launch a browser instance for Claude automation.
    
    Args:
        url: URL to navigate to
        config: Configuration object (optional)
    
    Returns:
        True if browser launched successfully, False otherwise
    """
    try:
        # Use standardized configuration handling
        browser_config = get_browser_config(config)
        chrome_path = browser_config.chrome_path
        profile_dir = browser_config.profile_dir
        startup_wait = browser_config.startup_wait
        
        # Use shared Chrome path finder if not in config
        if not chrome_path:
            chrome_path = core_get_chrome_path()
            if not chrome_path:
                logging.error("Could not find Chrome browser")
                return False
        
        # Ensure profile directory exists
        os.makedirs(profile_dir, exist_ok=True)
        
        # Launch browser with the specified profile
        logging.info(f"Launching browser: {chrome_path}")
        logging.info(f"Using profile: {profile_dir}")
        logging.info(f"Navigating to: {url}")
        
        cmd = [
            chrome_path,
            f"--user-data-dir={profile_dir}",
            "--start-maximized",
            "--disable-extensions",
            url
        ]
        
        process = subprocess.Popen(cmd)
        
        # Check if process started successfully
        if process.poll() is not None:
            logging.error(f"Browser process exited with code {process.returncode}")
            return False
        
        # Wait for browser to initialize
        logging.info(f"Waiting {startup_wait} seconds for browser to initialize")
        time.sleep(startup_wait)
        
        return True
    
    except Exception as e:
        logging.error(f"Failed to launch browser: {e}")
        return False

def get_chrome_path():
    """
    Get the default Chrome browser path based on the operating system.
    
    Returns:
        Path to Chrome executable or None if not found
    """
    # Use shared implementation
    return core_get_chrome_path()

def close_browser():
    """
    Close all Chrome browser instances.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logging.info("Attempting to close browser")
        
        # Use shared browser closing logic
        success = close_browser_process()
        
        # Allow time for browsers to close
        time.sleep(BROWSER_CLOSE_WAIT_TIME)
        
        # Verify browser is closed
        is_closed = verify_browser_closed()
        if is_closed:
            logging.info("Browser closed successfully")
        else:
            logging.warning("Browser may not have closed properly")
            
        return is_closed
    
    except Exception as e:
        logging.error(f"Failed to close browser: {e}")
        return False

def verify_browser_closed():
    """
    Verify that Chrome browser is no longer running.
    
    Returns:
        True if browser is closed, False if still running
    """
    try:
        # Use shared verification logic
        return core_verify_browser_closed()
    
    except Exception as e:
        logging.error(f"Error verifying browser closure: {e}")
        # If verification fails, assume browser may still be running
        return False

def check_browser_ready(ui_element, timeout=DEFAULT_BROWSER_READY_TIMEOUT):
    """
    Check if the browser is fully loaded and ready for interaction.
    
    Args:
        ui_element: UIElement to look for (like prompt_box)
        timeout: Maximum time to wait in seconds
    
    Returns:
        True if browser is ready, False if timeout
    """
    logging.info(f"Checking if browser is ready (timeout: {timeout}s)")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        if find_element(ui_element):
            logging.info("Browser is ready")
            return True
        
        logging.debug("Browser not ready yet, waiting...")
        time.sleep(1)
    
    logging.warning("Browser ready check timed out")
    return False

def refresh_page():
    """
    Refresh the current page.
    """
    from pyautogui import hotkey
    logging.info("Refreshing page")
    hotkey('f5')
    time.sleep(PAGE_REFRESH_DELAY)  # Wait for page to reload