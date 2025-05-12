#!/usr/bin/env python3
"""
Browser Core Utilities

Centralized browser management functions for Claude GUI automation.
Provides platform-independent implementations for Chrome detection,
browser launching, and browser closing operations.
"""

import subprocess
import time
import logging
import os
import platform
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List

def get_chrome_path() -> Optional[str]:
    """
    Get the default Chrome browser path based on the operating system.
    
    Returns:
        Path to Chrome executable or None if not found
    """
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

def launch_browser(url: str, profile_dir: Optional[str] = None, 
                  config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Launch a browser instance for Claude automation.
    
    Args:
        url: URL to navigate to
        profile_dir: Path to browser profile directory (optional)
        config: Configuration object (optional)
    
    Returns:
        True if browser launched successfully, False otherwise
    """
    try:
        # Use configuration if provided, otherwise use defaults
        if config:
            chrome_path = config.get("chrome_path", "")
            profile_dir = profile_dir or config.get("browser_profile", "")
            startup_wait = config.get("browser_launch_wait", 10)
        else:
            chrome_path = ""
            profile_dir = profile_dir or os.path.join(os.path.expanduser("~"), "ClaudeProfile")
            startup_wait = 10
        
        # Determine browser path based on platform if not in config
        if not chrome_path:
            chrome_path = get_chrome_path()
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

def close_browser(force: bool = False) -> bool:
    """
    Close all Chrome browser instances.
    
    Args:
        force: Whether to use force closing methods if regular close fails
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logging.info("Attempting to close browser")
        system = platform.system()
        
        if system == "Windows":
            force_arg = "/f" if force else ""
            subprocess.run(["taskkill", force_arg, "/im", "chrome.exe"], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL)
        else:
            # On Unix systems, pkill with -9 for force, or without for graceful
            force_arg = "-9" if force else ""
            pkill_cmd = ["pkill"]
            if force_arg:
                pkill_cmd.append(force_arg)
            pkill_cmd.extend(["-f", "chrome"])
            subprocess.run(pkill_cmd, 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL)
        
        # Allow time for browsers to close
        time.sleep(2)
        
        # Verify browser is closed
        is_closed = verify_browser_closed()
        
        # Retry with force if normal close failed and force wasn't already tried
        if not is_closed and not force:
            logging.warning("Browser didn't close normally, attempting force close")
            return close_browser(force=True)
            
        if is_closed:
            logging.info("Browser closed successfully")
        else:
            logging.warning("Browser may not have closed properly")
            
        return is_closed
        
    except Exception as e:
        logging.error(f"Failed to close browser: {e}")
        return False

def verify_browser_closed() -> bool:
    """
    Verify that Chrome browser is no longer running.
    
    Returns:
        True if browser is closed, False if still running
    """
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

def refresh_page() -> None:
    """
    Refresh the current page using F5 key.
    """
    try:
        import pyautogui
        logging.info("Refreshing page")
        pyautogui.hotkey('f5')
        time.sleep(5)  # Wait for page to reload
    except ImportError:
        logging.warning("PyAutoGUI not available, can't refresh page")
    except Exception as e:
        logging.error(f"Error refreshing page: {e}")

def check_browser_ready(ui_element: Any, timeout: int = 30) -> bool:
    """
    Check if the browser is fully loaded and ready for interaction.
    
    Args:
        ui_element: Element to look for (like prompt_box)
        timeout: Maximum time to wait in seconds
    
    Returns:
        True if browser is ready, False if timeout
    """
    try:
        # Import here to avoid circular imports
        from src.automation.recognition import find_element
        
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
    except ImportError:
        logging.warning("Recognition module not available, can't check browser ready state")
        return True  # Assume ready if module not available
    except Exception as e:
        logging.error(f"Error checking browser ready state: {e}")
        return False