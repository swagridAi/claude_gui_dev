#!/usr/bin/env python3
"""
Browser management module for Claude automation.

Provides browser operations needed for Claude automation with support for
browser pooling, adaptive timing, region-based screenshots, and standardized error recovery.
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
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Set, NamedTuple
import uuid
from functools import wraps
from contextlib import contextmanager
from enum import Enum, auto

# Import pyautogui for screenshots and basic interaction
import pyautogui

# Try to import numpy for advanced image processing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Try to import PIL for image operations
try:
    from PIL import Image, ImageChops
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Try to import from main automation, use simplified versions if not available
try:
    from src.automation.browser import get_chrome_path as main_get_chrome_path
    from src.automation.browser import close_browser as main_close_browser
    MAIN_IMPORTS_AVAILABLE = True
except ImportError:
    MAIN_IMPORTS_AVAILABLE = False


# === Exception Hierarchy ===

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


class BrowserVerificationError(BrowserError):
    """Exception raised when visual verification fails."""
    pass


# === Constants and Configuration ===

class BrowserConstants:
    """Central constants for browser module."""
    
    # Default timeouts and intervals
    DEFAULT_STARTUP_DELAY = 10
    DEFAULT_PAGE_LOAD_TIMEOUT = 15
    DEFAULT_VERIFICATION_TIMEOUT = 30
    DEFAULT_CHECK_INTERVAL = 0.5
    
    # Visual verification thresholds
    PIXEL_CHANGE_THRESHOLD = 0.05  # 5% pixel difference threshold
    MIN_PIXELS_CHANGED = 1000
    STABILITY_THRESHOLD = 0.01  # 1% change threshold for stability
    
    # Browser pool settings
    DEFAULT_POOL_SIZE = 3
    MAX_BROWSER_AGE = 3600  # 1 hour in seconds
    
    # Standard region definitions (relative to screen size)
    STANDARD_REGIONS = {
        "full_screen": (0, 0, 1, 1),  # Full screen (x, y, width, height as percentages)
        "prompt_area": (0.05, 0.7, 0.9, 0.25),  # Bottom area where prompt box is located
        "response_area": (0.05, 0.1, 0.9, 0.6),  # Area where Claude's response appears
        "screen_center": (0.25, 0.25, 0.5, 0.5),  # Center of the screen
        "claude_thinking": (0.4, 0.4, 0.2, 0.2)  # Area where thinking indicator appears
    }


class BrowserHealthStatus(Enum):
    """Represents the health status of a browser instance."""
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


# === Utility Classes ===

class Region(NamedTuple):
    """Represents a screen region with absolute coordinates."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def coordinates(self) -> Tuple[int, int, int, int]:
        """Get region as a tuple of coordinates."""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center point of the region."""
        return (self.x + self.width // 2, self.y + self.height // 2)


class RegionManager:
    """Manages screen regions for efficient screenshot capturing and element detection."""
    
    def __init__(self):
        """Initialize the region manager."""
        self.screen_size = pyautogui.size()
        self.regions: Dict[str, Region] = {}
        self.region_cache: Dict[str, Dict[str, Any]] = {}
        self._register_standard_regions()
    
    def _register_standard_regions(self) -> None:
        """Register standard regions based on screen size."""
        for name, relative_region in BrowserConstants.STANDARD_REGIONS.items():
            self.register_region_from_relative(name, relative_region)
    
    def register_region(self, name: str, region: Region) -> None:
        """
        Register a region with a name.
        
        Args:
            name: Name to associate with the region
            region: Region object
        """
        self.regions[name] = region
        # Clear cache for this region if it exists
        if name in self.region_cache:
            del self.region_cache[name]
    
    def register_region_from_coordinates(self, name: str, coordinates: Tuple[int, int, int, int]) -> None:
        """
        Register a region from raw coordinates.
        
        Args:
            name: Name to associate with the region
            coordinates: Tuple of (x, y, width, height)
        """
        self.register_region(name, Region(*coordinates))
    
    def register_region_from_relative(self, name: str, relative_region: Tuple[float, float, float, float]) -> None:
        """
        Register a region based on relative screen coordinates (0-1).
        
        Args:
            name: Name to associate with the region
            relative_region: Tuple of (x_percent, y_percent, width_percent, height_percent)
        """
        screen_width, screen_height = self.screen_size
        x_pct, y_pct, width_pct, height_pct = relative_region
        
        # Convert percentages to absolute pixel values
        x = int(screen_width * x_pct)
        y = int(screen_height * y_pct)
        width = int(screen_width * width_pct)
        height = int(screen_height * height_pct)
        
        self.register_region(name, Region(x, y, width, height))
    
    def get_region(self, region_identifier: Union[str, Tuple[int, int, int, int], Region]) -> Region:
        """
        Get a region by name or create from coordinates.
        
        Args:
            region_identifier: Region name, coordinates tuple, or Region object
            
        Returns:
            Region object
            
        Raises:
            ValueError: If region name not found or invalid coordinates
        """
        if isinstance(region_identifier, str):
            # Look up by name
            if region_identifier in self.regions:
                return self.regions[region_identifier]
            raise ValueError(f"Region '{region_identifier}' not registered")
            
        elif isinstance(region_identifier, tuple) and len(region_identifier) == 4:
            # Direct coordinates
            return Region(*region_identifier)
            
        elif isinstance(region_identifier, Region):
            # Already a Region object
            return region_identifier
            
        raise ValueError(f"Invalid region identifier: {region_identifier}")
    
    def update_screen_size(self) -> None:
        """Update screen size and recalculate all regions."""
        new_size = pyautogui.size()
        if new_size != self.screen_size:
            self.screen_size = new_size
            # Recalculate all standard regions
            self._register_standard_regions()
            # Clear cache completely
            self.region_cache.clear()
    
    def capture_screenshot(self, region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None) -> Image.Image:
        """
        Capture a screenshot of the specified region.
        
        Args:
            region_identifier: Region to capture (name, coordinates, or Region object)
            
        Returns:
            PIL Image object
        """
        if region_identifier is None:
            # Full screen screenshot
            return pyautogui.screenshot()
        
        # Get region and capture screenshot
        region = self.get_region(region_identifier)
        return pyautogui.screenshot(region=region.coordinates)


class ScreenshotComparer:
    """Compares screenshots to detect changes."""
    
    @staticmethod
    def calculate_difference(img1: Image.Image, img2: Image.Image) -> Tuple[float, int]:
        """
        Calculate the difference between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Tuple of (difference_percentage, changed_pixels)
        """
        if not NUMPY_AVAILABLE or not PIL_AVAILABLE:
            # Fallback method using PIL only
            diff = ImageChops.difference(img1, img2)
            diff_bbox = diff.getbbox()
            
            if diff_bbox is None:
                return 0.0, 0  # Images are identical
            
            # Rough estimate based on bounding box
            changed_area = (diff_bbox[2] - diff_bbox[0]) * (diff_bbox[3] - diff_bbox[1])
            total_area = img1.width * img1.height
            return changed_area / total_area, changed_area
        
        # Use numpy for faster and more accurate comparison
        np_img1 = np.array(img1)
        np_img2 = np.array(img2)
        
        # Ensure images are the same size
        if np_img1.shape != np_img2.shape:
            # Resize second image to match first if necessary
            img2_resized = img2.resize((img1.width, img1.height))
            np_img2 = np.array(img2_resized)
        
        # Calculate difference
        diff = np.abs(np_img1.astype(np.int16) - np_img2.astype(np.int16))
        
        # Count changed pixels (pixels with any change in any channel)
        changed_pixels = np.sum(np.any(diff > 10, axis=2))
        
        # Calculate percentage
        total_pixels = diff.shape[0] * diff.shape[1]
        difference_percentage = changed_pixels / total_pixels
        
        return difference_percentage, int(changed_pixels)
    
    @staticmethod
    def is_significant_change(img1: Image.Image, img2: Image.Image, 
                            threshold: float = BrowserConstants.PIXEL_CHANGE_THRESHOLD,
                            min_pixels: int = BrowserConstants.MIN_PIXELS_CHANGED) -> bool:
        """
        Determine if the change between images is significant.
        
        Args:
            img1: First image
            img2: Second image
            threshold: Difference threshold (0-1)
            min_pixels: Minimum number of pixels that must change
            
        Returns:
            True if change is significant
        """
        diff_percentage, changed_pixels = ScreenshotComparer.calculate_difference(img1, img2)
        return diff_percentage > threshold and changed_pixels > min_pixels


class VisualVerifier:
    """Provides visual verification of browser state through screenshot analysis."""
    
    def __init__(self, region_manager: Optional[RegionManager] = None):
        """
        Initialize the visual verifier.
        
        Args:
            region_manager: Optional region manager for screenshot capturing
        """
        self.region_manager = region_manager or RegionManager()
        self.reference_images: Dict[str, Dict[str, Any]] = {}
        self.comparer = ScreenshotComparer()
    
    def take_reference_screenshot(self, key: str, 
                                region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None) -> Image.Image:
        """
        Take a reference screenshot for later comparison.
        
        Args:
            key: Identifier for this reference
            region_identifier: Region to capture
            
        Returns:
            Captured image
        """
        screenshot = self.region_manager.capture_screenshot(region_identifier)
        
        # Store in reference dictionary
        self.reference_images[key] = {
            'image': screenshot,
            'timestamp': time.time(),
            'region': region_identifier
        }
        
        return screenshot
    
    def check_for_changes(self, reference_key: str, 
                         region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None,
                         threshold: float = BrowserConstants.PIXEL_CHANGE_THRESHOLD,
                         min_pixels: int = BrowserConstants.MIN_PIXELS_CHANGED) -> Tuple[bool, float, int]:
        """
        Check if a region has changed compared to a reference screenshot.
        
        Args:
            reference_key: Reference screenshot identifier
            region_identifier: Region to check (uses reference's region if None)
            threshold: Difference threshold (0-1)
            min_pixels: Minimum number of pixels that must change
            
        Returns:
            Tuple of (is_significant_change, difference_percentage, changed_pixels)
        """
        # If no reference exists, take one now
        if reference_key not in self.reference_images:
            self.take_reference_screenshot(reference_key, region_identifier)
            return False, 0.0, 0
        
        # Use the reference's region if none specified
        if region_identifier is None:
            region_identifier = self.reference_images[reference_key].get('region')
        
        # Take current screenshot
        current = self.region_manager.capture_screenshot(region_identifier)
        
        # Get reference image
        reference = self.reference_images[reference_key]['image']
        
        # Calculate difference
        diff_percentage, changed_pixels = ScreenshotComparer.calculate_difference(reference, current)
        
        # Determine if change is significant
        is_significant = diff_percentage > threshold and changed_pixels > min_pixels
        
        # If significant change, update reference
        if is_significant:
            self.reference_images[reference_key]['image'] = current
            self.reference_images[reference_key]['timestamp'] = time.time()
        
        return is_significant, diff_percentage, changed_pixels
    
    def wait_for_change(self, reference_key: str, 
                       region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None,
                       timeout: float = BrowserConstants.DEFAULT_VERIFICATION_TIMEOUT,
                       check_interval: float = BrowserConstants.DEFAULT_CHECK_INTERVAL,
                       threshold: float = BrowserConstants.PIXEL_CHANGE_THRESHOLD,
                       min_pixels: int = BrowserConstants.MIN_PIXELS_CHANGED) -> Tuple[bool, float, float]:
        """
        Wait for visual change in a region, with timeout.
        
        Args:
            reference_key: Reference screenshot identifier
            region_identifier: Region to monitor
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds
            threshold: Difference threshold
            min_pixels: Minimum pixels that must change
            
        Returns:
            Tuple of (change_detected, difference_percentage, time_elapsed)
        """
        # Take initial reference if not exists
        if reference_key not in self.reference_images:
            self.take_reference_screenshot(reference_key, region_identifier)
        
        start_time = time.time()
        end_time = start_time + timeout
        
        # Use adaptive checking interval (starts frequent, gets less frequent)
        current_interval = check_interval
        last_log_time = start_time
        
        while time.time() < end_time:
            # Check for changes
            changed, diff_percentage, _ = self.check_for_changes(
                reference_key, 
                region_identifier,
                threshold,
                min_pixels
            )
            
            if changed:
                elapsed = time.time() - start_time
                return True, diff_percentage, elapsed
            
            # Log progress periodically (every 5 seconds)
            current_time = time.time()
            if current_time - last_log_time >= 5:
                elapsed = current_time - start_time
                remaining = end_time - current_time
                logging.debug(f"Waiting for change in {reference_key}: {int(elapsed)}s elapsed, {int(remaining)}s remaining")
                last_log_time = current_time
            
            # Adapt check interval based on elapsed time
            elapsed = current_time - start_time
            # Start with frequent checks, gradually increase interval
            current_interval = min(check_interval + (elapsed / 20), 2.0)
            
            # Sleep before next check
            time.sleep(current_interval)
        
        # If we get here, timeout occurred
        elapsed = time.time() - start_time
        return False, 0.0, elapsed
    
    def wait_for_stability(self, region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None,
                          duration: float = 1.0,
                          check_interval: float = 0.2,
                          threshold: float = BrowserConstants.STABILITY_THRESHOLD) -> bool:
        """
        Wait for a region to become visually stable (no significant changes).
        
        Args:
            region_identifier: Region to monitor
            duration: Duration of stability required
            check_interval: Time between checks
            threshold: Maximum allowed change
            
        Returns:
            True if region became stable for required duration
        """
        stability_key = f"stability_{uuid.uuid4().hex[:8]}"
        
        # Take initial reference
        self.take_reference_screenshot(stability_key, region_identifier)
        
        # Track start of current stability period
        stability_start = None
        
        # Check for stability over time
        check_start = time.time()
        while True:
            # Small sleep between checks
            time.sleep(check_interval)
            
            # Check for changes
            changed, diff_percentage, _ = self.check_for_changes(
                stability_key,
                region_identifier,
                threshold,
                0  # No minimum pixel count for stability check
            )
            
            current_time = time.time()
            
            if not changed:
                # Region is stable
                if stability_start is None:
                    # Start of stability period
                    stability_start = current_time
                elif current_time - stability_start >= duration:
                    # Stable for required duration
                    return True
            else:
                # Reset stability period
                stability_start = None
                
                # Update reference image after change
                self.take_reference_screenshot(stability_key, region_identifier)
                
                # Prevent endless loop
                if current_time - check_start > 30:  # 30 second timeout
                    return False
        
        return False


# === Retry Utility ===

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
                        actual_delay = max(0.1, actual_delay)  # Ensure minimum delay
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


# === Browser Interfaces and Implementations ===

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
    def is_page_loaded(self, region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None, 
                       timeout: float = 5) -> bool:
        """Check if the current page is loaded."""
        pass
    
    @abstractmethod
    def wait_for_page_load(self, region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None, 
                          timeout: float = 15) -> bool:
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
    def get_screenshot(self, region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None) -> Optional[Image.Image]:
        """Take a screenshot of the current browser window or region."""
        pass
    
    @abstractmethod
    def clear_input(self) -> bool:
        """Clear any active input field."""
        pass
    
    @abstractmethod
    def type_text(self, text: str, delay: Optional[float] = None) -> bool:
        """Type text with optional delay between characters."""
        pass
    
    @abstractmethod
    def send_prompt(self) -> bool:
        """Submit the current prompt (e.g., press Enter)."""
        pass
    
    @abstractmethod
    def wait_for_visual_change(self, reference_key: str, 
                              region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None,
                              timeout: float = 30) -> bool:
        """Wait for visual change in specified region."""
        pass
    
    @abstractmethod
    def wait_for_visual_stability(self, region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None,
                                 duration: float = 1.0) -> bool:
        """Wait for visual stability in specified region."""
        pass
    
    @abstractmethod
    def get_health_status(self) -> BrowserHealthStatus:
        """Get the health status of the browser."""
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
        self.startup_delay = self.config.get("browser_launch_wait", BrowserConstants.DEFAULT_STARTUP_DELAY)
        
        # Browser state tracking
        self.id = str(uuid.uuid4())[:8]  # Unique ID for this instance
        self.state = "initialized"  # initialized, running, closed, error
        self.launch_time = None
        self.error = None
        
        # Initialize region manager
        self.region_manager = RegionManager()
        
        # Initialize visual verifier
        self.visual_verifier = VisualVerifier(self.region_manager)
        
        # Health monitoring
        self.health_status = BrowserHealthStatus.UNKNOWN
        self.health_check_time = None
        self.error_count = 0
    
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
            
            # Take reference screenshots after launch
            try:
                self.visual_verifier.take_reference_screenshot("initial_page", "screen_center")
                self.update_health_status(BrowserHealthStatus.HEALTHY)
            except Exception as e:
                logging.warning(f"Could not take initial screenshots: {e}")
                self.update_health_status(BrowserHealthStatus.UNKNOWN)
            
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
                self.update_health_status(BrowserHealthStatus.UNKNOWN)
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
    
    def get_screenshot(self, region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None) -> Optional[Image.Image]:
        """
        Take a screenshot of the current browser window or region.
        
        Args:
            region_identifier: Region to capture or None for full screen
            
        Returns:
            PIL Image object or None if failed
        """
        try:
            # Capture screenshot using region manager
            return self.region_manager.capture_screenshot(region_identifier)
        except Exception as e:
            logging.error(f"Error taking screenshot: {e}")
            self.error_count += 1
            if self.error_count > 3:
                self.update_health_status(BrowserHealthStatus.DEGRADED)
            return None
    
    def is_page_loaded(self, region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None, 
                      timeout: float = 5) -> bool:
        """
        Check if the current page is loaded by detecting visual changes.
        
        Args:
            region_identifier: Region to check or None for screen center
            timeout: Maximum time to wait for check
            
        Returns:
            True if page appears to be loaded
        """
        # Default to screen center if no region specified
        if region_identifier is None:
            region_identifier = "screen_center"
        
        reference_key = f"page_load_{uuid.uuid4().hex[:8]}"
        
        # Take initial reference
        self.visual_verifier.take_reference_screenshot(reference_key, region_identifier)
        
        # Wait briefly
        time.sleep(min(1.0, timeout / 2))
        
        # Check for changes
        changed, _, _ = self.visual_verifier.check_for_changes(reference_key, region_identifier)
        return changed
    
    def wait_for_page_load(self, region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None, 
                          timeout: float = BrowserConstants.DEFAULT_PAGE_LOAD_TIMEOUT) -> bool:
        """
        Wait for the page to load by periodically checking for visual changes.
        
        Args:
            region_identifier: Region to check or None for screen center
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if page loaded within timeout
            
        Raises:
            BrowserNotReadyError: If page not loaded within timeout
        """
        # Default to screen center if no region specified
        if region_identifier is None:
            region_identifier = "screen_center"
            
        logging.info(f"Waiting up to {timeout} seconds for page to load")
        
        # Create a unique reference key for this wait operation
        reference_key = f"page_load_wait_{uuid.uuid4().hex[:8]}"
        
        # Wait for visual change
        changed, _, elapsed = self.visual_verifier.wait_for_change(
            reference_key,
            region_identifier,
            timeout=timeout
        )
        
        if changed:
            # After content loads, wait for stability
            logging.debug(f"Page content changed after {elapsed:.1f}s, waiting for stability")
            stable = self.visual_verifier.wait_for_stability(region_identifier)
            
            if stable:
                logging.info(f"Page load complete and stable after {time.time() - self.launch_time:.1f}s total")
                return True
            else:
                logging.warning("Page content loaded but did not stabilize")
                return True  # Still return success as content has changed
        
        logging.warning(f"No significant page change detected after {timeout} seconds")
        raise BrowserNotReadyError(f"Page content did not load within {timeout} seconds")
    
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
            try:
                self.wait_for_page_load()
                return True
            except BrowserNotReadyError:
                logging.warning("Page refresh may not have completed, but continuing")
                return True
            
        except Exception as e:
            self.error_count += 1
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
            
            # Try using clipboard for reliability if available
            clipboard_available = False
            try:
                import pyperclip
                original_clipboard = pyperclip.paste()
                pyperclip.copy(url)
                clipboard_available = True
            except ImportError:
                clipboard_available = False
            
            # Open new tab
            pyautogui.hotkey('ctrl', 't')
            time.sleep(1)
            
            if clipboard_available:
                # Paste URL from clipboard
                pyautogui.hotkey('ctrl', 'v')
                time.sleep(0.5)
            else:
                # Type URL manually
                self.type_text(url)
                time.sleep(0.5)
            
            # Press Enter to navigate
            pyautogui.press('enter')
            
            # Update the last known URL
            self.last_known_url = url
            
            # Wait for page to load
            try:
                self.wait_for_page_load()
            except BrowserNotReadyError:
                logging.warning("Page may not have loaded completely, but continuing")
            
            # Restore original clipboard content if needed
            if clipboard_available:
                try:
                    pyperclip.copy(original_clipboard)
                except:
                    pass
            
            return True
            
        except BrowserNotReadyError as e:
            # Convert to navigation error
            self.error_count += 1
            raise BrowserNavigationError(f"Navigation timeout: {e}")
        except Exception as e:
            logging.error(f"Error navigating to URL: {e}")
            self.error_count += 1
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
            self.error_count += 1
            return False
    
    def type_text(self, text: str, delay: Optional[float] = None) -> bool:
        """
        Type text with optional delay between characters.
        
        Args:
            text: Text to type
            delay: Delay between characters in seconds (None for system default)
            
        Returns:
            True if typing succeeded
        """
        try:
            if delay is not None:
                for char in text:
                    pyautogui.write(char)
                    time.sleep(delay)
            else:
                pyautogui.write(text)
            return True
        except Exception as e:
            logging.error(f"Error typing text: {e}")
            self.error_count += 1
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
            self.error_count += 1
            return False
    
    def wait_for_visual_change(self, reference_key: str, 
                              region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None,
                              timeout: float = BrowserConstants.DEFAULT_VERIFICATION_TIMEOUT) -> bool:
        """
        Wait for visual change in a region.
        
        Args:
            reference_key: Reference identifier
            region_identifier: Region to monitor
            timeout: Maximum time to wait
            
        Returns:
            True if change detected within timeout
        """
        try:
            changed, _, _ = self.visual_verifier.wait_for_change(
                reference_key,
                region_identifier,
                timeout=timeout
            )
            return changed
        except Exception as e:
            logging.error(f"Error waiting for visual change: {e}")
            self.error_count += 1
            return False
    
    def wait_for_visual_stability(self, region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None,
                                 duration: float = 1.0) -> bool:
        """
        Wait for a region to become visually stable.
        
        Args:
            region_identifier: Region to monitor
            duration: Duration of stability required
            
        Returns:
            True if region became stable
        """
        try:
            return self.visual_verifier.wait_for_stability(region_identifier, duration)
        except Exception as e:
            logging.error(f"Error waiting for visual stability: {e}")
            self.error_count += 1
            return False
    
    def update_health_status(self, status: BrowserHealthStatus) -> None:
        """
        Update the browser's health status.
        
        Args:
            status: New health status
        """
        self.health_status = status
        self.health_check_time = time.time()
        
        if status != BrowserHealthStatus.HEALTHY:
            logging.warning(f"Browser health status changed to {status.name}")
    
    def get_health_status(self) -> BrowserHealthStatus:
        """
        Get the health status of the browser.
        
        Returns:
            Current health status
        """
        # If status is unknown or check is old, perform a health check
        if (self.health_status == BrowserHealthStatus.UNKNOWN or 
            self.health_check_time is None or 
            time.time() - self.health_check_time > 60):  # Recheck after 60 seconds
            self._perform_health_check()
            
        return self.health_status
    
    def _perform_health_check(self) -> None:
        """Perform a health check and update status."""
        # Skip if browser is not running
        if self.state != "running":
            self.update_health_status(BrowserHealthStatus.UNKNOWN)
            return
        
        # Check if process is still running
        if self.process and self.process.poll() is not None:
            self.update_health_status(BrowserHealthStatus.UNHEALTHY)
            return
        
        # Try to take a screenshot
        try:
            screenshot = self.get_screenshot("screen_center")
            if screenshot is None:
                self.update_health_status(BrowserHealthStatus.DEGRADED)
                return
                
            # If error count is high, consider degraded
            if self.error_count > 5:
                self.update_health_status(BrowserHealthStatus.DEGRADED)
            else:
                self.update_health_status(BrowserHealthStatus.HEALTHY)
                
        except Exception:
            self.update_health_status(BrowserHealthStatus.DEGRADED)


class BrowserPool:
    """
    Manages a pool of browser instances for efficient reuse.
    """
    
    def __init__(self, max_size: int = BrowserConstants.DEFAULT_POOL_SIZE, config: Optional[Dict[str, Any]] = None):
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
        
        # Track browser usage
        self.browser_usage: Dict[str, int] = {}  # Browser ID -> usage count
        self.browser_last_used: Dict[str, float] = {}  # Browser ID -> timestamp
    
    def get_browser(self) -> ChromeBrowser:
        """
        Get a browser instance, reusing an existing one or creating new.
        
        Returns:
            ChromeBrowser instance
        """
        # Filter out closed browsers
        self._remove_closed_browsers()
        
        # Check for available browsers that are healthy
        available_browsers = []
        for bid, browser in self.browsers.items():
            if (browser.state == "running" and 
                not browser.is_closed() and 
                browser.get_health_status() != BrowserHealthStatus.UNHEALTHY):
                available_browsers.append(bid)
        
        if available_browsers:
            # Use the browser with the least usage
            available_browsers.sort(key=lambda bid: self.browser_usage.get(bid, 0))
            browser_id = available_browsers[0]
            self.active_browser = browser_id
            
            # Update usage statistics
            self.browser_usage[browser_id] = self.browser_usage.get(browser_id, 0) + 1
            self.browser_last_used[browser_id] = time.time()
            
            logging.debug(f"Reusing browser instance {browser_id} (used {self.browser_usage[browser_id]} times)")
            return self.browsers[browser_id]
        
        # Create a new browser
        browser = ChromeBrowser(self.config)
        self.browsers[browser.id] = browser
        self.active_browser = browser.id
        
        # Initialize usage statistics
        self.browser_usage[browser.id] = 1
        self.browser_last_used[browser.id] = time.time()
        
        # Enforce pool size limit
        if len(self.browsers) > self.max_size:
            self._cleanup_excess_browsers()
            
        return browser
    
    def _remove_closed_browsers(self) -> None:
        """Remove browsers that are already closed from the pool."""
        for browser_id in list(self.browsers.keys()):
            if (self.browsers[browser_id].state == "closed" or 
                self.browsers[browser_id].is_closed()):
                # Remove from pool
                del self.browsers[browser_id]
                # Remove from statistics
                if browser_id in self.browser_usage:
                    del self.browser_usage[browser_id]
                if browser_id in self.browser_last_used:
                    del self.browser_last_used[browser_id]
    
    def _cleanup_excess_browsers(self) -> None:
        """Close oldest or least used browsers when pool exceeds max size."""
        # Don't close the active browser
        browsers_to_close = [
            bid for bid in self.browsers.keys() 
            if bid != self.active_browser
        ]
        
        # Sort by criteria: unhealthy first, then by last used time (oldest first)
        def sort_key(browser_id):
            browser = self.browsers[browser_id]
            health_priority = {
                BrowserHealthStatus.UNHEALTHY: 0,
                BrowserHealthStatus.DEGRADED: 1,
                BrowserHealthStatus.UNKNOWN: 2,
                BrowserHealthStatus.HEALTHY: 3
            }
            health = health_priority.get(browser.get_health_status(), 2)
            last_used = self.browser_last_used.get(browser_id, 0)
            return (health, last_used)
            
        browsers_to_close.sort(key=sort_key)
        
        # Close browsers until we're under the limit
        browsers_to_remove = len(self.browsers) - self.max_size
        
        for browser_id in browsers_to_close[:browsers_to_remove]:
            try:
                logging.info(f"Closing excess browser {browser_id}")
                self.browsers[browser_id].close()
                
                # Remove from dictionaries
                del self.browsers[browser_id]
                if browser_id in self.browser_usage:
                    del self.browser_usage[browser_id]
                if browser_id in self.browser_last_used:
                    del self.browser_last_used[browser_id]
                    
            except Exception as e:
                logging.warning(f"Error closing excess browser {browser_id}: {e}")
    
    def close_all(self) -> None:
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
        self.browser_usage = {}
        self.browser_last_used = {}
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Get a health report for all browsers in the pool.
        
        Returns:
            Dictionary with health statistics
        """
        total = len(self.browsers)
        healthy = sum(1 for b in self.browsers.values() 
                    if b.get_health_status() == BrowserHealthStatus.HEALTHY)
        degraded = sum(1 for b in self.browsers.values() 
                     if b.get_health_status() == BrowserHealthStatus.DEGRADED)
        unhealthy = sum(1 for b in self.browsers.values() 
                      if b.get_health_status() == BrowserHealthStatus.UNHEALTHY)
        
        return {
            "total": total,
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
            "health_percentage": (healthy / total * 100) if total > 0 else 0
        }


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
        
        # Track session state
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = None
        self.end_time = None
        self.is_active = False
    
    def launch(self, url: str, profile_dir: Optional[str] = None) -> bool:
        """
        Launch a browser and start a session.
        
        Args:
            url: URL to navigate to
            profile_dir: Browser profile directory
            
        Returns:
            True if session started successfully
        """
        try:
            self.start_time = time.time()
            self.is_active = True
            
            if self.reuse_browser and self.pool:
                # Get browser from pool
                self.browser = self.pool.get_browser()
                
                # If browser is already running, navigate to URL
                if self.browser.state == "running":
                    try:
                        result = self.browser.navigate(url)
                        return result
                    except BrowserNavigationError:
                        # If navigation fails, try launching fresh
                        logging.warning("Navigation failed, will try launching fresh browser")
                        pass
            else:
                # Create a new browser instance
                self.browser = ChromeBrowser(self.config)
            
            # Launch the browser
            result = self.browser.launch(url, profile_dir)
            return result
            
        except Exception as e:
            logging.error(f"Error launching browser session: {e}")
            self.is_active = False
            return False
    
    def close(self) -> bool:
        """
        End the current browser session and clean up resources.
        
        Returns:
            True if session ended successfully
        """
        if not self.browser:
            self.is_active = False
            return True
            
        try:
            self.end_time = time.time()
            self.is_active = False
            
            # For pooled browsers, don't close automatically
            if self.reuse_browser and self.pool:
                return True
                
            # Close browser directly
            result = self.browser.close()
            self.browser = None
            return result
            
        except Exception as e:
            logging.error(f"Error closing browser session: {e}")
            return False
    
    def close_all(self) -> bool:
        """
        Close all browser instances and clean up resources.
        
        Returns:
            True if closed successfully
        """
        success = True
        self.is_active = False
        
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
            logging.error(f"Error closing all browsers: {e}")
            return False
    
    @contextmanager
    def session(self, url: str, profile_dir: Optional[str] = None):
        """
        Context manager for browser sessions.
        
        Args:
            url: URL to navigate to
            profile_dir: Browser profile directory
            
        Yields:
            The BrowserSession instance
        """
        try:
            self.launch(url, profile_dir)
            yield self
        finally:
            self.close()
    
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
    
    def type_text(self, text: str, delay: Optional[float] = None) -> bool:
        """Type text with optional delay between characters."""
        if not self.browser:
            return False
        return self.browser.type_text(text, delay)
    
    def send_prompt(self) -> bool:
        """Submit the current prompt."""
        if not self.browser:
            return False
        return self.browser.send_prompt()
    
    def wait_for_page_load(self, region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None, 
                           timeout: float = BrowserConstants.DEFAULT_PAGE_LOAD_TIMEOUT) -> bool:
        """Wait for the page to load."""
        if not self.browser:
            return False
        try:
            return self.browser.wait_for_page_load(region_identifier, timeout)
        except BrowserNotReadyError:
            return False
    
    def get_screenshot(self, region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None) -> Optional[Image.Image]:
        """Take a screenshot of the browser window or region."""
        if not self.browser:
            return None
        return self.browser.get_screenshot(region_identifier)
    
    def wait_for_visual_change(self, reference_key: str, 
                              region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None,
                              timeout: float = BrowserConstants.DEFAULT_VERIFICATION_TIMEOUT) -> bool:
        """Wait for visual change in a region."""
        if not self.browser:
            return False
        return self.browser.wait_for_visual_change(reference_key, region_identifier, timeout)
    
    def wait_for_visual_stability(self, region_identifier: Optional[Union[str, Tuple[int, int, int, int], Region]] = None,
                                 duration: float = 1.0) -> bool:
        """Wait for a region to become visually stable."""
        if not self.browser:
            return False
        return self.browser.wait_for_visual_stability(region_identifier, duration)
    
    def is_browser_ready(self) -> bool:
        """Check if browser is ready for interaction."""
        if not self.browser:
            return False
        return (self.browser.state == "running" and 
                not self.browser.is_closed() and
                self.browser.get_health_status() != BrowserHealthStatus.UNHEALTHY)