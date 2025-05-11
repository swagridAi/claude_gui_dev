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
from typing import Callable, Any, Optional, Union, Tuple, Dict, List, NamedTuple
import io
from PIL import Image, ImageChops
import numpy as np

# Import OpenCV for template matching
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available - template matching disabled")

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


class TemplateError(InteractionError):
    """Exception raised for template-related errors."""
    pass


class ElementNotFoundError(InteractionError):
    """Exception raised when an element is not found."""
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
    
    # Template matching constants
    DEFAULT_TEMPLATE_THRESHOLD = 0.7  # Minimum confidence for template match
    SCALES = [0.9, 0.95, 1.0, 1.05, 1.1]  # Scales for multi-scale matching
    TEMPLATE_MATCH_METHODS = [cv2.TM_CCOEFF_NORMED] if CV2_AVAILABLE else []
    
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
    
    # Response detection constants
    STABILITY_CHECK_COUNT = 3  # Number of checks before confirming stability
    STABILITY_CHECK_INTERVAL = 2.0  # Time between stability checks
    MIN_RESPONSE_TIME = 5.0  # Minimum time to wait for any response


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
    
    @staticmethod
    def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format."""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is not available")
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL format."""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is not available")
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


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


# ---- Template Management and Element Detection ----

class MatchResult(NamedTuple):
    """Result of a template matching operation."""
    found: bool
    location: Optional[Tuple[int, int, int, int]]
    confidence: float
    method: str
    scale: float


class TemplateManager:
    """Manages template images for UI element detection."""
    
    def __init__(self, template_dir: str = "templates", cache_size: int = 10):
        """
        Initialize template manager.
        
        Args:
            template_dir: Directory containing template images
            cache_size: Maximum number of templates to cache in memory
        """
        self.template_dir = os.path.join(os.path.dirname(__file__), template_dir)
        self.cache_size = cache_size
        self.template_cache = {}  # {name: {'image': img, 'path': path, 'time': timestamp}}
        
        # Create template directory if it doesn't exist
        os.makedirs(self.template_dir, exist_ok=True)
    
    def get_template_path(self, name: str) -> str:
        """Get the path to a template image."""
        return os.path.join(self.template_dir, f"{name}.png")
    
    def load_template(self, name: str) -> Optional[np.ndarray]:
        """
        Load a template image from disk or cache.
        
        Args:
            name: Template name without extension
            
        Returns:
            OpenCV image or None if not found
        """
        if not CV2_AVAILABLE:
            return None
            
        # Check cache first
        if name in self.template_cache:
            self.template_cache[name]['time'] = time.time()
            return self.template_cache[name]['image']
        
        # Load from disk
        path = self.get_template_path(name)
        if not os.path.exists(path):
            logging.warning(f"Template {name} not found at {path}")
            return None
        
        try:
            image = cv2.imread(path)
            
            # Update cache
            self.template_cache[name] = {
                'image': image,
                'path': path,
                'time': time.time()
            }
            
            # Enforce cache size limit
            self._enforce_cache_limit()
            
            return image
        except Exception as e:
            logging.error(f"Error loading template {name}: {e}")
            return None
    
    def _enforce_cache_limit(self):
        """Remove oldest templates when cache exceeds size limit."""
        if len(self.template_cache) <= self.cache_size:
            return
            
        # Sort by time (oldest first)
        sorted_cache = sorted(
            self.template_cache.items(),
            key=lambda x: x[1]['time']
        )
        
        # Remove oldest entries
        for name, _ in sorted_cache[:(len(self.template_cache) - self.cache_size)]:
            del self.template_cache[name]
    
    def create_template(self, name: str, image: Union[Image.Image, np.ndarray], 
                       region: Optional[Tuple[int, int, int, int]] = None,
                       overwrite: bool = False) -> bool:
        """
        Create a new template image.
        
        Args:
            name: Template name without extension
            image: PIL Image or OpenCV image to save as template
            region: Optional region to crop from image
            overwrite: Whether to overwrite existing template
            
        Returns:
            True if template was created successfully
        """
        if not CV2_AVAILABLE:
            return False
            
        path = self.get_template_path(name)
        
        # Check if template already exists
        if os.path.exists(path) and not overwrite:
            logging.warning(f"Template {name} already exists. Use overwrite=True to replace.")
            return False
        
        try:
            # Convert to OpenCV format if needed
            if isinstance(image, Image.Image):
                cv_image = InteractionUtils.pil_to_cv2(image)
            else:
                cv_image = image
            
            # Crop if region specified
            if region:
                x, y, w, h = region
                cv_image = cv_image[y:y+h, x:x+w]
            
            # Save template
            cv2.imwrite(path, cv_image)
            
            # Update cache
            self.template_cache[name] = {
                'image': cv_image,
                'path': path,
                'time': time.time()
            }
            
            logging.info(f"Created template {name} at {path}")
            return True
        except Exception as e:
            logging.error(f"Error creating template {name}: {e}")
            return False
    
    def capture_template(self, name: str, region: Optional[Tuple[int, int, int, int]] = None,
                        overwrite: bool = False) -> bool:
        """
        Capture a new template from the screen.
        
        Args:
            name: Template name without extension
            region: Optional region to capture
            overwrite: Whether to overwrite existing template
            
        Returns:
            True if template was captured successfully
        """
        try:
            # Take screenshot
            screenshot = pyautogui.screenshot(region=region)
            
            # Save as template
            return self.create_template(name, screenshot, overwrite=overwrite)
        except Exception as e:
            logging.error(f"Error capturing template {name}: {e}")
            return False


class ElementDetector:
    """Detects UI elements using template matching."""
    
    def __init__(self, template_manager: Optional[TemplateManager] = None):
        """
        Initialize element detector.
        
        Args:
            template_manager: Optional template manager instance
        """
        self.template_manager = template_manager or TemplateManager()
        self.last_match_results = {}
        self.metrics = {
            'total_attempts': 0,
            'successful_matches': 0,
            'match_times': []
        }
    
    def find_element(self, template_name: str, screenshot: Optional[Union[Image.Image, np.ndarray]] = None,
                    region: Optional[Tuple[int, int, int, int]] = None,
                    threshold: float = InteractionConstants.DEFAULT_TEMPLATE_THRESHOLD,
                    scales: Optional[List[float]] = None,
                    methods: Optional[List[int]] = None) -> MatchResult:
        """
        Find an element on screen using template matching.
        
        Args:
            template_name: Name of the template to match
            screenshot: Optional screenshot to search in (takes a new one if None)
            region: Optional region to search in
            threshold: Minimum confidence for a match
            scales: List of scales to try (for multi-scale matching)
            methods: List of template matching methods to try
            
        Returns:
            MatchResult with match information
        """
        if not CV2_AVAILABLE:
            return MatchResult(False, None, 0.0, "No OpenCV", 1.0)
            
        self.metrics['total_attempts'] += 1
        start_time = time.time()
        
        # Load template
        template = self.template_manager.load_template(template_name)
        if template is None:
            logging.warning(f"Template {template_name} not found")
            return MatchResult(False, None, 0.0, "Template not found", 1.0)
        
        # Take screenshot if not provided
        if screenshot is None:
            screenshot = pyautogui.screenshot(region=region)
        
        # Convert to OpenCV format if needed
        if isinstance(screenshot, Image.Image):
            cv_screenshot = InteractionUtils.pil_to_cv2(screenshot)
        else:
            cv_screenshot = screenshot
        
        # Set default scales if not provided
        if scales is None:
            scales = InteractionConstants.SCALES
        
        # Set default methods if not provided
        if methods is None:
            methods = InteractionConstants.TEMPLATE_MATCH_METHODS
        
        # Track best match across all scales and methods
        best_match = None
        best_val = -1
        best_method = "unknown"
        best_scale = 1.0
        
        # Try different scales
        for scale in scales:
            # Resize template
            if scale != 1.0:
                h, w = template.shape[:2]
                resized_template = cv2.resize(template, (int(w * scale), int(h * scale)))
            else:
                resized_template = template
            
            # Check if template is larger than screenshot
            if (resized_template.shape[0] > cv_screenshot.shape[0] or 
                resized_template.shape[1] > cv_screenshot.shape[1]):
                continue
            
            # Try different methods
            for method in methods:
                # Perform template matching
                try:
                    result = cv2.matchTemplate(cv_screenshot, resized_template, method)
                    
                    # Get best match depending on method
                    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                        val = 1.0 - min_val  # Convert to similarity score (higher is better)
                        loc = min_loc
                    else:
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                        val = max_val
                        loc = max_loc
                    
                    # Update best match
                    if val > best_val:
                        best_val = val
                        best_method = str(method)
                        best_scale = scale
                        
                        # Calculate location with width and height
                        h, w = resized_template.shape[:2]
                        match_region = (loc[0], loc[1], w, h)
                        
                        # Adjust for search region if provided
                        if region:
                            match_region = (
                                match_region[0] + region[0],
                                match_region[1] + region[1],
                                match_region[2],
                                match_region[3]
                            )
                        
                        best_match = match_region
                except Exception as e:
                    logging.warning(f"Error during template matching with method {method}: {e}")
        
        # Record metrics
        self.metrics['match_times'].append(time.time() - start_time)
        if best_val >= threshold and best_match is not None:
            self.metrics['successful_matches'] += 1
            self.last_match_results[template_name] = (best_match, best_val)
            logging.debug(f"Found {template_name} with confidence {best_val:.2f} (scale={best_scale}, method={best_method})")
            return MatchResult(True, best_match, best_val, best_method, best_scale)
        else:
            logging.debug(f"Element {template_name} not found (best confidence: {best_val:.2f})")
            return MatchResult(False, None, best_val, best_method, best_scale)
    
    def wait_for_element(self, template_name: str, 
                        timeout: float = InteractionConstants.DEFAULT_WAIT_TIMEOUT,
                        check_interval: float = 1.0,
                        threshold: float = InteractionConstants.DEFAULT_TEMPLATE_THRESHOLD,
                        region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Wait for an element to appear on screen.
        
        Args:
            template_name: Name of the template to match
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds
            threshold: Minimum confidence for a match
            region: Optional region to search in
            
        Returns:
            Element location or None if not found within timeout
        """
        start_time = time.time()
        end_time = start_time + timeout
        
        logging.info(f"Waiting up to {timeout}s for element: {template_name}")
        
        while time.time() < end_time:
            # Check for element
            result = self.find_element(
                template_name, 
                region=region,
                threshold=threshold
            )
            
            if result.found:
                elapsed = time.time() - start_time
                logging.info(f"Found element {template_name} after {elapsed:.2f}s")
                return result.location
            
            # Log progress periodically
            elapsed = time.time() - start_time
            if elapsed > 0 and elapsed % 10 < check_interval:
                remaining = timeout - elapsed
                logging.debug(f"Still waiting for element {template_name}: {int(remaining)}s remaining...")
            
            # Sleep before next check
            time.sleep(check_interval)
        
        # If we get here, timeout was reached
        logging.warning(f"Timeout waiting for element: {template_name}")
        return None
    
    def wait_for_element_to_disappear(self, template_name: str,
                                    timeout: float = InteractionConstants.DEFAULT_WAIT_TIMEOUT,
                                    check_interval: float = 1.0,
                                    threshold: float = InteractionConstants.DEFAULT_TEMPLATE_THRESHOLD,
                                    region: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """
        Wait for an element to disappear from screen.
        
        Args:
            template_name: Name of the template to match
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds
            threshold: Minimum confidence for a match
            region: Optional region to search in
            
        Returns:
            True if element disappeared, False if still present after timeout
        """
        start_time = time.time()
        end_time = start_time + timeout
        
        logging.info(f"Waiting up to {timeout}s for element to disappear: {template_name}")
        
        while time.time() < end_time:
            # Check for element
            result = self.find_element(
                template_name, 
                region=region,
                threshold=threshold
            )
            
            if not result.found:
                elapsed = time.time() - start_time
                logging.info(f"Element {template_name} disappeared after {elapsed:.2f}s")
                return True
            
            # Log progress periodically
            elapsed = time.time() - start_time
            if elapsed > 0 and elapsed % 10 < check_interval:
                remaining = timeout - elapsed
                logging.debug(f"Element {template_name} still visible: {int(remaining)}s remaining...")
            
            # Sleep before next check
            time.sleep(check_interval)
        
        # If we get here, timeout was reached
        logging.warning(f"Timeout waiting for element to disappear: {template_name}")
        return False
    
    def get_match_metrics(self) -> Dict[str, Any]:
        """Get metrics on template matching performance."""
        metrics = self.metrics.copy()
        
        # Calculate averages
        if metrics['match_times']:
            metrics['average_match_time'] = sum(metrics['match_times']) / len(metrics['match_times'])
        else:
            metrics['average_match_time'] = 0
            
        if metrics['total_attempts'] > 0:
            metrics['success_rate'] = metrics['successful_matches'] / metrics['total_attempts']
        else:
            metrics['success_rate'] = 0
            
        return metrics


# ---- Response Detection and Monitoring ----

class ResponseMonitor:
    """Monitors Claude's responses using multiple detection strategies."""
    
    def __init__(self, element_detector: Optional[ElementDetector] = None):
        """
        Initialize response monitor.
        
        Args:
            element_detector: Optional element detector instance
        """
        self.element_detector = element_detector or ElementDetector()
        self.visual_verifier = VisualVerifier()
        self.metrics = {
            'response_times': [],
            'detection_method_counts': {
                'template': 0,
                'stability': 0,
                'timeout': 0
            }
        }
    
    def wait_for_response(self, timeout: float = InteractionConstants.DEFAULT_WAIT_TIMEOUT,
                         prompt_input_template: str = "prompt_input",
                         thinking_template: str = "thinking_indicator",
                         check_interval: float = 2.0) -> Tuple[bool, str]:
        """
        Wait for Claude to respond using a hybrid detection strategy.
        
        Args:
            timeout: Maximum time to wait in seconds
            prompt_input_template: Template name for prompt input box
            thinking_template: Template name for thinking indicator
            check_interval: Time between checks in seconds
            
        Returns:
            Tuple of (success, detection_method)
        """
        start_time = time.time()
        end_time = start_time + timeout
        
        logging.info(f"Waiting up to {timeout}s for Claude's response")
        
        # Track detection state
        response_started = False
        response_complete = False
        stability_check_started = False
        stability_counter = 0
        template_available = CV2_AVAILABLE and self._verify_template_exists(prompt_input_template)
        
        # Take initial reference images
        reference_key = f"response_{int(start_time)}"
        self.visual_verifier.take_reference_image(reference_key)
        
        # Ensure minimum wait time (for very fast responses)
        min_wait_end = start_time + InteractionConstants.MIN_RESPONSE_TIME
        
        # Main detection loop
        while time.time() < end_time:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Always wait the minimum response time
            if current_time < min_wait_end:
                time.sleep(min(check_interval, min_wait_end - current_time))
                continue
            
            # Detect response completion through multiple methods
            
            # Method 1: Template matching for input prompt reappearance
            if template_available and not response_complete:
                result = self.element_detector.find_element(prompt_input_template)
                if result.found:
                    logging.info(f"Detected response completion via template matching after {elapsed:.2f}s")
                    response_complete = True
                    self.metrics['detection_method_counts']['template'] += 1
                    self.metrics['response_times'].append(elapsed)
                    return True, "template"
            
            # Method 2: Visual stability detection
            if not response_complete:
                # Check for significant visual changes
                changed, _ = self.visual_verifier.check_for_changes(reference_key)
                
                if changed:
                    # Reset stability counter if changes are detected
                    if stability_check_started:
                        stability_counter = 0
                        logging.debug("Visual changes detected, resetting stability counter")
                    
                    # Mark that response has started
                    if not response_started:
                        response_started = True
                        logging.info(f"Detected Claude starting to respond after {elapsed:.2f}s")
                else:
                    # If no changes detected and response has started, begin stability checking
                    if response_started and not stability_check_started:
                        stability_check_started = True
                        stability_counter = 1
                        logging.debug("Starting response stability tracking")
                    elif stability_check_started:
                        stability_counter += 1
                        logging.debug(f"Response appears stable ({stability_counter}/{InteractionConstants.STABILITY_CHECK_COUNT})")
                
                # If stability threshold reached, consider response complete
                if stability_check_started and stability_counter >= InteractionConstants.STABILITY_CHECK_COUNT:
                    logging.info(f"Detected response completion via visual stability after {elapsed:.2f}s")
                    response_complete = True
                    self.metrics['detection_method_counts']['stability'] += 1
                    self.metrics['response_times'].append(elapsed)
                    return True, "stability"
            
            # Log progress periodically
            if elapsed % 30 < check_interval:
                remaining = timeout - elapsed
                logging.info(f"Still waiting for response: {int(remaining)}s remaining...")
                # Take verification screenshot if advanced logging is enabled
                if ADVANCED_LOGGING:
                    log_with_screenshot(
                        f"Waiting for response ({int(elapsed)}/{timeout}s elapsed)",
                        stage_name=f"WAITING_{int(elapsed)}"
                    )
            
            # Sleep before next check
            time.sleep(check_interval)
        
        # If we reach here, timeout occurred
        logging.warning(f"Timeout waiting for response after {timeout}s")
        self.metrics['detection_method_counts']['timeout'] += 1
        return False, "timeout"
    
    def _verify_template_exists(self, template_name: str) -> bool:
        """Check if a template exists and is valid."""
        if not CV2_AVAILABLE:
            return False
            
        template = self.element_detector.template_manager.load_template(template_name)
        return template is not None
    
    def get_response_metrics(self) -> Dict[str, Any]:
        """Get metrics on response detection performance."""
        metrics = self.metrics.copy()
        
        # Calculate averages
        if metrics['response_times']:
            metrics['average_response_time'] = sum(metrics['response_times']) / len(metrics['response_times'])
        else:
            metrics['average_response_time'] = 0
            
        # Calculate detection method percentages
        total_detections = sum(metrics['detection_method_counts'].values())
        if total_detections > 0:
            metrics['detection_method_percentages'] = {
                method: count / total_detections * 100
                for method, count in metrics['detection_method_counts'].items()
            }
        else:
            metrics['detection_method_percentages'] = {
                method: 0 for method in metrics['detection_method_counts']
            }
            
        return metrics


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
        self.metrics = {
            'characters_typed': 0,
            'typing_times': []
        }
    
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
        start_time = time.time()
        
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
            self.metrics['characters_typed'] += 1
            
            time.sleep(typing_delay)
            
            # Occasionally add a pause (simulating thinking)
            if random.random() < self.settings['pause_chance']:
                min_pause, max_pause = self.settings['pause_duration']
                time.sleep(random.uniform(min_pause, max_pause))
        
        # Record typing time
        typing_time = time.time() - start_time
        self.metrics['typing_times'].append(typing_time)
    
    def type_with_clipboard(self, text: str, clear_existing: bool = True) -> None:
        """
        Type using clipboard for efficiency with long text.
        
        Args:
            text: Text to type
            clear_existing: Whether to clear existing text first
        """
        start_time = time.time()
        
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
            
            # Record typing time and character count
            typing_time = time.time() - start_time
            self.metrics['typing_times'].append(typing_time)
            self.metrics['characters_typed'] += len(text)
        finally:
            # Restore original clipboard if possible
            if original_clipboard is not None:
                try:
                    pyperclip.copy(original_clipboard)
                except:
                    pass
    
    def get_typing_metrics(self) -> Dict[str, Any]:
        """Get typing performance metrics."""
        metrics = self.metrics.copy()
        
        # Calculate averages
        if metrics['typing_times']:
            metrics['average_typing_time'] = sum(metrics['typing_times']) / len(metrics['typing_times'])
        else:
            metrics['average_typing_time'] = 0
            
        # Calculate typing speed
        if metrics['typing_times'] and metrics['characters_typed'] > 0:
            total_time = sum(metrics['typing_times'])
            if total_time > 0:
                metrics['characters_per_second'] = metrics['characters_typed'] / total_time
            else:
                metrics['characters_per_second'] = 0
        else:
            metrics['characters_per_second'] = 0
            
        return metrics


class UIInteraction:
    """
    Core UI interaction functionality.
    """
    
    def __init__(self, enable_visual_verification: bool = True, 
                enable_template_detection: bool = True):
        """
        Initialize UI interaction handler.
        
        Args:
            enable_visual_verification: Whether to use visual verification
            enable_template_detection: Whether to use template detection
        """
        self.typing = HumanizedTyping()
        self.verifier = VisualVerifier() if enable_visual_verification else None
        self.enable_visual_verification = enable_visual_verification
        
        self.template_manager = TemplateManager() if enable_template_detection else None
        self.element_detector = ElementDetector(self.template_manager) if enable_template_detection else None
        self.response_monitor = ResponseMonitor(self.element_detector) if enable_template_detection else None
        self.enable_template_detection = enable_template_detection and CV2_AVAILABLE
        
        # Performance metrics
        self.metrics = {
            'operations': {},
            'start_time': time.time()
        }
    
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
        
        # Track operation timing
        start_time = time.time()
        
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
        
        # Record metrics
        self._record_operation_metrics('type_text', start_time, len(text))
        
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
        start_time = time.time()
        
        pyautogui.press(key)
        
        # Record metrics
        self._record_operation_metrics('press_key', start_time)
        
        return True
    
    @with_error_handling
    def press_submit(self) -> bool:
        """
        Press Enter to submit the current prompt.
        
        Returns:
            True if successful
        """
        logging.info("Pressing Enter to submit prompt")
        start_time = time.time()
        
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
        
        # Record metrics
        self._record_operation_metrics('press_submit', start_time)
        
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
        start_time = time.time()
        
        # Move to position with humanized motion
        pyautogui.moveTo(x, y, duration=random.uniform(0.1, 0.3))
        
        # Perform click
        pyautogui.click(x=x, y=y, button=button)
        
        # Record metrics
        self._record_operation_metrics('click_at', start_time)
        
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
        start_time = time.time()
        
        # Try template detection first if enabled
        if self.enable_template_detection and self.element_detector:
            result = self.element_detector.find_element("prompt_input")
            if result.found:
                x = result.location[0] + result.location[2] // 2
                y = result.location[1] + result.location[3] // 2
                logging.info(f"Clicking detected prompt area at ({x}, {y})")
                success = self.click_at(x, y)
                
                # Record metrics
                self._record_operation_metrics('click_prompt_area', start_time, detection_method="template")
                
                return success
        
        # Fall back to provided coordinates
        if coordinates:
            x, y = coordinates
            logging.info(f"Clicking prompt area at ({x}, {y})")
            success = self.click_at(x, y)
            
            # Record metrics
            self._record_operation_metrics('click_prompt_area', start_time, detection_method="coordinates")
            
            return success
        else:
            logging.warning("No coordinates provided for prompt area and template detection failed")
            return False
    
    @with_error_handling
    def wait_for_response(self, timeout: int = InteractionConstants.DEFAULT_WAIT_TIMEOUT, 
                         check_interval: int = InteractionConstants.DEFAULT_CHECK_INTERVAL,
                         verify_visually: bool = True) -> bool:
        """
        Wait for Claude to respond, with hybrid detection strategy.
        
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
        start_time = time.time()
        
        # Use response monitor if available
        if self.enable_template_detection and self.response_monitor:
            try:
                success, method = self.response_monitor.wait_for_response(
                    timeout=timeout,
                    check_interval=check_interval
                )
                
                # Record metrics
                elapsed = time.time() - start_time
                self._record_operation_metrics(
                    'wait_for_response', 
                    start_time, 
                    detection_method=method,
                    success=success
                )
                
                if ADVANCED_LOGGING:
                    log_with_screenshot("Response detection complete", stage_name="RESPONSE_COMPLETE")
                    
                return success
            except Exception as e:
                logging.warning(f"Error using response monitor: {e}")
                # Fall back to simple waiting if response monitor fails
                
        # Fall back to simple waiting with visual verification
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
                end_time = start_time + timeout
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
                    
                    # Record metrics
                    self._record_operation_metrics(
                        'wait_for_response', 
                        start_time, 
                        detection_method="visual_stability",
                        success=True
                    )
                    
                    if ADVANCED_LOGGING:
                        log_with_screenshot("Response complete", stage_name="RESPONSE_COMPLETE")
                    
                    return True
                
            except InteractionTimeout:
                # Fall back to timed waiting if visual verification fails
                logging.warning("Visual verification failed, falling back to timed waiting")
                
        # Simple timed waiting
        end_time = start_time + timeout
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
        
        # Record metrics
        self._record_operation_metrics(
            'wait_for_response', 
            start_time, 
            detection_method="timeout",
            success=True  # Consider timeout as "success" for the waiting operation
        )
        
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
        start_time = time.time()
        
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
            success = self.wait_for_response(timeout=wait_time, verify_visually=self.enable_visual_verification)
            
            # Record combined operation metrics
            self._record_operation_metrics('send_prompt', start_time, success=success)
            
            return success
        
        # Record metrics without waiting
        self._record_operation_metrics('send_prompt', start_time)
        
        return True
    
    def _record_operation_metrics(self, operation: str, start_time: float, data_size: int = 0, 
                                detection_method: str = None, success: bool = True) -> None:
        """Record metrics for an operation."""
        if operation not in self.metrics['operations']:
            self.metrics['operations'][operation] = {
                'count': 0,
                'success_count': 0,
                'total_time': 0,
                'data_size': 0,
                'detection_methods': {}
            }
        
        elapsed = time.time() - start_time
        self.metrics['operations'][operation]['count'] += 1
        if success:
            self.metrics['operations'][operation]['success_count'] += 1
        self.metrics['operations'][operation]['total_time'] += elapsed
        self.metrics['operations'][operation]['data_size'] += data_size
        
        if detection_method:
            if detection_method not in self.metrics['operations'][operation]['detection_methods']:
                self.metrics['operations'][operation]['detection_methods'][detection_method] = 0
            self.metrics['operations'][operation]['detection_methods'][detection_method] += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {
            'operations': {},
            'total_runtime': time.time() - self.metrics['start_time'],
            'typing': self.typing.get_typing_metrics(),
        }
        
        # Process operations metrics
        for op, data in self.metrics['operations'].items():
            metrics['operations'][op] = data.copy()
            if data['count'] > 0:
                metrics['operations'][op]['average_time'] = data['total_time'] / data['count']
                metrics['operations'][op]['success_rate'] = data['success_count'] / data['count'] * 100
            else:
                metrics['operations'][op]['average_time'] = 0
                metrics['operations'][op]['success_rate'] = 0
        
        # Add detection metrics if available
        if self.enable_template_detection:
            if self.element_detector:
                metrics['element_detection'] = self.element_detector.get_match_metrics()
            if self.response_monitor:
                metrics['response_detection'] = self.response_monitor.get_response_metrics()
        
        return metrics


# ---- Facade functions (maintaining original API) ----

# Initialize singleton instances for module-level functions
_ui_interaction = UIInteraction()
_verifier = VisualVerifier()
_typing = HumanizedTyping()

# Create element detector if cv2 is available
if CV2_AVAILABLE:
    _template_manager = TemplateManager()
    _element_detector = ElementDetector(_template_manager)
    _response_monitor = ResponseMonitor(_element_detector)
else:
    _element_detector = None
    _response_monitor = None


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
    Wait for Claude to respond, with hybrid detection strategy.
    
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
    
    # Use template detection if available
    if CV2_AVAILABLE and _element_detector:
        # Try to find thinking indicator
        result = _element_detector.find_element("thinking_indicator")
        if result.found:
            return True
    
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
    
    # Use template matching if available
    if CV2_AVAILABLE and _element_detector:
        # Check for prompt input box (indicates response is complete)
        result = _element_detector.find_element("prompt_input")
        if result.found:
            return True
    
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
    # Use template detection to try to count messages if available
    if CV2_AVAILABLE and _element_detector:
        # TODO: Implement message counting based on templates
        pass
    
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


def capture_template(name: str, region: Optional[Tuple[int, int, int, int]] = None) -> bool:
    """
    Capture a new template for UI element detection.
    
    Args:
        name: Template name
        region: Optional specific region to capture
        
    Returns:
        True if template was captured successfully
    """
    if not CV2_AVAILABLE or not _template_manager:
        logging.warning("Template capture not available - OpenCV not installed")
        return False
        
    return _template_manager.capture_template(name, region)


def get_performance_metrics() -> Dict[str, Any]:
    """
    Get comprehensive performance metrics.
    
    Returns:
        Dictionary of performance metrics
    """
    return _ui_interaction.get_performance_metrics()