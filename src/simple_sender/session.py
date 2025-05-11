#!/usr/bin/env python3
"""
Session Management Module for Simple Claude Sender

This module provides classes and functions for managing Claude AI automation sessions,
including session execution, tracking, resumption, and reporting.
"""

import logging
import time
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Type
import random
from enum import Enum, auto
from pathlib import Path


# ---- Constants ----

class SessionConstants:
    """Centralized constants for session management."""
    
    # Timing constants
    DEFAULT_PROMPT_DELAY = 300
    DEFAULT_SESSION_DELAY = 10
    MIN_DETECTION_INTERVAL = 2
    MAX_DETECTION_INTERVAL = 10
    
    # Retry constants
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 2
    DEFAULT_RETRY_BACKOFF = 1.5
    DEFAULT_RETRY_JITTER = 0.5
    
    # Checkpoint constants
    CHECKPOINT_VERSION = "1.1"
    
    # Detection constants
    MIN_STABLE_DURATION = 5
    DEFAULT_DETECTION_TIMEOUT = 300
    VISUAL_CHANGE_THRESHOLD = 0.05


# ---- Event Definitions ----

class ProgressEvents:
    """Standardized event names for progress callbacks."""
    
    # Session events
    SESSION_STARTING = "session_starting"
    SESSION_STARTED = "session_started"
    SESSION_COMPLETED = "session_completed"
    SESSION_FAILED = "session_failed"
    
    # Prompt events
    PROMPT_STARTED = "prompt_started"
    PROMPT_COMPLETED = "prompt_completed"
    PROMPT_FAILED = "prompt_failed"
    
    # Response events
    RESPONSE_STARTED = "response_started"
    RESPONSE_DETECTED = "response_detected"
    RESPONSE_COMPLETED = "response_completed"
    RESPONSE_TIMEOUT = "response_timeout"
    
    # Run events
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


# ---- State Enumerations ----

class SessionState(Enum):
    """Represents the possible states of a session."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    INTERRUPTED = auto()


class PromptState(Enum):
    """Represents the possible states of a prompt execution."""
    PENDING = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()


class DetectionMethod(Enum):
    """Represents the method used to detect response completion."""
    TIMEOUT = auto()         # Fixed timeout
    VISUAL_CHANGE = auto()   # Visual change detection
    TEMPLATE_MATCH = auto()  # Template matching
    CONTENT_STABLE = auto()  # Content stabilization
    HYBRID = auto()          # Combination of methods


# ---- Error Hierarchy ----

class SessionError(Exception):
    """Base exception for session-related errors."""
    pass


class SessionConfigError(SessionError):
    """Exception raised for session configuration errors."""
    pass


class SessionExecutionError(SessionError):
    """Exception raised when session execution fails."""
    pass


class BrowserError(SessionError):
    """Exception raised for browser-related errors."""
    pass


class PromptExecutionError(SessionError):
    """Exception raised when prompt execution fails."""
    pass


class CheckpointError(SessionError):
    """Exception raised for checkpoint-related errors."""
    pass


class DetectionError(SessionError):
    """Exception raised for response detection errors."""
    pass


# ---- Core Classes ----

class ResponseDetector:
    """
    Detects when Claude has completed its response using multiple strategies.
    Uses a hybrid approach combining visual detection and timeouts.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize response detector with optional configuration."""
        self.config = config or {}
        self.last_change_time = 0
        self.detection_start_time = 0
        self.consecutive_stable_checks = 0
        self.detection_history = []
        self.template_detection_available = False
        
        # Try to check if template detection is available
        try:
            # Check if interaction module has template detection
            # This is a lightweight check that doesn't import unless necessary
            local_vars = {}
            exec("from src.simple_sender.interaction import verify_response_complete", globals(), local_vars)
            self.template_detection_available = "verify_response_complete" in local_vars
        except ImportError:
            self.template_detection_available = False
    
    def detect_response_completion(self, browser_handler, timeout: int = SessionConstants.DEFAULT_DETECTION_TIMEOUT) -> Tuple[bool, DetectionMethod]:
        """
        Detect when Claude has completed its response using various strategies.
        
        Args:
            browser_handler: Handler for browser interactions
            timeout: Maximum time to wait for response completion
            
        Returns:
            Tuple of (success, detection_method)
        """
        # Reset detection state
        self.detection_start_time = time.time()
        self.last_change_time = time.time()
        self.consecutive_stable_checks = 0
        self.detection_history = []
        
        # Log start of detection
        logging.info(f"Starting response detection (timeout: {timeout}s)")
        
        # Determine available detection methods
        methods = self._get_available_detection_methods(browser_handler)
        
        # First wait for initial response (Claude thinking/typing)
        initial_response_detected = self._wait_for_initial_response(browser_handler, timeout)
        if not initial_response_detected:
            logging.warning("Could not detect initial response from Claude")
            # Continue anyway, as Claude might be responding without visual indicators
        
        # Main detection loop
        end_time = self.detection_start_time + timeout
        check_interval = SessionConstants.MIN_DETECTION_INTERVAL
        
        while time.time() < end_time:
            # Try each detection method
            for method in methods:
                result = self._try_detection_method(method, browser_handler)
                if result:
                    elapsed = time.time() - self.detection_start_time
                    logging.info(f"Response completed (detected by {method.name}, elapsed: {elapsed:.1f}s)")
                    return True, method
                
            # Pause before next check (adaptive interval)
            elapsed = time.time() - self.detection_start_time
            # Increase check interval as time passes (less frequent checks)
            check_interval = min(
                SessionConstants.MIN_DETECTION_INTERVAL + (elapsed / 30), 
                SessionConstants.MAX_DETECTION_INTERVAL
            )
            
            # Log progress periodically
            if int(elapsed) % 30 < check_interval:
                remaining = timeout - elapsed
                logging.info(f"Still waiting for response completion: {int(remaining)}s remaining...")
            
            time.sleep(check_interval)
        
        # If we reach here, timeout was reached
        logging.warning(f"Response detection timed out after {timeout}s")
        return False, DetectionMethod.TIMEOUT
    
    def _get_available_detection_methods(self, browser_handler) -> List[DetectionMethod]:
        """Determine which detection methods are available."""
        methods = []
        
        # Template matching if available
        if self.template_detection_available:
            methods.append(DetectionMethod.TEMPLATE_MATCH)
        
        # Visual change detection
        if hasattr(browser_handler, 'get_screenshot'):
            methods.append(DetectionMethod.VISUAL_CHANGE)
        
        # Content stability detection
        methods.append(DetectionMethod.CONTENT_STABLE)
        
        # Always include timeout as fallback
        if DetectionMethod.TIMEOUT not in methods:
            methods.append(DetectionMethod.TIMEOUT)
        
        return methods
    
    def _wait_for_initial_response(self, browser_handler, timeout: int) -> bool:
        """Wait for initial response (Claude thinking/typing)."""
        start_time = time.time()
        max_wait = min(30, timeout * 0.2)  # Wait up to 30s or 20% of timeout
        
        if self.template_detection_available:
            try:
                # Use specialized detection if available
                from src.simple_sender.interaction import verify_claude_responding
                return verify_claude_responding()
            except (ImportError, AttributeError):
                pass
        
        # Simple waiting strategy
        time.sleep(2)  # Small initial wait
        
        # Take reference screenshot if possible
        reference_screenshot = None
        if hasattr(browser_handler, 'get_screenshot'):
            reference_screenshot = browser_handler.get_screenshot()
        
        # Check for visual changes
        if reference_screenshot:
            time.sleep(2)  # Wait before comparing
            current_screenshot = browser_handler.get_screenshot()
            if self._detect_visual_change(reference_screenshot, current_screenshot):
                return True
        
        # If no detection method worked, assume response started
        return True
    
    def _try_detection_method(self, method: DetectionMethod, browser_handler) -> bool:
        """
        Try a specific detection method.
        
        Args:
            method: The detection method to try
            browser_handler: Browser handler for interactions
            
        Returns:
            True if response completion detected, False otherwise
        """
        try:
            if method == DetectionMethod.TEMPLATE_MATCH:
                # Use template matching if available
                from src.simple_sender.interaction import verify_response_complete
                return verify_response_complete()
            
            elif method == DetectionMethod.VISUAL_CHANGE:
                # Check for visual stability
                return self._check_visual_stability(browser_handler)
            
            elif method == DetectionMethod.CONTENT_STABLE:
                # Check for content stability
                return self._check_content_stability(browser_handler)
            
            else:
                # Unknown method
                return False
                
        except Exception as e:
            logging.warning(f"Error using detection method {method.name}: {e}")
            return False
    
    def _check_visual_stability(self, browser_handler) -> bool:
        """
        Check if the screen has been visually stable (unchanged).
        
        Args:
            browser_handler: Browser handler with screenshot capability
            
        Returns:
            True if screen is stable, False otherwise
        """
        if not hasattr(browser_handler, 'get_screenshot'):
            return False
        
        # Get current and previous screenshots
        current_screenshot = browser_handler.get_screenshot()
        if not current_screenshot:
            return False
        
        if not self.detection_history:
            # First check, save screenshot and return
            self.detection_history.append(current_screenshot)
            return False
        
        # Compare with most recent screenshot
        prev_screenshot = self.detection_history[-1]
        
        # Check for significant changes
        has_changed = self._detect_visual_change(prev_screenshot, current_screenshot)
        
        # Save current screenshot for next comparison
        self.detection_history.append(current_screenshot)
        if len(self.detection_history) > 3:
            self.detection_history.pop(0)  # Keep only last 3 screenshots
        
        if has_changed:
            # Reset stability counter if changes detected
            self.consecutive_stable_checks = 0
            self.last_change_time = time.time()
            return False
        else:
            # Increment stability counter
            self.consecutive_stable_checks += 1
            
            # If stable for several checks, consider response complete
            if self.consecutive_stable_checks >= 3:
                stable_duration = time.time() - self.last_change_time
                if stable_duration >= SessionConstants.MIN_STABLE_DURATION:
                    return True
        
        return False
    
    def _check_content_stability(self, browser_handler) -> bool:
        """
        Check if content has been stable for sufficient time.
        This is a simpler alternative when screenshots aren't available.
        
        Args:
            browser_handler: Browser handler
            
        Returns:
            True if content appears stable
        """
        # If we've seen no changes for a while, assume response is complete
        stable_duration = time.time() - self.last_change_time
        
        # Criteria: No changes for at least MIN_STABLE_DURATION seconds
        # and at least MINIMUM_RESPONSE_TIME has passed
        if stable_duration >= SessionConstants.MIN_STABLE_DURATION:
            elapsed = time.time() - self.detection_start_time
            if elapsed >= 10:  # Minimum plausible response time
                return True
        
        return False
    
    def _detect_visual_change(self, img1, img2) -> bool:
        """
        Detect if there's a significant visual change between two images.
        
        Args:
            img1: First image (PIL Image)
            img2: Second image (PIL Image)
            
        Returns:
            True if significant change detected
        """
        try:
            import numpy as np
            from PIL import ImageChops
            
            # Calculate difference
            diff = ImageChops.difference(img1, img2)
            diff_array = np.array(diff)
            
            # Calculate statistics
            non_zero_pixels = np.count_nonzero(diff_array)
            total_pixels = diff_array.size / 3  # RGB has 3 values per pixel
            difference_percentage = non_zero_pixels / total_pixels
            
            # Determine if change is significant
            return difference_percentage > SessionConstants.VISUAL_CHANGE_THRESHOLD
            
        except (ImportError, Exception) as e:
            logging.warning(f"Error comparing images: {e}")
            return False


class PromptResult:
    """Represents the result of a single prompt execution."""
    
    def __init__(self, prompt_text: str, index: int):
        self.prompt_text = prompt_text
        self.index = index
        self.state = PromptState.PENDING
        self.start_time = None
        self.end_time = None
        self.error = None
        self.duration = None
        self.attempt_count = 0
        self.detection_method = None
        self.response_start_time = None
        self.response_detected_time = None
        self.response_complete_time = None
        
    def start(self):
        """Mark the start of prompt execution."""
        self.start_time = datetime.now()
        self.state = PromptState.EXECUTING
        self.attempt_count += 1
        return self
        
    def response_started(self):
        """Mark when Claude starts responding."""
        self.response_start_time = datetime.now()
        return self
    
    def response_detected(self):
        """Mark when Claude's response is detected."""
        self.response_detected_time = datetime.now()
        return self
        
    def complete(self, success: bool, error: Optional[str] = None, detection_method: Optional[DetectionMethod] = None):
        """Mark the completion of prompt execution with results."""
        self.end_time = datetime.now()
        self.state = PromptState.COMPLETED if success else PromptState.FAILED
        self.error = error
        self.detection_method = detection_method
        self.response_complete_time = self.end_time
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
        return self
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "index": self.index,
            "prompt": self.prompt_text[:100] + "..." if len(self.prompt_text) > 100 else self.prompt_text,
            "state": self.state.name,
            "success": self.state == PromptState.COMPLETED,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "response_start_time": self.response_start_time.isoformat() if self.response_start_time else None,
            "response_detected_time": self.response_detected_time.isoformat() if self.response_detected_time else None,
            "response_complete_time": self.response_complete_time.isoformat() if self.response_complete_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "attempt_count": self.attempt_count,
            "detection_method": self.detection_method.name if self.detection_method else None,
            "error": self.error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptResult':
        """Create a PromptResult instance from dictionary data."""
        result = cls(data.get("prompt", ""), data.get("index", 0))
        result.state = PromptState[data.get("state", "PENDING")]
        result.attempt_count = data.get("attempt_count", 0)
        result.error = data.get("error")
        result.duration = data.get("duration")
        
        # Parse detection method if available
        detection_method = data.get("detection_method")
        if detection_method:
            try:
                result.detection_method = DetectionMethod[detection_method]
            except (KeyError, ValueError):
                pass
        
        # Parse timestamps if they exist
        for attr in ["start_time", "end_time", "response_start_time", 
                   "response_detected_time", "response_complete_time"]:
            if data.get(attr):
                try:
                    setattr(result, attr, datetime.fromisoformat(data[attr]))
                except (ValueError, TypeError):
                    pass
            
        return result


class CheckpointManager:
    """Manages session checkpoints for resumable execution."""
    
    def __init__(self, checkpoint_dir: str = "logs/simple_sender/checkpoints"):
        """Initialize the checkpoint manager."""
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.version = SessionConstants.CHECKPOINT_VERSION
        self._cache = {}  # Cache for recently accessed checkpoints
    
    def checkpoint_path(self, session_id: str) -> str:
        """Get the path to a session's checkpoint file."""
        return os.path.join(self.checkpoint_dir, f"{session_id}_checkpoint.json")
    
    def exists(self, session_id: str) -> bool:
        """Check if a checkpoint exists for the given session."""
        return os.path.exists(self.checkpoint_path(session_id))
    
    def save(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Save a session checkpoint.
        
        Args:
            session_id: Session identifier
            data: Session data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add version information
            checkpoint_data = {
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "session": data
            }
            
            checkpoint_path = self.checkpoint_path(session_id)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Update cache
            self._cache[session_id] = data
            
            logging.debug(f"Saved checkpoint for session '{session_id}'")
            return True
        except Exception as e:
            logging.warning(f"Failed to save checkpoint for session '{session_id}': {e}")
            return False
    
    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a session checkpoint.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found or error
            
        Raises:
            CheckpointError: If there's an error loading the checkpoint
        """
        # Check cache first
        if session_id in self._cache:
            return self._cache[session_id]
            
        checkpoint_path = self.checkpoint_path(session_id)
        if not os.path.exists(checkpoint_path):
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Check version compatibility
            version = checkpoint_data.get("version", "0.0")
            if version != self.version:
                logging.warning(f"Checkpoint version mismatch: expected {self.version}, got {version}")
            
            # Get session data and update cache
            session_data = checkpoint_data.get("session")
            self._cache[session_id] = session_data
            
            return session_data
        except Exception as e:
            logging.warning(f"Failed to load checkpoint for session '{session_id}': {e}")
            raise CheckpointError(f"Failed to load checkpoint: {e}")
    
    def delete(self, session_id: str) -> bool:
        """
        Delete a session checkpoint.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        checkpoint_path = self.checkpoint_path(session_id)
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                
                # Remove from cache if present
                if session_id in self._cache:
                    del self._cache[session_id]
                    
                logging.debug(f"Deleted checkpoint for session '{session_id}'")
                return True
            except Exception as e:
                logging.warning(f"Failed to delete checkpoint for session '{session_id}': {e}")
                return False
        return True


class PromptExecutor:
    """Executes prompts through a browser handler with error handling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize prompt executor with optional configuration."""
        self.config = config or {}
        self.response_detector = ResponseDetector(config)
    
    def execute_prompt(self, prompt_text: str, browser_handler, 
                     timeout: int = SessionConstants.DEFAULT_DETECTION_TIMEOUT, 
                     retry_count: int = 0) -> Tuple[bool, Optional[str], Optional[DetectionMethod]]:
        """
        Execute a single prompt and wait for response.
        
        Args:
            prompt_text: The prompt to execute
            browser_handler: Handler for browser interactions
            timeout: Maximum time to wait for response in seconds
            retry_count: Current retry attempt (for logging)
            
        Returns:
            Tuple of (success, error_message, detection_method)
        """
        try:
            # Clear any current content
            browser_handler.clear_input()
            time.sleep(0.5)
            
            # Type the prompt with human-like delays
            prefix = "Retry #" + str(retry_count) + ": " if retry_count > 0 else ""
            logging.info(f"{prefix}Typing prompt: {prompt_text[:50]}..." if len(prompt_text) > 50 
                        else f"{prefix}Typing prompt: {prompt_text}")
            
            # Typing should be handled by browser_handler to separate concerns
            self._type_text(browser_handler, prompt_text)
            time.sleep(1)
            
            # Send prompt
            logging.info(f"{prefix}Sending prompt")
            browser_handler.send_prompt()
            
            # Wait for response with improved detection
            return self._wait_for_response(browser_handler, timeout)
            
        except Exception as e:
            error_msg = f"Error executing prompt: {str(e)}"
            logging.error(error_msg)
            return False, error_msg, None
    
    def _type_text(self, browser_handler, text: str) -> bool:
        """
        Type text using the browser handler.
        
        Args:
            browser_handler: Browser handler
            text: Text to type
            
        Returns:
            True if successful
        """
        # Use type_text method if available
        if hasattr(browser_handler, 'type_text'):
            return browser_handler.type_text(text)
            
        # Fall back to interaction module if browser handler doesn't have typing
        try:
            from src.simple_sender.interaction import type_text
            return type_text(text)
        except ImportError:
            # Last resort: character-by-character typing
            if hasattr(browser_handler, 'type_character'):
                for char in text:
                    browser_handler.type_character(char)
                    time.sleep(0.03)  # Small delay between characters
                return True
                
        logging.error("No method available to type text")
        return False
    
    def _wait_for_response(self, browser_handler, timeout: int) -> Tuple[bool, Optional[str], Optional[DetectionMethod]]:
        """
        Wait for Claude's response using hybrid detection.
        
        Args:
            browser_handler: Browser handler
            timeout: Maximum wait time in seconds
            
        Returns:
            Tuple of (success, error_message, detection_method)
        """
        # First check if browser has built-in response detection
        if hasattr(browser_handler, 'wait_for_response_complete'):
            try:
                success = browser_handler.wait_for_response_complete(timeout)
                if success:
                    return True, None, DetectionMethod.HYBRID
                else:
                    return False, "Response timeout or verification failed", DetectionMethod.TIMEOUT
            except Exception as e:
                logging.warning(f"Browser detection failed: {e}")
                # Fall through to our detection methods
        
        # Use our hybrid detection approach
        try:
            success, detection_method = self.response_detector.detect_response_completion(browser_handler, timeout)
            if success:
                return True, None, detection_method
            else:
                return False, "Response detection timeout", DetectionMethod.TIMEOUT
        except Exception as e:
            logging.warning(f"Hybrid detection failed: {e}")
            # Fall back to simple waiting
        
        # Fall back to simple waiting if all else fails
        self._wait_with_logging(timeout)
        return True, None, DetectionMethod.TIMEOUT
    
    def _wait_with_logging(self, wait_time: int):
        """Wait with periodic logging."""
        start_time = time.time()
        while time.time() - start_time < wait_time:
            time.sleep(30)  # Check every 30 seconds
            elapsed = time.time() - start_time
            remaining = wait_time - elapsed
            if remaining > 0:
                logging.info(f"Still waiting: {int(remaining)} seconds remaining...")


class Session:
    """Represents a single Claude AI automation session."""
    
    def __init__(self, session_id: str, config: Dict[str, Any], global_config: Dict[str, Any]):
        """
        Initialize a session with configuration.
        
        Args:
            session_id: Session identifier
            config: Session-specific configuration
            global_config: Global configuration for fallback values
        """
        self.id = session_id
        self.name = config.get("name", session_id)
        self.url = config.get("claude_url", global_config.get("claude_url", "https://claude.ai"))
        self.prompts = config.get("prompts", [])
        self.current_prompt_index = 0
        self.state = SessionState.PENDING
        self.results: List[PromptResult] = []
        self.start_time = None
        self.end_time = None
        self.profile_dir = global_config.get("browser_profile")
        self.error = None
        
        # Performance metrics
        self.total_wait_time = 0
        self.total_processing_time = 0
        self.detection_method_counts = {method: 0 for method in DetectionMethod}
        
        self._validate_config()
        self._initialize_results()
    
    def _validate_config(self):
        """Validate session configuration."""
        if not self.url:
            raise SessionConfigError(f"No URL specified for session '{self.id}'")
        
        if not self.prompts:
            raise SessionConfigError(f"No prompts specified for session '{self.id}'")
    
    def _initialize_results(self):
        """Initialize prompt results list."""
        self.results = [PromptResult(prompt, idx) for idx, prompt in enumerate(self.prompts)]
    
    def get_next_prompt(self) -> Optional[Tuple[int, str]]:
        """Get the next prompt to execute."""
        if self.current_prompt_index >= len(self.prompts):
            return None
        
        return (self.current_prompt_index, self.prompts[self.current_prompt_index])
    
    def advance_prompt(self):
        """Advance to the next prompt."""
        self.current_prompt_index += 1
    
    def start(self):
        """Mark the start of session execution."""
        self.start_time = datetime.now()
        self.transition_to(SessionState.RUNNING)
        logging.info(f"=== Starting session: {self.id} ({self.name}) ===")
        logging.info(f"Session URL: {self.url}")
        logging.info(f"Number of prompts: {len(self.prompts)}")
    
    def complete(self, success: bool, error: Optional[str] = None):
        """Mark the completion of session execution."""
        self.end_time = datetime.now()
        self.transition_to(SessionState.COMPLETED if success else SessionState.FAILED)
        self.error = error
        
        # Calculate success statistics
        total_prompts = len(self.prompts)
        completed_prompts = sum(1 for r in self.results if r.state == PromptState.COMPLETED)
        attempted_prompts = sum(1 for r in self.results if r.attempt_count > 0)
        
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0
        minutes, seconds = divmod(duration, 60)
        hours, minutes = divmod(minutes, 60)
        
        logging.info(f"=== Session {self.id} {self.state.name} ===")
        logging.info(f"Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        logging.info(f"Prompts: {completed_prompts}/{attempted_prompts}/{total_prompts} "
                    f"(successful/attempted/total)")
        
        # Log detection methods used
        if hasattr(self, 'detection_method_counts'):
            methods = [f"{method.name}: {count}" for method, count in self.detection_method_counts.items() if count > 0]
            if methods:
                logging.info(f"Detection methods used: {', '.join(methods)}")
        
        if error:
            logging.info(f"Error: {error}")
    
    def transition_to(self, new_state: SessionState):
        """
        Transition the session to a new state with validation.
        
        Args:
            new_state: The target state
            
        Raises:
            SessionError: If the transition is invalid
        """
        # Define valid transitions
        valid_transitions = {
            SessionState.PENDING: [SessionState.RUNNING],
            SessionState.RUNNING: [SessionState.COMPLETED, SessionState.FAILED, SessionState.INTERRUPTED],
            SessionState.COMPLETED: [],  # Terminal state
            SessionState.FAILED: [],     # Terminal state
            SessionState.INTERRUPTED: [SessionState.RUNNING]  # Can resume from interrupted
        }
        
        if new_state not in valid_transitions.get(self.state, []):
            raise SessionError(f"Invalid state transition: {self.state.name} -> {new_state.name}")
        
        # Apply the transition
        old_state = self.state
        self.state = new_state
        logging.debug(f"Session '{self.id}' transitioned: {old_state.name} -> {new_state.name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "state": self.state.name,
            "current_prompt_index": self.current_prompt_index,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error,
            "prompts_count": len(self.prompts),
            "results": [r.to_dict() for r in self.results],
            "metrics": {
                "total_wait_time": self.total_wait_time,
                "total_processing_time": self.total_processing_time,
                "detection_methods": {method.name: count for method, count in self.detection_method_counts.items()}
            }
        }
    
    def record_detection_method(self, method: Optional[DetectionMethod]):
        """Record the detection method used for a prompt."""
        if method:
            self.detection_method_counts[method] = self.detection_method_counts.get(method, 0) + 1
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Dict[str, Any], global_config: Dict[str, Any]) -> 'Session':
        """
        Create a Session instance from dictionary data.
        
        Args:
            data: Session data dictionary
            config: Session-specific configuration
            global_config: Global configuration
            
        Returns:
            Session instance
        """
        session = cls(data["id"], config, global_config)
        
        # Restore session state
        try:
            session.state = SessionState[data.get("state", "PENDING")]
        except (KeyError, ValueError):
            session.state = SessionState.PENDING
            
        session.current_prompt_index = data.get("current_prompt_index", 0)
        session.error = data.get("error")
        
        # Restore timestamps if they exist
        if data.get("start_time"):
            try:
                session.start_time = datetime.fromisoformat(data["start_time"])
            except (ValueError, TypeError):
                pass
                
        if data.get("end_time"):
            try:
                session.end_time = datetime.fromisoformat(data["end_time"])
            except (ValueError, TypeError):
                pass
        
        # Restore prompt results if they exist
        if "results" in data and isinstance(data["results"], list):
            session.results = [
                PromptResult.from_dict(result_data) 
                for result_data in data["results"]
            ]
        
        # Restore metrics if they exist
        if "metrics" in data:
            metrics = data["metrics"]
            session.total_wait_time = metrics.get("total_wait_time", 0)
            session.total_processing_time = metrics.get("total_processing_time", 0)
            
            # Restore detection method counts
            if "detection_methods" in metrics:
                for method_name, count in metrics["detection_methods"].items():
                    try:
                        method = DetectionMethod[method_name]
                        session.detection_method_counts[method] = count
                    except (KeyError, ValueError):
                        pass
        
        return session


class SessionExecutionEngine:
    """
    Handles prompt execution and response monitoring separate from state management.
    Acts as an execution layer between SessionManager and PromptExecutor.
    """
    
    def __init__(self, session: Session, browser_handler, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the session execution engine.
        
        Args:
            session: Session to execute
            browser_handler: Browser handler for interactions
            config: Optional configuration dictionary
        """
        self.session = session
        self.browser_handler = browser_handler
        self.config = config or {}
        self.prompt_executor = PromptExecutor(config)
        self.retry_settings = self._get_retry_settings()
        self.callbacks = {}
    
    def _get_retry_settings(self) -> Dict[str, Any]:
        """Get retry settings from configuration."""
        return {
            "max_retries": self.config.get("max_retries", SessionConstants.DEFAULT_MAX_RETRIES),
            "retry_delay": self.config.get("retry_delay", SessionConstants.DEFAULT_RETRY_DELAY),
            "retry_backoff": self.config.get("retry_backoff", SessionConstants.DEFAULT_RETRY_BACKOFF),
            "retry_jitter": self.config.get("retry_jitter", SessionConstants.DEFAULT_RETRY_JITTER)
        }
    
    def execute_next_prompt(self) -> bool:
        """
        Execute the next prompt in the session.
        
        Returns:
            True if successful, False if failed or no more prompts
        """
        # Get the next prompt
        next_prompt = self.session.get_next_prompt()
        if not next_prompt:
            return False
            
        prompt_index, prompt_text = next_prompt
        prompt_result = self.session.results[prompt_index].start()
        
        # Trigger callback for prompt start
        self._trigger_callback(ProgressEvents.PROMPT_STARTED, self.session.id, prompt_index, len(self.session.prompts))
        
        # Get retry settings
        max_retries = self.retry_settings["max_retries"]
        
        success = False
        detection_method = None
        for retry in range(max_retries):
            try:
                success, error, detection_method = self.prompt_executor.execute_prompt(
                    prompt_text, 
                    self.browser_handler,
                    timeout=self.config.get("prompt_delay", SessionConstants.DEFAULT_PROMPT_DELAY),
                    retry_count=retry
                )
                
                if success:
                    break
                    
                # If this is not the last retry, wait before trying again
                if retry < max_retries - 1:
                    # Calculate retry delay with exponential backoff and jitter
                    base_delay = self.retry_settings["retry_delay"]
                    backoff = self.retry_settings["retry_backoff"]
                    jitter = self.retry_settings["retry_jitter"]
                    
                    delay = min(
                        base_delay * (backoff ** retry),
                        self.config.get("max_retry_delay", 30)
                    )
                    
                    # Add jitter
                    jitter_factor = random.uniform(1 - jitter, 1 + jitter)
                    actual_delay = delay * jitter_factor
                    
                    logging.warning(f"Retry {retry+1}/{max_retries} after {actual_delay:.1f}s: {error}")
                    time.sleep(actual_delay)
                    
            except Exception as e:
                error = f"Error processing prompt: {str(e)}"
                logging.error(error)
                
                # Trigger callback for error
                self._trigger_callback(ProgressEvents.PROMPT_FAILED, self.session.id, prompt_index, str(e))
                
                # If this is not the last retry, try again
                if retry < max_retries - 1:
                    logging.warning(f"Retry {retry+1}/{max_retries} after exception")
                    time.sleep(self.retry_settings["retry_delay"])
        
        # Mark prompt as completed or failed
        prompt_result.complete(success, None if success else error, detection_method)
        
        # Record detection method
        self.session.record_detection_method(detection_method)
        
        # Trigger appropriate callback
        if success:
            self._trigger_callback(ProgressEvents.PROMPT_COMPLETED, self.session.id, prompt_index, len(self.session.prompts))
        else:
            self._trigger_callback(ProgressEvents.PROMPT_FAILED, self.session.id, prompt_index, len(self.session.prompts))
        
        # Advance to next prompt if successful
        if success:
            self.session.advance_prompt()
            return True
        
        return False
    
    def register_callback(self, event: str, callback: Callable):
        """
        Register a callback for a specific event.
        
        Args:
            event: Event name (use constants from ProgressEvents)
            callback: Callback function to call when event occurs
        """
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """
        Trigger callbacks for an event.
        
        Args:
            event: Event name
            *args, **kwargs: Arguments to pass to callback
        """
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(event, *args, **kwargs)
                except Exception as e:
                    logging.warning(f"Error in callback for event {event}: {e}")
    
    def execute_remaining_prompts(self) -> bool:
        """
        Execute all remaining prompts in the session.
        
        Returns:
            True if all prompts executed successfully
        """
        success = True
        
        while self.session.current_prompt_index < len(self.session.prompts):
            # Take a small break between prompts
            if self.session.current_prompt_index > 0:
                time.sleep(2)
                
            # Execute next prompt
            prompt_success = self.execute_next_prompt()
            
            # If prompt failed, mark overall execution as failed
            if not prompt_success:
                success = False
                break
        
        return success


class SessionManager:
    """Manages multiple Claude AI automation sessions."""
    
    def __init__(self, config_facade, prompt_executor=None, 
                checkpoint_manager=None, session_tracker=None):
        """
        Initialize session manager with dependencies.
        
        Args:
            config_facade: Configuration facade for accessing settings
            prompt_executor: Optional prompt executor (created if None)
            checkpoint_manager: Optional checkpoint manager (created if None)
            session_tracker: Optional session tracker for persistence
        """
        self.config_facade = config_facade
        self.prompt_executor = prompt_executor
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.session_tracker = session_tracker
        self.sessions: Dict[str, Session] = {}
        self.progress_callback = None
        self.browser_handlers = {}
    
    def load_sessions(self, session_ids=None, run_all=False, skip_completed=True) -> List[str]:
        """
        Load sessions from configuration.
        
        Args:
            session_ids: Optional list of session IDs to load
            run_all: Whether to load all available sessions
            skip_completed: Whether to skip previously completed sessions
            
        Returns:
            List of loaded session IDs
        """
        all_sessions = self.config_facade.get_all_sessions()
        global_config = self.config_facade.get_global_config()
        
        if not all_sessions:
            # If no sessions defined, create a default session with global prompts
            global_prompts = self.config_facade.get_global_prompts()
            if not global_prompts:
                logging.error("No prompts found in configuration")
                return []
            
            default_session = {
                "name": "Default",
                "prompts": global_prompts,
                "claude_url": self.config_facade.get_claude_url()
            }
            all_sessions = {"default": default_session}
        
        # Determine which sessions to load
        if run_all:
            sessions_to_load = list(all_sessions.keys())
        elif session_ids:
            sessions_to_load = [
                sid for sid in session_ids if sid in all_sessions
            ]
        else:
            # Default: load all sessions
            sessions_to_load = list(all_sessions.keys())
        
        # Filter out completed sessions if requested
        if skip_completed and self.session_tracker:
            sessions_to_load = [
                sid for sid in sessions_to_load 
                if not self.session_tracker.is_completed(sid)
            ]
        
        # Load sessions
        loaded_sessions = []
        for session_id in sessions_to_load:
            session_config = self.config_facade.get_session_config(session_id)
            try:
                session = Session(session_id, session_config, global_config)
                self.sessions[session_id] = session
                loaded_sessions.append(session_id)
                logging.debug(f"Loaded session: {session_id}")
                
                # Check for checkpoint and resume if available
                if self.checkpoint_manager.exists(session_id):
                    self._load_session_checkpoint(session_id)
            except SessionConfigError as e:
                logging.error(f"Failed to load session '{session_id}': {e}")
        
        logging.info(f"Loaded {len(loaded_sessions)} sessions")
        return loaded_sessions
    
    def get_session(self, session_id) -> Optional[Session]:
        """Get a session by ID."""
        return self.sessions.get(session_id)
    
    def execute_session(self, session_id: str, browser_handler, 
                       prompt_delay: Optional[int] = None, 
                       resume: bool = True,
                       progress_callback: Optional[Callable] = None,
                       take_screenshots: bool = True) -> bool:
        """
        Execute a single session from start to finish.
        
        Args:
            session_id: ID of the session to execute
            browser_handler: Browser handler instance for browser operations
            prompt_delay: Delay between prompts in seconds (None for config default)
            resume: Whether to resume from checkpoint if available
            progress_callback: Optional callback for progress updates
            take_screenshots: Whether to capture screenshots during execution
            
        Returns:
            True if successful, False otherwise
        """
        session = self.sessions.get(session_id)
        if not session:
            logging.error(f"Session '{session_id}' not found")
            return False
        
        # Get delay from configuration if not specified
        if prompt_delay is None:
            prompt_delay = self.config_facade.get_delay_between_prompts(session_id)
        
        # Store browser handler for this session
        self.browser_handlers[session_id] = browser_handler
        
        # Set progress callback
        self.progress_callback = progress_callback
        
        try:
            # Report progress if callback provided
            self._trigger_callback(ProgressEvents.SESSION_STARTED, session_id)
            
            # Start the session if not already running
            if session.state != SessionState.RUNNING:
                session.start()
            
            # Save initial checkpoint
            self._save_session_checkpoint(session_id)
            
            # Launch browser for this session
            try:
                # Use start method if available, otherwise launch
                if hasattr(browser_handler, 'start'):
                    if not browser_handler.start(session.url, session.profile_dir):
                        error_msg = f"Failed to start browser for session '{session_id}'"
                        logging.error(error_msg)
                        session.complete(False, error_msg)
                        self._trigger_callback(ProgressEvents.SESSION_FAILED, session_id, error_msg)
                        return False
                elif hasattr(browser_handler, 'launch'):
                    if not browser_handler.launch(session.url, session.profile_dir):
                        error_msg = f"Failed to launch browser for session '{session_id}'"
                        logging.error(error_msg)
                        session.complete(False, error_msg)
                        self._trigger_callback(ProgressEvents.SESSION_FAILED, session_id, error_msg)
                        return False
                else:
                    error_msg = "Browser handler has no start/launch method"
                    logging.error(error_msg)
                    session.complete(False, error_msg)
                    self._trigger_callback(ProgressEvents.SESSION_FAILED, session_id, error_msg)
                    return False
            except Exception as e:
                error_msg = f"Browser launch error: {str(e)}"
                logging.error(error_msg)
                session.complete(False, error_msg)
                self._trigger_callback(ProgressEvents.SESSION_FAILED, session_id, error_msg)
                return False
            
            # Create execution engine with config and callbacks
            config = {
                "prompt_delay": prompt_delay,
                "take_screenshots": take_screenshots
            }
            engine = SessionExecutionEngine(session, browser_handler, config)
            
            # Register progress callbacks if provided
            if progress_callback:
                engine.register_callback(ProgressEvents.PROMPT_STARTED, self._handle_callback)
                engine.register_callback(ProgressEvents.PROMPT_COMPLETED, self._handle_callback)
                engine.register_callback(ProgressEvents.PROMPT_FAILED, self._handle_callback)
            
            # Process all prompts
            success = engine.execute_remaining_prompts()
            
            session.complete(success)
            if success and self.session_tracker:
                self.session_tracker.mark_completed(session_id, True)
                self._trigger_callback(ProgressEvents.SESSION_COMPLETED, session_id)
            
            # Clean up checkpoint if successful
            if success:
                self.checkpoint_manager.delete(session_id)
            
            return success
            
        except SessionExecutionError as e:
            error_msg = f"Error executing session '{session_id}': {str(e)}"
            logging.error(error_msg)
            session.complete(False, error_msg)
            self._trigger_callback(ProgressEvents.SESSION_FAILED, session_id, error_msg)
            return False
        except KeyboardInterrupt:
            error_msg = "Execution interrupted by user"
            logging.warning(error_msg)
            session.transition_to(SessionState.INTERRUPTED)
            self._save_session_checkpoint(session_id)
            raise
        finally:
            # Always close the browser in case of error
            try:
                # Use end/close method based on what's available
                if hasattr(browser_handler, 'end'):
                    browser_handler.end()
                elif hasattr(browser_handler, 'close'):
                    browser_handler.close()
            except Exception as e:
                logging.error(f"Error closing browser: {e}")
            
            # Remove browser handler reference
            if session_id in self.browser_handlers:
                del self.browser_handlers[session_id]
    
    def run_default_session(self, prompts: List[str], browser_handler, 
                          prompt_delay: int = SessionConstants.DEFAULT_PROMPT_DELAY, 
                          resume: bool = False,
                          take_screenshots: bool = True) -> bool:
        """
        Run a default session with the provided prompts.
        
        Args:
            prompts: List of prompts to execute
            browser_handler: Browser handler
            prompt_delay: Delay between prompts
            resume: Whether to resume from checkpoint
            take_screenshots: Whether to take screenshots
            
        Returns:
            True if successful
        """
        # Create a default session
        session_id = "default"
        session_config = {
            "name": "Default Session",
            "prompts": prompts,
            "claude_url": self.config_facade.get_claude_url()
        }
        global_config = self.config_facade.get_global_config()
        
        # Create or update session
        if session_id in self.sessions:
            # Update existing session
            self.sessions[session_id] = Session(session_id, session_config, global_config)
        else:
            # Create new session
            self.sessions[session_id] = Session(session_id, session_config, global_config)
        
        # Execute the session
        return self.execute_session(
            session_id,
            browser_handler,
            prompt_delay,
            resume,
            self.progress_callback,
            take_screenshots
        )
    
    def run_sessions(self, browser_handler, 
                     prompt_delay: int = SessionConstants.DEFAULT_PROMPT_DELAY, 
                     session_delay: int = SessionConstants.DEFAULT_SESSION_DELAY,
                     progress_callback: Optional[Callable] = None,
                     take_screenshots: bool = True) -> Dict[str, bool]:
        """
        Run all loaded sessions.
        
        Args:
            browser_handler: Browser handler instance
            prompt_delay: Delay between prompts in seconds
            session_delay: Delay between sessions in seconds
            progress_callback: Optional callback for progress updates
            take_screenshots: Whether to capture screenshots
            
        Returns:
            Dictionary mapping session IDs to success status
        """
        if not self.sessions:
            logging.error("No sessions loaded")
            return {}
        
        # Store progress callback
        self.progress_callback = progress_callback
        
        results = {}
        session_ids = list(self.sessions.keys())
        
        # Report overall progress if callback provided
        self._trigger_callback(ProgressEvents.RUN_STARTED, session_ids)
        
        for i, session_id in enumerate(session_ids):
            # Report session progress if callback provided
            self._trigger_callback(ProgressEvents.SESSION_STARTING, session_id, i, len(session_ids))
            
            # Execute session
            try:
                success = self.execute_session(
                    session_id, 
                    browser_handler, 
                    prompt_delay,
                    progress_callback=progress_callback,
                    take_screenshots=take_screenshots
                )
                results[session_id] = success
            except KeyboardInterrupt:
                logging.warning(f"Session '{session_id}' interrupted by user")
                results[session_id] = False
                break
            
            # Wait between sessions if there are more to process
            if i < len(session_ids) - 1:
                logging.info(f"Waiting {session_delay} seconds before next session...")
                time.sleep(session_delay)
        
        # Log summary
        self._log_summary(results)
        
        # Report completion if callback provided
        self._trigger_callback(ProgressEvents.RUN_COMPLETED, results)
        
        return results
    
    def _save_session_checkpoint(self, session_id: str) -> bool:
        """Save checkpoint for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        session_data = session.to_dict()
        return self.checkpoint_manager.save(session_id, session_data)
    
    def _load_session_checkpoint(self, session_id: str) -> bool:
        """Load checkpoint for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        try:
            checkpoint_data = self.checkpoint_manager.load(session_id)
            if not checkpoint_data:
                return False
            
            # Update session state
            if checkpoint_data.get("state") == SessionState.COMPLETED.name:
                # Already completed, skip
                logging.info(f"Session '{session_id}' was already completed according to checkpoint")
                return True
            
            # Update current prompt index
            current_index = checkpoint_data.get("current_prompt_index", 0)
            session.current_prompt_index = current_index
            
            # Update state
            try:
                session.state = SessionState[checkpoint_data.get("state", "PENDING")]
            except (KeyError, ValueError):
                session.state = SessionState.PENDING
            
            # Update timestamps if available
            if checkpoint_data.get("start_time"):
                try:
                    session.start_time = datetime.fromisoformat(checkpoint_data["start_time"])
                except (ValueError, TypeError):
                    pass
            
            # Update prompt results if available
            if "results" in checkpoint_data and isinstance(checkpoint_data["results"], list):
                for result_data in checkpoint_data["results"]:
                    index = result_data.get("index", 0)
                    if 0 <= index < len(session.results):
                        session.results[index] = PromptResult.from_dict(result_data)
            
            # Update metrics if available
            if "metrics" in checkpoint_data:
                metrics = checkpoint_data.get("metrics", {})
                if isinstance(metrics, dict):
                    session.total_wait_time = metrics.get("total_wait_time", 0)
                    session.total_processing_time = metrics.get("total_processing_time", 0)
                    
                    # Update detection method counts
                    if "detection_methods" in metrics:
                        detection_methods = metrics.get("detection_methods", {})
                        if isinstance(detection_methods, dict):
                            for method_name, count in detection_methods.items():
                                try:
                                    method = DetectionMethod[method_name]
                                    session.detection_method_counts[method] = count
                                except (KeyError, ValueError):
                                    pass
            
            if current_index > 0:
                # Resume from checkpoint
                logging.info(f"Resuming session '{session_id}' from prompt {current_index + 1}/{len(session.prompts)}")
            
            return True
            
        except Exception as e:
            logging.warning(f"Failed to load checkpoint for session '{session_id}': {e}")
            return False
    
    def _log_summary(self, results: Dict[str, bool]):
        """Log summary of session results."""
        logging.info("=== SESSION RESULTS SUMMARY ===")
        successful = sum(1 for success in results.values() if success)
        logging.info(f"Total sessions: {len(results)}")
        logging.info(f"Successful: {successful}")
        logging.info(f"Failed: {len(results) - successful}")
        
        for session_id, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            logging.info(f"Session '{session_id}': {status}")
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """Trigger progress callback with given event and data."""
        if self.progress_callback:
            try:
                self.progress_callback(event, *args, **kwargs)
            except Exception as e:
                logging.warning(f"Error in progress callback: {e}")
    
    def _handle_callback(self, event: str, *args, **kwargs):
        """Handle callback from execution engine and forward to progress callback."""
        # Simply forward to main callback
        self._trigger_callback(event, *args, **kwargs)
    
    def generate_session_report(self, session_id: str) -> Dict[str, Any]:
        """Generate a detailed report for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": f"Session '{session_id}' not found"}
        
        # Get basic session data
        report = session.to_dict()
        
        # Add additional analytics
        if session.start_time and session.end_time:
            duration_seconds = (session.end_time - session.start_time).total_seconds()
            report["duration_seconds"] = duration_seconds
            report["duration_formatted"] = self._format_duration(duration_seconds)
        
        # Add prompt statistics
        completed_prompts = sum(1 for r in session.results if r.state == PromptState.COMPLETED)
        failed_prompts = sum(1 for r in session.results if r.state == PromptState.FAILED)
        pending_prompts = sum(1 for r in session.results if r.state == PromptState.PENDING)
        
        report["prompt_stats"] = {
            "total": len(session.prompts),
            "completed": completed_prompts,
            "failed": failed_prompts,
            "pending": pending_prompts,
            "success_rate": completed_prompts / len(session.prompts) if session.prompts else 0
        }
        
        # Add performance metrics
        report["performance"] = {
            "average_response_time": self._calculate_average_response_time(session),
            "timeout_count": self._count_timeouts(session),
            "detection_success_rate": self._calculate_detection_success_rate(session),
            "time_saved": self._estimate_time_saved(session),
            "detection_methods": {method.name: count for method, count in session.detection_method_counts.items()}
        }
        
        # Add detailed prompt data
        report["prompts"] = [r.to_dict() for r in session.results]
        
        return report
    
    def _calculate_average_response_time(self, session: Session) -> float:
        """Calculate average response time for a session."""
        response_times = []
        
        for result in session.results:
            if (result.state == PromptState.COMPLETED and 
                result.start_time and result.end_time):
                response_times.append((result.end_time - result.start_time).total_seconds())
        
        if not response_times:
            return 0
            
        return sum(response_times) / len(response_times)
    
    def _count_timeouts(self, session: Session) -> int:
        """Count the number of timeout detections in a session."""
        timeout_count = 0
        
        for result in session.results:
            if result.detection_method == DetectionMethod.TIMEOUT:
                timeout_count += 1
                
        return timeout_count
    
    def _calculate_detection_success_rate(self, session: Session) -> float:
        """Calculate the success rate of non-timeout detections."""
        detection_attempts = sum(1 for r in session.results if r.state == PromptState.COMPLETED)
        
        if not detection_attempts:
            return 0
            
        non_timeout_detections = sum(1 for r in session.results 
                                   if r.state == PromptState.COMPLETED and 
                                   r.detection_method != DetectionMethod.TIMEOUT)
        
        return non_timeout_detections / detection_attempts if detection_attempts else 0
    
    def _estimate_time_saved(self, session: Session) -> float:
        """Estimate time saved compared to fixed waiting."""
        fixed_wait_time = SessionConstants.DEFAULT_PROMPT_DELAY
        total_saved = 0
        
        for result in session.results:
            if (result.state == PromptState.COMPLETED and 
                result.start_time and result.end_time and
                result.detection_method != DetectionMethod.TIMEOUT):
                
                actual_time = (result.end_time - result.start_time).total_seconds()
                saved = fixed_wait_time - actual_time
                if saved > 0:
                    total_saved += saved
        
        return total_saved
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to a human-readable string."""
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def get_session_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the status of a session.
        
        Args:
            session_id: Session identifier (None for all sessions)
            
        Returns:
            Dictionary with session status
        """
        if self.session_tracker:
            if session_id:
                return self.session_tracker.get_session_status(session_id)
            else:
                return self.session_tracker.get_all_session_status()
        else:
            # Return session state from memory
            if session_id:
                session = self.sessions.get(session_id)
                if not session:
                    return {}
                return {
                    "completed": session.state == SessionState.COMPLETED,
                    "success": session.state == SessionState.COMPLETED,
                    "state": session.state.name,
                    "current_prompt_index": session.current_prompt_index,
                    "prompts_completed": sum(1 for r in session.results if r.state == PromptState.COMPLETED)
                }
            else:
                return {
                    sid: {
                        "completed": s.state == SessionState.COMPLETED,
                        "success": s.state == SessionState.COMPLETED,
                        "state": s.state.name,
                        "current_prompt_index": s.current_prompt_index,
                        "prompts_completed": sum(1 for r in s.results if r.state == PromptState.COMPLETED)
                    }
                    for sid, s in self.sessions.items()
                }
    
    def get_all_session_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all sessions."""
        return self.get_session_status(None)