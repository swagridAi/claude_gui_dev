#!/usr/bin/env python3
"""
Constants module for Claude GUI Automation.

This module centralizes all constants used throughout the application
to eliminate duplication and ensure consistency. Constants are organized
into classes by domain area.
"""

class BrowserConstants:
    """Browser-related constants for timing and behavior."""
    
    # Launch and initialization
    STARTUP_WAIT = 10  # Seconds to wait after browser launch
    PAGE_LOAD_TIMEOUT = 15  # Maximum seconds to wait for page load
    PAGE_LOAD_CHECK_INTERVAL = 0.5  # Seconds between page load checks
    
    # Navigation
    NAVIGATION_TIMEOUT = 20  # Maximum seconds to wait for navigation
    TAB_SWITCH_DELAY = 0.5  # Seconds to wait after switching tabs
    
    # Verification
    BROWSER_CHECK_INTERVAL = 1  # Seconds between browser state checks
    VERIFICATION_ATTEMPTS = 3  # Number of attempts to verify browser state


class RetryConstants:
    """Constants for retry mechanism with exponential backoff."""
    
    # Core retry parameters
    MAX_ATTEMPTS = 3  # Maximum number of retry attempts
    INITIAL_DELAY = 2.0  # Initial delay in seconds
    MAX_DELAY = 30.0  # Maximum delay cap in seconds
    BACKOFF_FACTOR = 1.5  # Multiplier for delay after each failure
    JITTER_FACTOR = 0.5  # Random factor to add/subtract from delay (0-1)
    
    # Specialized retry profiles
    NETWORK_RETRY = {
        'max_attempts': 5,
        'initial_delay': 3.0,
        'backoff_factor': 2.0,
        'jitter_factor': 0.3,
        'max_delay': 60.0
    }
    
    UI_RETRY = {
        'max_attempts': 3,
        'initial_delay': 1.0,
        'backoff_factor': 1.5,
        'jitter_factor': 0.2,
        'max_delay': 10.0
    }


class TypingConstants:
    """Constants for human-like typing behavior."""
    
    # Default typing parameter
    DEFAULT_DELAY = 0.03  # Base delay between keystrokes in seconds
    
    # Typing profiles with timing characteristics
    PROFILES = {
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
    
    # Special character typing delays
    SPECIAL_CHAR_EXTRA_DELAY = 0.05  # Extra delay for special characters


class UIConstants:
    """Constants for UI interactions and behavior."""
    
    # General timing
    DEFAULT_ACTION_DELAY = 0.5  # Seconds between UI actions
    HUMANIZED_MIN_DELAY = 0.1  # Minimum random delay
    HUMANIZED_MAX_DELAY = 0.3  # Maximum random delay
    
    # Wait timeouts
    DEFAULT_WAIT_TIMEOUT = 300  # Default timeout in seconds (5 minutes)
    RESPONSE_WAIT_TIMEOUT = 300  # Timeout for Claude to respond
    THINKING_TIMEOUT = 60  # Timeout for initial "thinking" indicator
    
    # Check intervals
    DEFAULT_CHECK_INTERVAL = 10  # Seconds between status checks
    VISUAL_CHECK_INTERVAL = 0.5  # Seconds between visual checks
    
    # Visual verification
    VERIFICATION_THRESHOLD = 0.1  # Pixel difference threshold (0-1)
    MIN_PIXEL_CHANGE = 1000  # Minimum pixels that must change
    VERIFICATION_ATTEMPTS = 3  # Number of verification attempts


class SessionConstants:
    """Constants for session management."""
    
    # Delays and timeouts
    DEFAULT_PROMPT_DELAY = 180  # Seconds to wait after sending prompt
    DEFAULT_SESSION_DELAY = 10  # Seconds between sessions
    SESSION_TIMEOUT = 3600  # Maximum session duration in seconds (1 hour)
    
    # Checkpointing
    CHECKPOINT_INTERVAL = 600  # Seconds between session checkpoints (10 minutes)


class LoggingConstants:
    """Constants for logging configuration."""
    
    # Screenshot intervals
    SCREENSHOT_INTERVAL = 30  # Seconds between progress screenshots
    MAX_SCREENSHOTS = 100  # Maximum screenshots per session
    
    # Debug settings
    DEBUG_CLICK_REGION_SIZE = 100  # Pixel size of click debug regions
    DEBUG_IMAGE_QUALITY = 85  # JPEG quality for debug images (0-100)