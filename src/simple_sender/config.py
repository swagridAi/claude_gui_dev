#!/usr/bin/env python3
"""
Configuration module for the simple sender.

Provides a simplified interface to the Claude automation configuration system
with support for caching, validation, and session management.
"""

import os
import logging
import copy
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Set, Callable, TypedDict, cast
from dataclasses import dataclass
from functools import wraps
from contextlib import contextmanager

# ===== MODULE 1: Data Models =====
# In a multi-file implementation, these would be in models.py

@dataclass
class SessionConfig:
    """Session configuration model with typed attributes."""
    id: str
    name: str
    url: str
    prompts: List[str]
    delay: int = 180
    
    @classmethod
    def from_dict(cls, session_id: str, data: Dict[str, Any]) -> 'SessionConfig':
        """Create SessionConfig from dictionary data."""
        return cls(
            id=session_id,
            name=data.get("name", session_id),
            url=data.get("claude_url", "https://claude.ai"),
            prompts=data.get("prompts", []),
            delay=data.get("delay_between_prompts", 180)
        )


@dataclass
class BrowserConfig:
    """Browser configuration model with typed attributes."""
    profile_dir: str
    chrome_path: Optional[str] = None
    startup_delay: int = 10
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BrowserConfig':
        """Create BrowserConfig from dictionary data."""
        profile = data.get("browser_profile")
        if not profile:
            profile = os.path.join(os.path.expanduser("~"), "ClaudeProfile")
            
        return cls(
            profile_dir=profile,
            chrome_path=data.get("chrome_path"),
            startup_delay=data.get("browser_launch_wait", 10)
        )


@dataclass
class RetryConfig:
    """Retry configuration model with typed attributes."""
    max_retries: int = 3
    retry_delay: float = 2.0
    max_retry_delay: float = 30.0
    retry_jitter: float = 0.5
    retry_backoff: float = 1.5
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetryConfig':
        """Create RetryConfig from dictionary data."""
        return cls(
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 2.0),
            max_retry_delay=data.get("max_retry_delay", 30.0),
            retry_jitter=data.get("retry_jitter", 0.5),
            retry_backoff=data.get("retry_backoff", 1.5)
        )


@dataclass
class VisualConfig:
    """Visual verification configuration model."""
    enabled: bool = True
    template_dir: str = "templates"
    confidence_threshold: float = 0.8
    max_wait_time: int = 300
    check_interval: float = 2.0
    template_matching_method: str = "cv2.TM_CCOEFF_NORMED"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VisualConfig':
        """Create VisualConfig from dictionary data."""
        visual_config = data.get("visual_verification", {})
        return cls(
            enabled=visual_config.get("enabled", True),
            template_dir=visual_config.get("template_dir", "templates"),
            confidence_threshold=visual_config.get("confidence_threshold", 0.8),
            max_wait_time=visual_config.get("max_wait_time", 300),
            check_interval=visual_config.get("check_interval", 2.0),
            template_matching_method=visual_config.get("template_matching_method", "cv2.TM_CCOEFF_NORMED")
        )


class ValidationResult:
    """Result of a validation operation with detailed error information."""
    
    def __init__(self, valid: bool = True, errors: Optional[List[str]] = None):
        self.valid = valid
        self.errors = errors or []
    
    def __bool__(self):
        return self.valid
    
    def add_error(self, error: str):
        """Add an error message and mark validation as failed."""
        self.errors.append(error)
        self.valid = False
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one."""
        if not other.valid:
            self.valid = False
            self.errors.extend(other.errors)


# ===== MODULE 2: Custom Exceptions =====
# In a multi-file implementation, these would be in exceptions.py

class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass


class SessionConfigError(ConfigError):
    """Exception for session configuration errors."""
    pass


class ValidationError(ConfigError):
    """Exception for validation errors."""
    def __init__(self, errors):
        self.errors = errors
        message = f"Validation failed: {'; '.join(errors)}"
        super().__init__(message)


class ConfigValidationError(ConfigError):
    """Detailed validation error with field information."""
    def __init__(self, field: str, message: str, details: Optional[List[str]] = None):
        self.field = field
        self.details = details or []
        self.message = message
        super().__init__(f"{field}: {message}")


# ===== MODULE 3: Validation Functions =====
# In a multi-file implementation, these would be in validation.py

class ConfigValidator:
    """Centralized validation logic for configuration objects."""
    
    @staticmethod
    def validate_prompts(prompts: List[Any]) -> ValidationResult:
        """
        Validate a list of prompts.
        
        Args:
            prompts: List of prompts to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        if not prompts:
            result.add_error("Empty prompts list")
            return result
        
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                result.add_error(f"Prompt {i+1} is not a string")
            elif not prompt.strip():
                result.add_error(f"Prompt {i+1} is empty")
        
        return result
    
    @staticmethod
    def validate_session_config(session_id: str, config_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate session configuration.
        
        Args:
            session_id: Session identifier
            config_data: Session configuration data
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        # Check essential fields
        if not config_data:
            result.add_error(f"Empty configuration for session '{session_id}'")
            return result
            
        # Validate prompts
        prompts = config_data.get("prompts", [])
        if not prompts:
            result.add_error(f"No prompts defined for session '{session_id}'")
        else:
            prompt_validation = ConfigValidator.validate_prompts(prompts)
            result.merge(prompt_validation)
        
        # Validate URL
        url = config_data.get("claude_url")
        if url and not isinstance(url, str):
            result.add_error(f"URL must be a string, got {type(url).__name__}")
        elif url and not url.startswith(("http://", "https://")):
            result.add_error(f"Invalid URL format: {url}")
        
        # Validate delay
        delay = config_data.get("delay_between_prompts")
        if delay is not None:
            if not isinstance(delay, (int, float)):
                result.add_error(f"Delay must be a number, got {type(delay).__name__}")
            elif delay < 0:
                result.add_error(f"Delay cannot be negative: {delay}")
        
        return result
    
    @staticmethod
    def validate_browser_config(config_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate browser configuration.
        
        Args:
            config_data: Browser configuration data
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult()
        
        # Validate chrome_path
        chrome_path = config_data.get("chrome_path")
        if chrome_path and not isinstance(chrome_path, str):
            result.add_error(f"Chrome path must be a string, got {type(chrome_path).__name__}")
        elif chrome_path and not os.path.exists(chrome_path):
            result.add_error(f"Chrome path does not exist: {chrome_path}")
        
        # Validate browser_profile
        profile = config_data.get("browser_profile")
        if profile and not isinstance(profile, str):
            result.add_error(f"Browser profile must be a string, got {type(profile).__name__}")
        
        # Validate startup_wait
        startup_wait = config_data.get("browser_launch_wait")
        if startup_wait is not None:
            if not isinstance(startup_wait, (int, float)):
                result.add_error(f"Browser launch wait must be a number, got {type(startup_wait).__name__}")
            elif startup_wait < 0:
                result.add_error(f"Browser launch wait cannot be negative: {startup_wait}")
        
        return result


# ===== MODULE 4: Cache Management =====
# In a multi-file implementation, these would be in cache.py

class CacheManager:
    """Manages caching of configuration values."""
    
    def __init__(self, ttl: int = 60):
        """
        Initialize the cache manager.
        
        Args:
            ttl: Default time-to-live in seconds for cache entries
        """
        self._cache = {}
        self._timestamps = {}
        self._default_ttl = ttl
    
    def is_valid(self, key: str) -> bool:
        """Check if a cached item is still valid."""
        if key not in self._cache or key not in self._timestamps:
            return False
        age = time.time() - self._timestamps[key]
        return age < self._default_ttl
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from cache if it exists and is valid."""
        if self.is_valid(key):
            return self._cache[key]
        return default
    
    def set(self, key: str, value: Any) -> None:
        """Store a value in cache with timestamp."""
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def invalidate(self, key: Optional[str] = None) -> None:
        """Invalidate specific cache key or entire cache."""
        if key:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
        else:
            self._cache.clear()
            self._timestamps.clear()


def cached_method(ttl: Optional[int] = None):
    """
    Decorator for caching method results.
    
    Args:
        ttl: Optional override for cache time-to-live in seconds
        
    Returns:
        Decorated method with caching
    """
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Generate a cache key based on method name and arguments
            key_parts = [method.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = "_".join(key_parts)
            
            # Get cache manager from instance
            if not hasattr(self, '_cache'):
                # Create cache manager if it doesn't exist
                self._cache = CacheManager(ttl or 60)
            
            # Check if value is in cache
            cached_value = self._cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Call the original method
            result = method(self, *args, **kwargs)
            
            # Cache the result
            self._cache.set(cache_key, result)
            return result
        return wrapper
    return decorator


# ===== MODULE 5: Fallback Implementations =====
# In a multi-file implementation, these would be in fallbacks.py

class SimpleConfigManager:
    """Simple fallback config manager when main one is unavailable."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    import yaml  # Local import to avoid global dependency
                    self.config = yaml.safe_load(f) or {}
                    logging.info(f"Loaded configuration from {self.config_path}")
            else:
                logging.warning(f"Configuration file not found: {self.config_path}")
                self.config = create_default_simple_config()
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            self.config = create_default_simple_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with dot notation support."""
        if "." in key:
            parts = key.split(".")
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        return self.config.get(key, default)
    
    def is_in_session_mode(self) -> bool:
        """Stub for compatibility with ConfigManager."""
        return False
    
    def enter_session_mode(self, session_id: Optional[str] = None) -> None:
        """Stub for compatibility with ConfigManager."""
        pass
    
    def exit_session_mode(self) -> None:
        """Stub for compatibility with ConfigManager."""
        pass
    
    def save(self) -> bool:
        """Save configuration to file."""
        try:
            import yaml  # Local import to avoid global dependency
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            return True
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            return False


class SimpleSessionTracker:
    """Simple fallback session tracker when main one is unavailable."""
    
    def __init__(self, tracker_path: str):
        self.tracker_path = tracker_path
        self.session_status = {}
        self._load_status()
    
    def _load_status(self) -> None:
        """Load session status from file."""
        try:
            if os.path.exists(self.tracker_path):
                with open(self.tracker_path, 'r') as f:
                    self.session_status = json.load(f)
        except Exception as e:
            logging.error(f"Error loading session status: {e}")
            self.session_status = {}
    
    def _save_status(self) -> bool:
        """Save session status to file."""
        try:
            os.makedirs(os.path.dirname(self.tracker_path), exist_ok=True)
            with open(self.tracker_path, 'w') as f:
                json.dump(self.session_status, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving session status: {e}")
            return False
    
    def is_completed(self, session_id: str) -> bool:
        """Check if a session has been completed."""
        return session_id in self.session_status and self.session_status[session_id].get('completed', False)
    
    def mark_completed(self, session_id: str, success: bool = True, notes: Optional[str] = None) -> bool:
        """Mark a session as completed."""
        if session_id not in self.session_status:
            self.session_status[session_id] = {}
            
        self.session_status[session_id].update({
            'completed': True,
            'success': success,
            'completion_time': time.strftime("%Y-%m-%dT%H:%M:%S"),
            'notes': notes or ""
        })
        
        return self._save_status()
    
    def reset_session(self, session_id: str) -> bool:
        """Reset a session's completion status."""
        if session_id in self.session_status:
            # Store history if exists
            if 'history' not in self.session_status[session_id]:
                self.session_status[session_id]['history'] = []
                
            # Add current status to history
            if self.session_status[session_id].get('completed'):
                history_entry = {
                    'completed': self.session_status[session_id].get('completed', False),
                    'success': self.session_status[session_id].get('success', False),
                    'completion_time': self.session_status[session_id].get('completion_time', ''),
                    'notes': self.session_status[session_id].get('notes', '')
                }
                self.session_status[session_id]['history'].append(history_entry)
            
            # Reset status
            self.session_status[session_id]['completed'] = False
            self.session_status[session_id]['success'] = False
            self.session_status[session_id]['notes'] = f"Reset on {time.strftime('%Y-%m-%dT%H:%M:%S')}"
            if 'completion_time' in self.session_status[session_id]:
                del self.session_status[session_id]['completion_time']
            
            return self._save_status()
        return False


def create_default_simple_config() -> Dict[str, Any]:
    """
    Create a default configuration suitable for simple sender.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "claude_url": "https://claude.ai",
        "browser_profile": os.path.join(os.path.expanduser("~"), "ClaudeProfile"),
        "delay_between_prompts": 180,
        "max_retries": 3,
        "retry_delay": 2,
        "max_retry_delay": 30,
        "retry_jitter": 0.5,
        "retry_backoff": 1.5,
        "prompts": [
            "Hello, Claude! This is a test prompt from the simple sender."
        ],
        "visual_verification": {
            "enabled": True,
            "template_dir": "templates",
            "confidence_threshold": 0.8,
            "max_wait_time": 300,
            "check_interval": 2.0
        }
    }


def resolve_session_priority(available_sessions: List[str], 
                             specified_session: Optional[str] = None, 
                             run_one: bool = False,
                             skip_completed: bool = True,
                             config: Optional['ConfigFacade'] = None) -> List[str]:
    """
    Determine which sessions to run and in what order.
    
    Args:
        available_sessions: List of available session IDs
        specified_session: Specific session to prioritize (optional)
        run_one: Whether to run only one session
        skip_completed: Whether to skip completed sessions
        config: ConfigFacade instance for checking completion status
        
    Returns:
        Ordered list of session IDs to run
    """
    if not available_sessions:
        return []
    
    # If no configuration provided, we can't check completion status
    if config is None:
        skip_completed = False
    
    # Filter out completed sessions if requested
    sessions_to_run = []
    if skip_completed and config:
        sessions_to_run = [s for s in available_sessions if not config.is_session_complete(s)]
    else:
        sessions_to_run = copy.copy(available_sessions)
    
    # If no sessions left after filtering, return empty list or all sessions if force run
    if not sessions_to_run:
        if skip_completed:
            logging.info("All sessions are already completed")
            return []
        else:
            sessions_to_run = copy.copy(available_sessions)
    
    # Handle specific session prioritization
    if specified_session:
        if specified_session in sessions_to_run:
            # Move specified session to the front
            sessions_to_run.remove(specified_session)
            sessions_to_run.insert(0, specified_session)
        elif specified_session in available_sessions:
            # Session exists but was filtered out (likely completed)
            if skip_completed and config and config.is_session_complete(specified_session):
                logging.warning(f"Session '{specified_session}' is already completed")
                # Add it anyway at the front if we're supposed to run it
                sessions_to_run.insert(0, specified_session)
            else:
                # Some other reason for filtering
                logging.warning(f"Session '{specified_session}' was filtered out")
        else:
            # Session doesn't exist
            logging.warning(f"Session '{specified_session}' not found")
    
    # If run_one is True, keep only the first session
    if run_one and sessions_to_run:
        sessions_to_run = [sessions_to_run[0]]
    
    return sessions_to_run


# ===== MODULE 6: Main Facade Class =====
# In a multi-file implementation, this would be in facade.py

class ConfigFacade:
    """
    A simplified configuration interface for Claude automation.
    
    Acts as a facade over the more complex ConfigManager, providing
    domain-specific methods for common configuration operations.
    """
    
    def __init__(self, config_path: Optional[str] = None, config_manager=None, 
                 session_tracker=None, preserve_original: bool = True, 
                 strict_mode: bool = False, cache_ttl: int = 60):
        """
        Initialize the configuration facade.
        
        Args:
            config_path: Path to configuration file (optional)
            config_manager: Optional ConfigManager instance
            session_tracker: Optional SessionTracker instance
            preserve_original: Whether to preserve original config when saving
            strict_mode: Whether to raise exceptions on validation errors
            cache_ttl: Cache time-to-live in seconds
        """
        self.config_path = config_path or "config/user_config.yaml"
        self.preserve_original = preserve_original
        self.strict_mode = strict_mode
        
        # Initialize caching system
        self._cache = CacheManager(cache_ttl)
        
        # Initialize the underlying configuration manager
        if config_manager:
            self.config_manager = config_manager
        else:
            try:
                from src.utils.config_manager import ConfigManager
                self.config_manager = ConfigManager(self.config_path)
            except ImportError:
                logging.warning("Main ConfigManager not available, using fallback implementation")
                self.config_manager = SimpleConfigManager(self.config_path)
        
        # Initialize session tracker for progress monitoring
        if session_tracker:
            self.session_tracker = session_tracker
        else:
            try:
                from src.utils.session_tracker import SessionTracker
                tracker_path = os.path.join(os.path.dirname(self.config_path), "session_status.json")
                self.session_tracker = SessionTracker(tracker_path)
            except ImportError:
                logging.warning("Main SessionTracker not available, using fallback implementation")
                tracker_path = os.path.join(os.path.dirname(self.config_path), "simple_session_status.json")
                self.session_tracker = SimpleSessionTracker(tracker_path)

    @contextmanager
    def session_context(self, session_id: Optional[str] = None):
        """Context manager for session operations that preserves state."""
        try:
            if not self.config_manager.is_in_session_mode():
                self.config_manager.enter_session_mode(session_id)
            yield
        finally:
            if self.config_manager.is_in_session_mode():
                self.config_manager.exit_session_mode()

    # ==== Session Handling Methods ====
    
    @cached_method()
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of all available sessions.
        
        Returns:
            Dictionary mapping session IDs to session configurations
        """
        return self.config_manager.get("sessions", {})

    @cached_method()
    def get_session_ids(self) -> List[str]:
        """
        Get a list of all available session IDs.
        
        Returns:
            List of session IDs
        """
        return list(self.get_all_sessions().keys())

    @cached_method()
    def get_session_config(self, session_id: str) -> Union[SessionConfig, Dict[str, Any]]:
        """
        Get the configuration for a specific session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            SessionConfig object or empty dict if not found
            
        Raises:
            SessionConfigError: If session config is invalid and strict_mode is enabled
        """
        # Handle default session specially
        if session_id == "default":
            return self._get_default_session_config()
        
        # Get session data
        session_data = self.config_manager.get("sessions", {}).get(session_id, {})
        
        # Validate session configuration
        validation_result = ConfigValidator.validate_session_config(session_id, session_data)
        
        if not validation_result.valid:
            if self.strict_mode:
                raise SessionConfigError(f"Invalid session '{session_id}': {'; '.join(validation_result.errors)}")
            logging.warning(f"Using default values for invalid session '{session_id}': {'; '.join(validation_result.errors)}")
            return {}
        
        # Create SessionConfig and return
        try:
            return SessionConfig.from_dict(session_id, session_data)
        except Exception as e:
            logging.error(f"Error creating SessionConfig for '{session_id}': {e}")
            if self.strict_mode:
                raise
            return session_data  # Return raw data as fallback

    def _get_default_session_config(self) -> SessionConfig:
        """
        Create a default session configuration using global settings.
        
        Returns:
            SessionConfig object
        """
        prompts = self.config_manager.get("prompts", [])
        url = self.config_manager.get("claude_url", "https://claude.ai")
        delay = self.config_manager.get("delay_between_prompts", 180)
        
        return SessionConfig(
            id="default",
            name="Default Session",
            url=url,
            prompts=prompts,
            delay=delay
        )

    def validate_session(self, session_id: str) -> ValidationResult:
        """
        Validate a session configuration.
        
        Args:
            session_id: The session identifier
            
        Returns:
            ValidationResult with validation outcome
        """
        # Check if session exists
        if session_id != "default" and session_id not in self.get_session_ids():
            result = ValidationResult(False)
            result.add_error(f"Session '{session_id}' not found in configuration")
            return result
        
        # For default session, check global prompts
        if session_id == "default":
            prompts = self.config_manager.get("prompts", [])
            if not prompts:
                result = ValidationResult(False)
                result.add_error("No prompts defined in global configuration")
                return result
            
            # Validate prompts
            return ConfigValidator.validate_prompts(prompts)
        
        # Get session configuration
        session_data = self.config_manager.get("sessions", {}).get(session_id, {})
        
        # Validate session configuration
        return ConfigValidator.validate_session_config(session_id, session_data)

    def is_valid_session(self, session_id: str) -> bool:
        """
        Check if a session is valid.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if session is valid, False otherwise
        """
        return self.validate_session(session_id).valid

    @cached_method()
    def get_claude_url(self, session_id: Optional[str] = None) -> str:
        """
        Get the Claude URL for a session or the global default.
        
        Args:
            session_id: The session identifier (optional)
            
        Returns:
            Claude URL string
        """
        if session_id and session_id != "default":
            # Try to get session-specific URL first
            session_config = self.get_session_config(session_id)
            if isinstance(session_config, SessionConfig):
                return session_config.url
            elif isinstance(session_config, dict):
                url = session_config.get("claude_url")
                if url:
                    return url
        
        # Fall back to global URL
        return self.config_manager.get("claude_url", "https://claude.ai")

    @cached_method()
    def get_prompts(self, session_id: Optional[str] = None) -> List[str]:
        """
        Get the prompts for a session or global prompts.
        
        Args:
            session_id: The session identifier (optional)
            
        Returns:
            List of prompt strings
        """
        if session_id and session_id != "default":
            # Try to get session-specific prompts
            session_config = self.get_session_config(session_id)
            if isinstance(session_config, SessionConfig):
                return session_config.prompts
            elif isinstance(session_config, dict):
                prompts = session_config.get("prompts", [])
                if prompts:
                    return prompts
        
        # Fall back to global prompts
        return self.config_manager.get("prompts", [])
    
    @cached_method()
    def get_global_prompts(self) -> List[str]:
        """
        Get global prompts defined at the root level.
        
        Returns:
            List of prompt strings
        """
        return self.config_manager.get("prompts", [])

    def get_session_name(self, session_id: str) -> str:
        """
        Get the human-readable name for a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Session name (or session_id if no name is defined)
        """
        if session_id == "default":
            return "Default Session"
            
        session_config = self.get_session_config(session_id)
        if isinstance(session_config, SessionConfig):
            return session_config.name
        elif isinstance(session_config, dict):
            return session_config.get("name", session_id)
        return session_id

    # ==== Configuration Access Methods ====
    
    def get_validated_config(self, key: str, validator_func: Callable, 
                           default: Any = None) -> Any:
        """
        Get configuration with validation.
        
        Args:
            key: Configuration key
            validator_func: Function that validates configuration value
            default: Default value if key not found or validation fails
            
        Returns:
            Validated configuration value or default
            
        Raises:
            ValidationError: If validation fails and strict_mode is enabled
        """
        try:
            value = self.config_manager.get(key, default)
            result = validator_func(value)
            
            if not result.valid:
                if self.strict_mode:
                    raise ValidationError(result.errors)
                logging.warning(f"Invalid configuration for {key}: {', '.join(result.errors)}")
                return default
                
            return value
        except Exception as e:
            if not isinstance(e, ValidationError):
                logging.error(f"Error accessing configuration {key}: {e}")
            if self.strict_mode:
                raise
            return default

    @cached_method()
    def get_browser_config(self) -> BrowserConfig:
        """
        Get browser configuration.
        
        Returns:
            BrowserConfig object
        """
        # Get all config that might be needed for browser configuration
        config_data = {
            "browser_profile": self.config_manager.get("browser_profile"),
            "chrome_path": self.config_manager.get("chrome_path"),
            "browser_launch_wait": self.config_manager.get("browser_launch_wait", 10)
        }
        
        # Validate browser configuration
        validation_result = ConfigValidator.validate_browser_config(config_data)
        if not validation_result.valid and self.strict_mode:
            raise ValidationError(validation_result.errors)
        elif not validation_result.valid:
            logging.warning(f"Invalid browser configuration: {', '.join(validation_result.errors)}. Using defaults.")
        
        return BrowserConfig.from_dict(config_data)

    @cached_method()
    def get_retry_settings(self) -> RetryConfig:
        """
        Get retry settings.
        
        Returns:
            RetryConfig object
        """
        # Get all retry-related config
        config_data = {
            "max_retries": self.config_manager.get("max_retries", 3),
            "retry_delay": self.config_manager.get("retry_delay", 2),
            "max_retry_delay": self.config_manager.get("max_retry_delay", 30),
            "retry_jitter": self.config_manager.get("retry_jitter", 0.5),
            "retry_backoff": self.config_manager.get("retry_backoff", 1.5)
        }
        
        return RetryConfig.from_dict(config_data)

    @cached_method()
    def get_visual_config(self) -> VisualConfig:
        """
        Get visual verification configuration.
        
        Returns:
            VisualConfig object
        """
        # Get all visual verification related config
        visual_data = self.config_manager.get("visual_verification", {})
        
        # Create VisualConfig
        return VisualConfig.from_dict({"visual_verification": visual_data})

    def get_delay_between_prompts(self, session_id: Optional[str] = None) -> int:
        """
        Get the delay between prompts for a session or global default.
        
        Args:
            session_id: The session identifier (optional)
            
        Returns:
            Delay in seconds
        """
        if session_id and session_id != "default":
            # Try to get session-specific delay
            session_config = self.get_session_config(session_id)
            if isinstance(session_config, SessionConfig):
                return session_config.delay
            elif isinstance(session_config, dict):
                delay = session_config.get("delay_between_prompts")
                if delay is not None:
                    return delay
        
        # Fall back to global delay or default
        return self.config_manager.get("delay_between_prompts", 180)

    def get_template_path(self, template_name: str) -> str:
        """
        Get the full path to a template image.
        
        Args:
            template_name: Template image name without extension
            
        Returns:
            Path to template image file
        """
        visual_config = self.get_visual_config()
        template_dir = visual_config.template_dir
        return os.path.join(template_dir, f"{template_name}.png")

    def get_global_config(self) -> Dict[str, Any]:
        """
        Get global configuration settings.
        
        Returns:
            Dictionary with global configuration
        """
        config = self.config_manager.get_all() if hasattr(self.config_manager, 'get_all') else {}
        
        # Remove sessions key if present
        if 'sessions' in config:
            config = config.copy()
            del config['sessions']
            
        return config

    # ==== Session Tracking Methods ====
    
    def is_session_complete(self, session_id: str) -> bool:
        """
        Check if a session has been marked as complete.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if session is complete, False otherwise
        """
        return self.session_tracker.is_completed(session_id)

    def mark_session_complete(self, session_id: str, success: bool = True, 
                           notes: Optional[str] = None) -> bool:
        """
        Mark a session as complete.
        
        Args:
            session_id: The session identifier
            success: Whether the session completed successfully
            notes: Optional notes about the session execution
            
        Returns:
            True if status was successfully saved
        """
        return self.session_tracker.mark_completed(session_id, success, notes)

    def reset_session(self, session_id: str) -> bool:
        """
        Reset a session's completion status.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if successful, False otherwise
        """
        return self.session_tracker.reset_session(session_id)

    def get_incomplete_sessions(self) -> List[str]:
        """
        Get a list of sessions that haven't been completed.
        
        Returns:
            List of session IDs
        """
        all_sessions = self.get_all_sessions()
        return [s for s in all_sessions if not self.is_session_complete(s)]

    def get_session_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status information for a session or all sessions.
        
        Args:
            session_id: The session identifier (optional)
            
        Returns:
            Dictionary with session status information
        """
        if hasattr(self.session_tracker, 'get_session_status'):
            # Use main SessionTracker implementation
            return self.session_tracker.get_session_status(session_id)
        elif isinstance(self.session_tracker, SimpleSessionTracker):
            # Use fallback implementation
            if session_id:
                return self.session_tracker.session_status.get(session_id, {})
            else:
                return self.session_tracker.session_status
        else:
            # Unknown implementation, return empty result
            return {} if session_id else {}

    def save(self) -> bool:
        """
        Save any changes to the configuration.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_context():
                return self.config_manager.save()
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            return False
    
    # ==== Factory Methods ====
    
    @classmethod
    def create(cls, config_path: Optional[str] = None, strict_mode: bool = False) -> 'ConfigFacade':
        """
        Factory method to create a properly configured instance.
        
        Args:
            config_path: Path to config file (optional)
            strict_mode: Whether to enable strict validation mode
        
        Returns:
            ConfigFacade instance
        """
        instance = cls(config_path, strict_mode=strict_mode)
        
        # Preload commonly used configurations to prime the cache
        try:
            instance.get_browser_config()
            instance.get_retry_settings()
            instance.get_visual_config()
            instance.get_session_ids()
            instance.get_global_prompts()
        except Exception as e:
            logging.warning(f"Error preloading configuration: {e}")
        
        return instance


# ===== MODULE 7: Backwards Compatibility ====
# In a multi-file implementation, this would be in __init__.py

# Alias for backwards compatibility
SimpleConfig = ConfigFacade
SimpleConfigFacade = ConfigFacade