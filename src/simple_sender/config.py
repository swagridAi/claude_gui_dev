#!/usr/bin/env python3
"""
Configuration facade for the simple sender module.

Provides a simplified interface to the more complex configuration system
while maintaining compatibility with existing configuration files.
"""

import os
import logging
import copy
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass


# Custom exceptions for better error handling
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


# Domain models for typed configuration access
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


# Simplified fallback implementations for reduced coupling
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
    
    def enter_session_mode(self) -> None:
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


# Validation functions
def validate_prompts(prompts: List[Any]) -> Tuple[bool, List[str]]:
    """
    Validate a list of prompts and return whether they're valid.
    
    Args:
        prompts: List of prompts to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if not prompts:
        errors.append("Empty prompts list")
        return False, errors
    
    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, str):
            errors.append(f"Prompt {i+1} is not a string")
        elif not prompt.strip():
            errors.append(f"Prompt {i+1} is empty")
    
    return len(errors) == 0, errors


def validate_session_config(session_id: str, config_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate session configuration.
    
    Args:
        session_id: Session identifier
        config_data: Session configuration data
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check essential fields
    if not config_data:
        errors.append(f"Empty configuration for session '{session_id}'")
        return False, errors
        
    # Validate prompts
    prompts = config_data.get("prompts", [])
    if not prompts:
        errors.append(f"No prompts defined for session '{session_id}'")
    else:
        valid, prompt_errors = validate_prompts(prompts)
        if not valid:
            errors.extend(prompt_errors)
    
    # Validate URL
    url = config_data.get("claude_url")
    if url and not isinstance(url, str):
        errors.append(f"URL must be a string, got {type(url).__name__}")
    elif url and not url.startswith(("http://", "https://")):
        errors.append(f"Invalid URL format: {url}")
    
    # Validate delay
    delay = config_data.get("delay_between_prompts")
    if delay is not None:
        if not isinstance(delay, (int, float)):
            errors.append(f"Delay must be a number, got {type(delay).__name__}")
        elif delay < 0:
            errors.append(f"Delay cannot be negative: {delay}")
    
    return len(errors) == 0, errors


def validate_browser_config(config_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate browser configuration.
    
    Args:
        config_data: Browser configuration data
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Validate chrome_path
    chrome_path = config_data.get("chrome_path")
    if chrome_path and not isinstance(chrome_path, str):
        errors.append(f"Chrome path must be a string, got {type(chrome_path).__name__}")
    elif chrome_path and not os.path.exists(chrome_path):
        errors.append(f"Chrome path does not exist: {chrome_path}")
    
    # Validate browser_profile
    profile = config_data.get("browser_profile")
    if profile and not isinstance(profile, str):
        errors.append(f"Browser profile must be a string, got {type(profile).__name__}")
    
    # Validate startup_wait
    startup_wait = config_data.get("browser_launch_wait")
    if startup_wait is not None:
        if not isinstance(startup_wait, (int, float)):
            errors.append(f"Browser launch wait must be a number, got {type(startup_wait).__name__}")
        elif startup_wait < 0:
            errors.append(f"Browser launch wait cannot be negative: {startup_wait}")
    
    return len(errors) == 0, errors


# Main configuration facade class
class SimpleConfigFacade:
    """
    A simplified configuration interface for the simple sender module.
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
        self.cache_ttl = cache_ttl
        
        # Initialize caching system
        self._cache = {}
        self._cache_timestamps = {}
        
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

    # Cache management methods
    def _is_cache_valid(self, key: str) -> bool:
        """Check if a cached item is still valid."""
        if key not in self._cache or key not in self._cache_timestamps:
            return False
        age = time.time() - self._cache_timestamps[key]
        return age < self.cache_ttl

    def _cache_value(self, key: str, value: Any) -> None:
        """Store a value in cache with timestamp."""
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()

    def invalidate_cache(self, key: Optional[str] = None) -> None:
        """Invalidate specific cache key or entire cache."""
        if key:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
        else:
            self._cache.clear()
            self._cache_timestamps.clear()

    def get_config_with_validation(self, key: str, validator_func: Callable, 
                                  default: Any = None) -> Any:
        """
        Get configuration with consistent validation and error handling.
        
        Args:
            key: Configuration key
            validator_func: Function that validates configuration value
            default: Default value if key not found or validation fails
            
        Returns:
            Validated configuration value or default
            
        Raises:
            ValidationError: If validation fails and strict_mode is enabled
        """
        cache_key = f"validated_{key}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
            
        try:
            value = self.config_manager.get(key, default)
            valid, errors = validator_func(value)
            
            if not valid:
                if self.strict_mode:
                    raise ValidationError(errors)
                logging.warning(f"Invalid configuration for {key}: {', '.join(errors)}")
                return default
                
            self._cache_value(cache_key, value)
            return value
        except Exception as e:
            if not isinstance(e, ValidationError):
                logging.error(f"Error accessing configuration {key}: {e}")
            if self.strict_mode:
                raise
            return default

    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of all available sessions.
        
        Returns:
            Dictionary mapping session IDs to session configurations
        """
        cache_key = "all_sessions"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
            
        sessions = self.config_manager.get("sessions", {})
        self._cache_value(cache_key, sessions)
        return sessions

    def get_session_ids(self) -> List[str]:
        """
        Get a list of all available session IDs.
        
        Returns:
            List of session IDs
        """
        return list(self.get_all_sessions().keys())

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
        # Check cache first
        cache_key = f"session_{session_id}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        # Handle default session specially
        if session_id == "default":
            config = self._get_default_session_config()
            self._cache_value(cache_key, config)
            return config
        
        # Get session data
        session_data = self.config_manager.get("sessions", {}).get(session_id, {})
        
        # Validate session configuration
        valid, errors = validate_session_config(session_id, session_data)
        
        if not valid:
            if self.strict_mode:
                raise SessionConfigError(f"Invalid session '{session_id}': {', '.join(errors)}")
            logging.warning(f"Using default values for invalid session '{session_id}': {', '.join(errors)}")
            return {}
        
        # Create SessionConfig and cache
        try:
            config = SessionConfig.from_dict(session_id, session_data)
            self._cache_value(cache_key, config)
            return config
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

    def validate_session(self, session_id: str) -> Tuple[bool, str]:
        """
        Validate a session configuration and return whether it's valid.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if session exists
        if session_id != "default" and session_id not in self.get_session_ids():
            return False, f"Session '{session_id}' not found in configuration"
        
        # For default session, check global prompts
        if session_id == "default":
            prompts = self.config_manager.get("prompts", [])
            if not prompts:
                return False, "No prompts defined in global configuration"
            
            # Validate prompts
            is_valid, errors = validate_prompts(prompts)
            if not is_valid:
                return False, f"Invalid global prompts: {', '.join(errors)}"
                
            return True, ""
        
        # Get session configuration
        session_data = self.config_manager.get("sessions", {}).get(session_id, {})
        
        # Validate session configuration
        valid, errors = validate_session_config(session_id, session_data)
        if not valid:
            return False, f"Invalid session '{session_id}': {', '.join(errors)}"
        
        return True, ""

    def is_valid_session(self, session_id: str) -> bool:
        """
        Check if a session is valid.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if session is valid, False otherwise
        """
        valid, _ = self.validate_session(session_id)
        return valid

    def get_url(self, session_id: Optional[str] = None) -> str:
        """
        Get the Claude URL for a session or the global default.
        
        Args:
            session_id: The session identifier (optional)
            
        Returns:
            Claude URL string
        """
        # Use cache if available
        cache_key = f"url_{session_id or 'global'}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        if session_id and session_id != "default":
            # Try to get session-specific URL first
            session_config = self.get_session_config(session_id)
            if isinstance(session_config, SessionConfig):
                url = session_config.url
                self._cache_value(cache_key, url)
                return url
            elif isinstance(session_config, dict):
                url = session_config.get("claude_url")
                if url:
                    self._cache_value(cache_key, url)
                    return url
        
        # Fall back to global URL
        url = self.config_manager.get("claude_url", "https://claude.ai")
        self._cache_value(cache_key, url)
        return url

    def get_prompts(self, session_id: Optional[str] = None) -> List[str]:
        """
        Get the prompts for a session or global prompts.
        
        Args:
            session_id: The session identifier (optional)
            
        Returns:
            List of prompt strings
        """
        cache_key = f"prompts_{session_id or 'global'}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
            
        if session_id and session_id != "default":
            # Try to get session-specific prompts
            session_config = self.get_session_config(session_id)
            if isinstance(session_config, SessionConfig):
                prompts = session_config.prompts
                self._cache_value(cache_key, prompts)
                return prompts
            elif isinstance(session_config, dict):
                prompts = session_config.get("prompts", [])
                if prompts:
                    self._cache_value(cache_key, prompts)
                    return prompts
        
        # Fall back to global prompts
        prompts = self.config_manager.get("prompts", [])
        self._cache_value(cache_key, prompts)
        return prompts
        
    def get_global_prompts(self) -> List[str]:
        """
        Get global prompts defined at the root level.
        
        Returns:
            List of prompt strings
        """
        return self.config_manager.get("prompts", [])

    def get_browser_config(self) -> BrowserConfig:
        """
        Get browser configuration.
        
        Returns:
            BrowserConfig object
        """
        cache_key = "browser_config"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
            
        # Get all config that might be needed for browser configuration
        config_data = {
            "browser_profile": self.config_manager.get("browser_profile"),
            "chrome_path": self.config_manager.get("chrome_path"),
            "browser_launch_wait": self.config_manager.get("browser_launch_wait", 10)
        }
        
        # Validate browser configuration
        valid, errors = validate_browser_config(config_data)
        if not valid and self.strict_mode:
            raise ValidationError(errors)
        elif not valid:
            logging.warning(f"Invalid browser configuration: {', '.join(errors)}. Using defaults.")
        
        browser_config = BrowserConfig.from_dict(config_data)
        self._cache_value(cache_key, browser_config)
        return browser_config

    def get_retry_settings(self) -> RetryConfig:
        """
        Get retry settings.
        
        Returns:
            RetryConfig object
        """
        cache_key = "retry_config"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
            
        # Get all retry-related config
        config_data = {
            "max_retries": self.config_manager.get("max_retries", 3),
            "retry_delay": self.config_manager.get("retry_delay", 2),
            "max_retry_delay": self.config_manager.get("max_retry_delay", 30),
            "retry_jitter": self.config_manager.get("retry_jitter", 0.5),
            "retry_backoff": self.config_manager.get("retry_backoff", 1.5)
        }
        
        retry_config = RetryConfig.from_dict(config_data)
        self._cache_value(cache_key, retry_config)
        return retry_config

    def get_delay_between_prompts(self, session_id: Optional[str] = None) -> int:
        """
        Get the delay between prompts for a session or global default.
        
        Args:
            session_id: The session identifier (optional)
            
        Returns:
            Delay in seconds
        """
        cache_key = f"delay_{session_id or 'global'}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
            
        if session_id and session_id != "default":
            # Try to get session-specific delay
            session_config = self.get_session_config(session_id)
            if isinstance(session_config, SessionConfig):
                delay = session_config.delay
                self._cache_value(cache_key, delay)
                return delay
            elif isinstance(session_config, dict):
                delay = session_config.get("delay_between_prompts")
                if delay is not None:
                    self._cache_value(cache_key, delay)
                    return delay
        
        # Fall back to global delay or default
        delay = self.config_manager.get("delay_between_prompts", 180)
        self._cache_value(cache_key, delay)
        return delay

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

    # Session tracking methods
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
        # Set preservation mode if needed
        if self.preserve_original and hasattr(self.config_manager, 'is_in_session_mode'):
            if not self.config_manager.is_in_session_mode():
                self.config_manager.enter_session_mode()
        
        # Save configuration
        result = self.config_manager.save()
        
        # Exit session mode if we entered it
        if self.preserve_original and hasattr(self.config_manager, 'is_in_session_mode'):
            if self.config_manager.is_in_session_mode():
                self.config_manager.exit_session_mode()
            
        return result
    
    # Factory methods
    @classmethod
    def create(cls, config_path: Optional[str] = None, strict_mode: bool = False) -> 'SimpleConfigFacade':
        """
        Factory method to create a properly configured instance.
        
        Args:
            config_path: Path to config file (optional)
            strict_mode: Whether to enable strict validation mode
        
        Returns:
            SimpleConfigFacade instance
        """
        instance = cls(config_path, strict_mode=strict_mode)
        
        # Preload commonly used configurations to prime the cache
        try:
            instance.get_browser_config()
            instance.get_retry_settings()
            instance.get_session_ids()
            instance.get_global_prompts()
        except Exception as e:
            logging.warning(f"Error preloading configuration: {e}")
        
        return instance


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
        ]
    }


def resolve_session_priority(available_sessions: List[str], 
                             specified_session: Optional[str] = None, 
                             run_one: bool = False,
                             skip_completed: bool = True,
                             config: Optional[SimpleConfigFacade] = None) -> List[str]:
    """
    Determine which sessions to run and in what order.
    
    Args:
        available_sessions: List of available session IDs
        specified_session: Specific session to prioritize (optional)
        run_one: Whether to run only one session
        skip_completed: Whether to skip completed sessions
        config: SimpleConfigFacade instance for checking completion status
        
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


# Compatibility alias for backward compatibility
SimpleConfig = SimpleConfigFacade