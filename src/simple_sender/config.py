#!/usr/bin/env python3
"""
Configuration facade for the simple sender module.

Provides a simplified interface to the more complex configuration system
while maintaining compatibility with existing configuration files.
"""

import os
import logging
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

# Import from the main project
from src.utils.config_manager import ConfigManager
from src.utils.session_tracker import SessionTracker


class SimpleConfigFacade:
    """
    A simplified configuration interface for the simple sender module.
    Acts as a facade over the more complex ConfigManager, providing
    domain-specific methods for common configuration operations.
    """
    
    def __init__(self, config_path: Optional[str] = None, preserve_original: bool = True):
        """
        Initialize the configuration facade.
        
        Args:
            config_path: Path to configuration file (optional)
            preserve_original: Whether to preserve original config when saving
        """
        self.config_path = config_path or "config/user_config.yaml"
        self.preserve_original = preserve_original
        
        # Initialize the underlying configuration manager
        self.config_manager = ConfigManager(self.config_path)
        
        # Initialize session tracker for progress monitoring
        tracker_path = os.path.join(os.path.dirname(self.config_path), "session_status.json")
        self.session_tracker = SessionTracker(tracker_path)
        
        # Cache for frequently accessed values
        self._cache = {}

    def get_all_sessions(self) -> List[str]:
        """
        Get a list of all available session IDs.
        
        Returns:
            List of session IDs
        """
        return list(self.config_manager.get("sessions", {}).keys())

    def get_session_config(self, session_id: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Session configuration dictionary or empty dict if not found
        """
        return self.config_manager.get("sessions", {}).get(session_id, {})

    def validate_session(self, session_id: str) -> Tuple[bool, str]:
        """
        Validate a session configuration and return whether it's valid.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if session exists
        if session_id != "default" and session_id not in self.get_all_sessions():
            return False, f"Session '{session_id}' not found in configuration"
        
        # For default session, check global prompts
        if session_id == "default":
            prompts = self.config_manager.get("prompts", [])
            if not prompts:
                return False, "No prompts defined in global configuration"
            
            # Validate prompts
            is_valid, error = validate_prompts(prompts)
            if not is_valid:
                return False, f"Invalid global prompts: {error}"
                
            return True, ""
        
        # Get session configuration
        session_config = self.get_session_config(session_id)
        
        # Check for prompts
        if "prompts" not in session_config or not session_config["prompts"]:
            return False, f"No prompts defined for session '{session_id}'"
        
        # Validate prompts
        is_valid, error = validate_prompts(session_config["prompts"])
        if not is_valid:
            return False, f"Invalid prompts for session '{session_id}': {error}"
        
        return True, ""

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
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if session_id and session_id != "default":
            # Try to get session-specific URL first
            session_config = self.get_session_config(session_id)
            url = session_config.get("claude_url")
            if url:
                self._cache[cache_key] = url
                return url
        
        # Fall back to global URL
        url = self.config_manager.get("claude_url", "https://claude.ai")
        self._cache[cache_key] = url
        return url

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
            prompts = session_config.get("prompts", [])
            if prompts:
                return prompts
        
        # Fall back to global prompts
        return self.config_manager.get("prompts", [])

    def get_browser_profile(self) -> str:
        """
        Get the browser profile directory.
        
        Returns:
            Profile directory path
        """
        # Use cache if available
        if "browser_profile" in self._cache:
            return self._cache["browser_profile"]
        
        profile = self.config_manager.get("browser_profile")
        if not profile:
            # Create default profile path
            profile = os.path.join(os.path.expanduser("~"), "ClaudeProfile")
        
        self._cache["browser_profile"] = profile
        return profile

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
            delay = session_config.get("delay_between_prompts")
            if delay is not None:
                return delay
        
        # Fall back to global delay or default
        return self.config_manager.get("delay_between_prompts", 180)

    def get_retry_settings(self) -> Dict[str, Any]:
        """
        Get retry settings.
        
        Returns:
            Dictionary with retry settings
        """
        return {
            "max_retries": self.config_manager.get("max_retries", 3),
            "retry_delay": self.config_manager.get("retry_delay", 2),
            "max_retry_delay": self.config_manager.get("max_retry_delay", 30),
            "retry_jitter": self.config_manager.get("retry_jitter", 0.5),
            "retry_backoff": self.config_manager.get("retry_backoff", 1.5)
        }

    def get_session_name(self, session_id: str) -> str:
        """
        Get the human-readable name for a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Session name (or session_id if no name is defined)
        """
        session_config = self.get_session_config(session_id)
        return session_config.get("name", session_id)

    def is_session_complete(self, session_id: str) -> bool:
        """
        Check if a session has been marked as complete.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if session is complete, False otherwise
        """
        return self.session_tracker.is_completed(session_id)

    def mark_session_complete(self, session_id: str, success: bool = True, notes: Optional[str] = None) -> bool:
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

    def save(self) -> bool:
        """
        Save any changes to the configuration.
        
        Returns:
            True if successful, False otherwise
        """
        # Set preservation mode if needed
        if self.preserve_original and not self.config_manager.is_in_session_mode():
            self.config_manager.enter_session_mode()
        
        # Save configuration
        result = self.config_manager.save()
        
        # Exit session mode if we entered it
        if self.preserve_original and self.config_manager.is_in_session_mode():
            self.config_manager.exit_session_mode()
            
        return result


def validate_prompts(prompts: List[Any]) -> Tuple[bool, str]:
    """
    Validate a list of prompts and return whether they're valid.
    
    Args:
        prompts: List of prompts to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not prompts:
        return False, "Empty prompts list"
    
    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, str):
            return False, f"Prompt {i+1} is not a string"
        
        if not prompt.strip():
            return False, f"Prompt {i+1} is empty"
    
    return True, ""


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


def get_simple_config(config_path: Optional[str] = None) -> SimpleConfigFacade:
    """
    Factory function to create and return a SimpleConfigFacade instance.
    Handles path resolution and initialization.
    
    Args:
        config_path: Path to config file (optional)
        
    Returns:
        Initialized SimpleConfigFacade instance
    """
    # Resolve config path
    if not config_path:
        config_path = "config/user_config.yaml"
    
    # Create configuration facade
    return SimpleConfigFacade(config_path)