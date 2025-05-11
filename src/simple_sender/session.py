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
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import random
from enum import Enum, auto

# State enumerations for type-safe state management
class SessionState(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    INTERRUPTED = auto()

class PromptState(Enum):
    PENDING = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()

# Enhanced error hierarchy
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
        
    def start(self):
        """Mark the start of prompt execution."""
        self.start_time = datetime.now()
        self.state = PromptState.EXECUTING
        self.attempt_count += 1
        return self
        
    def complete(self, success: bool, error: Optional[str] = None):
        """Mark the completion of prompt execution with results."""
        self.end_time = datetime.now()
        self.state = PromptState.COMPLETED if success else PromptState.FAILED
        self.error = error
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
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "attempt_count": self.attempt_count,
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
        
        # Parse timestamps if they exist
        if data.get("start_time"):
            result.start_time = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            result.end_time = datetime.fromisoformat(data["end_time"])
            
        return result


class CheckpointManager:
    """Manages session checkpoints for resumable execution."""
    
    def __init__(self, checkpoint_dir: str = "logs/simple_sender/checkpoints"):
        """Initialize the checkpoint manager."""
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.version = "1.0"  # Checkpoint format version
    
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
            
            with open(self.checkpoint_path(session_id), 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
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
        """
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
            
            return checkpoint_data.get("session")
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
                logging.debug(f"Deleted checkpoint for session '{session_id}'")
                return True
            except Exception as e:
                logging.warning(f"Failed to delete checkpoint for session '{session_id}': {e}")
                return False
        return True


class PromptExecutor:
    """Executes prompts through a browser handler with error handling."""
    
    def execute_prompt(self, prompt_text: str, browser_handler, 
                     timeout: int = 300, retry_count: int = 0) -> Tuple[bool, Optional[str]]:
        """
        Execute a single prompt and wait for response.
        
        Args:
            prompt_text: The prompt to execute
            browser_handler: Handler for browser interactions
            timeout: Maximum time to wait for response in seconds
            retry_count: Current retry attempt (for logging)
            
        Returns:
            Tuple of (success, error_message)
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
            browser_handler.type_text(prompt_text)
            time.sleep(1)
            
            # Send prompt
            logging.info(f"{prefix}Sending prompt")
            browser_handler.send_prompt()
            
            # Wait for response with visual verification if available
            if hasattr(browser_handler, 'wait_for_response_complete'):
                success = browser_handler.wait_for_response_complete(timeout)
                if not success:
                    return False, "Response timeout or verification failed"
            else:
                # Fall back to simple waiting if visual verification is not available
                self._wait_with_logging(timeout)
            
            return True, None
            
        except Exception as e:
            error_msg = f"Error executing prompt: {str(e)}"
            logging.error(error_msg)
            return False, error_msg
    
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
            "results": [r.to_dict() for r in self.results]
        }
    
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
        session.state = SessionState[data.get("state", "PENDING")]
        session.current_prompt_index = data.get("current_prompt_index", 0)
        session.error = data.get("error")
        
        # Restore timestamps if they exist
        if data.get("start_time"):
            session.start_time = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            session.end_time = datetime.fromisoformat(data["end_time"])
        
        # Restore prompt results if they exist
        if "results" in data and isinstance(data["results"], list):
            session.results = [
                PromptResult.from_dict(result_data) 
                for result_data in data["results"]
            ]
        
        return session


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
        self.prompt_executor = prompt_executor or PromptExecutor()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.session_tracker = session_tracker
        self.sessions: Dict[str, Session] = {}
    
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
                       progress_callback: Optional[Callable] = None) -> bool:
        """
        Execute a single session from start to finish.
        
        Args:
            session_id: ID of the session to execute
            browser_handler: Browser handler instance for browser operations
            prompt_delay: Delay between prompts in seconds (None for config default)
            resume: Whether to resume from checkpoint if available
            progress_callback: Optional callback for progress updates
            
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
        
        try:
            # Report progress if callback provided
            if progress_callback:
                progress_callback("session_started", session_id)
            
            # Start the session if not already running
            if session.state != SessionState.RUNNING:
                session.start()
            
            # Save initial checkpoint
            self._save_session_checkpoint(session_id)
            
            # Launch browser for this session
            try:
                if not browser_handler.launch(session.url, session.profile_dir):
                    error_msg = f"Failed to launch browser for session '{session_id}'"
                    logging.error(error_msg)
                    session.complete(False, error_msg)
                    if progress_callback:
                        progress_callback("session_failed", session_id, error_msg)
                    return False
            except Exception as e:
                error_msg = f"Browser launch error: {str(e)}"
                logging.error(error_msg)
                session.complete(False, error_msg)
                if progress_callback:
                    progress_callback("session_failed", session_id, error_msg)
                return False
            
            # Process all prompts
            success = self._process_prompts(session, browser_handler, prompt_delay, progress_callback)
            
            session.complete(success)
            if success and self.session_tracker:
                self.session_tracker.mark_completed(session_id, True)
                if progress_callback:
                    progress_callback("session_completed", session_id)
            
            # Clean up checkpoint if successful
            if success:
                self.checkpoint_manager.delete(session_id)
            
            return success
            
        except SessionExecutionError as e:
            error_msg = f"Error executing session '{session_id}': {str(e)}"
            logging.error(error_msg)
            session.complete(False, error_msg)
            if progress_callback:
                progress_callback("session_failed", session_id, error_msg)
            return False
        finally:
            # Always close the browser in case of error
            try:
                browser_handler.close()
            except Exception as e:
                logging.error(f"Error closing browser: {e}")
    
    def run_sessions(self, browser_handler, 
                     prompt_delay: int = 300, 
                     session_delay: int = 10,
                     progress_callback: Optional[Callable] = None) -> Dict[str, bool]:
        """
        Run all loaded sessions.
        
        Args:
            browser_handler: Browser handler instance
            prompt_delay: Delay between prompts in seconds
            session_delay: Delay between sessions in seconds
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping session IDs to success status
        """
        if not self.sessions:
            logging.error("No sessions loaded")
            return {}
        
        results = {}
        session_ids = list(self.sessions.keys())
        
        # Report overall progress if callback provided
        if progress_callback:
            progress_callback("run_started", session_ids)
        
        for i, session_id in enumerate(session_ids):
            # Report session progress if callback provided
            if progress_callback:
                progress_callback("session_starting", session_id, i, len(session_ids))
            
            # Execute session
            success = self.execute_session(
                session_id, 
                browser_handler, 
                prompt_delay,
                progress_callback=progress_callback
            )
            results[session_id] = success
            
            # Wait between sessions if there are more to process
            if i < len(session_ids) - 1:
                logging.info(f"Waiting {session_delay} seconds before next session...")
                time.sleep(session_delay)
        
        # Log summary
        self._log_summary(results)
        
        # Report completion if callback provided
        if progress_callback:
            progress_callback("run_completed", results)
        
        return results
    
    def _process_prompts(self, session: Session, browser_handler,
                       prompt_delay: int,
                       progress_callback: Optional[Callable] = None) -> bool:
        """
        Process all prompts in a session.
        
        Args:
            session: Session to process
            browser_handler: Browser handler instance
            prompt_delay: Delay between prompts in seconds
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if all prompts were processed successfully, False otherwise
        """
        # Wait for initial page load and login
        logging.info("Waiting 15 seconds for page to load and login...")
        time.sleep(15)
        
        # Get retry settings
        retry_settings = self.config_facade.get_retry_settings()
        max_retries = retry_settings.get("max_retries", 3)
        
        # Process each prompt
        total_prompts = len(session.prompts)
        while True:
            next_prompt = session.get_next_prompt()
            if not next_prompt:
                break
                
            prompt_index, prompt_text = next_prompt
            prompt_result = session.results[prompt_index].start()
            
            # Report prompt progress if callback provided
            if progress_callback:
                progress_callback("prompt_started", session.id, prompt_index, total_prompts)
            
            success = False
            for retry in range(max_retries):
                try:
                    success, error = self.prompt_executor.execute_prompt(
                        prompt_text, 
                        browser_handler,
                        timeout=prompt_delay,
                        retry_count=retry
                    )
                    
                    if success:
                        break
                        
                    # If this is not the last retry, wait before trying again
                    if retry < max_retries - 1:
                        # Calculate retry delay with exponential backoff and jitter
                        base_delay = retry_settings.get("retry_delay", 2)
                        backoff = retry_settings.get("retry_backoff", 1.5)
                        jitter = retry_settings.get("retry_jitter", 0.5)
                        
                        delay = min(
                            base_delay * (backoff ** retry),
                            retry_settings.get("max_retry_delay", 30)
                        )
                        
                        # Add jitter
                        jitter_factor = random.uniform(1 - jitter, 1 + jitter)
                        actual_delay = delay * jitter_factor
                        
                        logging.warning(f"Retry {retry+1}/{max_retries} after {actual_delay:.1f}s: {error}")
                        time.sleep(actual_delay)
                        
                except Exception as e:
                    error = f"Error processing prompt: {str(e)}"
                    logging.error(error)
                    
                    # Report error if callback provided
                    if progress_callback:
                        progress_callback("prompt_error", session.id, prompt_index, str(e))
                    
                    # If this is not the last retry, try again
                    if retry < max_retries - 1:
                        logging.warning(f"Retry {retry+1}/{max_retries} after exception")
                        time.sleep(retry_settings.get("retry_delay", 2))
            
            # Mark prompt as completed or failed
            prompt_result.complete(success, None if success else error)
            
            # Report prompt completion if callback provided
            if progress_callback:
                progress_callback(
                    "prompt_completed" if success else "prompt_failed", 
                    session.id, prompt_index, total_prompts
                )
            
            # If prompt failed after all retries, end session
            if not success:
                logging.error(f"Failed to process prompt {prompt_index + 1} after {max_retries} attempts")
                return False
            
            # Advance to next prompt and save checkpoint
            session.advance_prompt()
            self._save_session_checkpoint(session.id)
        
        return True
    
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
            session.state = SessionState[checkpoint_data.get("state", "PENDING")]
            
            # Update timestamps if available
            if checkpoint_data.get("start_time"):
                session.start_time = datetime.fromisoformat(checkpoint_data["start_time"])
            
            # Update prompt results if available
            if "results" in checkpoint_data and isinstance(checkpoint_data["results"], list):
                for result_data in checkpoint_data["results"]:
                    index = result_data.get("index", 0)
                    if 0 <= index < len(session.results):
                        session.results[index] = PromptResult.from_dict(result_data)
            
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
        
        # Add detailed prompt data
        report["prompts"] = [r.to_dict() for r in session.results]
        
        return report
    
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