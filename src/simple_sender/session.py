#!/usr/bin/env python3
"""
Session Management Module for Simple Claude Sender

This module provides classes and functions for managing Claude AI automation sessions,
including session execution, tracking, resuming, and reporting.
"""

import logging
import time
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import random

class SessionError(Exception):
    """Base exception for session-related errors."""
    pass

class SessionConfigError(SessionError):
    """Exception raised for session configuration errors."""
    pass

class SessionExecutionError(SessionError):
    """Exception raised when session execution fails."""
    pass

class PromptResult:
    """Represents the result of a single prompt execution."""
    
    def __init__(self, prompt_text: str, index: int):
        self.prompt_text = prompt_text
        self.index = index
        self.success = False
        self.start_time = None
        self.end_time = None
        self.error = None
        self.duration = None
        
    def start(self):
        """Mark the start of prompt execution."""
        self.start_time = datetime.now()
        return self
        
    def complete(self, success: bool, error: Optional[str] = None):
        """Mark the completion of prompt execution with results."""
        self.end_time = datetime.now()
        self.success = success
        self.error = error
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
        return self
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "index": self.index,
            "prompt": self.prompt_text[:100] + "..." if len(self.prompt_text) > 100 else self.prompt_text,
            "success": self.success,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "error": self.error
        }


class Session:
    """Represents a single Claude AI automation session."""
    
    def __init__(self, session_id: str, config: Dict[str, Any], global_config: Dict[str, Any]):
        self.id = session_id
        self.name = config.get("name", session_id)
        self.url = config.get("claude_url", global_config.get("claude_url", "https://claude.ai"))
        self.prompts = config.get("prompts", [])
        self.current_prompt_index = 0
        self.status = "pending"  # pending, running, completed, failed
        self.results: List[PromptResult] = []
        self.start_time = None
        self.end_time = None
        self.profile_dir = global_config.get("browser_profile")
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
        self.status = "running"
        logging.info(f"=== Starting session: {self.id} ({self.name}) ===")
        logging.info(f"Session URL: {self.url}")
        logging.info(f"Number of prompts: {len(self.prompts)}")
    
    def complete(self, success: bool):
        """Mark the completion of session execution."""
        self.end_time = datetime.now()
        self.status = "completed" if success else "failed"
        
        # Calculate success statistics
        total_prompts = len(self.prompts)
        completed_prompts = sum(1 for r in self.results if r.end_time is not None)
        successful_prompts = sum(1 for r in self.results if r.success)
        
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0
        minutes, seconds = divmod(duration, 60)
        hours, minutes = divmod(minutes, 60)
        
        logging.info(f"=== Session {self.id} {self.status} ===")
        logging.info(f"Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        logging.info(f"Prompts: {successful_prompts}/{completed_prompts}/{total_prompts} "
                    f"(successful/attempted/total)")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "status": self.status,
            "current_prompt_index": self.current_prompt_index,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "prompts_count": len(self.prompts),
            "results": [r.to_dict() for r in self.results]
        }


class SessionManager:
    """Manages multiple Claude AI automation sessions."""
    
    def __init__(self, config_manager, session_tracker=None):
        """
        Initialize session manager.
        
        Args:
            config_manager: Configuration manager instance
            session_tracker: Optional session tracker for persistence
        """
        self.config_manager = config_manager
        self.session_tracker = session_tracker
        self.sessions: Dict[str, Session] = {}
        self.checkpoint_dir = "logs/simple_sender/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
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
        config = self.config_manager.get_all()
        all_sessions = config.get("sessions", {})
        global_config = {k: v for k, v in config.items() if k != "sessions"}
        
        if not all_sessions:
            # If no sessions defined, create a default session with global prompts
            global_prompts = config.get("prompts", [])
            if not global_prompts:
                logging.error("No prompts found in configuration")
                return []
            
            default_session = {
                "name": "Default",
                "prompts": global_prompts,
                "claude_url": config.get("claude_url", "https://claude.ai")
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
            session_config = all_sessions.get(session_id, {})
            try:
                session = Session(session_id, session_config, global_config)
                self.sessions[session_id] = session
                loaded_sessions.append(session_id)
                logging.debug(f"Loaded session: {session_id}")
                
                # Check for checkpoint and resume if available
                if self._checkpoint_exists(session_id):
                    self._load_checkpoint(session_id)
            except SessionConfigError as e:
                logging.error(f"Failed to load session '{session_id}': {e}")
        
        logging.info(f"Loaded {len(loaded_sessions)} sessions")
        return loaded_sessions
    
    def get_session(self, session_id) -> Optional[Session]:
        """Get a session by ID."""
        return self.sessions.get(session_id)
    
    def execute_session(self, session_id: str, prompt_delay: int, 
                        browser_handler) -> bool:
        """
        Execute a single session from start to finish.
        
        Args:
            session_id: ID of the session to execute
            prompt_delay: Delay between prompts in seconds
            browser_handler: Browser handler instance for browser operations
            
        Returns:
            True if successful, False otherwise
        """
        session = self.sessions.get(session_id)
        if not session:
            logging.error(f"Session '{session_id}' not found")
            return False
        
        try:
            session.start()
            self._save_checkpoint(session_id)
            
            # Launch browser for this session
            if not browser_handler.launch(session.url, session.profile_dir):
                logging.error(f"Failed to launch browser for session '{session_id}'")
                session.complete(False)
                return False
            
            # Process all prompts
            success = self._process_prompts(session, prompt_delay, browser_handler)
            
            session.complete(success)
            if success and self.session_tracker:
                self.session_tracker.mark_completed(session_id, True)
            
            # Clean up checkpoint if successful
            if success:
                self._delete_checkpoint(session_id)
            
            return success
            
        except SessionExecutionError as e:
            logging.error(f"Error executing session '{session_id}': {e}")
            session.complete(False)
            return False
        finally:
            # Always close the browser
            try:
                browser_handler.close()
            except Exception as e:
                logging.error(f"Error closing browser: {e}")
    
    def run_sessions(self, browser_reuse=False, prompt_delay=300, 
                     session_delay=10, browser_handler=None) -> Dict[str, bool]:
        """
        Run all loaded sessions.
        
        Args:
            browser_reuse: Whether to reuse browser instance between sessions
            prompt_delay: Delay between prompts in seconds
            session_delay: Delay between sessions in seconds
            browser_handler: Optional browser handler instance
            
        Returns:
            Dictionary mapping session IDs to success status
        """
        if not self.sessions:
            logging.error("No sessions loaded")
            return {}
        
        # Use provided browser handler or create a new one
        if not browser_handler:
            from src.simple_sender.browser import SimpleBrowserHandler
            browser_handler = SimpleBrowserHandler()
        
        results = {}
        session_ids = list(self.sessions.keys())
        
        for i, session_id in enumerate(session_ids):
            # Execute session
            success = self.execute_session(session_id, prompt_delay, browser_handler)
            results[session_id] = success
            
            # Wait between sessions if there are more to process
            if i < len(session_ids) - 1:
                logging.info(f"Waiting {session_delay} seconds before next session...")
                time.sleep(session_delay)
                
                # If not reusing browser, ensure it's closed
                if not browser_reuse:
                    browser_handler.close()
        
        # Log summary
        self._log_summary(results)
        return results
    
    def _process_prompts(self, session: Session, prompt_delay: int, 
                         browser_handler) -> bool:
        """
        Process all prompts in a session.
        
        Args:
            session: Session to process
            prompt_delay: Delay between prompts in seconds
            browser_handler: Browser handler instance
            
        Returns:
            True if all prompts were processed successfully, False otherwise
        """
        # Wait for initial page load and login
        logging.info("Waiting 15 seconds for page to load and login...")
        time.sleep(15)
        
        # Process each prompt
        while True:
            next_prompt = session.get_next_prompt()
            if not next_prompt:
                break
                
            prompt_index, prompt_text = next_prompt
            prompt_result = session.results[prompt_index].start()
            
            try:
                logging.info(f"Processing prompt {prompt_index + 1}/{len(session.prompts)}")
                
                # Clear any current content
                browser_handler.clear_input()
                time.sleep(0.5)
                
                # Type the prompt with human-like delays
                logging.info(f"Typing prompt: {prompt_text[:50]}..." if len(prompt_text) > 50 
                            else f"Typing prompt: {prompt_text}")
                self._type_humanized(browser_handler, prompt_text)
                time.sleep(1)
                
                # Send prompt
                logging.info("Sending prompt")
                browser_handler.send_prompt()
                
                # Wait for response
                logging.info(f"Waiting {prompt_delay} seconds for response...")
                self._wait_with_logging(prompt_delay)
                
                # Mark prompt as successful
                prompt_result.complete(True)
                session.advance_prompt()
                self._save_checkpoint(session.id)
                
            except Exception as e:
                error_msg = f"Error processing prompt {prompt_index + 1}: {str(e)}"
                logging.error(error_msg)
                prompt_result.complete(False, error_msg)
                return False
        
        return True
    
    def _wait_with_logging(self, wait_time: int):
        """Wait with periodic logging."""
        start_time = time.time()
        while time.time() - start_time < wait_time:
            time.sleep(30)  # Check every 30 seconds
            elapsed = time.time() - start_time
            remaining = wait_time - elapsed
            if remaining > 0:
                logging.info(f"Still waiting: {int(remaining)} seconds remaining...")
    
    def _type_humanized(self, browser_handler, text: str):
        """Type text with human-like delays."""
        for char in text:
            browser_handler.type_character(char)
            
            # Random delay between 0.02 and 0.1 seconds for human-like typing
            typing_delay = random.uniform(0.02, 0.1)
            time.sleep(typing_delay)
            
            # Occasionally add a slightly longer pause (simulating thinking)
            if random.random() < 0.05:  # 5% chance
                time.sleep(random.uniform(0.2, 0.5))
    
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
    
    def _checkpoint_path(self, session_id: str) -> str:
        """Get path to checkpoint file for a session."""
        return os.path.join(self.checkpoint_dir, f"{session_id}_checkpoint.json")
    
    def _checkpoint_exists(self, session_id: str) -> bool:
        """Check if a checkpoint exists for a session."""
        return os.path.exists(self._checkpoint_path(session_id))
    
    def _save_checkpoint(self, session_id: str):
        """Save checkpoint for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        checkpoint_data = session.to_dict()
        
        try:
            with open(self._checkpoint_path(session_id), 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            logging.debug(f"Saved checkpoint for session '{session_id}'")
        except Exception as e:
            logging.warning(f"Failed to save checkpoint for session '{session_id}': {e}")
    
    def _load_checkpoint(self, session_id: str):
        """Load checkpoint for a session."""
        checkpoint_path = self._checkpoint_path(session_id)
        if not os.path.exists(checkpoint_path):
            return
        
        session = self.sessions.get(session_id)
        if not session:
            return
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Restore session state
            session.current_prompt_index = checkpoint_data.get("current_prompt_index", 0)
            session.status = checkpoint_data.get("status", "pending")
            
            if session.status == "completed":
                # Already completed, skip
                logging.info(f"Session '{session_id}' was already completed according to checkpoint")
            elif session.current_prompt_index > 0:
                # Resume from checkpoint
                logging.info(f"Resuming session '{session_id}' from prompt {session.current_prompt_index + 1}")
            
            logging.debug(f"Loaded checkpoint for session '{session_id}'")
            
        except Exception as e:
            logging.warning(f"Failed to load checkpoint for session '{session_id}': {e}")
    
    def _delete_checkpoint(self, session_id: str):
        """Delete checkpoint for a session."""
        checkpoint_path = self._checkpoint_path(session_id)
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                logging.debug(f"Deleted checkpoint for session '{session_id}'")
            except Exception as e:
                logging.warning(f"Failed to delete checkpoint for session '{session_id}': {e}")
    
    def generate_session_report(self, session_id: str) -> Dict[str, Any]:
        """Generate a detailed report for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": f"Session '{session_id}' not found"}
        
        return session.to_dict()