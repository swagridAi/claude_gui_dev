#!/usr/bin/env python3
# src/utils/session_tracker.py
import os
import json
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, Any, Optional

# Module-level constants
DEFAULT_TRACKER_FILE = "config/session_status.json"
JSON_INDENT = 2
ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"

class SessionTracker:
    """Tracks which sessions have been completed and when."""
    
    def __init__(self, tracker_file: str = DEFAULT_TRACKER_FILE):
        """
        Initialize the session tracker.
        
        Args:
            tracker_file: Path to the JSON file tracking session status
        """
        self.tracker_file = tracker_file
        self.session_status = {}
        self._load_status()
    
    @contextmanager
    def _safe_operation(self, operation_name: str, fallback_value=False):
        """Context manager for safe operations with consistent error handling."""
        try:
            yield
        except Exception as e:
            logging.error(f"Error in {operation_name}: {e}")
            return fallback_value
    
    def _load_json(self) -> Dict[str, Any]:
        """Load JSON data from tracker file."""
        if not os.path.exists(self.tracker_file):
            logging.info(f"No existing session status file found at {self.tracker_file}")
            os.makedirs(os.path.dirname(self.tracker_file), exist_ok=True)
            return {}
        
        with open(self.tracker_file, 'r') as f:
            data = json.load(f)
        logging.debug(f"Loaded session status from {self.tracker_file}")
        return data
    
    def _save_json(self, data: Dict[str, Any]) -> bool:
        """Save JSON data to tracker file."""
        with open(self.tracker_file, 'w') as f:
            json.dump(data, f, indent=JSON_INDENT)
        logging.debug(f"Saved session status to {self.tracker_file}")
        return True
    
    def _ensure_session_exists(self, session_id: str) -> None:
        """Ensure session entry exists in status dictionary."""
        if session_id not in self.session_status:
            self.session_status[session_id] = {}
    
    def _add_to_history(self, session_id: str) -> None:
        """Add current session state to history before modification."""
        session = self.session_status[session_id]
        
        if 'history' not in session:
            session['history'] = []
        
        if session.get('completed'):
            history_entry = {
                'completed': session.get('completed', False),
                'success': session.get('success', False),
                'completion_time': session.get('completion_time', ''),
                'notes': session.get('notes', '')
            }
            session['history'].append(history_entry)
    
    def _load_status(self) -> None:
        """Load the session status from file."""
        with self._safe_operation('loading session status'):
            self.session_status = self._load_json()
    
    def _save_status(self) -> bool:
        """Save the session status to file."""
        with self._safe_operation('saving session status') as context:
            result = self._save_json(self.session_status)
            return result
        return False
    
    def is_completed(self, session_id: str) -> bool:
        """
        Check if a session has been completed.
        
        Args:
            session_id: The session identifier
            
        Returns:
            bool: True if the session is marked as completed
        """
        return session_id in self.session_status and self.session_status[session_id].get('completed', False)
    
    def mark_completed(self, session_id: str, success: bool = True, notes: Optional[str] = None) -> bool:
        """
        Mark a session as completed.
        
        Args:
            session_id: The session identifier
            success: Whether the session completed successfully
            notes: Optional notes about the session execution
            
        Returns:
            bool: True if status was successfully saved
        """
        with self._safe_operation('marking session completed') as context:
            self._ensure_session_exists(session_id)
            
            self.session_status[session_id].update({
                'completed': True,
                'success': success,
                'completion_time': datetime.now().strftime(ISO_FORMAT),
                'notes': notes or ""
            })
            
            return self._save_status()
        return False
    
    def reset_session(self, session_id: str) -> bool:
        """
        Reset a session's completion status.
        
        Args:
            session_id: The session identifier
            
        Returns:
            bool: True if status was successfully saved
        """
        with self._safe_operation('resetting session') as context:
            if session_id not in self.session_status:
                return False
            
            self._add_to_history(session_id)
            
            # Reset current status
            session = self.session_status[session_id]
            session['completed'] = False
            session['success'] = False
            session['notes'] = f"Reset on {datetime.now().strftime(ISO_FORMAT)}"
            if 'completion_time' in session:
                del session['completion_time']
            
            return self._save_status()
        return False
    
    def reset_all_sessions(self) -> bool:
        """
        Reset completion status for all sessions.
        
        Returns:
            bool: True if status was successfully saved
        """
        for session_id in list(self.session_status.keys()):
            self.reset_session(session_id)
        return True
    
    def get_session_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the status of a specific session or all sessions.
        
        Args:
            session_id: The session identifier or None for all sessions
            
        Returns:
            dict: Session status information
        """
        if session_id:
            return self.session_status.get(session_id, {})
        return self.session_status
    
    def get_completed_sessions(self) -> list:
        """
        Get a list of all completed sessions.
        
        Returns:
            list: Session IDs of completed sessions
        """
        return [session_id for session_id in self.session_status
                if self.session_status[session_id].get('completed', False)]
    
    def get_pending_sessions(self) -> list:
        """
        Get a list of all sessions that are not completed.
        
        Returns:
            list: Session IDs of pending sessions
        """
        return [session_id for session_id in self.session_status
                if not self.session_status[session_id].get('completed', False)]