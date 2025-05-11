# src/utils/session_tracker.py
import os
import json
import logging
from datetime import datetime

class SessionTracker:
    """Tracks which sessions have been completed and when."""
    
    def __init__(self, tracker_file="config/session_status.json"):
        """
        Initialize the session tracker.
        
        Args:
            tracker_file: Path to the JSON file tracking session status
        """
        self.tracker_file = tracker_file
        self.session_status = {}
        self._load_status()
    
    def _load_status(self):
        """Load the session status from file."""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r') as f:
                    self.session_status = json.load(f)
                logging.debug(f"Loaded session status from {self.tracker_file}")
            except Exception as e:
                logging.error(f"Error loading session status: {e}")
                self.session_status = {}
        else:
            logging.info(f"No existing session status file found at {self.tracker_file}")
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.tracker_file), exist_ok=True)
            self.session_status = {}
    
    def _save_status(self):
        """Save the session status to file."""
        try:
            with open(self.tracker_file, 'w') as f:
                json.dump(self.session_status, f, indent=2)
            logging.debug(f"Saved session status to {self.tracker_file}")
            return True
        except Exception as e:
            logging.error(f"Error saving session status: {e}")
            return False
    
    def is_completed(self, session_id):
        """
        Check if a session has been completed.
        
        Args:
            session_id: The session identifier
            
        Returns:
            bool: True if the session is marked as completed
        """
        return session_id in self.session_status and self.session_status[session_id].get('completed', False)
    
    def mark_completed(self, session_id, success=True, notes=None):
        """
        Mark a session as completed.
        
        Args:
            session_id: The session identifier
            success: Whether the session completed successfully
            notes: Optional notes about the session execution
            
        Returns:
            bool: True if status was successfully saved
        """
        if session_id not in self.session_status:
            self.session_status[session_id] = {}
        
        self.session_status[session_id].update({
            'completed': True,
            'success': success,
            'completion_time': datetime.now().isoformat(),
            'notes': notes or ""
        })
        
        return self._save_status()
    
    def reset_session(self, session_id):
        """
        Reset a session's completion status.
        
        Args:
            session_id: The session identifier
            
        Returns:
            bool: True if status was successfully saved
        """
        if session_id in self.session_status:
            # Archive the previous run info if it exists
            if 'history' not in self.session_status[session_id]:
                self.session_status[session_id]['history'] = []
                
            # Add current status to history before resetting
            if self.session_status[session_id].get('completed'):
                history_entry = {
                    'completed': self.session_status[session_id].get('completed', False),
                    'success': self.session_status[session_id].get('success', False),
                    'completion_time': self.session_status[session_id].get('completion_time', ''),
                    'notes': self.session_status[session_id].get('notes', '')
                }
                self.session_status[session_id]['history'].append(history_entry)
            
            # Reset current status
            self.session_status[session_id]['completed'] = False
            self.session_status[session_id]['success'] = False
            if 'completion_time' in self.session_status[session_id]:
                del self.session_status[session_id]['completion_time']
            self.session_status[session_id]['notes'] = "Reset on " + datetime.now().isoformat()
            
            return self._save_status()
        return False
    
    def reset_all_sessions(self):
        """
        Reset completion status for all sessions.
        
        Returns:
            bool: True if status was successfully saved
        """
        for session_id in list(self.session_status.keys()):
            self.reset_session(session_id)
        return True
    
    def get_session_status(self, session_id=None):
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
    
    def get_completed_sessions(self):
        """
        Get a list of all completed sessions.
        
        Returns:
            list: Session IDs of completed sessions
        """
        return [session_id for session_id in self.session_status
                if self.session_status[session_id].get('completed', False)]
    
    def get_pending_sessions(self):
        """
        Get a list of all sessions that are not completed.
        
        Returns:
            list: Session IDs of pending sessions
        """
        return [session_id for session_id in self.session_status
                if not self.session_status[session_id].get('completed', False)]