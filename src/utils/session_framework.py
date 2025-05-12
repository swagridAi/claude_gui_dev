#!/usr/bin/env python3
"""
Unified Session Management Framework

Consolidates session tracking, state management, and persistence patterns
from across the codebase into a single, reusable component.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum, auto
from pathlib import Path
import copy
from dataclasses import dataclass, asdict, field


# Unified session states
class SessionState(Enum):
    """Standardized session states across the application."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    INTERRUPTED = auto()


class PromptState(Enum):
    """States for individual prompt execution."""
    PENDING = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class SessionMetrics:
    """Performance metrics for session execution."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    prompts_attempted: int = 0
    prompts_completed: int = 0
    prompts_failed: int = 0
    retry_count: int = 0
    
    def calculate_duration(self):
        """Calculate duration if both timestamps exist."""
        if self.start_time and self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    def success_rate(self) -> float:
        """Calculate prompt success rate."""
        if self.prompts_attempted == 0:
            return 0.0
        return self.prompts_completed / self.prompts_attempted


@dataclass
class SessionRecord:
    """Unified session tracking record."""
    id: str
    name: str = ""
    state: SessionState = SessionState.PENDING
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    completion_time: Optional[str] = None
    success: bool = False
    notes: str = ""
    prompts: List[str] = field(default_factory=list)
    current_prompt_index: int = 0
    error_messages: List[str] = field(default_factory=list)
    
    def mark_started(self):
        """Mark session as started."""
        self.state = SessionState.RUNNING
        self.metrics.start_time = datetime.now()
    
    def mark_completed(self, success: bool = True, notes: str = ""):
        """Mark session as completed."""
        self.state = SessionState.COMPLETED if success else SessionState.FAILED
        self.success = success
        self.completion_time = datetime.now().isoformat()
        self.metrics.end_time = datetime.now()
        self.metrics.calculate_duration()
        if notes:
            self.notes = notes
    
    def add_error(self, error: str):
        """Add error message to session record."""
        self.error_messages.append(f"{datetime.now().isoformat()}: {error}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to strings
        if self.metrics.start_time:
            data['metrics']['start_time'] = self.metrics.start_time.isoformat()
        if self.metrics.end_time:
            data['metrics']['end_time'] = self.metrics.end_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionRecord':
        """Create instance from dictionary."""
        # Convert string timestamps back to datetime
        if 'metrics' in data:
            metrics_data = data['metrics']
            if metrics_data.get('start_time'):
                metrics_data['start_time'] = datetime.fromisoformat(metrics_data['start_time'])
            if metrics_data.get('end_time'):
                metrics_data['end_time'] = datetime.fromisoformat(metrics_data['end_time'])
            data['metrics'] = SessionMetrics(**metrics_data)
        
        if 'state' in data and isinstance(data['state'], str):
            data['state'] = SessionState[data['state']]
        
        return cls(**data)


class SessionPersistence:
    """Handles session data persistence with atomic operations."""
    
    def __init__(self, storage_path: str):
        """Initialize with storage path."""
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_valid_json()
    
    def _ensure_valid_json(self):
        """Ensure storage file is valid JSON."""
        if not self.storage_path.exists():
            self._save_data({})
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data from storage with error handling."""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"Error loading session data: {e}, starting fresh")
            return {}
    
    def _save_data(self, data: Dict[str, Any]) -> bool:
        """Save data to storage with atomic write."""
        temp_path = self.storage_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.storage_path)
            return True
        except Exception as e:
            logging.error(f"Error saving session data: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    def save_session(self, session: SessionRecord) -> bool:
        """Save single session record."""
        data = self._load_data()
        data[session.id] = session.to_dict()
        return self._save_data(data)
    
    def load_session(self, session_id: str) -> Optional[SessionRecord]:
        """Load single session record."""
        data = self._load_data()
        session_data = data.get(session_id)
        if session_data:
            return SessionRecord.from_dict(session_data)
        return None
    
    def get_all_sessions(self) -> Dict[str, SessionRecord]:
        """Get all session records."""
        data = self._load_data()
        return {sid: SessionRecord.from_dict(sdata) for sid, sdata in data.items()}
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session record."""
        data = self._load_data()
        if session_id in data:
            del data[session_id]
            return self._save_data(data)
        return True


class SessionTracker:
    """Unified session tracking and management."""
    
    def __init__(self, storage_path: str = "config/session_status.json", 
                 checkpoint_dir: str = "logs/checkpoints"):
        """Initialize session tracker."""
        self.persistence = SessionPersistence(storage_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session(self, session_id: str, name: str = "", 
                      prompts: Optional[List[str]] = None) -> SessionRecord:
        """Create new session record."""
        session = SessionRecord(
            id=session_id,
            name=name or session_id,
            prompts=prompts or []
        )
        self.persistence.save_session(session)
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Get session record by ID."""
        return self.persistence.load_session(session_id)
    
    def update_session(self, session: SessionRecord) -> bool:
        """Update existing session record."""
        return self.persistence.save_session(session)
    
    def is_completed(self, session_id: str) -> bool:
        """Check if session is completed."""
        session = self.get_session(session_id)
        return session is not None and session.state == SessionState.COMPLETED
    
    def mark_completed(self, session_id: str, success: bool = True, 
                      notes: Optional[str] = None) -> bool:
        """Mark session as completed."""
        session = self.get_session(session_id)
        if not session:
            # Create minimal session record if doesn't exist
            session = self.create_session(session_id)
        
        session.mark_completed(success, notes or "")
        return self.update_session(session)
    
    def get_completed_sessions(self) -> List[str]:
        """Get list of completed session IDs."""
        return [
            sid for sid, session in self.persistence.get_all_sessions().items()
            if session.state == SessionState.COMPLETED
        ]
    
    def get_pending_sessions(self) -> List[str]:
        """Get list of pending session IDs."""
        return [
            sid for sid, session in self.persistence.get_all_sessions().items()
            if session.state in [SessionState.PENDING, SessionState.RUNNING, SessionState.INTERRUPTED]
        ]
    
    def get_session_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status for specific session or all sessions."""
        if session_id:
            session = self.get_session(session_id)
            return session.to_dict() if session else {}
        else:
            return {sid: session.to_dict() 
                   for sid, session in self.persistence.get_all_sessions().items()}
    
    def reset_session(self, session_id: str) -> bool:
        """Reset session status for re-running."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Archive current state in notes
        if session.state != SessionState.PENDING:
            archive_note = f"Reset from {session.state.name} at {datetime.now().isoformat()}"
            session.notes = f"{session.notes}\n{archive_note}" if session.notes else archive_note
        
        # Reset status
        session.state = SessionState.PENDING
        session.success = False
        session.completion_time = None
        session.current_prompt_index = 0
        session.metrics = SessionMetrics()
        
        return self.update_session(session)
    
    def save_checkpoint(self, session: SessionRecord) -> bool:
        """Save session checkpoint for resumability."""
        checkpoint_path = self.checkpoint_dir / f"{session.id}_checkpoint.json"
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving checkpoint for {session.id}: {e}")
            return False
    
    def load_checkpoint(self, session_id: str) -> Optional[SessionRecord]:
        """Load session checkpoint if exists."""
        checkpoint_path = self.checkpoint_dir / f"{session_id}_checkpoint.json"
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            return SessionRecord.from_dict(data)
        except Exception as e:
            logging.error(f"Error loading checkpoint for {session_id}: {e}")
            return None
    
    def cleanup_checkpoints(self, max_age_days: int = 30):
        """Clean up old checkpoint files."""
        max_age_seconds = max_age_days * 24 * 60 * 60
        current_time = time.time()
        
        for checkpoint_file in self.checkpoint_dir.glob("*_checkpoint.json"):
            file_age = current_time - checkpoint_file.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    checkpoint_file.unlink()
                    logging.debug(f"Removed old checkpoint: {checkpoint_file}")
                except Exception as e:
                    logging.warning(f"Error removing checkpoint {checkpoint_file}: {e}")


class SessionExecutor:
    """Handles session execution with progress tracking."""
    
    def __init__(self, tracker: SessionTracker):
        """Initialize with session tracker."""
        self.tracker = tracker
        self.current_session: Optional[SessionRecord] = None
    
    def start_session(self, session_id: str) -> bool:
        """Start session execution."""
        session = self.tracker.get_session(session_id)
        if not session:
            logging.error(f"Session {session_id} not found")
            return False
        
        # Load from checkpoint if available
        checkpoint = self.tracker.load_checkpoint(session_id)
        if checkpoint:
            session = checkpoint
            logging.info(f"Resuming session {session_id} from checkpoint")
        
        session.mark_started()
        self.current_session = session
        self.tracker.save_checkpoint(session)
        return True
    
    def complete_prompt(self, success: bool = True, error: Optional[str] = None):
        """Mark current prompt as completed."""
        if not self.current_session:
            return
        
        if success:
            self.current_session.metrics.prompts_completed += 1
        else:
            self.current_session.metrics.prompts_failed += 1
            if error:
                self.current_session.add_error(error)
        
        self.current_session.current_prompt_index += 1
        self.tracker.save_checkpoint(self.current_session)
    
    def finish_session(self, success: bool = True, notes: str = ""):
        """Complete session execution."""
        if not self.current_session:
            return
        
        # Calculate final metrics
        total_prompts = len(self.current_session.prompts)
        self.current_session.metrics.prompts_attempted = min(
            self.current_session.current_prompt_index, total_prompts
        )
        
        self.current_session.mark_completed(success, notes)
        self.tracker.update_session(self.current_session)
        
        # Clean up checkpoint since session is complete
        checkpoint_path = self.tracker.checkpoint_dir / f"{self.current_session.id}_checkpoint.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        self.current_session = None
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current session progress."""
        if not self.current_session:
            return {}
        
        total_prompts = len(self.current_session.prompts)
        return {
            "session_id": self.current_session.id,
            "current_prompt": self.current_session.current_prompt_index,
            "total_prompts": total_prompts,
            "progress_percentage": (self.current_session.current_prompt_index / total_prompts * 100) if total_prompts > 0 else 0,
            "prompts_completed": self.current_session.metrics.prompts_completed,
            "prompts_failed": self.current_session.metrics.prompts_failed,
            "success_rate": self.current_session.metrics.success_rate()
        }


# Factory function for simplified usage
def create_session_framework(storage_path: str = "config/session_status.json",
                           checkpoint_dir: str = "logs/checkpoints") -> Tuple[SessionTracker, SessionExecutor]:
    """Create pre-configured session framework components."""
    tracker = SessionTracker(storage_path, checkpoint_dir)
    executor = SessionExecutor(tracker)
    return tracker, executor