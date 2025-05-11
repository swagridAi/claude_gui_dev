#!/usr/bin/env python3
"""
Simple Claude Prompt Sender - Main Entry Point

A simplified interface for sending prompts to Claude without relying 
on complex image recognition. Provides a balance between simplicity
and reliability with configurable options.

Supports hybrid waiting system that combines template matching with
fallback timeouts for optimal efficiency when waiting for responses.
"""

import argparse
import logging
import sys
import time
import functools
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from datetime import datetime


# === Custom Exceptions ===
class SimpleSenderError(Exception):
    """Base exception for the simple sender module."""
    pass


class ConfigurationError(SimpleSenderError):
    """Exception for configuration-related errors."""
    pass


class BrowserError(SimpleSenderError):
    """Exception for browser-related errors."""
    pass


class SessionError(SimpleSenderError):
    """Exception for session-related errors."""
    pass


class TemplateError(SimpleSenderError):
    """Exception for template-related errors."""
    pass


class VisualVerificationError(SimpleSenderError):
    """Exception for visual verification errors."""
    pass


# === Dependency Interfaces ===
class ConfigProvider:
    """Interface for configuration access."""
    
    def __init__(self, config_path: str = "config/user_config.yaml"):
        # Import here to avoid circular imports
        from src.simple_sender.config import SimpleConfig
        self.config = SimpleConfig(config_path)
        self._cache = {}
    
    def get_session_ids(self) -> List[str]:
        """Get all available session IDs."""
        return self.config.get_session_ids()
    
    def get_sessions(self) -> Dict[str, Any]:
        """Get all sessions with their configuration."""
        if "sessions" not in self._cache:
            self._cache["sessions"] = self.config.get_sessions()
        return self._cache["sessions"]
    
    def get_global_prompts(self) -> List[str]:
        """Get global prompts defined in configuration."""
        if "global_prompts" not in self._cache:
            self._cache["global_prompts"] = self.config.get_global_prompts()
        return self._cache["global_prompts"]
    
    def is_valid_session(self, session_id: str) -> bool:
        """Check if a session is valid."""
        return self.config.is_valid_session(session_id)
    
    def get_session_name(self, session_id: str) -> str:
        """Get the name of a session."""
        sessions = self.get_sessions()
        if session_id in sessions:
            return sessions[session_id].get("name", session_id)
        return session_id
    
    def get_template_settings(self) -> Dict[str, Any]:
        """Get template matching and visual verification settings."""
        if "template_settings" not in self._cache:
            self._cache["template_settings"] = {
                "enabled": self.config.get("template_matching.enabled", False),
                "template_dir": self.config.get("template_matching.template_dir", "assets/templates"),
                "threshold": self.config.get("template_matching.threshold", 0.8),
                "scale_factor": self.config.get("template_matching.scale_factor", 1.0),
                "use_edge_detection": self.config.get("template_matching.use_edge_detection", False),
                "match_method": self.config.get("template_matching.match_method", "TM_CCOEFF_NORMED")
            }
        return self._cache["template_settings"]
    
    def get_waiting_strategy(self) -> Dict[str, Any]:
        """Get waiting strategy configuration."""
        if "waiting_strategy" not in self._cache:
            self._cache["waiting_strategy"] = {
                "use_hybrid": self.config.get("waiting.use_hybrid", False),
                "max_wait": self.config.get("waiting.max_wait", 300),
                "check_interval": self.config.get("waiting.check_interval", 5),
                "stability_duration": self.config.get("waiting.stability_duration", 10),
                "stability_threshold": self.config.get("waiting.stability_threshold", 0.02)
            }
        return self._cache["waiting_strategy"]
    
    def update_template_settings(self, settings: Dict[str, Any]) -> bool:
        """Update template settings in configuration."""
        try:
            for key, value in settings.items():
                self.config.set(f"template_matching.{key}", value)
            
            # Invalidate cache
            if "template_settings" in self._cache:
                del self._cache["template_settings"]
                
            # Save changes
            return self.config.save()
        except Exception as e:
            logging.error(f"Failed to update template settings: {e}")
            return False
    
    def get_metrics_settings(self) -> Dict[str, Any]:
        """Get metrics collection settings."""
        if "metrics_settings" not in self._cache:
            self._cache["metrics_settings"] = {
                "enabled": self.config.get("metrics.enabled", True),
                "store_dir": self.config.get("metrics.store_dir", "logs/metrics"),
                "retention_days": self.config.get("metrics.retention_days", 30)
            }
        return self._cache["metrics_settings"]


class BrowserFactory:
    """Factory for creating browser sessions."""
    
    def create_browser(self, reuse: bool = False) -> Any:
        """Create a new browser session."""
        # Import here to avoid circular imports
        from src.simple_sender.browser import BrowserSession
        return BrowserSession(reuse_browser=reuse)


class SessionManager:
    """Interface for session management."""
    
    def __init__(self, config_provider: ConfigProvider):
        self.config = config_provider
        # Import here to avoid circular imports
        from src.simple_sender.session import SessionManager as RealSessionManager
        self.session_manager = RealSessionManager(self.config.config)
    
    def run_session(self, session_id: str, browser_session: Any, 
                   prompt_delay: int = 300, resume: bool = False, 
                   take_screenshots: bool = True,
                   use_hybrid_waiting: bool = False,
                   metrics_collector: Optional[Any] = None) -> bool:
        """Run a single session."""
        try:
            # Get waiting strategy configuration
            waiting_strategy = self.config.get_waiting_strategy()
            hybrid_enabled = use_hybrid_waiting or waiting_strategy.get("use_hybrid", False)
            
            # Log waiting strategy
            if hybrid_enabled:
                logging.info("Using hybrid waiting strategy (template detection with fallback timeout)")
            else:
                logging.info(f"Using fixed timeout waiting strategy ({prompt_delay}s)")
            
            # Forward to session manager with appropriate parameters
            return self.session_manager.run_session(
                session_id, 
                browser_session, 
                prompt_delay, 
                resume, 
                take_screenshots,
                use_hybrid_waiting=hybrid_enabled,
                metrics_collector=metrics_collector
            )
        except Exception as e:
            logging.error(f"Error running session: {e}")
            return False
    
    def run_default_session(self, prompts: List[str], browser_session: Any,
                           prompt_delay: int = 300, resume: bool = False,
                           take_screenshots: bool = True,
                           use_hybrid_waiting: bool = False,
                           metrics_collector: Optional[Any] = None) -> bool:
        """Run the default session with global prompts."""
        try:
            # Get waiting strategy configuration
            waiting_strategy = self.config.get_waiting_strategy()
            hybrid_enabled = use_hybrid_waiting or waiting_strategy.get("use_hybrid", False)
            
            # Log waiting strategy
            if hybrid_enabled:
                logging.info("Using hybrid waiting strategy (template detection with fallback timeout)")
            else:
                logging.info(f"Using fixed timeout waiting strategy ({prompt_delay}s)")
            
            # Forward to session manager with appropriate parameters
            return self.session_manager.run_default_session(
                prompts, 
                browser_session, 
                prompt_delay, 
                resume, 
                take_screenshots,
                use_hybrid_waiting=hybrid_enabled,
                metrics_collector=metrics_collector
            )
        except Exception as e:
            logging.error(f"Error running default session: {e}")
            return False
    
    def get_session_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the status of a session."""
        return self.session_manager.get_session_status(session_id)
    
    def get_all_session_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of all sessions."""
        return self.session_manager.get_all_session_status()


class TemplateManager:
    """Interface for template management."""
    
    def __init__(self, config_provider: ConfigProvider):
        self.config = config_provider
        # Import template module only if templates are enabled
        self.template_system = None
        self._initialize()
    
    def _initialize(self) -> bool:
        """Initialize the template system if enabled."""
        template_settings = self.config.get_template_settings()
        if not template_settings.get("enabled", False):
            return False
        
        try:
            # Import template module only when needed
            try:
                from src.simple_sender.templates import TemplateSystem
                self.template_system = TemplateSystem(template_settings)
                return True
            except ImportError as e:
                logging.warning(f"Template system unavailable: {e}")
                return False
        except Exception as e:
            logging.error(f"Error initializing template system: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if template system is available."""
        return self.template_system is not None
    
    def capture_template(self, template_name: str, region: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """Capture a template for UI element."""
        if not self.is_available():
            raise TemplateError("Template system is not available")
        
        try:
            return self.template_system.capture_template(template_name, region)
        except Exception as e:
            raise TemplateError(f"Error capturing template: {e}")
    
    def list_templates(self) -> Dict[str, Any]:
        """List all available templates with metadata."""
        if not self.is_available():
            raise TemplateError("Template system is not available")
        
        try:
            return self.template_system.list_templates()
        except Exception as e:
            raise TemplateError(f"Error listing templates: {e}")
    
    def test_template(self, template_name: str) -> Dict[str, Any]:
        """Test a template against current screen."""
        if not self.is_available():
            raise TemplateError("Template system is not available")
        
        try:
            return self.template_system.test_template(template_name)
        except Exception as e:
            raise TemplateError(f"Error testing template: {e}")


class MetricsCollector:
    """Collects and saves performance metrics."""
    
    def __init__(self, config_provider: ConfigProvider):
        self.config = config_provider
        self.metrics = {
            "sessions": {},
            "started_at": datetime.now().isoformat(),
            "settings": {}
        }
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize metrics collection."""
        settings = self.config.get_metrics_settings()
        if settings.get("enabled", True):
            # Create storage directory if needed
            store_dir = settings.get("store_dir", "logs/metrics")
            os.makedirs(store_dir, exist_ok=True)
            
            # Store settings in metrics
            self.metrics["settings"] = {
                "hybrid_waiting": self.config.get_waiting_strategy().get("use_hybrid", False),
                "template_enabled": self.config.get_template_settings().get("enabled", False),
                "waiting_timeout": self.config.get_waiting_strategy().get("max_wait", 300)
            }
    
    def record_session_start(self, session_id: str) -> None:
        """Record session start time."""
        self.metrics["sessions"][session_id] = {
            "start_time": datetime.now().isoformat(),
            "prompts": [],
            "status": "running"
        }
    
    def record_session_end(self, session_id: str, success: bool) -> None:
        """Record session end time and success status."""
        if session_id in self.metrics["sessions"]:
            self.metrics["sessions"][session_id]["end_time"] = datetime.now().isoformat()
            self.metrics["sessions"][session_id]["status"] = "completed" if success else "failed"
            
            # Calculate total duration and average prompt time
            prompts = self.metrics["sessions"][session_id]["prompts"]
            if prompts:
                total_prompt_time = sum(p.get("duration", 0) for p in prompts)
                average_prompt_time = total_prompt_time / len(prompts)
                self.metrics["sessions"][session_id]["avg_prompt_time"] = average_prompt_time
                
                # Calculate time saved if using hybrid waiting
                if self.metrics["settings"].get("hybrid_waiting", False):
                    max_wait = self.metrics["settings"].get("waiting_timeout", 300)
                    saved_time = (max_wait * len(prompts)) - total_prompt_time
                    if saved_time > 0:
                        self.metrics["sessions"][session_id]["estimated_time_saved"] = saved_time
    
    def record_prompt(self, session_id: str, prompt_idx: int, prompt_text: str) -> None:
        """Record prompt execution start."""
        if session_id in self.metrics["sessions"]:
            self.metrics["sessions"][session_id]["prompts"].append({
                "index": prompt_idx,
                "text": prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text,
                "start_time": datetime.now().isoformat(),
                "status": "running"
            })
    
    def record_prompt_completion(self, session_id: str, prompt_idx: int, 
                               success: bool, detection_method: str, 
                               duration: float) -> None:
        """Record prompt completion with metrics."""
        if session_id in self.metrics["sessions"]:
            prompts = self.metrics["sessions"][session_id]["prompts"]
            for prompt in prompts:
                if prompt["index"] == prompt_idx:
                    prompt["end_time"] = datetime.now().isoformat()
                    prompt["status"] = "completed" if success else "failed"
                    prompt["duration"] = duration
                    prompt["detection_method"] = detection_method
                    break
    
    def save_metrics(self) -> Optional[str]:
        """Save metrics to a file and return filename."""
        settings = self.config.get_metrics_settings()
        if not settings.get("enabled", True):
            return None
            
        try:
            import json
            
            # Add completion timestamp
            self.metrics["completed_at"] = datetime.now().isoformat()
            
            # Calculate overall metrics
            total_sessions = len(self.metrics["sessions"])
            successful_sessions = sum(1 for s in self.metrics["sessions"].values() 
                                    if s.get("status") == "completed")
            total_prompts = sum(len(s.get("prompts", [])) for s in self.metrics["sessions"].values())
            
            self.metrics["summary"] = {
                "total_sessions": total_sessions,
                "successful_sessions": successful_sessions,
                "total_prompts": total_prompts,
                "success_rate": successful_sessions / total_sessions if total_sessions > 0 else 0
            }
            
            # Calculate time saved if using hybrid waiting
            if self.metrics["settings"].get("hybrid_waiting", False):
                total_saved = sum(s.get("estimated_time_saved", 0) for s in self.metrics["sessions"].values())
                if total_saved > 0:
                    self.metrics["summary"]["total_time_saved"] = total_saved
                    minutes, seconds = divmod(int(total_saved), 60)
                    hours, minutes = divmod(minutes, 60)
                    self.metrics["summary"]["readable_time_saved"] = f"{hours}h {minutes}m {seconds}s"
            
            # Generate filename with timestamp
            store_dir = settings.get("store_dir", "logs/metrics")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(store_dir, f"metrics_{timestamp}.json")
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(self.metrics, f, indent=2)
                
            return filename
            
        except Exception as e:
            logging.error(f"Error saving metrics: {e}")
            return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of collected metrics."""
        total_sessions = len(self.metrics["sessions"])
        successful_sessions = sum(1 for s in self.metrics["sessions"].values() 
                                if s.get("status") == "completed")
        total_prompts = sum(len(s.get("prompts", [])) for s in self.metrics["sessions"].values())
        
        summary = {
            "total_sessions": total_sessions,
            "successful_sessions": successful_sessions,
            "total_prompts": total_prompts,
            "success_rate": successful_sessions / total_sessions if total_sessions > 0 else 0,
            "hybrid_waiting_enabled": self.metrics["settings"].get("hybrid_waiting", False)
        }
        
        # Calculate time saved if using hybrid waiting
        if self.metrics["settings"].get("hybrid_waiting", False):
            total_saved = sum(s.get("estimated_time_saved", 0) for s in self.metrics["sessions"].values())
            if total_saved > 0:
                summary["total_time_saved"] = total_saved
                minutes, seconds = divmod(int(total_saved), 60)
                hours, minutes = divmod(minutes, 60)
                summary["readable_time_saved"] = f"{hours}h {minutes}m {seconds}s"
        
        return summary


# === Error Handling Decorator ===
def with_error_handling(func: Callable) -> Callable:
    """Decorator for standardized error handling."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BrowserError as e:
            logging.error(f"Browser error: {e}")
            return False
        except SessionError as e:
            logging.error(f"Session error: {e}")
            return False
        except ConfigurationError as e:
            logging.error(f"Configuration error: {e}")
            return False
        except TemplateError as e:
            logging.error(f"Template error: {e}")
            return False
        except VisualVerificationError as e:
            logging.error(f"Visual verification error: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            return False
    return wrapper


# === Progress Tracking ===
class ProgressTracker:
    """Track and display progress for long-running operations."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.detection_methods = {}  # Count detection methods used
    
    def update(self, increment: int = 1, detection_method: Optional[str] = None):
        """Update progress and display."""
        self.current += increment
        
        # Track detection method if provided
        if detection_method:
            self.detection_methods[detection_method] = self.detection_methods.get(detection_method, 0) + 1
            
        self._display_progress()
    
    def _display_progress(self):
        """Display progress to console."""
        if self.total == 0:
            return
        
        percent = min(100, int((self.current / self.total) * 100))
        elapsed = time.time() - self.start_time
        
        if self.current > 0 and elapsed > 0:
            items_per_sec = self.current / elapsed
            remaining_items = self.total - self.current
            eta_seconds = remaining_items / items_per_sec if items_per_sec > 0 else 0
            
            minutes, seconds = divmod(int(eta_seconds), 60)
            hours, minutes = divmod(minutes, 60)
            eta = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            eta = "Unknown"
        
        # Add detection method info if available
        method_info = ""
        if self.detection_methods:
            method_counts = [f"{method}: {count}" for method, count in self.detection_methods.items()]
            method_info = f" (Detection methods: {', '.join(method_counts)})"
        
        logging.info(f"{self.description}: {percent}% complete - ETA: {eta}{method_info}")
    
    def get_detection_stats(self) -> Dict[str, int]:
        """Get statistics on detection methods used."""
        return self.detection_methods


# === Command Pattern Implementation ===
class Command:
    """Base class for command pattern."""
    
    def __init__(self, config_provider: ConfigProvider, 
                session_manager: SessionManager,
                browser_factory: BrowserFactory,
                template_manager: Optional[TemplateManager] = None,
                metrics_collector: Optional[MetricsCollector] = None):
        self.config = config_provider
        self.session_manager = session_manager
        self.browser_factory = browser_factory
        self.template_manager = template_manager
        self.metrics_collector = metrics_collector
    
    def execute(self, args: argparse.Namespace) -> bool:
        """Execute the command with given arguments."""
        raise NotImplementedError("Subclasses must implement execute()")


class ListSessionsCommand(Command):
    """Command to list available sessions."""
    
    @with_error_handling
    def execute(self, args: argparse.Namespace) -> bool:
        """List all available sessions with optional details."""
        sessions = self.config.get_sessions()
        
        if not sessions:
            print("No sessions defined in configuration file.")
            return False
        
        print(f"\nFound {len(sessions)} sessions:")
        for session_id, session_info in sessions.items():
            session_name = session_info.get("name", session_id)
            prompt_count = len(session_info.get("prompts", []))
            
            # Basic session info
            print(f"  - {session_id}: {session_name} ({prompt_count} prompts)")
            
            # Detailed session info if verbose
            if args.verbose and prompt_count > 0:
                print(f"    URL: {session_info.get('claude_url', 'default')}")
                for i, prompt in enumerate(session_info.get("prompts", []), 1):
                    prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
                    print(f"    {i}. {prompt_preview}")
                print()
        
        return True


class ShowStatusCommand(Command):
    """Command to show session status."""
    
    @with_error_handling
    def execute(self, args: argparse.Namespace) -> bool:
        """Show completion status for sessions."""
        if args.session:
            # Show status for specific session
            status = self.session_manager.get_session_status(args.session)
            if not status:
                print(f"Session '{args.session}' not found.")
                return False
            
            print(f"Status for session '{args.session}':")
            print(f"  Completed: {status.get('completed', False)}")
            print(f"  Success: {status.get('success', False)}")
            if 'completion_time' in status:
                print(f"  Completed at: {status['completion_time']}")
            if 'prompts_completed' in status:
                print(f"  Prompts completed: {status['prompts_completed']}")
            
            return True
        else:
            # Show summary of all sessions
            all_status = self.session_manager.get_all_session_status()
            completed = [sid for sid, status in all_status.items() 
                        if status.get('completed', False)]
            pending = [sid for sid, status in all_status.items() 
                      if not status.get('completed', False)]
            
            print(f"Total sessions: {len(all_status)}")
            print(f"Completed: {len(completed)}")
            print(f"Pending: {len(pending)}")
            
            if completed:
                print("\nCompleted sessions:")
                for sid in completed:
                    print(f"  - {sid}")
            
            if pending:
                print("\nPending sessions:")
                for sid in pending:
                    print(f"  - {sid}")
            
            return True


class RunSessionsCommand(Command):
    """Command to run sessions."""
    
    def _select_sessions_to_run(self, args: argparse.Namespace) -> List[str]:
        """Determine which sessions to run based on arguments."""
        all_sessions = self.config.get_session_ids()
        
        if args.session and args.run_one:
            # Run only the specified session
            if args.session in all_sessions:
                logging.info(f"Running only session: {args.session}")
                return [args.session]
            else:
                logging.warning(f"Session '{args.session}' not found.")
                return []
        elif args.session:
            # Use specified session as the first one, then run all others
            if args.session in all_sessions:
                # Move the specified session to the front
                remaining = [s for s in all_sessions if s != args.session]
                sessions_to_run = [args.session] + remaining
                logging.info(f"Running {args.session} first, then all other sessions")
                return sessions_to_run
            else:
                # If specified session doesn't exist, run all
                logging.warning(f"Session '{args.session}' not found. Running all sessions.")
                return all_sessions
        else:
            # Run all sessions by default
            logging.info("Running all sessions")
            return all_sessions
    
    def _filter_valid_sessions(self, sessions: List[str]) -> List[str]:
        """Filter out invalid sessions."""
        return [session_id for session_id in sessions 
                if self.config.is_valid_session(session_id)]
    
    def _execute_single_session(self, session_id: str, browser_session: Any, 
                               args: argparse.Namespace) -> bool:
        """Execute a single session with proper setup and teardown."""
        try:
            # Start metrics collection if available
            if self.metrics_collector:
                self.metrics_collector.record_session_start(session_id)
            
            # Run the session
            success = self.session_manager.run_session(
                session_id,
                browser_session,
                prompt_delay=args.delay,
                resume=args.resume,
                take_screenshots=not args.disable_screenshots,
                use_hybrid_waiting=args.use_hybrid_waiting,
                metrics_collector=self.metrics_collector
            )
            
            # Record completion in metrics
            if self.metrics_collector:
                self.metrics_collector.record_session_end(session_id, success)
                
            return success
        except KeyboardInterrupt:
            logging.warning("Execution interrupted by user.")
            # Record interruption in metrics
            if self.metrics_collector:
                self.metrics_collector.record_session_end(session_id, False)
            return False
        except Exception as e:
            logging.error(f"Error running session '{session_id}': {e}", exc_info=True)
            # Record failure in metrics
            if self.metrics_collector:
                self.metrics_collector.record_session_end(session_id, False)
            return False
    
    def _execute_default_session(self, browser_session: Any, 
                                args: argparse.Namespace) -> bool:
        """Execute default session using global prompts."""
        global_prompts = self.config.get_global_prompts()
        
        if not global_prompts:
            logging.error("No prompts found in configuration. Exiting.")
            return False
        
        try:
            # Start metrics collection if available
            if self.metrics_collector:
                self.metrics_collector.record_session_start("default")
            
            logging.info("No valid sessions found. Using global prompts.")
            success = self.session_manager.run_default_session(
                global_prompts,
                browser_session,
                prompt_delay=args.delay,
                resume=args.resume,
                take_screenshots=not args.disable_screenshots,
                use_hybrid_waiting=args.use_hybrid_waiting,
                metrics_collector=self.metrics_collector
            )
            
            # Record completion in metrics
            if self.metrics_collector:
                self.metrics_collector.record_session_end("default", success)
                
            return success
        except Exception as e:
            logging.error(f"Error running default session: {e}")
            # Record failure in metrics
            if self.metrics_collector:
                self.metrics_collector.record_session_end("default", False)
            return False
    
    def _report_results(self, results: Dict[str, bool]) -> None:
        """Report session execution results."""
        logging.info("=== SESSION RESULTS SUMMARY ===")
        successful = sum(1 for success in results.values() if success)
        logging.info(f"Total sessions: {len(results)}")
        logging.info(f"Successful: {successful}")
        logging.info(f"Failed: {len(results) - successful}")
        
        for session_id, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            logging.info(f"Session '{session_id}': {status}")
        
        # Report metrics if available
        if self.metrics_collector:
            metrics_summary = self.metrics_collector.get_summary()
            
            if metrics_summary.get("hybrid_waiting_enabled"):
                if "readable_time_saved" in metrics_summary:
                    logging.info(f"Hybrid waiting saved approximately {metrics_summary['readable_time_saved']}")
                
                # Save metrics to file
                metrics_file = self.metrics_collector.save_metrics()
                if metrics_file:
                    logging.info(f"Detailed metrics saved to: {metrics_file}")
    
    @with_error_handling
    def execute(self, args: argparse.Namespace) -> bool:
        """Run specified sessions based on command-line arguments."""
        # Select sessions to run
        sessions_to_run = self._select_sessions_to_run(args)
        valid_sessions = self._filter_valid_sessions(sessions_to_run)
        
        if not valid_sessions:
            # If no valid sessions, try using global prompts
            browser_session = self.browser_factory.create_browser(reuse_browser=False)
            try:
                return self._execute_default_session(browser_session, args)
            finally:
                browser_session.close()
        
        # Initialize browser session manager
        browser_session = self.browser_factory.create_browser(reuse_browser=args.reuse_browser)
        
        # Log session plan
        logging.info(f"Will run {len(valid_sessions)} sessions: {', '.join(valid_sessions)}")
        
        # Create progress tracker
        progress = ProgressTracker(len(valid_sessions), "Sessions")
        
        # Run each session
        results = {}
        try:
            for i, session_id in enumerate(valid_sessions):
                logging.info(f"Starting session {i+1}/{len(valid_sessions)}: {session_id}")
                
                # Run the session
                success = self._execute_single_session(session_id, browser_session, args)
                results[session_id] = success
                
                # Update progress
                progress.update()
                
                # Wait between sessions if there are more to process
                if i < len(valid_sessions) - 1:
                    delay = args.session_delay
                    logging.info(f"Waiting {delay} seconds before next session...")
                    time.sleep(delay)
        finally:
            # Always ensure browser is closed at the end
            browser_session.close()
        
        # Report results
        self._report_results(results)
        
        return all(results.values())


class TemplateCommand(Command):
    """Command for template management operations."""
    
    @with_error_handling
    def execute(self, args: argparse.Namespace) -> bool:
        """Execute template management operations."""
        # Check if template system is available
        if not self.template_manager or not self.template_manager.is_available():
            logging.error("Template system is not available. Make sure it's enabled in configuration.")
            print("Template system is not available. Enable it with --enable-templates")
            return False
        
        # Execute subcommand based on args
        if args.template_action == "capture":
            return self._capture_template(args)
        elif args.template_action == "list":
            return self._list_templates(args)
        elif args.template_action == "test":
            return self._test_template(args)
        elif args.template_action == "enable":
            return self._enable_templates(args)
        else:
            logging.error(f"Unknown template action: {args.template_action}")
            return False
    
    def _capture_template(self, args: argparse.Namespace) -> bool:
        """Capture a template."""
        if not args.template_name:
            logging.error("Template name is required")
            return False
        
        print(f"Capturing template '{args.template_name}'")
        print("Position your mouse over the target UI element and wait...")
        
        # Give user time to position mouse
        for i in range(5, 0, -1):
            print(f"Capturing in {i} seconds...")
            time.sleep(1)
            
        # Get current mouse position for region
        try:
            import pyautogui
            x, y = pyautogui.position()
            
            # Define region around mouse position
            region_size = args.region_size or 100
            region = (x - region_size//2, y - region_size//2, region_size, region_size)
            
            # Capture template
            success = self.template_manager.capture_template(args.template_name, region)
            
            if success:
                print(f"Successfully captured template '{args.template_name}'")
                print(f"Region: {region}")
                return True
            else:
                print(f"Failed to capture template '{args.template_name}'")
                return False
                
        except Exception as e:
            logging.error(f"Error capturing template: {e}")
            print(f"Error capturing template: {e}")
            return False
    
    def _list_templates(self, args: argparse.Namespace) -> bool:
        """List available templates."""
        try:
            templates = self.template_manager.list_templates()
            
            if not templates:
                print("No templates found.")
                return True
                
            print(f"Found {len(templates)} templates:")
            for name, info in templates.items():
                print(f"  - {name}")
                if args.verbose:
                    print(f"    Created: {info.get('created', 'Unknown')}")
                    print(f"    Region: {info.get('region', 'Unknown')}")
                    print(f"    Resolution: {info.get('resolution', 'Unknown')}")
                    print()
            
            return True
            
        except Exception as e:
            logging.error(f"Error listing templates: {e}")
            print(f"Error listing templates: {e}")
            return False
    
    def _test_template(self, args: argparse.Namespace) -> bool:
        """Test a template against current screen."""
        if not args.template_name:
            logging.error("Template name is required")
            return False
            
        try:
            print(f"Testing template '{args.template_name}'")
            result = self.template_manager.test_template(args.template_name)
            
            if result.get("found", False):
                print(f"Template '{args.template_name}' found!")
                print(f"Location: {result.get('location')}")
                print(f"Confidence: {result.get('confidence', 0):.2f}")
                return True
            else:
                print(f"Template '{args.template_name}' not found on screen")
                return False
                
        except Exception as e:
            logging.error(f"Error testing template: {e}")
            print(f"Error testing template: {e}")
            return False
    
    def _enable_templates(self, args: argparse.Namespace) -> bool:
        """Enable template system and configure settings."""
        try:
            # Update template settings
            template_settings = {
                "enabled": True,
                "template_dir": args.template_dir or "assets/templates"
            }
            
            if args.threshold is not None:
                template_settings["threshold"] = args.threshold
                
            success = self.config.update_template_settings(template_settings)
            
            if success:
                print("Template system has been enabled.")
                print(f"Template directory: {template_settings['template_dir']}")
                if args.threshold is not None:
                    print(f"Matching threshold: {args.threshold}")
                print("\nRestart the application for changes to take effect.")
                return True
            else:
                print("Failed to update template settings.")
                return False
                
        except Exception as e:
            logging.error(f"Error enabling templates: {e}")
            print(f"Error enabling templates: {e}")
            return False


class CommandRegistry:
    """Registry for command pattern implementation."""
    
    def __init__(self, config_provider: ConfigProvider, 
                session_manager: SessionManager,
                browser_factory: BrowserFactory,
                template_manager: Optional[TemplateManager] = None,
                metrics_collector: Optional[MetricsCollector] = None):
        self.commands = {}
        self.config = config_provider
        self.session_manager = session_manager
        self.browser_factory = browser_factory
        self.template_manager = template_manager
        self.metrics_collector = metrics_collector
        self._register_commands()
    
    def _register_commands(self) -> None:
        """Register all available commands."""
        self.register('list', ListSessionsCommand(
            self.config, self.session_manager, self.browser_factory, 
            self.template_manager, self.metrics_collector))
        self.register('status', ShowStatusCommand(
            self.config, self.session_manager, self.browser_factory, 
            self.template_manager, self.metrics_collector))
        self.register('run', RunSessionsCommand(
            self.config, self.session_manager, self.browser_factory, 
            self.template_manager, self.metrics_collector))
        self.register('template', TemplateCommand(
            self.config, self.session_manager, self.browser_factory, 
            self.template_manager, self.metrics_collector))
    
    def register(self, name: str, command: Command) -> None:
        """Register a command with a name."""
        self.commands[name] = command
    
    def get_command(self, name: str) -> Optional[Command]:
        """Get a command by name."""
        return self.commands.get(name)
    
    def execute(self, name: str, args: argparse.Namespace) -> bool:
        """Execute a command by name."""
        command = self.get_command(name)
        if not command:
            logging.error(f"Unknown command: {name}")
            return False
        return command.execute(args)


# === Command Line Interface ===
class CommandLineInterface:
    """Command line interface for the simple sender."""
    
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command-line arguments with support for subcommands."""
        parser = argparse.ArgumentParser(
            description="Simple Claude Prompt Sender",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Global options
        parser.add_argument("--config", help="Path to config file", default="config/user_config.yaml")
        parser.add_argument("--debug", help="Enable debug mode", action="store_true")
        parser.add_argument("--disable-screenshots", help="Disable screenshot logging", action="store_true")
        
        # Create subparsers for commands
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        
        # Run command - for executing sessions
        run_parser = subparsers.add_parser("run", help="Run one or more sessions")
        run_parser.add_argument("--session", help="Specific session to run")
        run_parser.add_argument("--run-one", action="store_true", help="Run only the specified session")
        run_parser.add_argument("--delay", type=int, help="Delay between prompts (seconds)", default=300)
        run_parser.add_argument("--session-delay", type=int, help="Delay between sessions (seconds)", default=10)
        run_parser.add_argument("--reuse-browser", action="store_true", help="Reuse browser instance between sessions")
        run_parser.add_argument("--resume", action="store_true", help="Resume from last completed prompt")
        
        # New arguments for hybrid waiting
        run_parser.add_argument("--use-hybrid-waiting", action="store_true", 
                               help="Use template detection to determine when responses are complete")
        run_parser.add_argument("--max-wait", type=int, 
                               help="Maximum time to wait for response with hybrid waiting")
        run_parser.add_argument("--no-metrics", action="store_true", help="Disable metrics collection")
        
        # List command - for displaying available sessions
        list_parser = subparsers.add_parser("list", help="List available sessions")
        list_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed session information")
        
        # Status command - for checking session status
        status_parser = subparsers.add_parser("status", help="Check session completion status")
        status_parser.add_argument("--session", help="Specific session to check status for")
        
        # Template command - for managing templates
        template_parser = subparsers.add_parser("template", help="Manage UI templates for visual detection")
        template_subparsers = template_parser.add_subparsers(dest="template_action", help="Template action")
        
        # Template capture command
        capture_parser = template_subparsers.add_parser("capture", help="Capture a new template")
        capture_parser.add_argument("template_name", help="Name for the template")
        capture_parser.add_argument("--region-size", type=int, help="Size of capture region around cursor", default=100)
        
        # Template list command
        list_template_parser = template_subparsers.add_parser("list", help="List available templates")
        list_template_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed template information")
        
        # Template test command
        test_parser = template_subparsers.add_parser("test", help="Test a template against current screen")
        test_parser.add_argument("template_name", help="Name of template to test")
        
        # Template enable command
        enable_parser = template_subparsers.add_parser("enable", help="Enable template system")
        enable_parser.add_argument("--template-dir", help="Directory for template storage")
        enable_parser.add_argument("--threshold", type=float, help="Matching threshold (0-1)", default=0.8)
        
        # Set default command to 'run' if none specified
        args = parser.parse_args()
        if not args.command:
            args.command = 'run'
        
        return args


# === Setup Logging ===
def setup_logging(debug: bool = False) -> str:
    """Set up logging with optional debug mode."""
    try:
        from src.simple_sender.logging import setup_logging as real_setup_logging
        return real_setup_logging(debug=debug)
    except ImportError:
        # Simple fallback if logging module not available
        log_dir = "logs/simple_sender"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"run_{timestamp}.log")
        
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info(f"Logging initialized. Log file: {log_file}")
        return log_dir


# === Main Application ===
class SimpleSenderApplication:
    """Main application class."""
    
    def __init__(self):
        self.cli = CommandLineInterface()
        self.args = None
        self.config_provider = None
        self.session_manager = None
        self.browser_factory = None
        self.template_manager = None
        self.metrics_collector = None
        self.command_registry = None
    
    def initialize(self, args: argparse.Namespace) -> bool:
        """Initialize application components."""
        self.args = args
        
        # Set up logging
        log_dir = setup_logging(debug=args.debug)
        logging.info("=== CLAUDE SIMPLE SENDER STARTING ===")
        logging.info(f"Command: {args.command}")
        
        # Initialize components in sequence with better error handling
        if not self._initialize_config(args.config):
            return False
            
        if not self._initialize_browser_factory():
            return False
            
        if not self._initialize_session_manager():
            return False
            
        # Initialize template system if enabled
        self._initialize_template_system()
        
        # Initialize metrics collection if not disabled
        if not args.no_metrics if hasattr(args, 'no_metrics') else True:
            self._initialize_metrics()
        
        # Create command registry
        if not self._initialize_command_registry():
            return False
            
        return True
    
    def _initialize_config(self, config_path: str) -> bool:
        """Initialize configuration provider."""
        try:
            self.config_provider = ConfigProvider(config_path)
            logging.debug("Configuration provider initialized")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize configuration: {e}")
            return False
    
    def _initialize_browser_factory(self) -> bool:
        """Initialize browser factory."""
        try:
            self.browser_factory = BrowserFactory()
            logging.debug("Browser factory initialized")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize browser factory: {e}")
            return False
    
    def _initialize_session_manager(self) -> bool:
        """Initialize session manager."""
        try:
            if not self.config_provider:
                logging.error("Cannot initialize session manager: config provider not available")
                return False
            self.session_manager = SessionManager(self.config_provider)
            logging.debug("Session manager initialized")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize session manager: {e}")
            return False
    
    def _initialize_template_system(self) -> bool:
        """Initialize template system if enabled."""
        try:
            template_settings = self.config_provider.get_template_settings()
            if template_settings.get("enabled", False):
                self.template_manager = TemplateManager(self.config_provider)
                if self.template_manager.is_available():
                    logging.info("Template system initialized and enabled")
                    # Log hybrid waiting information
                    waiting_strategy = self.config_provider.get_waiting_strategy()
                    if waiting_strategy.get("use_hybrid", False):
                        logging.info("Hybrid waiting strategy is enabled")
                        logging.info(f"Max wait time: {waiting_strategy.get('max_wait', 300)} seconds")
                    return True
                else:
                    logging.warning("Template system enabled but not available")
            return False
        except Exception as e:
            logging.warning(f"Failed to initialize template system: {e}")
            return False
    
    def _initialize_metrics(self) -> bool:
        """Initialize metrics collection."""
        try:
            self.metrics_collector = MetricsCollector(self.config_provider)
            logging.debug("Metrics collector initialized")
            return True
        except Exception as e:
            logging.warning(f"Failed to initialize metrics collector: {e}")
            return False
    
    def _initialize_command_registry(self) -> bool:
        """Initialize command registry."""
        try:
            self.command_registry = CommandRegistry(
                self.config_provider,
                self.session_manager,
                self.browser_factory,
                self.template_manager,
                self.metrics_collector
            )
            logging.debug("Command registry initialized")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize command registry: {e}")
            return False
    
    def run(self) -> int:
        """Run the application."""
        args = self.cli.parse_arguments()
        
        if not self.initialize(args):
            return 1
        
        # Execute requested command
        success = False
        try:
            success = self.command_registry.execute(args.command, args)
        except Exception as e:
            logging.error(f"Error executing {args.command} command: {e}", exc_info=True)
            return 1
        finally:
            # Save metrics if collected
            if self.metrics_collector:
                metrics_file = self.metrics_collector.save_metrics()
                if metrics_file:
                    logging.info(f"Metrics saved to: {metrics_file}")
            
            logging.info("=== CLAUDE SIMPLE SENDER COMPLETED ===")
        
        return 0 if success else 1


def main() -> int:
    """Main entry point for the simple Claude prompt sender."""
    app = SimpleSenderApplication()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())