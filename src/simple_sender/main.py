#!/usr/bin/env python3
"""
Simple Claude Prompt Sender - Main Entry Point

A simplified interface for sending prompts to Claude without relying 
on complex image recognition. Provides a balance between simplicity
and reliability with configurable options.
"""

import argparse
import logging
import sys
import time
import functools
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple, Union


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
                   take_screenshots: bool = True) -> bool:
        """Run a single session."""
        return self.session_manager.run_session(
            session_id, browser_session, prompt_delay, resume, take_screenshots
        )
    
    def run_default_session(self, prompts: List[str], browser_session: Any,
                           prompt_delay: int = 300, resume: bool = False,
                           take_screenshots: bool = True) -> bool:
        """Run the default session with global prompts."""
        return self.session_manager.run_default_session(
            prompts, browser_session, prompt_delay, resume, take_screenshots
        )
    
    def get_session_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the status of a session."""
        return self.session_manager.get_session_status(session_id)
    
    def get_all_session_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of all sessions."""
        return self.session_manager.get_all_session_status()


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
    
    def update(self, increment: int = 1):
        """Update progress and display."""
        self.current += increment
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
        
        logging.info(f"{self.description}: {percent}% complete - ETA: {eta}")


# === Command Pattern Implementation ===
class Command:
    """Base class for command pattern."""
    
    def __init__(self, config_provider: ConfigProvider, 
                session_manager: SessionManager,
                browser_factory: BrowserFactory):
        self.config = config_provider
        self.session_manager = session_manager
        self.browser_factory = browser_factory
    
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
            # Run the session
            success = self.session_manager.run_session(
                session_id,
                browser_session,
                prompt_delay=args.delay,
                resume=args.resume,
                take_screenshots=not args.disable_screenshots
            )
            return success
        except KeyboardInterrupt:
            logging.warning("Execution interrupted by user.")
            return False
        except Exception as e:
            logging.error(f"Error running session '{session_id}': {e}", exc_info=True)
            return False
    
    def _execute_default_session(self, browser_session: Any, 
                                args: argparse.Namespace) -> bool:
        """Execute default session using global prompts."""
        global_prompts = self.config.get_global_prompts()
        
        if not global_prompts:
            logging.error("No prompts found in configuration. Exiting.")
            return False
        
        logging.info("No valid sessions found. Using global prompts.")
        return self.session_manager.run_default_session(
            global_prompts,
            browser_session,
            prompt_delay=args.delay,
            resume=args.resume,
            take_screenshots=not args.disable_screenshots
        )
    
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


class CommandRegistry:
    """Registry for command pattern implementation."""
    
    def __init__(self, config_provider: ConfigProvider, 
                session_manager: SessionManager,
                browser_factory: BrowserFactory):
        self.commands = {}
        self.config = config_provider
        self.session_manager = session_manager
        self.browser_factory = browser_factory
        self._register_commands()
    
    def _register_commands(self) -> None:
        """Register all available commands."""
        self.register('list', ListSessionsCommand(
            self.config, self.session_manager, self.browser_factory))
        self.register('status', ShowStatusCommand(
            self.config, self.session_manager, self.browser_factory))
        self.register('run', RunSessionsCommand(
            self.config, self.session_manager, self.browser_factory))
    
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
        
        # List command - for displaying available sessions
        list_parser = subparsers.add_parser("list", help="List available sessions")
        list_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed session information")
        
        # Status command - for checking session status
        status_parser = subparsers.add_parser("status", help="Check session completion status")
        status_parser.add_argument("--session", help="Specific session to check status for")
        
        # Set default command to 'run' if none specified
        args = parser.parse_args()
        if not args.command:
            args.command = 'run'
        
        return args


# === Setup Logging ===
def setup_logging(debug: bool = False) -> str:
    """Set up logging with optional debug mode."""
    from src.simple_sender.logging import setup_logging as real_setup_logging
    return real_setup_logging(debug=debug)


# === Main Application ===
class SimpleSenderApplication:
    """Main application class."""
    
    def __init__(self):
        self.cli = CommandLineInterface()
        self.args = None
        self.config_provider = None
        self.session_manager = None
        self.browser_factory = None
        self.command_registry = None
    
    def initialize(self, args: argparse.Namespace) -> bool:
        """Initialize application components."""
        self.args = args
        
        # Set up logging
        log_dir = setup_logging(debug=args.debug)
        logging.info("=== CLAUDE SIMPLE SENDER STARTING ===")
        logging.info(f"Command: {args.command}")
        
        # Initialize components
        try:
            self.config_provider = ConfigProvider(args.config)
            self.browser_factory = BrowserFactory()
            self.session_manager = SessionManager(self.config_provider)
            
            # Create command registry
            self.command_registry = CommandRegistry(
                self.config_provider,
                self.session_manager,
                self.browser_factory
            )
            
            return True
        except Exception as e:
            logging.error(f"Initialization error: {e}", exc_info=True)
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
            logging.info("=== CLAUDE SIMPLE SENDER COMPLETED ===")
        
        return 0 if success else 1


def main() -> int:
    """Main entry point for the simple Claude prompt sender."""
    app = SimpleSenderApplication()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())