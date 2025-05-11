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
from pathlib import Path

# Import from simple_sender package
from src.simple_sender.browser import BrowserSession
from src.simple_sender.config import SimpleConfig
from src.simple_sender.interaction import send_prompt, type_with_human_timing
from src.simple_sender.logging import setup_logging, log_action
from src.simple_sender.session import SessionManager

def parse_arguments():
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

def list_sessions(config, verbose=False):
    """List all available sessions with optional details."""
    sessions = config.get_sessions()
    
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
        if verbose and prompt_count > 0:
            print(f"    URL: {session_info.get('claude_url', 'default')}")
            for i, prompt in enumerate(session_info.get("prompts", []), 1):
                prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
                print(f"    {i}. {prompt_preview}")
            print()
    
    return True

def show_status(session_manager, session_id=None):
    """Show completion status for sessions."""
    if session_id:
        # Show status for specific session
        status = session_manager.get_session_status(session_id)
        if not status:
            print(f"Session '{session_id}' not found.")
            return False
        
        print(f"Status for session '{session_id}':")
        print(f"  Completed: {status.get('completed', False)}")
        print(f"  Success: {status.get('success', False)}")
        if 'completion_time' in status:
            print(f"  Completed at: {status['completion_time']}")
        if 'prompts_completed' in status:
            print(f"  Prompts completed: {status['prompts_completed']}")
        
        return True
    else:
        # Show summary of all sessions
        all_status = session_manager.get_all_session_status()
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

def run_sessions(config, session_manager, args):
    """Run specified sessions based on command-line arguments."""
    # Determine which sessions to run
    sessions_to_run = []
    
    if args.session and args.run_one:
        # Run only the specified session
        sessions_to_run = [args.session]
        logging.info(f"Running only session: {args.session}")
    elif args.session:
        # Use specified session as the first one, then run all others
        all_sessions = config.get_session_ids()
        if args.session in all_sessions:
            # Move the specified session to the front
            all_sessions.remove(args.session)
            sessions_to_run = [args.session] + all_sessions
            logging.info(f"Running {args.session} first, then all other sessions")
        else:
            # If specified session doesn't exist, run all
            sessions_to_run = all_sessions
            logging.info(f"Session '{args.session}' not found. Running all sessions.")
    else:
        # Run all sessions by default
        sessions_to_run = config.get_session_ids()
        logging.info("Running all sessions")
    
    # Filter out invalid sessions
    valid_sessions = []
    for session_id in sessions_to_run:
        if config.is_valid_session(session_id):
            valid_sessions.append(session_id)
        else:
            logging.warning(f"Session '{session_id}' not found or has no prompts. Skipping.")
    
    if not valid_sessions:
        # If no valid sessions defined, check if global prompts exist
        global_prompts = config.get_global_prompts()
        
        if not global_prompts:
            logging.error("No prompts found in configuration. Exiting.")
            return False
        
        # Run using global prompts
        logging.info("No valid sessions found. Using global prompts.")
        browser_session = BrowserSession(reuse_browser=False)
        success = session_manager.run_default_session(
            global_prompts,
            browser_session,
            prompt_delay=args.delay,
            resume=args.resume,
            take_screenshots=not args.disable_screenshots
        )
        return success
    
    # Initialize browser session manager
    browser_session = BrowserSession(reuse_browser=args.reuse_browser)
    
    # Log session plan
    logging.info(f"Will run {len(valid_sessions)} sessions: {', '.join(valid_sessions)}")
    log_action("Starting session execution", take_screenshots=not args.disable_screenshots)
    
    # Run each session
    results = {}
    for i, session_id in enumerate(valid_sessions):
        try:
            # Run the session
            success = session_manager.run_session(
                session_id,
                browser_session,
                prompt_delay=args.delay,
                resume=args.resume,
                take_screenshots=not args.disable_screenshots
            )
            results[session_id] = success
            
            # Wait between sessions if there are more to process
            if i < len(valid_sessions) - 1:
                delay = args.session_delay
                logging.info(f"Waiting {delay} seconds before next session...")
                time.sleep(delay)
                
        except KeyboardInterrupt:
            logging.warning("Execution interrupted by user.")
            results[session_id] = False
            break
        except Exception as e:
            logging.error(f"Error running session '{session_id}': {e}", exc_info=True)
            results[session_id] = False
    
    # Always ensure browser is closed at the end
    browser_session.close()
    
    # Log summary of results
    logging.info("=== SESSION RESULTS SUMMARY ===")
    successful = sum(1 for success in results.values() if success)
    logging.info(f"Total sessions: {len(results)}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {len(results) - successful}")
    
    for session_id, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logging.info(f"Session '{session_id}': {status}")
    
    return all(results.values())

def main():
    """Main entry point for the simple Claude prompt sender."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging
    log_dir = setup_logging(debug=args.debug)
    logging.info("=== CLAUDE SIMPLE SENDER STARTING ===")
    logging.info(f"Command: {args.command}")
    
    # Initialize configuration and session manager
    try:
        config = SimpleConfig(args.config)
        session_manager = SessionManager(config)
    except Exception as e:
        logging.error(f"Initialization error: {e}", exc_info=True)
        return 1
    
    # Execute requested command
    success = False
    try:
        if args.command == 'list':
            success = list_sessions(config, args.verbose)
        elif args.command == 'status':
            success = show_status(session_manager, args.session)
        elif args.command == 'run':
            success = run_sessions(config, session_manager, args)
        else:
            logging.error(f"Unknown command: {args.command}")
            return 1
    except Exception as e:
        logging.error(f"Error executing {args.command} command: {e}", exc_info=True)
        return 1
    finally:
        logging.info("=== CLAUDE SIMPLE SENDER COMPLETED ===")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())