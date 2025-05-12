#!/usr/bin/env python3
"""
Simple Claude Prompt Sender

A minimal script that sends prompts to Claude without image recognition.
Launches the browser for each session, types prompts, presses Enter, and
closes the browser before moving to the next session.
"""

import subprocess
import time
import argparse
import os
import logging
import sys
from datetime import datetime

# Import shared components
from src.utils.browser_core import get_chrome_path, launch_browser as launch_browser_core, close_browser as close_browser_core
from src.utils.config_base import ConfigBase
from src.utils.human_input import HumanizedKeyboard
from src.utils.constants import (
    DEFAULT_SESSION_DELAY, 
    DEFAULT_PROMPT_DELAY, 
    DEFAULT_BROWSER_STARTUP_WAIT,
    DEFAULT_LOGIN_WAIT,
    PROGRESS_LOG_INTERVAL
)


# Set up logging
def setup_logging():
    """Set up logging with timestamps and both file and console output."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", "simple_sender")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")


class SimpleSenderConfig(ConfigBase):
    """Configuration management for Simple Sender."""
    
    def get_browser_config(self):
        """Get browser-specific configuration."""
        return {
            'profile_dir': self.get("browser_profile"),
            'chrome_path': self.get("chrome_path"),
            'startup_wait': self.get("browser_launch_wait", DEFAULT_BROWSER_STARTUP_WAIT)
        }
    
    def get_session_config(self, session_id):
        """Get configuration for a specific session."""
        sessions = self.get("sessions", {})
        if session_id in sessions:
            return sessions[session_id]
        return {}


def launch_browser(url, profile_dir=None):
    """Launch Chrome browser with specified URL and profile."""
    chrome_path = get_chrome_path()
    if not chrome_path:
        logging.error("Chrome browser not found.")
        return False
    
    return launch_browser_core(url, profile_dir, chrome_path)


def send_prompts(prompts, session_id=None, delay_between_prompts=DEFAULT_PROMPT_DELAY):
    """Send each prompt and press Enter."""
    if not prompts:
        logging.error("No prompts to send.")
        return
    
    # Initialize humanized keyboard with normal profile
    keyboard = HumanizedKeyboard(profile='normal')
    
    # Log session information
    if session_id:
        logging.info(f"Running session: {session_id}")
        
    # Wait for page to fully load and for user to handle any login/captcha
    logging.info(f"Waiting {DEFAULT_LOGIN_WAIT} seconds for page to load and for user to handle any login...")
    time.sleep(DEFAULT_LOGIN_WAIT)
    
    # Process each prompt
    for i, prompt in enumerate(prompts):
        prompt_num = i + 1
        logging.info(f"Processing prompt {prompt_num}/{len(prompts)}")
        
        try:
            # Clear any current content
            keyboard.clear_input()
            
            # Type the prompt text with humanized timing
            logging.info(f"Typing prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Typing prompt: {prompt}")
            keyboard.type_text(prompt)
            
            time.sleep(1)
            
            # Press Enter to send
            logging.info("Pressing Enter to send prompt")
            keyboard.send()
            
            # Wait for response (fixed time)
            wait_time = delay_between_prompts
            logging.info(f"Waiting {wait_time} seconds for response...")
            
            # Log progress during the wait
            start_time = time.time()
            while time.time() - start_time < wait_time:
                time.sleep(PROGRESS_LOG_INTERVAL)
                elapsed = time.time() - start_time
                remaining = wait_time - elapsed
                if remaining > 0:
                    logging.info(f"Still waiting: {int(remaining)} seconds remaining...")
            
            logging.info(f"Completed prompt {prompt_num}/{len(prompts)}")
            
        except Exception as e:
            logging.error(f"Error sending prompt {prompt_num}: {e}")
    
    logging.info("All prompts processed.")


def close_browser():
    """Close Chrome browser."""
    return close_browser_core()


def run_session(session_id, session_config, global_config, prompt_delay):
    """Run a single session from start to finish."""
    logging.info(f"=== Starting session: {session_id} ===")
    
    # Get session URL and prompts
    session_url = session_config.get("claude_url", global_config.get("claude_url", "https://claude.ai"))
    session_prompts = session_config.get("prompts", [])
    browser_config = global_config.get_browser_config()
    
    if not session_prompts:
        logging.warning(f"No prompts found for session '{session_id}'. Skipping.")
        return False
    
    # Log session details
    session_name = session_config.get("name", session_id)
    logging.info(f"Session name: {session_name}")
    logging.info(f"Number of prompts: {len(session_prompts)}")
    
    success = False
    try:
        # Launch browser for this session
        if not launch_browser(session_url, browser_config.get('profile_dir')):
            logging.error(f"Failed to launch browser for session '{session_id}'.")
            return False
        
        # Send all prompts for this session
        send_prompts(session_prompts, session_id, prompt_delay)
        success = True
        
    except Exception as e:
        logging.error(f"Error in session '{session_id}': {e}")
        success = False
    finally:
        # Always close the browser before returning
        close_browser()
    
    status = "COMPLETED SUCCESSFULLY" if success else "FAILED"
    logging.info(f"=== Session {session_id} {status} ===")
    return success


def get_all_sessions(config):
    """Get all available sessions from the config."""
    sessions = config.get("sessions", {})
    if not sessions:
        logging.warning("No sessions defined in configuration file.")
    return list(sessions.keys())


def main():
    # Set up logging first
    setup_logging()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Simple Claude Prompt Sender")
    parser.add_argument("--config", help="Path to config file", default="config/user_config.yaml")
    parser.add_argument("--session", help="Specific session to run (default: run all)", default=None)
    parser.add_argument("--run-one", action="store_true", help="Run only the specified session, not all")
    parser.add_argument("--delay", type=int, help="Delay between prompts in seconds", default=DEFAULT_PROMPT_DELAY)
    parser.add_argument("--session-delay", type=int, help="Delay between sessions in seconds", default=DEFAULT_SESSION_DELAY)
    args = parser.parse_args()
    
    # Log startup information
    logging.info("=== CLAUDE SIMPLE SENDER STARTING ===")
    logging.info(f"Command line args: {args}")
    
    # Load configuration
    config = SimpleSenderConfig(args.config)
    
    # Determine which sessions to run
    sessions_to_run = []
    
    if args.session and args.run_one:
        # Run only the specified session
        sessions_to_run = [args.session]
        logging.info(f"Running only session: {args.session}")
    elif args.session:
        # Use specified session as the first one, then run all others
        all_sessions = get_all_sessions(config)
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
        sessions_to_run = get_all_sessions(config)
        logging.info("Running all sessions")
    
    if not sessions_to_run:
        # If no sessions defined, use default global prompts
        logging.info("No sessions defined in config. Using global prompts.")
        global_prompts = config.get("prompts", [])
        
        if not global_prompts:
            logging.error("No prompts found in configuration. Exiting.")
            return 1
        
        # Create a fake session with global settings
        default_session = {
            "name": "Default",
            "prompts": global_prompts,
            "claude_url": config.get("claude_url", "https://claude.ai")
        }
        
        run_session("default", default_session, config, args.delay)
        return 0
    
    # Count valid sessions
    valid_sessions = []
    for session_id in sessions_to_run:
        session_config = config.get_session_config(session_id)
        if not session_config:
            logging.warning(f"Session '{session_id}' not found in configuration. Skipping.")
            continue
        valid_sessions.append(session_id)
    
    if not valid_sessions:
        logging.error("No valid sessions found. Exiting.")
        return 1
    
    logging.info(f"Will run {len(valid_sessions)} sessions: {', '.join(valid_sessions)}")
    
    # Run each session with a separate browser instance
    results = {}
    for i, session_id in enumerate(valid_sessions):
        session_config = config.get_session_config(session_id)
        
        # Run the session
        success = run_session(session_id, session_config, config, args.delay)
        results[session_id] = success
        
        # Wait between sessions if there are more to process
        if i < len(valid_sessions) - 1:
            delay = args.session_delay
            logging.info(f"Waiting {delay} seconds before next session...")
            time.sleep(delay)
    
    # Log summary of results
    logging.info("=== SESSION RESULTS SUMMARY ===")
    successful = sum(1 for success in results.values() if success)
    logging.info(f"Total sessions: {len(results)}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {len(results) - successful}")
    
    for session_id, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logging.info(f"Session '{session_id}': {status}")
    
    logging.info("=== CLAUDE SIMPLE SENDER COMPLETED ===")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())