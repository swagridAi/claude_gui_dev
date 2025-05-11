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
import yaml
import pyautogui
import sys
from datetime import datetime
import random

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

def load_config(config_path="config/user_config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return {}

def get_chrome_path():
    """Find Chrome browser path based on OS."""
    import platform
    
    system = platform.system()
    
    if system == "Windows":
        paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Google\Chrome\Application\chrome.exe")
        ]
    elif system == "Darwin":  # macOS
        paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            os.path.expanduser("~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        ]
    else:  # Linux and others
        paths = [
            "/usr/bin/google-chrome",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium"
        ]
    
    # Return the first path that exists
    for path in paths:
        if os.path.exists(path):
            return path
    
    return None

def launch_browser(url, profile_dir=None):
    """Launch Chrome browser with specified URL and profile."""
    chrome_path = get_chrome_path()
    if not chrome_path:
        logging.error("Chrome browser not found.")
        return False
    
    if not profile_dir:
        profile_dir = os.path.join(os.path.expanduser("~"), "ClaudeProfile")
    
    # Ensure profile directory exists
    os.makedirs(profile_dir, exist_ok=True)
    
    cmd = [
        chrome_path,
        f"--user-data-dir={profile_dir}",
        "--start-maximized",
        "--disable-extensions",
        url
    ]
    
    try:
        logging.info(f"Launching Chrome: {chrome_path}")
        logging.info(f"Using profile: {profile_dir}")
        logging.info(f"Navigating to: {url}")
        
        process = subprocess.Popen(cmd)
        
        # Check if process started successfully
        if process.poll() is not None:
            logging.error(f"Browser process exited with code {process.returncode}")
            return False
        
        # Wait for browser to initialize
        logging.info("Waiting for browser to initialize (10 seconds)")
        time.sleep(10)
        
        return True
    except Exception as e:
        logging.error(f"Failed to launch browser: {e}")
        return False

def send_prompts(prompts, session_id=None, delay_between_prompts=300):
    """Send each prompt and press Enter."""
    if not prompts:
        logging.error("No prompts to send.")
        return
    
    # Log session information
    if session_id:
        logging.info(f"Running session: {session_id}")
        
    # Wait for page to fully load and for user to handle any login/captcha
    logging.info("Waiting 15 seconds for page to load and for user to handle any login...")
    time.sleep(15)
    
    # Process each prompt
    for i, prompt in enumerate(prompts):
        prompt_num = i + 1
        logging.info(f"Processing prompt {prompt_num}/{len(prompts)}")
        
        try:
            # Clear any current content with Ctrl+A and Delete
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(0.5)
            pyautogui.press('delete')
            time.sleep(0.5)
            
            # Type the prompt text character by character with random delays
            logging.info(f"Typing prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Typing prompt: {prompt}")
            
            # Type each character with a random delay
            for char in prompt:
                # Type the character
                pyautogui.write(char)
                
                # Random delay between 0.02 and 0.1 seconds for more human-like typing
                typing_delay = random.uniform(0.02, 0.1)
                time.sleep(typing_delay)
                
                # Occasionally add a slightly longer pause (simulating thinking)
                if random.random() < 0.05:  # 5% chance
                    time.sleep(random.uniform(0.2, 0.5))
            
            time.sleep(1)
            
            # Press Enter to send
            logging.info("Pressing Enter to send prompt")
            pyautogui.press('enter')
            
            # Wait for response (fixed time)
            wait_time = delay_between_prompts
            logging.info(f"Waiting {wait_time} seconds for response...")
            
            # Log progress during the wait
            start_time = time.time()
            while time.time() - start_time < wait_time:
                time.sleep(30)  # Check every 30 seconds
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
    try:
        logging.info("Closing browser...")
        import platform
        
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/f", "/im", "chrome.exe"], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["pkill", "-f", "chrome"], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL)
        
        # Wait for browser to close
        time.sleep(2)
        logging.info("Browser closed.")
        return True
        
    except Exception as e:
        logging.error(f"Error closing browser: {e}")
        return False

def run_session(session_id, session_config, global_config, prompt_delay):
    """Run a single session from start to finish."""
    logging.info(f"=== Starting session: {session_id} ===")
    
    # Get session URL and prompts
    session_url = session_config.get("claude_url", global_config.get("claude_url", "https://claude.ai"))
    session_prompts = session_config.get("prompts", [])
    profile_dir = global_config.get("browser_profile")
    
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
        if not launch_browser(session_url, profile_dir):
            logging.error(f"Failed to launch browser for session '{session_id}'.")
            return False
        
        # Send all prompts for this session
        success = send_prompts(session_prompts, session_id, prompt_delay)
        
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
    parser.add_argument("--delay", type=int, help="Delay between prompts in seconds", default=180)
    parser.add_argument("--session-delay", type=int, help="Delay between sessions in seconds", default=10)
    args = parser.parse_args()
    
    # Log startup information
    logging.info("=== CLAUDE SIMPLE SENDER STARTING ===")
    logging.info(f"Command line args: {args}")
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        logging.error("Failed to load configuration. Exiting.")
        return 1
    
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
        session_config = config.get("sessions", {}).get(session_id)
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
        session_config = config.get("sessions", {}).get(session_id, {})
        
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