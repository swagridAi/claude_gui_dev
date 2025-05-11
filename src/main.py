#!/usr/bin/env python3
from src.automation.state_machine import SimpleAutomationMachine
from src.utils.config_manager import ConfigManager
from src.utils.logging_util import setup_visual_logging
from src.utils.session_tracker import SessionTracker
import argparse
import logging
import os
import time
import copy
from src.utils.region_manager import RegionManager
from src.models.ui_element import UIElement
from src.utils.reference_manager import ReferenceImageManager

def parse_arguments():
    parser = argparse.ArgumentParser(description="Claude GUI Automation")
    parser.add_argument("--config", help="Path to config file", default="config/user_config.yaml")
    parser.add_argument("--debug", help="Enable debug mode", action="store_true")
    parser.add_argument("--calibrate", help="Run calibration before starting", action="store_true")
    parser.add_argument("--max-retries", type=int, help="Maximum retry attempts", default=None)
    parser.add_argument("--retry-delay", type=float, help="Initial delay between retries (seconds)", default=None)
    parser.add_argument("--skip-preprocessing", action="store_true", 
                      help="Skip reference image preprocessing")
    # Session selection arguments
    parser.add_argument("--session", help="Specific session name to run (overrides default behavior)", default="default")
    parser.add_argument("--list-sessions", action="store_true", help="List available sessions and exit")
    parser.add_argument("--run-one", action="store_true", 
                      help="Run only the specified session (don't run all)")
    parser.add_argument("--run-all", action="store_true", 
                      help="Force run all sessions (same as --run-all-sessions)")
    parser.add_argument("--run-all-sessions", action="store_true", 
                      help="Run all defined sessions sequentially")
    parser.add_argument("--skip-completed", action="store_true", 
                      help="Skip sessions that have been completed successfully")
    parser.add_argument("--no-skip-completed", action="store_false", dest="skip_completed",
                      help="Force run all sessions even if they've been completed")
    parser.add_argument("--session-delay", type=int, default=5,
                      help="Delay in seconds between running sessions")
    # Config preservation 
    parser.add_argument("--preserve-config", action="store_true", 
                      help="Preserve original configuration (don't overwrite)")
    parser.add_argument("--temp-config", help="Path to write temporary config", default=None)
    parser.add_argument("--restore-original-config", action="store_true",
                      help="Restore original configuration before running")
    parser.add_argument("--show-config-diff", action="store_true",
                      help="Show differences between original and current config")
    parser.add_argument("--cleanup-temp-configs", type=int, metavar="DAYS",
                      help="Remove temporary config files older than specified days")
    # Coordinate options
    parser.add_argument("--use-coordinates", action="store_true",
                      help="Prioritize coordinate-based clicking over visual recognition")
    parser.add_argument("--use-visual", action="store_true",
                      help="Prioritize visual recognition over coordinate-based clicking")
    parser.add_argument("--capture-coordinates", action="store_true",
                      help="Run coordinate capture tool before starting")
    return parser.parse_args()

def cleanup_temp_configs(days_old=7):
    """Remove temporary config files older than specified days."""
    temp_dir = "config/temp"
    if not os.path.exists(temp_dir):
        return
        
    current_time = time.time()
    max_age = days_old * 24 * 60 * 60  # Convert days to seconds
    
    logging.info(f"Cleaning up temporary config files older than {days_old} days")
    
    count = 0
    for filename in os.listdir(temp_dir):
        if filename.endswith(".yaml") and "temp" in filename:
            filepath = os.path.join(temp_dir, filename)
            file_age = current_time - os.path.getmtime(filepath)
            
            if file_age > max_age:
                try:
                    os.remove(filepath)
                    count += 1
                except Exception as e:
                    logging.error(f"Error removing temp file {filepath}: {e}")
    
    logging.info(f"Removed {count} temporary config files")

def run_session(session_id, config_manager, preserve_config=False, args=None):
    """
    Run a single session.
    
    Args:
        session_id: ID of the session to run
        config_manager: ConfigManager instance
        preserve_config: Whether to preserve original config
        args: Command line arguments
        
    Returns:
        bool: True if successful, False if failed
    """
    logging.info(f"Starting session: {session_id}")
    
    # Get all sessions
    sessions = config_manager.get("sessions", {})
    
    # Handle default session differently
    if session_id == "default" and (session_id not in sessions):
        logging.info("Using global configuration for default session")
        # No need to merge, just use the global configuration
        session_config = config_manager.get_all()
    else:
        # Enter session mode to isolate changes to this session
        if not config_manager.is_in_session_mode():
            config_manager.enter_session_mode(session_id)
        
        # Try to merge the session configuration
        if not config_manager.merge_session_config(session_id):
            logging.error(f"Failed to merge configuration for session '{session_id}'")
            return False
        
        # Get the merged configuration
        session_config = config_manager.get_all()
    
    # Initialize reference image manager
    reference_manager = ReferenceImageManager()
    
    # Preprocess existing reference images
    if not args or not args.skip_preprocessing:
        ui_elements_config = config_manager.get("ui_elements", {})
        try:
            if reference_manager.ensure_preprocessing(ui_elements_config, config_manager, preserve_config):
                logging.info("Completed automatic preprocessing of reference images")
            else:
                logging.info("Reference images already preprocessed, skipping preprocessing")
        except Exception as e:
            logging.error(f"Error during reference preprocessing: {e}")
            logging.warning("Continuing without preprocessing references")
    
    # Override config with command line arguments if provided
    if args and args.max_retries is not None:
        config_manager.set("max_retries", args.max_retries)
        logging.info(f"Setting max retries to {args.max_retries} from command line")
        
    if args and args.retry_delay is not None:
        config_manager.set("retry_delay", args.retry_delay)
        logging.info(f"Setting retry delay to {args.retry_delay} from command line")
    
    # Handle coordinate-based clicking command line options
    if args and args.use_coordinates:
        # Set global preference for coordinates
        config_manager.set("automation_settings.prefer_coordinates", True)
        
        # Set all elements to use coordinates first
        ui_elements_config = config_manager.get("ui_elements", {})
        for element_name, element_config in ui_elements_config.items():
            if "click_coordinates" in element_config:
                element_config["use_coordinates_first"] = True
    
    if args and args.use_visual:
        # Set global preference for visual recognition
        config_manager.set("automation_settings.prefer_coordinates", False)
        
        # Set all elements to use visual recognition first
        ui_elements_config = config_manager.get("ui_elements", {})
        for element_name, element_config in ui_elements_config.items():
            if "click_coordinates" in element_config:
                element_config["use_coordinates_first"] = False
    
    # Initialize region manager
    region_manager = RegionManager()
    
    # Parse config for UI elements with both absolute and relative regions
    ui_elements = {}
    for element_name, element_config in config_manager.get("ui_elements", {}).items():
        # Get coordinate-based properties
        click_coordinates = element_config.get("click_coordinates")
        use_coordinates_first = element_config.get("use_coordinates_first", True)
        
        ui_elements[element_name] = UIElement(
            name=element_name,
            reference_paths=element_config.get("reference_paths", []),
            region=element_config.get("region"),
            relative_region=element_config.get("relative_region"),
            parent=element_config.get("parent"),
            confidence=element_config.get("confidence", 0.7),
            click_coordinates=click_coordinates,
            use_coordinates_first=use_coordinates_first
        )
    
    # Register UI elements with region manager
    region_manager.set_ui_elements(ui_elements)
    
    # Display session information
    session_data = config_manager.get("sessions", {}).get(session_id, {})
    session_name = session_data.get("name", session_id)
    prompts = config_manager.get("prompts", [])
    prompt_count = len(prompts)
    logging.info(f"Running session: {session_name} with {prompt_count} prompts")
    logging.info(f"Claude URL: {config_manager.get('claude_url')}")
    
    # Initialize state machine with region manager
    state_machine = SimpleAutomationMachine(config_manager)
    state_machine.region_manager = region_manager
    
    # Set preservation flag if the state machine supports it
    if hasattr(state_machine, 'set_preserve_config') and callable(getattr(state_machine, 'set_preserve_config')):
        state_machine.set_preserve_config(preserve_config)
    
    # Start automation
    success = False
    try:
        state_machine.run()
        # Check if state_machine.AutomationState exists and is accessible
        if hasattr(state_machine, 'state') and hasattr(state_machine, 'AutomationState'):
            success = (state_machine.state == state_machine.AutomationState.COMPLETE)
        else:
            # Fall back to a simple success check based on no exceptions
            success = True
    except KeyboardInterrupt:
        logging.info("Automation stopped by user")
        success = False
    except Exception as e:
        logging.error(f"Automation failed: {e}", exc_info=True)
        success = False
    finally:
        # Ensure cleanup even if exceptions occur
        try:
            state_machine.cleanup()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
    
    logging.info(f"Session '{session_id}' completed with {'success' if success else 'failure'}")
    
    # Exit session mode if we're in it
    if config_manager.is_in_session_mode():
        config_manager.exit_session_mode()
    
    return success

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    args.debug = True  # Force debug mode for more verbose output
    log_dir = setup_visual_logging(debug=args.debug)
    logging.info("Starting Claude automation")
    
    # Clean up temporary configs if requested
    if args.cleanup_temp_configs:
        cleanup_temp_configs(args.cleanup_temp_configs)
        if not any([args.list_sessions, args.run_one, args.run_all, args.run_all_sessions]):
            logging.info("Cleanup completed. Exiting.")
            return
    
    # Load configuration
    config = ConfigManager(args.config)
    
    # Restore original configuration if requested
    if args.restore_original_config:
        if config.restore_original_config():
            logging.info("Restored original configuration")
            # Save the restored config
            if config.save():
                logging.info("Saved restored configuration")
            else:
                logging.error("Failed to save restored configuration")
        else:
            logging.warning("Failed to restore original configuration")
    
    # List available sessions if requested
    if args.list_sessions:
        sessions = config.get("sessions", {})
        if not sessions:
            print("No sessions defined in configuration. Add sessions to your config file.")
            return
        
        print("\nAvailable sessions:")
        for session_id, session_data in sessions.items():
            name = session_data.get("name", session_id)
            prompt_count = len(session_data.get("prompts", []))
            print(f"  - {session_id}: {name} ({prompt_count} prompts)")
        
        # Display config difference if requested
        if args.show_config_diff and hasattr(config, 'show_changes') and callable(config.show_changes):
            print("\nConfiguration differences:")
            config.show_changes()
            
        return
    
    # Initialize session tracker for tracking completion status
    session_tracker = SessionTracker()
    
    # Determine which sessions to run
    sessions_to_run = []
    
    # Check if we should use the default behavior (run all sessions, skip completed)
    # Default behavior: run all sessions unless explicitly asked to run just one
    should_run_all = args.run_all or args.run_all_sessions
    run_single_session = args.run_one or args.session != "default"
    
    # Default to running all sessions if no specific session selection is made
    if not run_single_session and not should_run_all:
        should_run_all = True
        args.skip_completed = True  # Skip completed by default
        logging.info("Using default behavior: running all sessions and skipping completed ones")
    
    if should_run_all:
        # Get all sessions from config
        sessions = config.get("sessions", {})
        if not sessions:
            logging.error("No sessions defined in configuration.")
            print("No sessions defined in configuration. Add sessions to your config file.")
            return
        
        # Determine the order to run sessions
        for session_id in sorted(sessions.keys()):
            # Skip completed sessions if requested
            if args.skip_completed and session_tracker.is_completed(session_id):
                logging.info(f"Skipping completed session: {session_id}")
                continue
            
            sessions_to_run.append(session_id)
        
        if not sessions_to_run:
            logging.info("No sessions to run (all have been completed)")
            print("All sessions have been completed.")
            print("To run them again, use: python -m src.main_sequential --no-skip-completed")
            return
    else:
        # Run just the specified session
        session_id = args.session
        sessions = config.get("sessions", {})
        
        # Handle the case where default session is requested but doesn't exist
        if session_id == "default" and session_id not in sessions:
            # This is fine - we'll use the global config
            pass
        elif session_id != "default" and session_id not in sessions:
            logging.error(f"Session '{session_id}' not found in configuration")
            print(f"Error: Session '{session_id}' not found. Use --list-sessions to see available sessions.")
            return
        
        sessions_to_run.append(session_id)
    
    logging.info(f"Will run {len(sessions_to_run)} sessions: {', '.join(sessions_to_run)}")
    
    # Set preserve_config flag
    preserve_config = args.preserve_config
    
    # Run sessions
    for i, session_id in enumerate(sessions_to_run):
        logging.info(f"Running session {i+1}/{len(sessions_to_run)}: {session_id}")
        
        # Run calibration if this is the first session and calibration is requested
        if i == 0 and args.calibrate:
            from src.utils.calibration import run_calibration
            try:
                # Pass preserve_config flag to calibration
                if run_calibration(config, preserve_config):
                    logging.info("Calibration completed successfully")
                else:
                    logging.warning("Calibration failed or was incomplete")
            except Exception as e:
                logging.error(f"Error during calibration: {e}")
                logging.warning("Continuing with previous calibration settings")
        
        # Run coordinate capture if requested for the first session
        if i == 0 and args.capture_coordinates:
            from tools.unified_calibration import run_coordinate_capture
            logging.info("Running coordinate capture tool")
            try:
                run_coordinate_capture(config, preserve_config)
                logging.info("Coordinate capture completed")
            except Exception as e:
                logging.error(f"Error during coordinate capture: {e}")
                logging.warning("Continuing with existing coordinates")
        
        # Create a working copy of config for temporary use if needed
        if preserve_config and args.temp_config:
            # Create directory if needed
            temp_dir = os.path.dirname(args.temp_config)
            if temp_dir and not os.path.exists(temp_dir):
                os.makedirs(temp_dir, exist_ok=True)
                
            # Save a temporary copy for this run
            logging.info(f"Creating temporary config at {args.temp_config}")
            config.save(args.temp_config)
            
            # Use this temporary config for the rest of the run
            working_config = ConfigManager(args.temp_config)
        else:
            # Use the original config
            working_config = config
        
        # Run the session
        success = run_session(session_id, working_config, preserve_config, args)
        
        # Track session completion
        session_tracker.mark_completed(session_id, success)
        
        # Wait between sessions if there are more to process
        if i < len(sessions_to_run) - 1:
            delay = args.session_delay
            logging.info(f"Waiting {delay} seconds before next session...")
            time.sleep(delay)
    
    # Show completion summary
    if len(sessions_to_run) > 1:
        logging.info("All sessions completed")
        # Report on each session
        for session_id in sessions_to_run:
            status = session_tracker.get_session_status(session_id)
            success = status.get('success', False)
            logging.info(f"Session '{session_id}': {'Successful' if success else 'Failed'}")
    
    # If we're preserving config and using the original file, save with preservation
    if preserve_config and not args.temp_config:
        try:
            logging.info("Saving configuration with session preservation")
            config.save_preserving_sessions()
            # Exit session mode if active
            if config.is_in_session_mode():
                config.exit_session_mode()
        except Exception as e:
            logging.error(f"Error saving preserved configuration: {e}")
    
    logging.info("Automation complete")

if __name__ == "__main__":
    main()