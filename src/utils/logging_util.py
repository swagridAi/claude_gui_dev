import logging
import os
import time
from datetime import datetime
import pyautogui
import cv2
import numpy as np

def setup_visual_logging(debug=False):
    """
    Set up logging with screenshot capability.
    
    Args:
        debug: Enable debug mode for more verbose logging
    
    Returns:
        Path to log directory
    """
    debug = True
    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up file and console logging
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "automation.log")),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized in {log_dir}")
    return log_dir

def log_with_screenshot(message, level=logging.INFO, region=None, stage_name=None):
    """
    Log a message and capture a screenshot at each program stage.
    
    Args:
        message: Log message
        level: Logging level (default: INFO)
        region: Optional region to capture (x, y, width, height)
        stage_name: Name of the current execution stage (used in filename)
    """
    # Log the message
    logging.log(level, message)
    
    try:
        # Create screenshots directory for this run
        timestamp_run = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_dir = os.path.join("logs", f"run_{timestamp_run}", "screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)
        
        # Generate timestamp for this specific screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Include stage name in filename if provided
        stage_prefix = f"{stage_name}_" if stage_name else ""
        filename = os.path.join(screenshot_dir, f"{stage_prefix}screenshot_{timestamp}.png")
        
        # Log before attempting to capture
        logging.debug(f"Attempting to capture screenshot for stage: {stage_name or 'unnamed'}")
        
        # Capture screenshot (full screen or region)
        try:
            screenshot = pyautogui.screenshot(region=region)
            logging.debug(f"Screenshot captured successfully")
        except Exception as screenshot_error:
            logging.error(f"Failed to capture screenshot: {screenshot_error}", exc_info=True)
            return
        
        # Save the screenshot
        try:
            screenshot.save(filename)
            logging.debug(f"Screenshot saved to {filename}")
        except Exception as save_error:
            logging.error(f"Failed to save screenshot: {save_error}", exc_info=True)
            return
        
        # Create an annotated version with timestamp and stage info
        try:
            # Convert to OpenCV format
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Add timestamp and stage information
            stage_info = f"Stage: {stage_name}" if stage_name else "Unnamed stage"
            time_info = f"Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
            
            # Add text to top of image
            cv2.putText(
                img, 
                f"{stage_info} | {time_info}",
                (10, 30),  # Position at top-left with padding
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Font size
                (0, 0, 255),  # Red color
                2  # Thickness
            )
            
            # Add message text
            cv2.putText(
                img, 
                message[:100] + "..." if len(message) > 100 else message,
                (10, 70),  # Position below stage info
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,  # Font size
                (0, 255, 0),  # Green color
                1  # Thickness
            )
            
            # If region is specified, draw rectangle
            if region:
                x, y, w, h = region
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            
            # Save the annotated screenshot
            annotated_filename = os.path.join(screenshot_dir, f"{stage_prefix}annotated_{timestamp}.png")
            cv2.imwrite(annotated_filename, img)
            logging.debug(f"Annotated screenshot saved to {annotated_filename}")
            
        except Exception as annotation_error:
            logging.error(f"Failed to create annotated screenshot: {annotation_error}", exc_info=True)
        
    except Exception as e:
        logging.error(f"Failed in log_with_screenshot: {e}", exc_info=True)