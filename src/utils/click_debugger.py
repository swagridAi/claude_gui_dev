import pyautogui
import cv2
import numpy as np
import os
import time
import logging
from PIL import Image
from datetime import datetime

# Visual debugging constants
DEBUG_DIR = "logs/click_debug"
ELEMENT_COLOR = (0, 255, 0)  # Green
CLICK_COLOR = (0, 0, 255)    # Red
MARKER_SIZE = 20
LINE_THICKNESS = 2
CIRCLE_RADIUS = 30
FONT_SCALE_SMALL = 0.5
FONT_SCALE_LARGE = 0.7

def _ensure_debug_directory():
    """Create debug directory if it doesn't exist."""
    os.makedirs(DEBUG_DIR, exist_ok=True)

def _capture_and_convert_screenshot():
    """Take screenshot and convert to OpenCV format."""
    screenshot = pyautogui.screenshot()
    img = np.array(screenshot)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def _calculate_click_position(location, offset):
    """Calculate click position from location and offset."""
    if len(location) >= 4:
        x, y, width, height = location
        center_x = x + width // 2 + offset[0]
        center_y = y + height // 2 + offset[1]
    else:
        center_x, center_y = location[0] + offset[0], location[1] + offset[1]
    return center_x, center_y

def _draw_element_bounds(img, location, name):
    """Draw element rectangle and label on image."""
    if len(location) >= 4:
        x, y, width, height = location
        cv2.rectangle(img, (x, y), (x + width, y + height), ELEMENT_COLOR, LINE_THICKNESS)
        
        cv2.putText(
            img, 
            f"Found: {name}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE_SMALL,
            ELEMENT_COLOR,
            1
        )

def _draw_click_indicators(img, center_x, center_y):
    """Draw click marker, circle, and coordinates on image."""
    # Draw crosshair marker
    cv2.drawMarker(
        img, 
        (center_x, center_y), 
        CLICK_COLOR,
        markerType=cv2.MARKER_CROSS, 
        markerSize=MARKER_SIZE, 
        thickness=LINE_THICKNESS
    )
    
    # Draw circle around click position
    cv2.circle(img, (center_x, center_y), CIRCLE_RADIUS, CLICK_COLOR, LINE_THICKNESS)
    
    # Add click coordinates text
    cv2.putText(
        img, 
        f"Click: ({center_x}, {center_y})",
        (center_x + 35, center_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE_LARGE,
        CLICK_COLOR,
        LINE_THICKNESS
    )

def _save_debug_image(img, name):
    """Save debug image with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(DEBUG_DIR, f"click_{name}_{timestamp}.png")
    cv2.imwrite(filename, img)
    return filename

def debug_click_location(location, offset=(0, 0), name="element"):
    """
    Debug click locations by taking a screenshot and marking where the click will occur.
    
    Args:
        location: Location tuple (x, y, width, height)
        offset: (x, y) offset from center
        name: Name of the element being clicked
    """
    try:
        # Ensure debug directory exists
        _ensure_debug_directory()
        
        # Calculate click position
        center_x, center_y = _calculate_click_position(location, offset)
        
        # Take screenshot and convert
        img = _capture_and_convert_screenshot()
        
        # Draw element bounds
        _draw_element_bounds(img, location, name)
        
        # Draw click indicators
        _draw_click_indicators(img, center_x, center_y)
        
        # Save debug image
        filename = _save_debug_image(img, name)
        
        logging.info(f"Click debug image saved to {filename}")
        logging.info(f"Clicking {name} at position ({center_x}, {center_y}), original location: {location}")
        
        return center_x, center_y
        
    except Exception as e:
        logging.error(f"Error in click debugging: {e}")
        return None