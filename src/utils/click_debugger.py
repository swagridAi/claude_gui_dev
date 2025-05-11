import pyautogui
import cv2
import numpy as np
import os
import time
import logging
from PIL import Image
from datetime import datetime

def debug_click_location(location, offset=(0, 0), name="element"):
    """
    Debug click locations by taking a screenshot and marking where the click will occur.
    
    Args:
        location: Location tuple (x, y, width, height)
        offset: (x, y) offset from center
        name: Name of the element being clicked
    """
    try:
        # Create debug directory
        debug_dir = "logs/click_debug"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Calculate click position
        if len(location) >= 4:
            x, y, width, height = location
            center_x = x + width // 2 + offset[0]
            center_y = y + height // 2 + offset[1]
        else:
            center_x, center_y = location[0] + offset[0], location[1] + offset[1]
        
        # Take a screenshot
        screenshot = pyautogui.screenshot()
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Draw element rectangle
        if len(location) >= 4:
            x, y, width, height = location
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Add text label
            cv2.putText(
                img, 
                f"Found: {name}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        # Draw click position with crosshair
        cv2.drawMarker(
            img, 
            (center_x, center_y), 
            (0, 0, 255),  # Red for click point
            markerType=cv2.MARKER_CROSS, 
            markerSize=20, 
            thickness=2
        )
        
        # Draw larger circle around click position
        cv2.circle(img, (center_x, center_y), 30, (0, 0, 255), 2)
        
        # Add click coordinates text
        cv2.putText(
            img, 
            f"Click: ({center_x}, {center_y})",
            (center_x + 35, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        
        # Save the debug image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(debug_dir, f"click_{name}_{timestamp}.png")
        cv2.imwrite(filename, img)
        
        logging.info(f"Click debug image saved to {filename}")
        logging.info(f"Clicking {name} at position ({center_x}, {center_y}), original location: {location}")
        
        return center_x, center_y
        
    except Exception as e:
        logging.error(f"Error in click debugging: {e}")
        return None