import pyautogui
import time
import logging
import random
from src.models.ui_element import UIElement
from src.automation.recognition import find_element

# Configure PyAutoGUI settings
pyautogui.PAUSE = 0.5  # Default delay between actions
pyautogui.FAILSAFE = True  # Move mouse to upper-left to abort

def click_at_coordinates(x, y, right_click=False, double_click=False, element_name="coordinates"):
    """
    Click directly at the specified coordinates.
    
    Args:
        x, y: Screen coordinates to click
        right_click: Whether to perform a right click
        double_click: Whether to perform a double click
        element_name: Name for logging and debugging
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Debug click location
        from src.utils.click_debugger import debug_click_location
        debug_click_location((x, y, 1, 1), name=element_name)
        
        # Add slight humanized movement
        humanize_mouse_movement(x, y)
        
        # Perform the click
        if right_click:
            pyautogui.rightClick(x, y)
            logging.debug(f"Right-clicked at coordinates ({x}, {y})")
        elif double_click:
            pyautogui.doubleClick(x, y)
            logging.debug(f"Double-clicked at coordinates ({x}, {y})")
        else:
            pyautogui.click(x, y)
            logging.debug(f"Clicked at coordinates ({x}, {y})")
        
        return True
    
    except Exception as e:
        logging.error(f"Click at coordinates failed: {e}")
        return False

def click_element(location, right_click=False, double_click=False, offset=(0, 0)):
    """
    Click on a UI element at the specified location.
    
    Args:
        location: Location object (x, y, width, height) or UIElement
        right_click: Whether to perform a right click
        double_click: Whether to perform a double click
        offset: (x, y) offset from center to click at
    
    Returns:
        True if successful, False otherwise
    """
    try:
        element_name = "unknown"
        
        # Handle UIElement objects
        if isinstance(location, UIElement):
            element_name = location.name
            
            # Try direct coordinates first if available and configured
            if location.click_coordinates and location.use_coordinates_first:
                coords = location.click_coordinates
                # Handle both tuple and list formats
                x, y = coords if isinstance(coords, tuple) else tuple(coords)
                logging.info(f"Using direct coordinates for {element_name}: ({x}, {y})")
                return click_at_coordinates(x, y, right_click, double_click, element_name)
            
            # Fall back to visual recognition
            element_location = find_element(location)
            if not element_location:
                # Try coordinates as fallback if available
                if location.click_coordinates:
                    coords = location.click_coordinates
                    # Handle both tuple and list formats
                    x, y = coords if isinstance(coords, tuple) else tuple(coords)
                    logging.info(f"Visual recognition failed. Using fallback coordinates for {element_name}: ({x}, {y})")
                    return click_at_coordinates(x, y, right_click, double_click, element_name)
                    
                logging.error(f"Could not find element {location.name} to click")
                return False
                
            location = element_location
        
        # Extract coordinates
        if len(location) >= 4:
            x, y, width, height = location
            center_x = x + width // 2 + offset[0]
            center_y = y + height // 2 + offset[1]
        else:
            center_x, center_y = location[0] + offset[0], location[1] + offset[1]
        
        # Debug click location
        from src.utils.click_debugger import debug_click_location
        debug_click_location(location, offset, element_name)
        
        # Add slight humanized movement
        humanize_mouse_movement(center_x, center_y)
        
        # Perform the click
        if right_click:
            pyautogui.rightClick(center_x, center_y)
            logging.debug(f"Right-clicked at ({center_x}, {center_y})")
        elif double_click:
            pyautogui.doubleClick(center_x, center_y)
            logging.debug(f"Double-clicked at ({center_x}, {center_y})")
        else:
            pyautogui.click(center_x, center_y)
            logging.debug(f"Clicked at ({center_x}, {center_y})")
        
        return True
    
    except Exception as e:
        logging.error(f"Click failed: {e}")
        return False

def send_text(text, target=None, delay=0.01):
    """
    Type text with a human-like delay between keystrokes.
    
    Args:
        text: Text to type
        target: Optional target element to click before typing
        delay: Delay between keystrokes (randomized)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Click the target first if provided
        if target:
            success = click_element(target)
            if not success:
                return False
        
        # Clear any existing text with Ctrl+A and Delete
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.press('delete')
        
        # Type the text with human-like delays
        for char in text:
            pyautogui.write(char, interval=delay * random.uniform(0.8, 1.2))
        
        logging.debug(f"Typed text: {text[:10]}..." if len(text) > 10 else f"Typed text: {text}")
        return True
    
    except Exception as e:
        logging.error(f"Text input failed: {e}")
        return False

def press_key(key):
    """
    Press a single key.
    
    Args:
        key: Key to press (e.g., 'enter', 'esc', 'tab')
    
    Returns:
        True if successful, False otherwise
    """
    try:
        pyautogui.press(key)
        logging.debug(f"Pressed key: {key}")
        return True
    
    except Exception as e:
        logging.error(f"Key press failed: {e}")
        return False

def press_hotkey(*keys):
    """
    Press a combination of keys.
    
    Args:
        *keys: Keys to press (e.g., 'ctrl', 'c' for copy)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        pyautogui.hotkey(*keys)
        logging.debug(f"Pressed hotkey: {'+'.join(keys)}")
        return True
    
    except Exception as e:
        logging.error(f"Hotkey press failed: {e}")
        return False

def scroll(clicks, target=None):
    """
    Scroll up or down.
    
    Args:
        clicks: Number of "clicks" to scroll (positive for down, negative for up)
        target: Optional target element to hover over before scrolling
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Move to target first if provided
        if target:
            if isinstance(target, UIElement):
                element_location = find_element(target)
                if not element_location:
                    logging.error(f"Could not find element {target.name} to scroll")
                    return False
                target = element_location
            
            x, y, width, height = target
            pyautogui.moveTo(x + width // 2, y + height // 2)
        
        # Perform scrolling
        pyautogui.scroll(clicks)
        direction = "down" if clicks < 0 else "up"
        logging.debug(f"Scrolled {direction} by {abs(clicks)} clicks")
        return True
    
    except Exception as e:
        logging.error(f"Scroll failed: {e}")
        return False

def drag_drop(start_location, end_location, duration=0.5):
    """
    Perform drag and drop operation.
    
    Args:
        start_location: Starting location (x, y, width, height) or UIElement
        end_location: Ending location (x, y, width, height) or UIElement
        duration: Duration of drag operation
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Handle start location
        if isinstance(start_location, UIElement):
            # Try coordinates first if available and configured
            if start_location.click_coordinates and start_location.use_coordinates_first:
                coords = start_location.click_coordinates
                # Handle both tuple and list formats
                start_x, start_y = coords if isinstance(coords, tuple) else tuple(coords)
            else:
                element_location = find_element(start_location)
                if not element_location:
                    logging.error(f"Could not find element {start_location.name} for drag")
                    return False
                start_location = element_location
                # Calculate center
                start_x = start_location[0] + start_location[2] // 2
                start_y = start_location[1] + start_location[3] // 2
        else:
            # Calculate center from region
            start_x = start_location[0] + start_location[2] // 2
            start_y = start_location[1] + start_location[3] // 2
        
        # Handle end location
        if isinstance(end_location, UIElement):
            # Try coordinates first if available and configured
            if end_location.click_coordinates and end_location.use_coordinates_first:
                coords = end_location.click_coordinates
                # Handle both tuple and list formats
                end_x, end_y = coords if isinstance(coords, tuple) else tuple(coords)
            else:
                element_location = find_element(end_location)
                if not element_location:
                    logging.error(f"Could not find element {end_location.name} for drop")
                    return False
                end_location = element_location
                # Calculate center
                end_x = end_location[0] + end_location[2] // 2
                end_y = end_location[1] + end_location[3] // 2
        else:
            # Calculate center from region
            end_x = end_location[0] + end_location[2] // 2
            end_y = end_location[1] + end_location[3] // 2
        
        # Perform drag and drop
        pyautogui.moveTo(start_x, start_y)
        pyautogui.dragTo(end_x, end_y, duration=duration)
        
        logging.debug(f"Drag and drop from ({start_x}, {start_y}) to ({end_x}, {end_y})")
        return True
    
    except Exception as e:
        logging.error(f"Drag and drop failed: {e}")
        return False

def humanize_mouse_movement(x, y, speed=0.5):
    """
    Move mouse to coordinates with human-like movement.
    
    Args:
        x: Target x coordinate
        y: Target y coordinate
        speed: Movement speed (lower is faster)
    """
    # Get current position
    current_x, current_y = pyautogui.position()
    
    # Add slight randomization to target (within 2 pixels)
    rand_x = x + random.randint(-2, 2)
    rand_y = y + random.randint(-2, 2)
    
    # Use PyAutoGUI's built-in easing function for humanized movement
    pyautogui.moveTo(rand_x, rand_y, duration=speed, tween=pyautogui.easeOutQuad)

def wait_and_click(ui_element, timeout=10, interval=0.5):
    """
    Wait for an element to appear and then click it.
    
    Args:
        ui_element: UIElement to wait for and click
        timeout: Maximum time to wait in seconds
        interval: Check interval in seconds
    
    Returns:
        True if found and clicked, False otherwise
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check if we should use coordinates first
        if ui_element.click_coordinates and ui_element.use_coordinates_first:
            coords = ui_element.click_coordinates
            # Handle both tuple and list formats
            x, y = coords if isinstance(coords, tuple) else tuple(coords)
            logging.info(f"Using direct coordinates for {ui_element.name}: ({x}, {y})")
            return click_at_coordinates(x, y, element_name=ui_element.name)
        
        # Use visual recognition
        location = find_element(ui_element)
        if location:
            return click_element(location)
        
        time.sleep(interval)
    
    # If element not found by visual recognition, try coordinates as fallback
    if ui_element.click_coordinates:
        coords = ui_element.click_coordinates
        # Handle both tuple and list formats
        x, y = coords if isinstance(coords, tuple) else tuple(coords)
        logging.info(f"Element not found visually. Using fallback coordinates for {ui_element.name}: ({x}, {y})")
        return click_at_coordinates(x, y, element_name=ui_element.name)
    
    logging.error(f"Timed out waiting for {ui_element.name}")
    return False

def retry_interaction(interaction_func, max_retries=3, *args, **kwargs):
    """
    Retry an interaction function multiple times.
    
    Args:
        interaction_func: Function to retry
        max_retries: Maximum number of retry attempts
        *args, **kwargs: Arguments to pass to the function
    
    Returns:
        Result of the interaction function or False if all retries failed
    """
    retries = 0
    while retries < max_retries:
        result = interaction_func(*args, **kwargs)
        if result:
            return result
        
        retries += 1
        logging.warning(f"Retry {retries}/{max_retries} for interaction")
        time.sleep(1)  # Wait before retrying
    
    return False