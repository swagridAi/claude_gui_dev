import pyautogui
import logging
import time

class RegionManager:
    """
    Manages regions for UI elements, providing conversion between 
    relative and absolute coordinates.
    """
    
    def __init__(self):
        """Initialize the region manager."""
        self.screen_size = pyautogui.size()
        self.ui_elements = {}
        self.anchor_points = {}
        self.last_update_time = 0
        
    def set_ui_elements(self, ui_elements):
        """
        Set the UI elements dictionary.
        
        Args:
            ui_elements: Dictionary of UIElement objects
        """
        self.ui_elements = ui_elements
        
    def update_screen_size(self):
        """Update the screen size information."""
        self.screen_size = pyautogui.size()
        logging.debug(f"Updated screen size: {self.screen_size}")
        self.last_update_time = time.time()
        
    def get_absolute_region(self, element_name):
        """
        Get the absolute region for an element.
        
        Args:
            element_name: Name of the UI element
            
        Returns:
            Tuple of (x, y, width, height) or None
        """
        # Ensure we have up-to-date screen information
        if time.time() - self.last_update_time > 60:  # Update every minute
            self.update_screen_size()
            
        # Get the element
        if element_name not in self.ui_elements:
            logging.warning(f"Element {element_name} not found")
            return None
            
        element = self.ui_elements[element_name]
        
        # Get effective region
        return element.get_effective_region(self.ui_elements, self.screen_size)
    
    def convert_to_relative(self, absolute_region, reference="screen"):
        """
        Convert an absolute region to relative coordinates.
        
        Args:
            absolute_region: Tuple of (x, y, width, height)
            reference: Reference for relative coordinates ('screen' or element name)
            
        Returns:
            Tuple of (x_pct, y_pct, width_pct, height_pct)
        """
        if reference == "screen":
            screen_width, screen_height = self.screen_size
            x, y, w, h = absolute_region
            
            return (
                x / screen_width,
                y / screen_height,
                w / screen_width,
                h / screen_height
            )
        else:
            # Relative to another element
            if reference not in self.ui_elements:
                logging.warning(f"Reference element {reference} not found")
                return None
                
            ref_region = self.get_absolute_region(reference)
            if not ref_region:
                return None
                
            ref_x, ref_y, ref_w, ref_h = ref_region
            x, y, w, h = absolute_region
            
            return (
                (x - ref_x) / ref_w,
                (y - ref_y) / ref_h,
                w / ref_w,
                h / ref_h
            )
    
    def set_anchor_point(self, name, location):
        """
        Set an anchor point for relative positioning.
        
        Args:
            name: Anchor point name
            location: Tuple of (x, y)
        """
        self.anchor_points[name] = location
        logging.debug(f"Set anchor point {name} at {location}")
    
    def get_region_relative_to_anchor(self, anchor_name, offsets):
        """
        Get a region relative to an anchor point.
        
        Args:
            anchor_name: Name of the anchor point
            offsets: Tuple of (x_offset, y_offset, width, height)
            
        Returns:
            Tuple of (x, y, width, height) or None
        """
        if anchor_name not in self.anchor_points:
            logging.warning(f"Anchor point {anchor_name} not found")
            return None
            
        anchor_x, anchor_y = self.anchor_points[anchor_name]
        x_offset, y_offset, width, height = offsets
        
        return (anchor_x + x_offset, anchor_y + y_offset, width, height)
        
    def calculate_adaptive_regions(self, base_element_name):
        """
        Calculate adaptive regions based on a successfully located base element.
        
        Args:
            base_element_name: Name of the base UI element
            
        Returns:
            Dictionary of calculated regions
        """
        if base_element_name not in self.ui_elements:
            return {}
            
        base_element = self.ui_elements[base_element_name]
        
        # Check if base element has been found
        if not base_element.last_match_location:
            return {}
            
        # Use the last match location as our reference point
        base_x, base_y, base_w, base_h = base_element.last_match_location
        
        # Calculate regions for elements that reference this element
        regions = {}
        
        for name, element in self.ui_elements.items():
            if element.parent == base_element_name and element.relative_region:
                # Calculate relative to parent
                if 0 <= element.relative_region[0] <= 1:  # Using percentages
                    # Percentage-based calculation
                    rx, ry, rw, rh = element.relative_region
                    regions[name] = (
                        base_x + int(base_w * rx),
                        base_y + int(base_h * ry),
                        int(base_w * rw),
                        int(base_h * rh)
                    )
                else:
                    # Offset-based calculation
                    ox, oy, w, h = element.relative_region
                    regions[name] = (
                        base_x + ox,
                        base_y + oy,
                        w,
                        h
                    )
        
        return regions
    
    def detect_window_position(self, window_title=None):
        """
        Detect the position of a window by title.
        
        Args:
            window_title: Title of the window to detect
            
        Returns:
            Tuple of (x, y, width, height) or None
        """
        if not window_title:
            return None
            
        try:
            # This requires additional libraries depending on OS
            import platform
            
            if platform.system() == "Windows":
                # Windows approach
                import win32gui
                
                def callback(hwnd, window_rect):
                    if win32gui.IsWindowVisible(hwnd) and window_title in win32gui.GetWindowText(hwnd):
                        rect = win32gui.GetWindowRect(hwnd)
                        window_rect[0] = rect
                        return False
                    return True
                
                window_rect = [None]
                win32gui.EnumWindows(callback, window_rect)
                
                if window_rect[0]:
                    left, top, right, bottom = window_rect[0]
                    return (left, top, right - left, bottom - top)
                
            elif platform.system() == "Darwin":  # macOS
                # AppleScript approach
                import subprocess
                
                script = f'''
                tell application "System Events"
                    set frontApp to first application process whose frontmost is true
                    set frontAppName to name of frontApp
                    set windowPosition to position of first window of frontApp
                    set windowSize to size of first window of frontApp
                    return windowPosition & windowSize
                end tell
                '''
                
                result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    values = [int(x) for x in result.stdout.strip().split(',')]
                    if len(values) == 4:
                        return tuple(values)
            
            # Linux approach would require tools like xwininfo or wmctrl
            
            return None
            
        except Exception as e:
            logging.error(f"Error detecting window position: {e}")
            return None