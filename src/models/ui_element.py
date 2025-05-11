class UIElement:
    """
    Class representing a UI element on screen.
    
    Attributes:
        name: Element name
        reference_paths: List of paths to reference images
        region: Screen region (x, y, width, height) or None for full screen
        relative_region: Region defined relative to parent or screen
        parent: Parent element to calculate relative position from
        confidence: Confidence threshold for recognition
        click_coordinates: (x, y) tuple for direct clicking
        use_coordinates_first: Whether to prioritize coordinates over visual recognition
    """
    
    def __init__(self, name, reference_paths=None, region=None, relative_region=None, 
                 parent=None, confidence=0.7, click_coordinates=None, use_coordinates_first=True):
        """
        Initialize a UI element.
        
        Args:
            name: Element name
            reference_paths: List of paths to reference images
            region: Absolute screen region (x, y, width, height) or None for full screen
            relative_region: Region defined relative to parent or screen
                Format: (x_pct, y_pct, width_pct, height_pct) or (x_offset, y_offset, width, height)
            parent: Parent element name or None for screen-relative
            confidence: Confidence threshold for recognition
            click_coordinates: (x, y) tuple for direct clicking
            use_coordinates_first: Whether to prioritize coordinates over visual recognition
        """
        self.name = name
        self.reference_paths = reference_paths or []
        self.region = region
        self.relative_region = relative_region
        self.parent = parent
        self.confidence = confidence
        # Add fields to track successful matches
        self.last_match_location = None
        self.last_match_time = 0
        
        # New coordinate-based properties
        self.click_coordinates = click_coordinates
        self.use_coordinates_first = use_coordinates_first
    
    def __str__(self):
        """String representation of the UI element."""
        if self.click_coordinates:
            return f"UIElement(name={self.name}, click_coordinates={self.click_coordinates}, use_coordinates_first={self.use_coordinates_first})"
        elif self.relative_region:
            return f"UIElement(name={self.name}, relative_region={self.relative_region}, parent={self.parent}, confidence={self.confidence})"
        else:
            return f"UIElement(name={self.name}, region={self.region}, confidence={self.confidence})"
    
    def __repr__(self):
        """Detailed representation of the UI element."""
        return f"UIElement(name='{self.name}', reference_paths={self.reference_paths}, region={self.region}, relative_region={self.relative_region}, parent={self.parent}, confidence={self.confidence}, click_coordinates={self.click_coordinates}, use_coordinates_first={self.use_coordinates_first})"
    
    def get_effective_region(self, ui_elements=None, screen_size=None):
        """
        Get the effective region by calculating relative positions.
        
        Args:
            ui_elements: Dictionary of all UI elements
            screen_size: Tuple of (width, height) if needed
            
        Returns:
            Tuple of (x, y, width, height) or None
        """
        # If absolute region is set, use it
        if self.region:
            return self.region
            
        # If no relative region, return None
        if not self.relative_region:
            return None
            
        # If we need to calculate based on parent
        if self.parent:
            if not ui_elements or self.parent not in ui_elements:
                return None
                
            parent_element = ui_elements[self.parent]
            parent_region = parent_element.get_effective_region(ui_elements, screen_size)
            
            if not parent_region:
                return None
                
            # Calculate region relative to parent
            return self._calculate_region_from_parent(parent_region)
        
        # Calculate region relative to screen
        if not screen_size:
            import pyautogui
            screen_size = pyautogui.size()
            
        return self._calculate_region_from_screen(screen_size)
    
    def _calculate_region_from_parent(self, parent_region):
        """Calculate absolute region based on parent region."""
        px, py, pw, ph = parent_region
        rx, ry, rw, rh = self.relative_region
        
        # Check if using percentages (values between 0 and 1)
        if 0 <= rx <= 1 and 0 <= ry <= 1 and 0 <= rw <= 1 and 0 <= rh <= 1:
            # Convert percentages to absolute values
            x = px + int(pw * rx)
            y = py + int(ph * ry)
            w = int(pw * rw)
            h = int(ph * rh)
        else:
            # Treat as offsets
            x = px + rx
            y = py + ry
            w = rw
            h = rh
            
        return (x, y, w, h)
    
    def _calculate_region_from_screen(self, screen_size):
        """Calculate absolute region based on screen size."""
        screen_width, screen_height = screen_size
        rx, ry, rw, rh = self.relative_region
        
        # Check if using percentages (values between 0 and 1)
        if 0 <= rx <= 1 and 0 <= ry <= 1 and 0 <= rw <= 1 and 0 <= rh <= 1:
            # Convert percentages to absolute values
            x = int(screen_width * rx)
            y = int(screen_height * ry)
            w = int(screen_width * rw)
            h = int(screen_height * rh)
        else:
            # Treat as direct values
            x = rx
            y = ry
            w = rw
            h = rh
            
        return (x, y, w, h)
    
    def update_from_match(self, location):
        """
        Update the element's region based on a successful match.
        
        Args:
            location: Match location (x, y, width, height)
        """
        import time
        self.last_match_location = location
        self.last_match_time = time.time()
        
        # If using absolute regions, we could update it here
        # but we'll keep the original region for stability