import tkinter as tk
from tkinter import ttk, messagebox
import os
import cv2
import numpy as np
import logging
import re
import glob
from PIL import Image, ImageTk

class AnalyzeTabView:
    """View component for the Analysis tab of the calibration tool."""
    
    def __init__(self, parent, app_controller, recognition_controller):
        """
        Initialize the Analyze tab view.
        
        Args:
            parent: Parent tkinter container
            app_controller: Main application controller
            recognition_controller: Controller for element recognition
        """
        self.parent = parent
        self.app_controller = app_controller
        self.recognition_controller = recognition_controller
        self.debug_images = {}
        self.current_image_path = None
        self.debug_scale = 1.0
        self.debug_image = None  # Reference to prevent garbage collection
        
        # UI components
        self.frame = None
        self.debug_listbox = None
        self.analyze_canvas = None
        self.match_info = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the complete analyze tab UI."""
        self.frame = ttk.Frame(self.parent)
        self._create_layout()
        self._create_debug_controls()
        self._create_match_info_panel()
        self._create_canvas()
        self._bind_events()
        
        return self.frame
    
    def _create_layout(self):
        """Create the paned layout for the analyze tab."""
        # Create a paned window
        pane = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Debug controls
        self.left_frame = ttk.Frame(pane, padding=10)
        pane.add(self.left_frame, weight=1)
        
        # Right panel - Debug visualization
        self.right_frame = ttk.LabelFrame(pane, text="Debug Visualization")
        pane.add(self.right_frame, weight=3)
        
        # Debug controls in left frame
        ttk.Label(self.left_frame, text="Debug Analysis", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))
    
    def _create_debug_controls(self):
        """Create debug image controls and listings."""
        # Debug image selection
        ttk.Label(self.left_frame, text="Select Debug Image:").pack(anchor="w", pady=(0, 5))
        
        # Debug image listbox
        self.debug_listbox = tk.Listbox(self.left_frame, height=10)
        self.debug_listbox.pack(fill=tk.X, pady=(0, 10))
        
        # Analysis controls
        ttk.Button(self.left_frame, text="Scan Debug Images", 
                  command=self.scan_debug_images).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(self.left_frame, text="Use Best Match Region", 
                  command=self._use_best_match_region).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(self.left_frame, text="Adjust Click Offset", 
                  command=self._adjust_click_offset).pack(fill=tk.X, pady=(0, 5))
        
        # Navigation buttons
        nav_frame = ttk.Frame(self.left_frame)
        nav_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(nav_frame, text="Back to Define", 
                  command=lambda: self.app_controller.select_tab(1)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Go to Verify", 
                  command=lambda: self.app_controller.select_tab(3)).pack(side=tk.RIGHT)
    
    def _create_match_info_panel(self):
        """Create the match information display panel."""
        self.match_info_frame = ttk.LabelFrame(self.left_frame, text="Match Information")
        self.match_info_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        self.match_info = tk.Text(self.match_info_frame, height=10, width=30, wrap=tk.WORD, state=tk.DISABLED)
        self.match_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _create_canvas(self):
        """Create the image analysis canvas."""
        self.analyze_canvas = tk.Canvas(self.right_frame, bg="gray90")
        self.analyze_canvas.pack(fill=tk.BOTH, expand=True)
    
    def _bind_events(self):
        """Bind all event handlers."""
        self.debug_listbox.bind('<<ListboxSelect>>', self._on_debug_image_select)
    
    def scan_debug_images(self):
        """Scan for debug images in the logs directory and update display."""
        self.debug_images = self.recognition_controller.scan_debug_images()
        self._populate_debug_listbox()
        
        # Show status
        element_count = len(self.debug_images)
        if element_count > 0:
            self.app_controller.show_status(f"Found debug images for {element_count} UI elements")
        else:
            self.app_controller.show_status("No debug images found matching UI elements")
    
    def _populate_debug_listbox(self):
        """Populate the debug listbox with found images."""
        self.debug_listbox.delete(0, tk.END)
        
        # Add entries for each element
        for element_name in sorted(self.debug_images.keys()):
            rec_count = len(self.debug_images[element_name]["recognition"])
            click_count = len(self.debug_images[element_name]["click"])
            
            self.debug_listbox.insert(tk.END, f"{element_name} (R:{rec_count}, C:{click_count})")
            
            # Add individual debug images with indent
            for i, path in enumerate(self.debug_images[element_name]["recognition"]):
                filename = os.path.basename(path)
                self.debug_listbox.insert(tk.END, f"  - R{i+1}: {filename}")
                
            for i, path in enumerate(self.debug_images[element_name]["click"]):
                filename = os.path.basename(path)
                self.debug_listbox.insert(tk.END, f"  - C{i+1}: {filename}")
    
    def _on_debug_image_select(self, event):
        """Handle selection of a debug image from the listbox."""
        selection = self.debug_listbox.curselection()
        if not selection:
            return
            
        # Get selected image entry
        selected_entry = self.debug_listbox.get(selection[0])
        
        # Check if it's an element or an image entry
        if selected_entry.startswith("  - "):
            # Process image entry (starts with "  - ")
            self._process_image_selection(selection[0], selected_entry)
        else:
            # Process element entry
            self.app_controller.show_status(f"Selected element: {selected_entry}")
    
    def _process_image_selection(self, index, selected_entry):
        """Process the selection of an image entry from the debug listbox."""
        # Parse image type (R or C) and index
        parts = selected_entry.strip()[4:].split(":", 1)  # Skip the "  - " prefix
        if len(parts) != 2:
            return
            
        image_ref = parts[0].strip()  # This should be something like "R1" or "C1"
        if not image_ref or len(image_ref) < 2:
            self.app_controller.show_status(f"Invalid image reference: {image_ref}")
            return
            
        image_type = image_ref[0]  # R or C
        try:
            # Make sure there's actually a number after the R or C
            if len(image_ref) < 2 or not image_ref[1:].isdigit():
                self.app_controller.show_status(f"Invalid image index format: {image_ref}")
                return
                
            image_index = int(image_ref[1:]) - 1  # Convert to 0-based index
        except ValueError:
            self.app_controller.show_status(f"Invalid image index: {image_ref[1:]}")
            return
        
        # Find the parent element entry
        parent_index = index - 1
        while parent_index >= 0:
            parent_entry = self.debug_listbox.get(parent_index)
            if not parent_entry.startswith("  - "):
                break
            parent_index -= 1
        
        if parent_index < 0:
            self.app_controller.show_status("Could not find parent element for this image")
            return
            
        # Extract element name from parent entry
        parent_parts = parent_entry.split(" ", 1)
        if len(parent_parts) != 2:
            self.app_controller.show_status("Invalid parent element format")
            return
            
        element_name = parent_parts[0]
        
        # Get the image path
        if element_name not in self.debug_images:
            self.app_controller.show_status(f"No debug images found for {element_name}")
            return
            
        if image_type == "R" and image_index < len(self.debug_images[element_name]["recognition"]):
            image_path = self.debug_images[element_name]["recognition"][image_index]
            self.display_debug_image(image_path, "recognition")
        elif image_type == "C" and image_index < len(self.debug_images[element_name]["click"]):
            image_path = self.debug_images[element_name]["click"][image_index]
            self.display_debug_image(image_path, "click")
        else:
            self.app_controller.show_status(f"Image index out of range: {image_ref}")
    
    def display_debug_image(self, image_path, image_type):
        """Display a debug image on the analyze canvas."""
        try:
            # Store the current image path
            self.current_image_path = image_path
            
            # Load the image
            img = Image.open(image_path)
            
            # Get canvas dimensions
            canvas_width = self.analyze_canvas.winfo_width()
            canvas_height = self.analyze_canvas.winfo_height()
            
            # If canvas not yet sized, use defaults
            if canvas_width <= 1:
                canvas_width = 800
                canvas_height = 600
            
            # Keep aspect ratio
            img_width, img_height = img.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize the image
            resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.debug_image = ImageTk.PhotoImage(resized)
            
            # Clear canvas
            self.analyze_canvas.delete("all")
            
            # Draw the image
            self.analyze_canvas.create_image(0, 0, anchor=tk.NW, image=self.debug_image)
            self.analyze_canvas.config(scrollregion=(0, 0, new_width, new_height))
            
            # Store the scale factor for coordinate conversion
            self.debug_scale = scale
            
            # Extract information from image
            self.analyze_debug_image(image_path, image_type)
            
            # Update status
            self.app_controller.show_status(f"Displaying debug image: {os.path.basename(image_path)}")
            
        except Exception as e:
            messagebox.showerror("Display Error", f"Error displaying debug image: {e}")
            self.app_controller.show_status(f"Error displaying debug image: {e}")
    
    def analyze_debug_image(self, image_path, image_type):
        """Analyze a debug image and update match information."""
        # Enable text widget for editing
        self.match_info.config(state=tk.NORMAL)
        
        # Clear current content
        self.match_info.delete(1.0, tk.END)
        
        try:
            # Extract filename and element name
            filename = os.path.basename(image_path)
            element_name = self._extract_element_name(filename, image_type)
            
            # Build info text
            info = f"Debug Image: {filename}\n"
            info += f"Type: {image_type.capitalize()}\n"
            info += f"Element: {element_name}\n\n"
            
            # Add additional analysis based on image type
            if image_type == "recognition":
                match_regions = self._analyze_recognition_image(image_path)
                
                if match_regions:
                    info += f"Found {len(match_regions)} match regions:\n"
                    for i, region in enumerate(match_regions):
                        x, y, w, h = region
                        confidence = 0.0  # Would need to extract from image or filename
                        info += f"Match {i+1}: ({x}, {y}, {w}, {h}), conf: {confidence:.2f}\n"
                else:
                    info += "No match regions detected."
            
            elif image_type == "click":
                click_points = self._analyze_click_image(image_path)
                
                if click_points:
                    info += f"Found {len(click_points)} click points:\n"
                    for i, point in enumerate(click_points):
                        x, y = point
                        info += f"Click Point {i+1}: ({x}, {y})\n"
                else:
                    info += "No click points detected."
            
            # Insert analysis
            self.match_info.insert(tk.END, info)
            
        except Exception as e:
            self.match_info.insert(tk.END, f"Error analyzing debug image: {e}")
            
        # Disable editing
        self.match_info.config(state=tk.DISABLED)
    
    def _extract_element_name(self, filename, image_type):
        """Extract element name from filename."""
        if image_type == "recognition":
            match = re.match(r"(\w+)_matches_\d+\.png", filename)
            if match:
                return match.group(1)
        else:  # click
            match = re.match(r"click_(\w+)_\d+.*\.png", filename)
            if match:
                return match.group(1)
        
        # Fallback - try to extract from any word characters before underscore
        match = re.match(r"(\w+)_", filename)
        return match.group(1) if match else "unknown"
    
    def _analyze_recognition_image(self, image_path):
        """Analyze a recognition debug image to extract match regions."""
        try:
            # Load the image using OpenCV to detect marked regions
            img = cv2.imread(image_path)
            
            if img is None:
                logging.error(f"Could not load image: {image_path}")
                return []
                
            # Convert to HSV for easier color detection
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Green mask (for found regions)
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract regions
            match_regions = []
            for contour in contours:
                # Get the bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                match_regions.append((x, y, w, h))
                
                # Draw rectangle on analyze canvas
                canvas_x = x * self.debug_scale
                canvas_y = y * self.debug_scale
                canvas_w = w * self.debug_scale
                canvas_h = h * self.debug_scale
                
                self.analyze_canvas.create_rectangle(
                    canvas_x, canvas_y, 
                    canvas_x + canvas_w, canvas_y + canvas_h,
                    outline="green", width=2, tags="match"
                )
                
                # Add label
                self.analyze_canvas.create_text(
                    canvas_x, canvas_y - 10,
                    text=f"Match: ({x}, {y}, {w}, {h})",
                    fill="green", anchor=tk.W, tags="match"
                )
            
            return match_regions
            
        except Exception as e:
            logging.error(f"Error analyzing recognition debug: {e}")
            return []
    
    def _analyze_click_image(self, image_path):
        """Analyze a click debug image to extract click locations."""
        try:
            # Load the image using OpenCV
            img = cv2.imread(image_path)
            
            if img is None:
                logging.error(f"Could not load image: {image_path}")
                return []
                
            # Look for the red circle/cross marking the click point
            # Extract red channel
            red_channel = img[:,:,2]
            _, thresh = cv2.threshold(red_channel, 200, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for circular contours
            click_points = []
            for contour in contours:
                # Calculate properties that can help identify the click marker
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # A circle has the maximum area for a given perimeter
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    
                    # Circles have circularity close to 1.0
                    if 0.5 < circularity < 1.2 and 100 < area < 5000:
                        # Get the center of the contour
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            click_points.append((cx, cy))
                            
                            # Draw marker on analyze canvas
                            canvas_x = cx * self.debug_scale
                            canvas_y = cy * self.debug_scale
                            
                            self.analyze_canvas.create_oval(
                                canvas_x - 10, canvas_y - 10, 
                                canvas_x + 10, canvas_y + 10,
                                outline="red", width=2, tags="click"
                            )
                            
                            # Add label
                            self.analyze_canvas.create_text(
                                canvas_x + 15, canvas_y,
                                text=f"Click: ({cx}, {cy})",
                                fill="red", anchor=tk.W, tags="click"
                            )
            
            return click_points
            
        except Exception as e:
            logging.error(f"Error analyzing click debug: {e}")
            return []
    
    def _use_best_match_region(self):
        """Use the best match region from the debug image for the current element."""
        # Get the current element name
        element_name = self.recognition_controller.get_current_element_name()
        if not element_name:
            messagebox.showwarning("No Element Selected", "Please select an element first.")
            return
        
        # Check if a debug image is displayed
        if not self.current_image_path:
            messagebox.showwarning("No Debug Image", "Please select a debug image first.")
            return
        
        # Look for match regions on canvas
        match_items = self.analyze_canvas.find_withtag("match")
        if not match_items:
            messagebox.showwarning("No Matches", "No match regions found in the current debug image.")
            return
        
        # Get the first (best) match region
        coords = self.analyze_canvas.coords(match_items[0])
        if len(coords) != 4:
            return
        
        # Convert to original image coordinates
        x1, y1, x2, y2 = coords
        w, h = x2 - x1, y2 - y1
        
        # Convert canvas coordinates to original image coordinates
        orig_x = int(x1 / self.debug_scale)
        orig_y = int(y1 / self.debug_scale)
        orig_w = int(w / self.debug_scale)
        orig_h = int(h / self.debug_scale)
        
        # Update element's region via controller
        self.recognition_controller.update_element_region(
            element_name, 
            (orig_x, orig_y, orig_w, orig_h)
        )
        
        self.app_controller.show_status(
            f"Updated region for {element_name} to best match: ({orig_x}, {orig_y}, {orig_w}, {orig_h})"
        )
    
    def _adjust_click_offset(self):
        """Open a dialog to adjust click offset for the current element."""
        element_name = self.recognition_controller.get_current_element_name()
        if not element_name:
            messagebox.showwarning("No Selection", "Please select an element first.")
            return
            
        self.recognition_controller.adjust_click_offset(element_name)
    
    def clear_display(self):
        """Clear the analysis display and canvas."""
        self.analyze_canvas.delete("all")
        self.match_info.config(state=tk.NORMAL)
        self.match_info.delete(1.0, tk.END)
        self.match_info.config(state=tk.DISABLED)
        self.current_image_path = None