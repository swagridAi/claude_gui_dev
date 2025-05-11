#!/usr/bin/env python3
"""
Recognition Controller for Claude GUI Automation calibration tool.
Handles UI element recognition, testing, and coordinate management.
"""

import logging
import cv2
import numpy as np
import os
import re
import time
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk

class RecognitionController:
    """
    Controller for managing UI element recognition and testing.
    Handles core recognition functionality including testing elements,
    analyzing debug images, and managing click coordinates.
    """
    
    def __init__(self, app_controller, element_controller, config_model):
        """
        Initialize the recognition controller.
        
        Args:
            app_controller: Main application controller reference
            element_controller: Element management controller reference
            config_model: Configuration model reference
        """
        self.app_controller = app_controller
        self.element_controller = element_controller
        self.config_model = config_model
        self.debug_images = {}
        self.debug_scale = 1.0
        self.current_debug_image = None
        self.current_debug_path = None
        self.current_debug_type = None
        
    def find_element(self, element, confidence_override=None, screenshot=None):
        """
        Find a UI element on screen using recognition techniques.
        
        Args:
            element: UIElement to find
            confidence_override: Optional override for confidence threshold
            screenshot: Optional screenshot to use (if None, takes new screenshot)
            
        Returns:
            Tuple of (x, y, width, height) if found, None otherwise
        """
        try:
            # Import here to avoid circular imports
            from src.automation.recognition import find_element as recognition_find_element
            
            # Call the imported function with our parameters
            location = recognition_find_element(element, confidence_override)
            
            if location:
                self.app_controller.show_status(f"Found element {element.name} at {location}")
            else:
                self.app_controller.show_status(f"Could not find element {element.name}")
                
            return location
            
        except Exception as e:
            self.app_controller.show_status(f"Error finding element: {e}")
            logging.error(f"Error in find_element: {e}")
            return None

    def test_all_elements(self, canvas, update_results_callback=None):
        """
        Test recognition for all defined elements and display results.
        
        Args:
            canvas: Canvas to draw results on
            update_results_callback: Optional callback to update results display
            
        Returns:
            Dictionary of test results by element name
        """
        elements = self.element_controller.get_elements()
        if not elements:
            self.app_controller.show_status("No elements defined to test")
            if update_results_callback:
                update_results_callback("No elements defined to test.")
            return {}
        
        results = {}
        
        # Clear existing markers on canvas
        canvas.delete("recognition_test")
        
        # For each element, try to find it
        for name, element in elements.items():
            self.app_controller.show_status(f"Testing element: {name}")
            
            # Skip elements without regions or references
            if not element.region or not element.reference_paths:
                results[name] = {"found": False, "reason": "Missing region or reference images"}
                continue
                
            # Try to find the element
            location = self.find_element(element)
            
            results[name] = {
                "found": location is not None,
                "location": location
            }
            
            # Draw the location on the canvas if found
            if location:
                x, y, w, h = location
                scale = getattr(self.app_controller, 'scale', 1.0)
                
                # Draw rectangle around found element
                canvas.create_rectangle(
                    x * scale, y * scale, 
                    (x + w) * scale, (y + h) * scale,
                    outline="green", width=2, tags=("recognition_test", name)
                )
                
                # Draw label
                canvas.create_text(
                    x * scale, y * scale - 10,
                    text=name,
                    fill="green", anchor=tk.W, tags=("recognition_test", name)
                )
        
        # Update results text if callback provided
        if update_results_callback:
            result_text = self._format_test_results(results)
            update_results_callback(result_text)
        
        # Return results for further processing
        return results
    
    def _format_test_results(self, results):
        """
        Format test results into a readable text summary.
        
        Args:
            results: Dictionary of test results by element name
            
        Returns:
            Formatted text summary
        """
        text = ""
        for name, result in results.items():
            status = "Found" if result.get("found", False) else "Not found"
            text += f"{name}: {status}\n"
            
            if result.get("found", False) and "location" in result:
                x, y, w, h = result["location"]
                text += f"  Location: ({x}, {y}, {w}, {h})\n\n"
            else:
                reason = result.get("reason", "Unknown reason")
                text += f"  Not found! {reason}\n\n"
        
        # Add summary
        found_count = sum(1 for r in results.values() if r.get("found", False))
        total_count = len(results)
        
        text += f"\nSummary: Found {found_count} out of {total_count} elements."
        return text
        
    def capture_click_coordinates(self, element_name, callback=None):
        """
        Capture mouse coordinates for direct clicking.
        
        Args:
            element_name: Name of the element to capture coordinates for
            callback: Optional callback function after capture
            
        Returns:
            True if coordinates captured successfully, False otherwise
        """
        import pyautogui
        
        if not element_name:
            self.app_controller.show_status("No element selected for coordinate capture")
            return False
            
        # Minimize application window
        self.app_controller.root.iconify()
        self.app_controller.show_status(f"Position mouse for {element_name} click and wait 3 seconds...")
        
        # Give user time to position mouse
        time.sleep(3)
        
        # Get mouse position
        try:
            x, y = pyautogui.position()
            
            # Update element configuration
            elements = self.element_controller.get_elements()
            if element_name in elements:
                elements[element_name].click_coordinates = (x, y)
                
                # Update element info display
                self.element_controller.update_element_info()
                
                # Restore window
                self.app_controller.root.deiconify()
                
                self.app_controller.show_status(f"Captured click coordinates for {element_name}: ({x}, {y})")
                
                # Call callback if provided
                if callback:
                    callback(element_name, x, y)
                    
                return True
            else:
                self.app_controller.root.deiconify()
                self.app_controller.show_status(f"Element {element_name} not found")
                return False
                
        except Exception as e:
            self.app_controller.root.deiconify()
            self.app_controller.show_status(f"Error capturing coordinates: {e}")
            logging.error(f"Error in capture_click_coordinates: {e}")
            return False
        
    def use_best_match_region(self, element_name=None):
        """
        Update element region based on best match from debug image.
        
        Args:
            element_name: Name of element to update, defaults to currently selected
            
        Returns:
            True if region updated successfully, False otherwise
        """
        if not element_name:
            element_name = self.element_controller.current_element
            
        if not element_name:
            self.app_controller.show_status("No element selected")
            return False
        
        if not self.current_debug_image or not self.current_debug_path:
            self.app_controller.show_status("No debug image selected")
            return False
        
        # Get match regions from current debug image
        match_regions = self.analyze_recognition_debug(self.current_debug_path)
        
        if not match_regions:
            self.app_controller.show_status("No match regions found in debug image")
            return False
            
        # Get the first (usually best) match region
        region = match_regions[0]
        
        # Convert to original image coordinates if using debug_scale
        if self.debug_scale != 1.0:
            orig_x = int(region[0] / self.debug_scale)
            orig_y = int(region[1] / self.debug_scale)
            orig_w = int(region[2] / self.debug_scale)
            orig_h = int(region[3] / self.debug_scale)
            region = (orig_x, orig_y, orig_w, orig_h)
        
        # Update element's region
        elements = self.element_controller.get_elements()
        if element_name in elements:
            elements[element_name].region = region
            
            # Update UI
            self.element_controller.update_element_info()
            
            self.app_controller.show_status(
                f"Updated region for {element_name} to best match: {region}"
            )
            return True
        
        return False
        
    def scan_debug_images(self, log_directory="logs"):
        """
        Scan for debug images in logs directory and categorize them.
        
        Args:
            log_directory: Base directory to scan for logs
            
        Returns:
            Dictionary of debug images by element name
        """
        self.debug_images = {}
        
        # Find all run directories first
        run_dirs = sorted(glob.glob(f"{log_directory}/run_*"), reverse=True)
        
        if not run_dirs:
            self.app_controller.show_status("No run directories found in logs folder")
            return self.debug_images
            
        self.app_controller.show_status(f"Found {len(run_dirs)} run directories")
        
        # Get list of defined elements
        elements = self.element_controller.get_elements()
        
        # Process each run directory
        for run_dir in run_dirs:
            screenshot_dir = os.path.join(run_dir, "screenshots")
            if not os.path.exists(screenshot_dir):
                continue
                
            # Scan all screenshots in this run directory
            for image_path in glob.glob(os.path.join(screenshot_dir, "*.png")):
                filename = os.path.basename(image_path)
                
                # Try to categorize screenshot based on filename patterns
                if "SEARCH_" in filename or "FOUND_" in filename:
                    # Recognition-related screenshot
                    # Extract element name from patterns like SEARCH_element_name_START or FOUND_element_name
                    match = re.search(r"(?:SEARCH|FOUND)_(\w+)", filename)
                    if match:
                        element_name = match.group(1)
                        if element_name not in self.debug_images:
                            self.debug_images[element_name] = {"recognition": [], "click": []}
                        self.debug_images[element_name]["recognition"].append(image_path)
                
                elif "AFTER_CLICKING" in filename or "click" in filename.lower():
                    # Click-related screenshot
                    match = re.search(r"(?:click|CLICK)_(\w+)", filename)
                    if match:
                        element_name = match.group(1)
                        if element_name not in self.debug_images:
                            self.debug_images[element_name] = {"recognition": [], "click": []}
                        self.debug_images[element_name]["click"].append(image_path)
                
                # Also look for screenshots related to specific elements
                for element_name in elements.keys():
                    if element_name.lower() in filename.lower():
                        if element_name not in self.debug_images:
                            self.debug_images[element_name] = {"recognition": [], "click": []}
                        # Categorize as recognition or click based on filename
                        if "click" in filename.lower():
                            self.debug_images[element_name]["click"].append(image_path)
                        else:
                            self.debug_images[element_name]["recognition"].append(image_path)
        
        # Show status with count of debug images found
        element_count = len(self.debug_images)
        if element_count > 0:
            self.app_controller.show_status(f"Found debug images for {element_count} UI elements")
        else:
            self.app_controller.show_status("No debug images found matching UI elements")
            
        return self.debug_images
        
    def display_debug_image(self, image_path, image_type, canvas):
        """
        Display a debug image on the provided canvas.
        
        Args:
            image_path: Path to debug image
            image_type: Type of debug image ('recognition' or 'click')
            canvas: Canvas to display on
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Store current debug image info
            self.current_debug_path = image_path
            self.current_debug_type = image_type
            
            # Load the image
            img = Image.open(image_path)
            
            # Get canvas dimensions
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            # If canvas not yet sized, use defaults
            if canvas_width <= 1:
                canvas_width = 800
                canvas_height = 600
            
            # Keep aspect ratio
            img_width, img_height = img.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Save the scale factor for coordinate conversion
            self.debug_scale = scale
            
            # Resize the image
            resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_debug_image = ImageTk.PhotoImage(resized)
            
            # Clear canvas
            canvas.delete("all")
            
            # Draw the image
            canvas.create_image(0, 0, anchor=tk.NW, image=self.current_debug_image)
            canvas.config(scrollregion=(0, 0, new_width, new_height))
            
            # Analyze the image based on its type
            self.analyze_debug_image(image_path, image_type, canvas)
            
            # Update status
            self.app_controller.show_status(f"Displaying debug image: {os.path.basename(image_path)}")
            
            return True
            
        except Exception as e:
            self.app_controller.show_status(f"Error displaying debug image: {e}")
            logging.error(f"Error in display_debug_image: {e}")
            return False
    
    def analyze_debug_image(self, image_path, image_type, canvas=None):
        """
        Analyze a debug image to extract matches and annotate the canvas.
        
        Args:
            image_path: Path to debug image
            image_type: Type of debug image ('recognition' or 'click')
            canvas: Optional canvas to draw analysis results
            
        Returns:
            Match data extracted from image
        """
        try:
            match_info = {}
            
            # Extract element name from path
            filename = os.path.basename(image_path)
            element_name = None
            
            if image_type == "recognition":
                match = re.match(r"(\w+)_matches_\d+\.png", filename)
                if match:
                    element_name = match.group(1)
                else:
                    match = re.search(r"(?:SEARCH|FOUND)_(\w+)", filename)
                    if match:
                        element_name = match.group(1)
            else:  # click
                match = re.match(r"click_(\w+)_\d+.*\.png", filename)
                if match:
                    element_name = match.group(1)
            
            match_info["element_name"] = element_name
            match_info["filename"] = filename
            match_info["type"] = image_type
            
            # Process based on image type
            if image_type == "recognition":
                regions = self.analyze_recognition_debug(image_path, canvas)
                match_info["regions"] = regions
            elif image_type == "click":
                points = self.analyze_click_debug(image_path, canvas)
                match_info["points"] = points
            
            return match_info
            
        except Exception as e:
            self.app_controller.show_status(f"Error analyzing debug image: {e}")
            logging.error(f"Error in analyze_debug_image: {e}")
            return {}
    
    def analyze_recognition_debug(self, image_path, canvas=None):
        """
        Analyze a recognition debug image to extract match regions.
        
        Args:
            image_path: Path to debug image
            canvas: Optional canvas to draw results
            
        Returns:
            List of (x, y, width, height) regions found
        """
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
                
                # Draw rectangle on canvas if provided
                if canvas:
                    canvas_x = x * self.debug_scale
                    canvas_y = y * self.debug_scale
                    canvas_w = w * self.debug_scale
                    canvas_h = h * self.debug_scale
                    
                    canvas.create_rectangle(
                        canvas_x, canvas_y, 
                        canvas_x + canvas_w, canvas_y + canvas_h,
                        outline="green", width=2, tags="match"
                    )
                    
                    # Add label
                    canvas.create_text(
                        canvas_x, canvas_y - 10,
                        text=f"Match: ({x}, {y}, {w}, {h})",
                        fill="green", anchor=tk.W, tags="match"
                    )
            
            return match_regions
            
        except Exception as e:
            logging.error(f"Error analyzing recognition debug: {e}")
            return []
    
    def analyze_click_debug(self, image_path, canvas=None):
        """
        Analyze a click debug image to extract click locations.
        
        Args:
            image_path: Path to debug image
            canvas: Optional canvas to draw results
            
        Returns:
            List of (x, y) click points found
        """
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
                            
                            # Draw marker on canvas if provided
                            if canvas:
                                canvas_x = cx * self.debug_scale
                                canvas_y = cy * self.debug_scale
                                
                                canvas.create_oval(
                                    canvas_x - 10, canvas_y - 10, 
                                    canvas_x + 10, canvas_y + 10,
                                    outline="red", width=2, tags="click"
                                )
                                
                                # Add label
                                canvas.create_text(
                                    canvas_x + 15, canvas_y,
                                    text=f"Click: ({cx}, {cy})",
                                    fill="red", anchor=tk.W, tags="click"
                                )
            
            return click_points
            
        except Exception as e:
            logging.error(f"Error analyzing click debug: {e}")
            return []
            
    def toggle_coordinate_preference(self, element_name):
        """
        Toggle whether to use coordinates first or visual recognition first.
        
        Args:
            element_name: Name of element to toggle setting for
            
        Returns:
            New preference value (True or False)
        """
        elements = self.element_controller.get_elements()
        if element_name not in elements:
            self.app_controller.show_status(f"Element {element_name} not found")
            return None
            
        # Toggle the preference
        current = elements[element_name].use_coordinates_first
        elements[element_name].use_coordinates_first = not current
        
        # Update the UI
        self.element_controller.update_element_info()
        
        self.app_controller.show_status(
            f"Set {element_name} to {'use coordinates first' if not current else 'use visual recognition first'}"
        )
        
        return not current
        
    def get_match_info_text(self, match_info):
        """
        Generate human-readable text for match information.
        
        Args:
            match_info: Dictionary of match information
            
        Returns:
            Formatted text for display
        """
        if not match_info:
            return "No match information available."
            
        info = f"Debug Image: {match_info.get('filename', 'Unknown')}\n"
        info += f"Type: {match_info.get('type', 'Unknown').capitalize()}\n"
        info += f"Element: {match_info.get('element_name', 'Unknown')}\n\n"
        
        if match_info.get('type') == "recognition":
            regions = match_info.get('regions', [])
            if regions:
                info += f"Found {len(regions)} match regions:\n"
                for i, region in enumerate(regions):
                    x, y, w, h = region
                    info += f"Match {i+1}: ({x}, {y}, {w}, {h})\n"
            else:
                info += "No match regions detected."
                
        elif match_info.get('type') == "click":
            points = match_info.get('points', [])
            if points:
                info += f"Found {len(points)} click points:\n"
                for i, point in enumerate(points):
                    x, y = point
                    info += f"Click Point {i+1}: ({x}, {y})\n"
            else:
                info += "No click points detected."
                
        return info