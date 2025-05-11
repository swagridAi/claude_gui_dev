# utils/image_processor.py

import os
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

import cv2
import pyautogui
from PIL import Image, ImageEnhance, ImageTk

class BaseImageProcessor:
    """Base class for all image processing operations in Claude Automation."""
    
    def __init__(self, config=None):
        """
        Initialize the base image processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
    
    def preprocess(self, image, options=None):
        """
        Apply sequence of processing steps based on options.
        
        Args:
            image: PIL Image or numpy array
            options: Dictionary of preprocessing options
            
        Returns:
            Processed image in the same format as input
        """
        options = options or {}
        processed = image
        
        # Determine input format and convert to numpy if needed
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            img_array = np.array(image)
        else:
            img_array = image.copy()
            
        # Apply requested processing in sequence
        if options.get("grayscale", False):
            img_array = self.convert_to_grayscale(img_array)
            
        if options.get("contrast", False):
            method = options.get("contrast_method", "clahe")
            factor = options.get("contrast_factor", 2.0)
            img_array = self.enhance_contrast(img_array, method, factor)
            
        if options.get("threshold", False):
            method = options.get("threshold_method", "adaptive")
            img_array = self.apply_threshold(img_array, method)
            
        if options.get("denoise", False):
            strength = options.get("denoise_strength", 10)
            img_array = self.denoise_image(img_array, strength)
        
        # Convert back to original format
        if is_pil:
            return Image.fromarray(img_array)
        return img_array
    
    def convert_to_grayscale(self, image):
        """
        Convert image to grayscale with format checking.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Grayscale image as numpy array
        """
        # Check if already grayscale
        if len(image.shape) == 2:
            return image
        
        # Convert to grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def enhance_contrast(self, image, method="clahe", factor=2.0):
        """
        Enhance image contrast using specified method.
        
        Args:
            image: Image as numpy array
            method: Contrast enhancement method ('clahe', 'stretch', 'simple')
            factor: Enhancement factor (for simple method)
            
        Returns:
            Contrast-enhanced image
        """
        # Ensure image is grayscale
        if len(image.shape) > 2:
            gray = self.convert_to_grayscale(image)
        else:
            gray = image.copy()
        
        if method == "clahe":
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(gray)
        
        elif method == "stretch":
            # Histogram stretching
            min_val = np.min(gray)
            max_val = np.max(gray)
            if min_val == max_val:
                return gray
            return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        else:  # Simple contrast adjustment
            # Convert back to PIL for simplicity with factor-based adjustment
            pil_img = Image.fromarray(gray)
            enhancer = ImageEnhance.Contrast(pil_img)
            enhanced = enhancer.enhance(factor)
            return np.array(enhanced)
    
    def apply_threshold(self, image, method="adaptive", block_size=11, c=2):
        """
        Apply thresholding using specified method.
        
        Args:
            image: Image as numpy array (grayscale)
            method: Thresholding method ('adaptive', 'otsu', 'binary')
            block_size: Block size for adaptive method
            c: Constant subtracted from mean for adaptive method
            
        Returns:
            Thresholded image
        """
        # Ensure image is grayscale
        if len(image.shape) > 2:
            gray = self.convert_to_grayscale(image)
        else:
            gray = image.copy()
        
        if method == "adaptive":
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, block_size, c
            )
        
        elif method == "otsu":
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        
        else:  # Simple binary threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return thresh
    
    def denoise_image(self, image, strength=10):
        """
        Apply noise reduction to image.
        
        Args:
            image: Image as numpy array
            strength: Denoising strength
            
        Returns:
            Denoised image
        """
        # Check if image is grayscale or color
        if len(image.shape) == 2:
            return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
        else:
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    
    def take_screenshot(self, region=None):
        """
        Capture a screenshot with improved error handling.
        
        Args:
            region: Optional tuple (x, y, width, height)
            
        Returns:
            PIL Image or None if failed
        """
        try:
            screenshot = pyautogui.screenshot(region=region)
            return screenshot
        except Exception as e:
            logging.error(f"Screenshot failed: {e}")
            return None
    
    def save_image(self, image, directory, filename=None, prefix=None):
        """
        Save image to file with proper directory handling.
        
        Args:
            image: PIL Image or numpy array
            directory: Directory to save to
            filename: Optional specific filename
            prefix: Optional filename prefix
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Generate filename if not provided
            if filename is None:
                timestamp = int(time.time())
                prefix = prefix or "image"
                filename = f"{prefix}_{timestamp}.png"
            
            # Form full path
            file_path = os.path.join(directory, filename)
            
            # Convert numpy to PIL if needed
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            
            # Save the image
            image.save(file_path)
            logging.debug(f"Image saved to {file_path}")
            return file_path
            
        except Exception as e:
            logging.error(f"Error saving image: {e}")
            return None


class UIImageProcessor(BaseImageProcessor):
    """Image processor specialized for UI elements and calibration."""
    
    def __init__(self, canvas=None, scale_factor=1.0, config=None):
        """
        Initialize UI image processor.
        
        Args:
            canvas: Optional Tkinter canvas for rendering
            scale_factor: Scale factor for rendering on canvas
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.canvas = canvas
        self.scale_factor = scale_factor
        self._image_references = {}  # Store references to prevent garbage collection
    
    def set_canvas(self, canvas):
        """Set the canvas for rendering."""
        self.canvas = canvas
    
    def set_scale_factor(self, scale_factor):
        """Set the scale factor for rendering."""
        self.scale_factor = scale_factor
    
    def update_canvas(self, canvas=None, image=None, scale=None):
        """
        Update a Tkinter canvas with an image at appropriate scale.
        
        Args:
            canvas: Tkinter canvas to update (uses self.canvas if None)
            image: PIL Image to display
            scale: Optional scale factor (uses self.scale_factor if None)
            
        Returns:
            The PhotoImage reference
        """
        canvas = canvas or self.canvas
        if canvas is None:
            logging.error("No canvas provided for update_canvas")
            return None
            
        scale = scale or self.scale_factor
        
        if image is None:
            return None
            
        # Get canvas dimensions
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # If canvas not yet sized, use defaults
        if canvas_width <= 1:
            canvas_width = 800
            canvas_height = 600
        
        # Resize the image to fit the canvas while maintaining aspect ratio
        img_width, img_height = image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize the image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to Tkinter PhotoImage
        photo_image = ImageTk.PhotoImage(resized)
        
        # Store reference to prevent garbage collection
        self._image_references[id(canvas)] = photo_image
        
        # Clear canvas
        canvas.delete("all")
        
        # Draw the image
        canvas.create_image(0, 0, anchor="nw", image=photo_image)
        canvas.config(scrollregion=(0, 0, new_width, new_height))
        
        # Store the scale factor for coordinate conversion
        self.scale_factor = scale
        
        return photo_image
    
    def capture_reference_image(self, source_image, region, output_dir=None, element_name=None):
        """
        Extract a region from an image to use as a reference template.
        
        Args:
            source_image: Source PIL Image
            region: Tuple (x, y, width, height) to extract
            output_dir: Optional directory to save extracted image
            element_name: Optional element name for filename
            
        Returns:
            Tuple of (extracted_image, file_path or None)
        """
        try:
            # Extract region from source image
            x, y, w, h = region
            cropped = source_image.crop((x, y, x + w, y + h))
            
            # Save if output directory provided
            file_path = None
            if output_dir:
                # Create directory if needed
                reference_dir = output_dir
                if element_name:
                    reference_dir = os.path.join(output_dir, element_name)
                
                os.makedirs(reference_dir, exist_ok=True)
                
                # Create filename
                timestamp = int(time.time())
                prefix = element_name or "reference"
                filename = f"{prefix}_{timestamp}.png"
                
                # Save image
                file_path = os.path.join(reference_dir, filename)
                cropped.save(file_path)
                logging.debug(f"Reference image saved to {file_path}")
            
            return cropped, file_path
            
        except Exception as e:
            logging.error(f"Error capturing reference image: {e}")
            return None, None
    
    def create_reference_variants(self, reference_path, output_dir=None):
        """
        Create standard variants of a reference image for improved recognition.
        
        Args:
            reference_path: Path to reference image
            output_dir: Directory to save variants (defaults to same as reference)
            
        Returns:
            List of paths to generated variants
        """
        try:
            # Load the reference image
            img = Image.open(reference_path)
            
            # Determine output directory
            if output_dir is None:
                output_dir = os.path.dirname(reference_path)
            
            # Get base filename without extension
            basename = os.path.splitext(os.path.basename(reference_path))[0]
            
            variants = [reference_path]  # Include original path
            
            # Create grayscale variant
            img_np = np.array(img)
            gray = self.convert_to_grayscale(img_np)
            gray_path = os.path.join(output_dir, f"{basename}_gray.png")
            Image.fromarray(gray).save(gray_path)
            variants.append(gray_path)
            
            # Create contrast-enhanced variant
            contrast = self.enhance_contrast(gray)
            contrast_path = os.path.join(output_dir, f"{basename}_contrast.png")
            Image.fromarray(contrast).save(contrast_path)
            variants.append(contrast_path)
            
            # Create thresholded variant
            thresh = self.apply_threshold(gray)
            thresh_path = os.path.join(output_dir, f"{basename}_thresh.png")
            Image.fromarray(thresh).save(thresh_path)
            variants.append(thresh_path)
            
            return variants
            
        except Exception as e:
            logging.error(f"Error creating reference variants: {e}")
            return [reference_path]  # Return original if processing fails
    
    def draw_regions(self, canvas, elements, current_element=None, scale=None):
        """
        Draw element regions on canvas with improved styling.
        
        Args:
            canvas: Tkinter canvas to draw on
            elements: Dictionary of element configurations
            current_element: Name of currently selected element
            scale: Optional scale factor (uses self.scale_factor if None)
        """
        if not canvas or not elements:
            return
            
        scale = scale or self.scale_factor
        
        # Clear previous regions
        canvas.delete("region")
        
        # Draw each element's region
        for name, config in elements.items():
            region = config.get("region")
            if not region:
                continue
                
            # Determine color (green for selected, blue for others)
            color = "green" if name == current_element else "blue"
            
            # Scale coordinates to canvas size
            x, y, w, h = region
            canvas_x = x * scale
            canvas_y = y * scale
            canvas_w = w * scale
            canvas_h = h * scale
            
            # Draw rectangle
            canvas.create_rectangle(
                canvas_x, canvas_y, 
                canvas_x + canvas_w, canvas_y + canvas_h,
                outline=color, width=2, tags="region"
            )
            
            # Draw label
            canvas.create_text(
                canvas_x + 5, canvas_y + 5,
                text=name, anchor="nw",
                fill=color, tags="region"
            )
    
    def draw_coordinate_markers(self, canvas, elements, scale=None):
        """
        Draw markers for click coordinates on canvas.
        
        Args:
            canvas: Tkinter canvas to draw on
            elements: Dictionary of element configurations
            scale: Optional scale factor (uses self.scale_factor if None)
        """
        if not canvas or not elements:
            return
            
        scale = scale or self.scale_factor
        
        # Clear previous markers
        canvas.delete("coordinate_marker")
        
        # Draw each element's coordinates
        for name, config in elements.items():
            if "click_coordinates" in config and config["click_coordinates"]:
                try:
                    x, y = config["click_coordinates"]
                    
                    # Scale to canvas size
                    canvas_x = x * scale
                    canvas_y = y * scale
                    
                    # Draw a crosshair
                    size = 10
                    canvas.create_line(
                        canvas_x - size, canvas_y, 
                        canvas_x + size, canvas_y, 
                        fill="red", width=2, tags="coordinate_marker"
                    )
                    canvas.create_line(
                        canvas_x, canvas_y - size, 
                        canvas_x, canvas_y + size, 
                        fill="red", width=2, tags="coordinate_marker"
                    )
                    
                    # Draw a circle
                    canvas.create_oval(
                        canvas_x - size, canvas_y - size,
                        canvas_x + size, canvas_y + size,
                        outline="red", width=2, tags="coordinate_marker"
                    )
                    
                    # Add label
                    canvas.create_text(
                        canvas_x + size + 5, canvas_y,
                        text=f"{name} ({x}, {y})",
                        fill="red", anchor="w", tags="coordinate_marker"
                    )
                except Exception as e:
                    logging.error(f"Error drawing coordinate marker for {name}: {e}")
    
    def analyze_debug_image(self, image_path, analysis_type="recognition"):
        """
        Extract information from debug screenshots with improved pattern recognition.
        
        Args:
            image_path: Path to debug image
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary with extracted information
        """
        try:
            # Load the image
            img = cv2.imread(image_path)
            
            if img is None:
                logging.error(f"Could not load image: {image_path}")
                return {"success": False, "error": "Failed to load image"}
                
            if analysis_type == "recognition":
                # Analyze recognition debug image (match regions)
                return self._analyze_recognition_debug(img, image_path)
            elif analysis_type == "click":
                # Analyze click debug image (click points)
                return self._analyze_click_debug(img, image_path)
            else:
                return {"success": False, "error": f"Unknown analysis type: {analysis_type}"}
                
        except Exception as e:
            logging.error(f"Error analyzing debug image: {e}")
            return {"success": False, "error": str(e)}
    
    def _analyze_recognition_debug(self, img, image_path):
        """
        Analyze a recognition debug image to extract match regions.
        
        Args:
            img: OpenCV image
            image_path: Original image path for reference
            
        Returns:
            Dictionary with match regions
        """
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
        
        return {
            "success": True,
            "type": "recognition",
            "matches": match_regions,
            "count": len(match_regions),
            "image_path": image_path
        }
    
    def _analyze_click_debug(self, img, image_path):
        """
        Analyze a click debug image to extract click locations.
        
        Args:
            img: OpenCV image
            image_path: Original image path for reference
            
        Returns:
            Dictionary with click points
        """
        # Extract red channel
        red_channel = img[:, :, 2]
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
        
        return {
            "success": True,
            "type": "click",
            "points": click_points,
            "count": len(click_points),
            "image_path": image_path
        }
    
    def convert_for_recognition(self, image, ui_element):
        """
        Convert an image to the optimal format for recognizing a specific element.
        
        Args:
            image: PIL Image or numpy array
            ui_element: UIElement or dict with element properties
            
        Returns:
            Processed image optimized for recognition
        """
        # Get element properties
        element_type = ui_element.get("type", "default") if isinstance(ui_element, dict) else getattr(ui_element, "type", "default")
        
        # Default preprocessing options
        options = {
            "grayscale": True,
            "contrast": True,
            "threshold": False,
            "denoise": True
        }
        
        # Customize preprocessing based on element type
        if element_type == "text":
            options["threshold"] = True
        elif element_type == "icon":
            options["contrast"] = False
        elif element_type == "button":
            options["threshold"] = True
            
        # Apply the preprocessing
        return self.preprocess(image, options)
    
    def preview_element_location(self, canvas, location, element_name=None, scale=None):
        """
        Draw highlighting to show where an element was found or will be clicked.
        
        Args:
            canvas: Tkinter canvas to draw on
            location: Tuple (x, y, width, height) of element location
            element_name: Optional name of element for labeling
            scale: Optional scale factor (uses self.scale_factor if None)
        """
        if not canvas or not location:
            return
            
        scale = scale or self.scale_factor
        
        # Clear previous highlighting
        canvas.delete("preview")
        
        # Draw rectangle around the element
        x, y, w, h = location
        canvas_x = x * scale
        canvas_y = y * scale
        canvas_w = w * scale
        canvas_h = h * scale
        
        # Draw rectangle
        canvas.create_rectangle(
            canvas_x, canvas_y, 
            canvas_x + canvas_w, canvas_y + canvas_h,
            outline="purple", width=2, tags="preview"
        )
        
        # Add label if provided
        if element_name:
            canvas.create_text(
                canvas_x, canvas_y - 10,
                text=element_name,
                fill="purple", anchor="w", tags="preview"
            )