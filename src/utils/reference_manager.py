import os
import time
import cv2
import numpy as np
import logging
from pathlib import Path
from PIL import Image, ImageEnhance
import pyautogui

class ReferenceImageManager:
    """
    Manages reference images for UI element detection.
    Provides functionality for finding, processing, creating variants,
    and updating reference images to improve recognition reliability.
    """
    
    def __init__(self, base_directory="assets/reference_images"):
        """
        Initialize the reference image manager.
        
        Args:
            base_directory: Base directory for reference images
        """
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)
        self.cache = {}  # Cache for processed images
        self.cache_expiry = 60 * 60  # 1 hour cache expiry
        
        # Create standard element directories
        self._ensure_element_directories()
        
        logging.info(f"Reference image manager initialized with base directory: {self.base_directory}")
    
    def _ensure_element_directories(self):
        """Create standard element directories if they don't exist."""
        standard_elements = ["prompt_box", "send_button", "thinking_indicator", "response_area"]
        for element in standard_elements:
            element_dir = self.base_directory / element
            element_dir.mkdir(exist_ok=True)
    
    def get_reference_paths(self, element_name):
        """
        Get all reference image paths for an element.
        
        Args:
            element_name: Name of the UI element
            
        Returns:
            List of paths to reference images
        """
        element_dir = self.base_directory / element_name
        if not element_dir.exists():
            logging.warning(f"No reference directory found for {element_name}")
            return []
            
        # Get all PNG files in the directory
        paths = sorted(str(p) for p in element_dir.glob("*.png"))
        
        if not paths:
            logging.warning(f"No reference images found for {element_name}")
        else:
            logging.debug(f"Found {len(paths)} reference images for {element_name}")
            
        return paths
    
    def create_scaled_variants(self, source_path, scales=(0.75, 0.9, 1.1, 1.25)):
        """
        Create scaled variants of a reference image.
        
        Args:
            source_path: Path to source image
            scales: Tuple of scale factors to create
            
        Returns:
            List of paths to all variants (including original)
        """
        if not os.path.exists(source_path):
            logging.error(f"Source image not found: {source_path}")
            return []
            
        try:
            # Load the source image
            img = cv2.imread(source_path)
            if img is None:
                logging.error(f"Failed to load image: {source_path}")
                return []
                
            # Get the filename without extension
            basename = os.path.splitext(os.path.basename(source_path))[0]
            parent_dir = os.path.dirname(source_path)
            
            # Create scaled variants
            paths = [source_path]  # Include the original
            for scale in scales:
                scaled_path = os.path.join(parent_dir, f"{basename}_s{int(scale*100)}.png")
                
                # Skip if it already exists and is recent
                if os.path.exists(scaled_path) and (time.time() - os.path.getmtime(scaled_path) < 86400):
                    paths.append(scaled_path)
                    continue
                    
                # Resize the image
                h, w = img.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                
                # Skip if dimensions are too small
                if new_h < 5 or new_w < 5:
                    logging.debug(f"Skipping scale {scale} for {source_path}: dimensions too small")
                    continue
                
                scaled_img = cv2.resize(img, (new_w, new_h))
                
                # Save the scaled variant
                cv2.imwrite(scaled_path, scaled_img)
                paths.append(scaled_path)
                logging.debug(f"Created scaled variant at {scale}x: {scaled_path}")
                
            return paths
            
        except Exception as e:
            logging.error(f"Error creating scaled variants: {e}")
            return [source_path]  # Return just the original
    
    def capture_reference_image(self, element_name, region=None, name_suffix=""):
        """
        Capture a new reference image for an element.
        
        Args:
            element_name: Name of the UI element
            region: Screen region to capture (x, y, width, height)
            name_suffix: Optional suffix for the filename
            
        Returns:
            Path to the captured image
        """
        try:
            # Create the element directory
            element_dir = self.base_directory / element_name
            element_dir.mkdir(exist_ok=True)
            
            # Generate a filename with timestamp
            timestamp = int(time.time())
            filename = f"{element_name}{name_suffix}_{timestamp}.png"
            filepath = str(element_dir / filename)
            
            # Capture the screenshot
            screenshot = pyautogui.screenshot(region=region)
            screenshot.save(filepath)
            
            logging.info(f"Captured reference image for {element_name}: {filepath}")
            
            # Create scaled variants
            self.create_scaled_variants(filepath)
            
            return filepath
            
        except Exception as e:
            logging.error(f"Error capturing reference image: {e}")
            return None
    
    def preprocess_reference_images(self, paths, preprocessing_options=None):
        """
        Preprocess reference images with various techniques to enhance recognition.
        
        Args:
            paths: List of paths to reference images
            preprocessing_options: Dictionary of preprocessing options
            
        Returns:
            List of paths to all variants (including originals)
        """
        if not preprocessing_options:
            preprocessing_options = {
                "grayscale": True,
                "contrast_enhance": True,
                "edge_detection": False,  # Keep edge detection off by default
                "threshold": True
            }
            
        all_paths = list(paths)  # Copy the original list
        
        for path in paths:
            if not os.path.exists(path):
                logging.warning(f"Reference image not found: {path}")
                continue
                
            basename = os.path.splitext(os.path.basename(path))[0]
            parent_dir = os.path.dirname(path)
            
            try:
                img = cv2.imread(path)
                if img is None:
                    logging.warning(f"Failed to load image: {path}")
                    continue
                
                # Apply preprocessing techniques
                if preprocessing_options.get("grayscale"):
                    gray_path = os.path.join(parent_dir, f"{basename}_gray.png")
                    
                    # Skip if it already exists and is recent
                    if os.path.exists(gray_path) and (time.time() - os.path.getmtime(gray_path) < 86400):
                        all_paths.append(gray_path)
                        continue
                        
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(gray_path, gray)
                    all_paths.append(gray_path)
                    logging.debug(f"Created grayscale variant: {gray_path}")
                
                if preprocessing_options.get("contrast_enhance"):
                    contrast_path = os.path.join(parent_dir, f"{basename}_contrast.png")
                    
                    # Skip if it already exists and is recent
                    if os.path.exists(contrast_path) and (time.time() - os.path.getmtime(contrast_path) < 86400):
                        all_paths.append(contrast_path)
                        continue
                        
                    # Convert to LAB color space for better contrast enhancement
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    cl = clahe.apply(l)
                    limg = cv2.merge((cl, a, b))
                    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                    
                    cv2.imwrite(contrast_path, enhanced)
                    all_paths.append(contrast_path)
                    logging.debug(f"Created contrast-enhanced variant: {contrast_path}")
                
                if preprocessing_options.get("edge_detection"):
                    edge_path = os.path.join(parent_dir, f"{basename}_edge.png")
                    
                    # Skip if it already exists and is recent
                    if os.path.exists(edge_path) and (time.time() - os.path.getmtime(edge_path) < 86400):
                        all_paths.append(edge_path)
                        continue
                        
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    cv2.imwrite(edge_path, edges)
                    all_paths.append(edge_path)
                    logging.debug(f"Created edge-detected variant: {edge_path}")
                
                if preprocessing_options.get("threshold"):
                    thresh_path = os.path.join(parent_dir, f"{basename}_thresh.png")
                    
                    # Skip if it already exists and is recent
                    if os.path.exists(thresh_path) and (time.time() - os.path.getmtime(thresh_path) < 86400):
                        all_paths.append(thresh_path)
                        continue
                        
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.adaptiveThreshold(
                        gray, 
                        255, 
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY, 
                        11, 
                        2
                    )
                    cv2.imwrite(thresh_path, thresh)
                    all_paths.append(thresh_path)
                    logging.debug(f"Created thresholded variant: {thresh_path}")
                    
            except Exception as e:
                logging.warning(f"Error preprocessing {path}: {e}")
                
        return all_paths
    
    def verify_references(self, ui_element, screen=None):
        """
        Verify if reference images are still valid by testing recognition.
        
        Args:
            ui_element: UIElement object to test
            screen: Optional screenshot to use (to avoid taking multiple screenshots)
            
        Returns:
            True if at least one reference is valid, False otherwise
        """
        # Import here to avoid circular imports
        from src.automation.recognition import find_element
        
        # Try to find the element
        result = find_element(ui_element)
        return result is not None
    
    def update_stale_references(self, ui_element, config, preserve_sessions=False):
        """
        Update stale reference images that are no longer recognized.
        
        Args:
            ui_element: UIElement object to update
            config: Configuration manager object
            preserve_sessions: Whether to preserve sessions when saving
            
        Returns:
            True if update successful, False otherwise
        """
        # Import here to avoid circular imports
        from src.automation.recognition import find_element
        
        # Try to find the element first
        location = find_element(ui_element)
        if location:
            # Element found, no need to update
            return True
            
        logging.warning(f"Reference images for {ui_element.name} may be stale. Attempting automatic update.")
        
        # If element has a region defined, capture a new reference in that region
        if ui_element.region:
            # Expand region slightly to ensure we capture the whole element
            x, y, w, h = ui_element.region
            expanded_region = (max(0, x-10), max(0, y-10), w+20, h+20)
            
            # Capture a new reference image
            new_ref = self.capture_reference_image(ui_element.name, expanded_region, "_auto")
            if new_ref:
                # Add the new reference to the element's references
                ui_element.reference_paths.append(new_ref)
                
                # Create variants of the new reference
                variants = self.preprocess_reference_images([new_ref])
                for variant in variants:
                    if variant != new_ref:  # Skip the original which we already added
                        ui_element.reference_paths.append(variant)
                
                # Update config
                element_config = config.get("ui_elements", {}).get(ui_element.name, {})
                if "reference_paths" in element_config:
                    # Add new references to config
                    for variant in variants:
                        if variant not in element_config["reference_paths"]:
                            element_config["reference_paths"].append(variant)
                    
                    # Save config with session preservation if requested
                    if preserve_sessions and not config.is_in_session_mode():
                        config.enter_session_mode()
                        config.save()
                        config.exit_session_mode()
                    else:
                        # Use standard save method (respects session mode if active)
                        config.save()
                    
                logging.info(f"Added new reference images for {ui_element.name}")
                return True
                
        return False

    def images_need_preprocessing(self, paths, preprocessing_options=None):
        """
        Check if reference images need preprocessing.
        
        Args:
            paths: List of paths to reference images
            preprocessing_options: Dictionary of preprocessing options
            
        Returns:
            True if any images need preprocessing, False otherwise
        """
        if not preprocessing_options:
            preprocessing_options = {
                "grayscale": True,
                "contrast_enhance": True,
                "edge_detection": False,
                "threshold": True
            }
        
        for path in paths:
            if not os.path.exists(path):
                continue
            
            # Only check original images (not variant images)
            if any(x in path for x in ['_gray', '_edge', '_contrast', '_thresh']):
                continue
                
            basename = os.path.splitext(os.path.basename(path))[0]
            parent_dir = os.path.dirname(path)
            
            # Check for each preprocessing type
            if preprocessing_options.get("grayscale"):
                gray_path = os.path.join(parent_dir, f"{basename}_gray.png")
                if not os.path.exists(gray_path) or (time.time() - os.path.getmtime(gray_path) >= 86400):
                    return True
            
            if preprocessing_options.get("contrast_enhance"):
                contrast_path = os.path.join(parent_dir, f"{basename}_contrast.png")
                if not os.path.exists(contrast_path) or (time.time() - os.path.getmtime(contrast_path) >= 86400):
                    return True
            
            if preprocessing_options.get("edge_detection"):
                edge_path = os.path.join(parent_dir, f"{basename}_edge.png")
                if not os.path.exists(edge_path) or (time.time() - os.path.getmtime(edge_path) >= 86400):
                    return True
            
            if preprocessing_options.get("threshold"):
                thresh_path = os.path.join(parent_dir, f"{basename}_thresh.png")
                if not os.path.exists(thresh_path) or (time.time() - os.path.getmtime(thresh_path) >= 86400):
                    return True
        
        # If we get here, all preprocessing variants exist and are recent
        return False

    # Update to ensure_preprocessing method in src/utils/reference_manager.py

    def ensure_preprocessing(self, ui_elements_config, config=None, preserve_sessions=False):
        """
        Ensure all reference images have been preprocessed.
        Only processes images that need it.
        
        Args:
            ui_elements_config: Dictionary of UI element configurations
            config: ConfigManager instance for saving updates
            preserve_sessions: Whether to preserve sessions when saving
            
        Returns:
            True if any preprocessing was done, False otherwise
        """
        any_preprocessing_done = False
        
        for element_name, element_config in ui_elements_config.items():
            if "reference_paths" in element_config and element_config["reference_paths"]:
                paths = element_config["reference_paths"]
                if self.images_need_preprocessing(paths):
                    logging.info(f"Preprocessing images for {element_name}...")
                    enhanced_paths = self.preprocess_reference_images(paths)
                    element_config["reference_paths"] = enhanced_paths
                    any_preprocessing_done = True
        
        # Save configuration if changes were made and config is provided
        if any_preprocessing_done and config:
            # Check if config is already in session mode or we explicitly want to preserve sessions
            if preserve_sessions and not config.is_in_session_mode():
                config.enter_session_mode()
                config.save()
                config.exit_session_mode()
            else:
                # Use standard save method (which will respect session mode if active)
                config.save()
            logging.info("Saved updated configuration with preprocessed image paths")
        
        return any_preprocessing_done

    def refresh_all_references(self, ui_elements, config, preserve_sessions=False):
        """
        Refresh all reference images for all UI elements.
        
        Args:
            ui_elements: Dictionary of UIElement objects
            config: Configuration manager object
            preserve_sessions: Whether to preserve sessions when saving
            
        Returns:
            Dictionary of update results by element name
        """
        results = {}
        
        for name, element in ui_elements.items():
            logging.info(f"Refreshing references for {name}")
            
            # Verify if element can be found with current references
            found = self.verify_references(element)
            
            if found:
                logging.info(f"Element {name} can be found with current references")
                results[name] = True
                continue
                
            # Try to update the element with session preservation
            updated = self.update_stale_references(element, config, preserve_sessions)
            results[name] = updated
            
            if updated:
                logging.info(f"Updated references for {name}")
            else:
                logging.warning(f"Failed to update references for {name}")
                
        return results

    def cleanup_old_variants(self, max_age_days=30):
        """
        Clean up old variant images that haven't been accessed recently.
        
        Args:
            max_age_days: Maximum age in days for variant files
            
        Returns:
            Number of files cleaned up
        """
        cleanup_count = 0
        max_age_seconds = max_age_days * 24 * 60 * 60
        current_time = time.time()
        
        # Walk through all reference image directories
        for root, _, files in os.walk(self.base_directory):
            for file in files:
                # Only process variant files (those with underscore in name)
                if "_" in file and (file.endswith(".png") or file.endswith(".jpg")):
                    full_path = os.path.join(root, file)
                    
                    # Check if file is a preprocessed variant
                    is_variant = any(suffix in file for suffix in 
                                    ["_gray", "_edge", "_contrast", "_thresh", "_s", "_scale"])
                    
                    if is_variant:
                        # Check file age (both modification and access time)
                        mtime = os.path.getmtime(full_path)
                        atime = os.path.getatime(full_path)
                        newest_time = max(mtime, atime)
                        
                        if current_time - newest_time > max_age_seconds:
                            try:
                                os.remove(full_path)
                                cleanup_count += 1
                                logging.debug(f"Cleaned up old variant: {full_path}")
                            except Exception as e:
                                logging.warning(f"Failed to clean up {full_path}: {e}")
                                
        logging.info(f"Cleaned up {cleanup_count} old variant files")
        return cleanup_count