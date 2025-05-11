#!/usr/bin/env python3
"""
Reference image preprocessing script for Claude GUI Automation.
Creates multiple variants of reference images to improve recognition reliability.
"""

import os
import cv2
import numpy as np
import logging
import time
import argparse
from pathlib import Path
from tqdm import tqdm  # For progress bars

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_directories():
    """Ensure all necessary directories exist."""
    base_dir = Path("assets/reference_images")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Standard UI elements
    ui_elements = ["prompt_box", "send_button", "thinking_indicator", "response_area"]
    for element in ui_elements:
        (base_dir / element).mkdir(exist_ok=True)
    
    return base_dir

def find_reference_images(base_dir):
    """Find all reference images in the base directory and its subdirectories."""
    image_paths = []
    
    # Walk through all subdirectories
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                # Skip already processed variants
                if any(x in file for x in ['_gray', '_edge', '_contrast', '_scale']):
                    continue
                    
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    
    return image_paths

def create_grayscale_variant(image_path):
    """Create a grayscale variant of the image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"Failed to load image: {image_path}")
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Create output path
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_gray.png"
        
        # Save the image
        cv2.imwrite(output_path, gray)
        return output_path
        
    except Exception as e:
        logging.error(f"Error creating grayscale variant for {image_path}: {e}")
        return None

def create_edge_variant(image_path):
    """Create an edge-detected variant of the image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Create output path
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_edge.png"
        
        # Save the image
        cv2.imwrite(output_path, edges)
        return output_path
        
    except Exception as e:
        logging.error(f"Error creating edge variant for {image_path}: {e}")
        return None

def create_contrast_variant(image_path):
    """Create a contrast-enhanced variant of the image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Create output path
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_contrast.png"
        
        # Save the image
        cv2.imwrite(output_path, enhanced)
        return output_path
        
    except Exception as e:
        logging.error(f"Error creating contrast variant for {image_path}: {e}")
        return None

def create_scaled_variants(image_path, scales=(0.75, 0.9, 1.1, 1.25)):
    """Create scaled variants of the image."""
    results = []
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []
            
        h, w = img.shape[:2]
        
        for scale in scales:
            try:
                # Calculate new dimensions
                new_h, new_w = int(h * scale), int(w * scale)
                
                # Skip if dimensions become too small
                if new_h < 5 or new_w < 5:
                    logging.warning(f"Skipping scale {scale} for {image_path}: too small")
                    continue
                
                # Resize the image
                scaled = cv2.resize(img, (new_w, new_h))
                
                # Create output path
                base_name = os.path.splitext(image_path)[0]
                scale_str = str(int(scale * 100))
                output_path = f"{base_name}_scale{scale_str}.png"
                
                # Save the image
                cv2.imwrite(output_path, scaled)
                results.append(output_path)
                
            except Exception as e:
                logging.error(f"Error creating scale {scale} for {image_path}: {e}")
                continue
                
        return results
        
    except Exception as e:
        logging.error(f"Error creating scaled variants for {image_path}: {e}")
        return []

def create_thresholded_variant(image_path):
    """Create a thresholded binary variant of the image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        # Create output path
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_thresh.png"
        
        # Save the image
        cv2.imwrite(output_path, thresh)
        return output_path
        
    except Exception as e:
        logging.error(f"Error creating thresholded variant for {image_path}: {e}")
        return None

def create_denoised_variant(image_path):
    """Create a denoised variant of the image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Apply denoising
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        # Create output path
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_denoised.png"
        
        # Save the image
        cv2.imwrite(output_path, denoised)
        return output_path
        
    except Exception as e:
        logging.error(f"Error creating denoised variant for {image_path}: {e}")
        return None

def process_all_images(image_paths, methods):
    """Process all images with the specified methods."""
    results = {}
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        results[image_path] = []
        
        # Apply each method
        for method_name, method_func in methods.items():
            try:
                tqdm.write(f"Applying {method_name} to {os.path.basename(image_path)}")
                result = method_func(image_path)
                
                if result:
                    if isinstance(result, list):
                        results[image_path].extend(result)
                    else:
                        results[image_path].append(result)
                        
            except Exception as e:
                logging.error(f"Error applying {method_name} to {image_path}: {e}")
                
    return results

def update_config_file(preprocessed_images):
    """Optionally update the configuration file with new reference images."""
    try:
        # Try to import the config manager
        import sys
        sys.path.append('.')  # Add project root to path
        
        from src.utils.config_manager import ConfigManager
        
        # Load configuration
        config = ConfigManager()
        ui_elements = config.get("ui_elements", {})
        
        # Update reference paths for each element
        for element_name, element_config in ui_elements.items():
            if "reference_paths" in element_config:
                # Find all new references for this element
                new_refs = []
                element_dir = f"assets/reference_images/{element_name}"
                
                for original_path in element_config["reference_paths"]:
                    if original_path in preprocessed_images:
                        new_refs.extend(preprocessed_images[original_path])
                
                # Add new references, avoiding duplicates
                for ref in new_refs:
                    if ref not in element_config["reference_paths"]:
                        element_config["reference_paths"].append(ref)
                
        # Save updated config
        config.save()
        logging.info("Updated configuration with new reference images")
        
    except Exception as e:
        logging.error(f"Error updating configuration: {e}")
        logging.info("Configuration not updated automatically")

def main():
    """Main function to run the preprocessing script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Preprocess reference images for GUI automation")
    parser.add_argument("--update-config", action="store_true", help="Update configuration with new variants")
    parser.add_argument("--grayscale", action="store_true", help="Create grayscale variants")
    parser.add_argument("--edge", action="store_true", help="Create edge detection variants")
    parser.add_argument("--contrast", action="store_true", help="Create contrast enhanced variants")
    parser.add_argument("--scale", action="store_true", help="Create scaled variants")
    parser.add_argument("--threshold", action="store_true", help="Create thresholded variants")
    parser.add_argument("--denoise", action="store_true", help="Create denoised variants")
    parser.add_argument("--all", action="store_true", help="Apply all preprocessing methods")
    args = parser.parse_args()
    
    # Default to all methods if none specified
    if not (args.grayscale or args.edge or args.contrast or args.scale or 
            args.threshold or args.denoise or args.all):
        args.all = True
    
    # Set up base directory
    base_dir = setup_directories()
    
    # Find all reference images
    logging.info("Finding reference images...")
    image_paths = find_reference_images(base_dir)
    logging.info(f"Found {len(image_paths)} reference images")
    
    # Skip if no images found
    if not image_paths:
        logging.warning("No reference images found. Exiting.")
        return
    
    # Set up methods based on arguments
    methods = {}
    
    if args.all or args.grayscale:
        methods["grayscale"] = create_grayscale_variant
    
    if args.all or args.edge:
        methods["edge detection"] = create_edge_variant
    
    if args.all or args.contrast:
        methods["contrast enhancement"] = create_contrast_variant
    
    if args.all or args.scale:
        methods["scaling"] = create_scaled_variants
    
    if args.all or args.threshold:
        methods["thresholding"] = create_thresholded_variant
    
    if args.all or args.denoise:
        methods["denoising"] = create_denoised_variant
    
    # Process all images
    logging.info(f"Starting preprocessing with methods: {', '.join(methods.keys())}")
    start_time = time.time()
    
    results = process_all_images(image_paths, methods)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Count total variants created
    total_variants = sum(len(variants) for variants in results.values())
    logging.info(f"Created {total_variants} variants from {len(image_paths)} original images")
    logging.info(f"Preprocessing completed in {duration:.2f} seconds")
    
    # Update configuration if requested
    if args.update_config:
        logging.info("Updating configuration with new reference images...")
        update_config_file(results)

if __name__ == "__main__":
    main()