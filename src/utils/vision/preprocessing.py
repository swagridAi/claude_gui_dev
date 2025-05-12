#!/usr/bin/env python3
"""
Image preprocessing utilities for improving recognition accuracy.
"""

import cv2
import numpy as np
import logging
from PIL import Image, ImageEnhance
from typing import Union, Optional, Dict, Any


def enhance_contrast(image: Union[Image.Image, np.ndarray], factor: float = 2.0) -> Union[Image.Image, np.ndarray]:
    """
    Enhance image contrast for better recognition.
    
    Args:
        image: PIL Image or numpy array
        factor: Contrast enhancement factor
        
    Returns:
        Enhanced image in same format as input
    """
    if isinstance(image, Image.Image):
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    else:
        # For numpy arrays, use OpenCV
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl, a, b))
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)


def denoise(image: Union[Image.Image, np.ndarray], strength: int = 10) -> Union[Image.Image, np.ndarray]:
    """
    Remove noise from image.
    
    Args:
        image: PIL Image or numpy array
        strength: Denoising strength (1-30)
        
    Returns:
        Denoised image in same format as input
    """
    if isinstance(image, Image.Image):
        # Convert to numpy for processing
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(img_array, None, strength, strength, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(img_array, None, strength, 7, 21)
        return Image.fromarray(denoised)
    else:
        # Already numpy array
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)


def apply_threshold(image: Union[Image.Image, np.ndarray], method: str = 'adaptive') -> np.ndarray:
    """
    Apply thresholding to an image.
    
    Args:
        image: PIL Image or numpy array
        method: Thresholding method ('adaptive', 'otsu', 'binary')
        
    Returns:
        Thresholded image as numpy array
    """
    # Convert to grayscale numpy array if needed
    if isinstance(image, Image.Image):
        img = np.array(image.convert('L'))
    elif len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image
    
    if method == 'adaptive':
        return cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == 'otsu':
        _, thresholded = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded
    elif method == 'binary':
        _, thresholded = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return thresholded
    else:
        logging.warning(f"Unknown threshold method: {method}, using adaptive")
        return cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )


def preprocess_image(image: Union[Image.Image, np.ndarray], 
                    operations: Optional[Dict[str, Any]] = None) -> Union[Image.Image, np.ndarray]:
    """
    Apply multiple preprocessing operations to an image.
    
    Args:
        image: PIL Image or numpy array
        operations: Dictionary of operations and their parameters
        
    Returns:
        Processed image in same format as input
    """
    if operations is None:
        operations = {
            'enhance_contrast': {'factor': 2.0},
            'denoise': {'strength': 10},
            'threshold': {'method': 'adaptive'}
        }
    
    result = image
    
    # Apply operations in order
    if 'enhance_contrast' in operations:
        result = enhance_contrast(result, **operations['enhance_contrast'])
    
    if 'denoise' in operations:
        result = denoise(result, **operations['denoise'])
    
    if 'threshold' in operations:
        # Note: threshold always returns numpy array
        was_pil = isinstance(result, Image.Image)
        result = apply_threshold(result, **operations['threshold'])
        if was_pil:
            result = Image.fromarray(result)
    
    return result


def create_grayscale_variants(image: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
    """
    Create grayscale version of an image.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        Grayscale image in same format as input
    """
    if isinstance(image, Image.Image):
        return image.convert('L')
    else:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return image  # Already grayscale


def detect_edges(image: Union[Image.Image, np.ndarray], 
                low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Detect edges in an image using Canny edge detection.
    
    Args:
        image: PIL Image or numpy array
        low_threshold: Lower threshold for edge linking
        high_threshold: Upper threshold for initial edge detection
        
    Returns:
        Edge image as numpy array
    """
    # Convert to grayscale numpy array if needed
    if isinstance(image, Image.Image):
        img = np.array(image.convert('L'))
    elif len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image
    
    return cv2.Canny(img, low_threshold, high_threshold)


def create_image_variants(image_path: str, scales: tuple = (0.75, 0.9, 1.1, 1.25),
                         preprocessors: Optional[list] = None) -> list:
    """
    Create multiple variants of an image for template matching.
    
    Args:
        image_path: Path to source image
        scales: Scale factors to apply
        preprocessors: List of preprocessing operations
        
    Returns:
        List of variant image paths
    """
    if not os.path.exists(image_path):
        logging.error(f"Source image not found: {image_path}")
        return []
    
    variants = [image_path]  # Include original
    base_name = os.path.splitext(image_path)[0]
    base_dir = os.path.dirname(image_path)
    
    try:
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Failed to load image: {image_path}")
            return variants
        
        # Create scaled variants
        for scale in scales:
            variant_path = f"{base_name}_s{int(scale*100)}.png"
            
            # Skip if already exists and recent
            if os.path.exists(variant_path) and \
               (time.time() - os.path.getmtime(variant_path) < 86400):
                variants.append(variant_path)
                continue
            
            # Resize image
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            if new_h < 5 or new_w < 5:
                continue  # Skip if too small
            
            scaled_img = cv2.resize(img, (new_w, new_h))
            cv2.imwrite(variant_path, scaled_img)
            variants.append(variant_path)
        
        # Create preprocessed variants
        if preprocessors:
            for processor_name in preprocessors:
                variant_path = f"{base_name}_{processor_name}.png"
                
                if os.path.exists(variant_path) and \
                   (time.time() - os.path.getmtime(variant_path) < 86400):
                    variants.append(variant_path)
                    continue
                
                processed_img = globals()[processor_name](img)
                if isinstance(processed_img, Image.Image):
                    processed_img.save(variant_path)
                else:
                    cv2.imwrite(variant_path, processed_img)
                variants.append(variant_path)
    
    except Exception as e:
        logging.error(f"Error creating image variants: {e}")
    
    return variants