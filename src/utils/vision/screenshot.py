#!/usr/bin/env python3
"""
Screenshot handling utilities for vision operations.
"""

import os
import time
import logging
from datetime import datetime
from typing import Optional, Tuple, Union, Dict, Any
from PIL import Image
import pyautogui
import cv2
import numpy as np


def take_screenshot(region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
    """
    Take a screenshot of specified region or full screen.
    
    Args:
        region: Optional tuple (x, y, width, height)
        
    Returns:
        PIL Image object
    """
    return pyautogui.screenshot(region=region)


def save_screenshot(image: Image.Image, name: str, directory: str = "logs/screenshots") -> Optional[str]:
    """
    Save screenshot with standardized naming and location.
    
    Args:
        image: PIL Image to save
        name: Base name for file
        directory: Directory to save in
        
    Returns:
        Path to saved file or None if failed
    """
    try:
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{name}_{timestamp}.png"
        filepath = os.path.join(directory, filename)
        image.save(filepath)
        logging.debug(f"Screenshot saved to {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Failed to save screenshot: {e}")
        return None


class ScreenshotManager:
    """Manages screenshot operations with caching and comparison capabilities."""
    
    def __init__(self):
        self.screenshot_cache: Dict[str, Image.Image] = {}
        self.timestamp_cache: Dict[str, float] = {}
        self.cache_ttl = 60.0  # Default cache time-to-live in seconds
    
    def take_and_cache(self, name: str, region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        """Take a screenshot and cache it with a name."""
        screenshot = take_screenshot(region)
        self.screenshot_cache[name] = screenshot
        self.timestamp_cache[name] = time.time()
        return screenshot
    
    def get_cached(self, name: str) -> Optional[Image.Image]:
        """Get a cached screenshot if it exists and hasn't expired."""
        if name not in self.screenshot_cache:
            return None
            
        if time.time() - self.timestamp_cache.get(name, 0) > self.cache_ttl:
            # Cache expired
            self.clear_cached(name)
            return None
            
        return self.screenshot_cache[name]
    
    def clear_cached(self, name: str):
        """Remove a specific cached screenshot."""
        self.screenshot_cache.pop(name, None)
        self.timestamp_cache.pop(name, None)
    
    def clear_all_cache(self):
        """Clear all cached screenshots."""
        self.screenshot_cache.clear()
        self.timestamp_cache.clear()
    
    def compare_images(self, image1: Image.Image, image2: Image.Image, 
                      threshold: float = 0.1) -> Tuple[bool, float]:
        """
        Compare two images and return if they're different.
        
        Args:
            image1: First image to compare
            image2: Second image to compare
            threshold: Difference threshold (0-1)
            
        Returns:
            Tuple of (is_different, difference_percentage)
        """
        # Convert to numpy arrays
        arr1 = np.array(image1)
        arr2 = np.array(image2)
        
        # Images must be same size
        if arr1.shape != arr2.shape:
            return True, 1.0
        
        # Calculate difference
        diff = np.abs(arr1.astype(float) - arr2.astype(float))
        total_diff = np.sum(diff)
        max_possible_diff = arr1.size * 255
        difference_percentage = total_diff / max_possible_diff
        
        return difference_percentage > threshold, difference_percentage