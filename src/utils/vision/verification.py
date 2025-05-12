#!/usr/bin/env python3
"""
Visual verification utilities for detecting UI state changes.
"""

import time
import logging
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import numpy as np
from .screenshot import ScreenshotManager


class VisualVerifier:
    """Unified visual verification for detecting UI changes."""
    
    def __init__(self, cache_ttl: float = 60.0):
        """
        Initialize the verifier.
        
        Args:
            cache_ttl: Time-to-live for cached reference images in seconds
        """
        self.screenshot_manager = ScreenshotManager()
        self.screenshot_manager.cache_ttl = cache_ttl
        self.reference_images: Dict[str, Dict[str, Any]] = {}
    
    def take_reference(self, name: str, region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        """
        Take a reference screenshot for later comparison.
        
        Args:
            name: Reference name
            region: Optional region to capture
            
        Returns:
            Captured image
        """
        image = self.screenshot_manager.take_and_cache(name, region)
        self.reference_images[name] = {
            'region': region,
            'timestamp': time.time()
        }
        return image
    
    def check_for_changes(self, reference_name: str, 
                         region: Optional[Tuple[int, int, int, int]] = None,
                         threshold: float = 0.1) -> Tuple[bool, float]:
        """
        Check if the screen has changed compared to a reference.
        
        Args:
            reference_name: Name of the reference image
            region: Region to check (uses reference's region if None)
            threshold: Difference threshold (0-1)
            
        Returns:
            Tuple of (has_changed, difference_percentage)
        """
        # Get reference image
        reference_image = self.screenshot_manager.get_cached(reference_name)
        if reference_image is None:
            # Take new reference if not found
            self.take_reference(reference_name, region)
            return False, 0.0
        
        # Use stored region if none provided
        if region is None and reference_name in self.reference_images:
            region = self.reference_images[reference_name]['region']
        
        # Take current screenshot
        current_image = self.screenshot_manager.take_and_cache(f"{reference_name}_current", region)
        
        # Compare images
        has_changed, diff_pct = self.screenshot_manager.compare_images(
            reference_image, current_image, threshold
        )
        
        if has_changed:
            # Update reference to current image for next comparison
            self.screenshot_manager.screenshot_cache[reference_name] = current_image
            self.screenshot_manager.timestamp_cache[reference_name] = time.time()
            
            logging.debug(f"Visual change detected for {reference_name}: {diff_pct:.4f}")
        
        return has_changed, diff_pct
    
    def wait_for_visual_change(self, reference_name: str, 
                              region: Optional[Tuple[int, int, int, int]] = None,
                              timeout: float = 60.0,
                              check_interval: float = 0.5,
                              threshold: float = 0.1) -> bool:
        """
        Wait for visual changes to occur within timeout period.
        
        Args:
            reference_name: Name for the reference image
            region: Region to monitor
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds
            threshold: Difference threshold to detect change
            
        Returns:
            True if change detected, False on timeout
        """
        # Take initial reference if needed
        if reference_name not in self.reference_images:
            self.take_reference(reference_name, region)
        
        start_time = time.time()
        end_time = start_time + timeout
        
        logging.info(f"Waiting up to {timeout}s for visual change: {reference_name}")
        
        while time.time() < end_time:
            has_changed, diff_pct = self.check_for_changes(reference_name, region, threshold)
            
            if has_changed:
                elapsed = time.time() - start_time
                logging.info(f"Visual change detected after {elapsed:.2f}s (diff: {diff_pct:.4f})")
                return True
            
            # Log progress periodically
            elapsed = time.time() - start_time
            if elapsed > 0 and elapsed % 30 < check_interval:
                remaining = timeout - elapsed
                logging.info(f"Still waiting for change: {int(remaining)}s remaining...")
            
            time.sleep(check_interval)
        
        logging.warning(f"Timeout waiting for visual change: {reference_name}")
        return False
    
    def wait_for_stability(self, reference_name: str,
                          region: Optional[Tuple[int, int, int, int]] = None,
                          stability_period: float = 3.0,
                          check_interval: float = 0.5,
                          threshold: float = 0.1,
                          timeout: float = 30.0) -> bool:
        """
        Wait for visual stability (no changes for a period).
        
        Args:
            reference_name: Name for the reference image
            region: Region to monitor
            stability_period: How long to wait without changes
            check_interval: Time between checks
            threshold: Difference threshold
            timeout: Maximum time to wait
            
        Returns:
            True if stability achieved, False on timeout
        """
        stable_since = None
        start_time = time.time()
        end_time = start_time + timeout
        
        logging.info(f"Waiting for {stability_period}s of visual stability: {reference_name}")
        
        while time.time() < end_time:
            has_changed, _ = self.check_for_changes(reference_name, region, threshold)
            
            if has_changed:
                # Reset stability timer
                stable_since = None
            else:
                # No change detected
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= stability_period:
                    elapsed = time.time() - start_time
                    logging.info(f"Visual stability achieved after {elapsed:.2f}s")
                    return True
            
            time.sleep(check_interval)
        
        logging.warning(f"Timeout waiting for visual stability: {reference_name}")
        return False
    
    def verify_ui_state(self, expected_state: str,
                       region: Optional[Tuple[int, int, int, int]] = None,
                       timeout: float = 10.0) -> bool:
        """
        Verify that the UI matches an expected state.
        
        Args:
            expected_state: Description of expected state
            region: Region to check
            timeout: Maximum time to verify
            
        Returns:
            True if verification successful
        """
        reference_name = f"ui_state_{expected_state}"
        
        try:
            # Wait for visual stabilization
            return self.wait_for_stability(
                reference_name, 
                region, 
                stability_period=2.0,
                timeout=timeout
            )
        except Exception as e:
            logging.error(f"Error verifying UI state '{expected_state}': {e}")
            return False
    
    def clear_cache(self):
        """Clear all cached references."""
        self.screenshot_manager.clear_all_cache()
        self.reference_images.clear()