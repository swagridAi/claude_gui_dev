#!/usr/bin/env python3
"""
Template matching utilities for finding UI elements.
"""

import os
import logging
from typing import List, Optional, Tuple, Dict, Any, Union
import cv2
import numpy as np
import pyautogui
from PIL import Image


class TemplateMatcher:
    """Unified template matching with multiple methods and adaptive confidence."""
    
    def __init__(self):
        self.methods = [
            cv2.TM_CCOEFF_NORMED,
            cv2.TM_CCORR_NORMED,
            cv2.TM_SQDIFF_NORMED
        ]
        self.scale_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
        
    def find_element(self, template_paths: List[str], screenshot: Union[str, Image.Image, np.ndarray],
                    confidence: float = 0.7, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Find a template in a screenshot using multiple methods.
        
        Args:
            template_paths: List of paths to template images
            screenshot: Screenshot as path, PIL Image, or numpy array
            confidence: Minimum confidence threshold (0-1)
            region: Optional region to search in
            
        Returns:
            Location tuple (x, y, width, height) or None if not found
        """
        # Convert screenshot to numpy array if needed
        if isinstance(screenshot, str):
            screenshot_cv = cv2.imread(screenshot)
        elif isinstance(screenshot, Image.Image):
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        else:
            screenshot_cv = screenshot
            
        # Apply region if specified
        if region:
            x, y, w, h = region
            screenshot_cv = screenshot_cv[y:y+h, x:x+w]
            offset = (x, y)
        else:
            offset = (0, 0)
        
        best_match = None
        best_score = 0
        best_method = None
        
        # Try each template
        for template_path in template_paths:
            if not os.path.exists(template_path):
                logging.warning(f"Template not found: {template_path}")
                continue
                
            template = cv2.imread(template_path)
            if template is None:
                logging.warning(f"Failed to load template: {template_path}")
                continue
            
            # Try different scales
            for scale in self.scale_factors:
                if scale != 1.0:
                    h, w = template.shape[:2]
                    template_scaled = cv2.resize(template, (int(w * scale), int(h * scale)))
                else:
                    template_scaled = template
                
                # Try different matching methods
                for method in self.methods:
                    result = self._match_template(screenshot_cv, template_scaled, method)
                    if result and result[2] > best_score:
                        best_match = result
                        best_score = result[2]
                        best_method = method
        
        if best_match and best_score >= confidence:
            x, y, score, w, h = best_match
            # Apply offset for region
            final_x = x + offset[0]
            final_y = y + offset[1]
            logging.debug(f"Found element at ({final_x}, {final_y}) with score {score:.3f} using {best_method}")
            return (final_x, final_y, w, h)
        
        return None
    
    def _match_template(self, image: np.ndarray, template: np.ndarray, method: int) -> Optional[Tuple[int, int, float, int, int]]:
        """
        Perform template matching using specified method.
        
        Returns:
            Tuple of (x, y, score, width, height) or None if no match
        """
        try:
            # Skip if template is larger than image
            if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
                return None
            
            result = cv2.matchTemplate(image, template, method)
            
            if method == cv2.TM_SQDIFF_NORMED:
                # For SQDIFF, lower values are better matches
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                score = 1.0 - min_val  # Convert to similarity score
                loc = min_loc
            else:
                # For other methods, higher values are better
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                score = max_val
                loc = max_loc
            
            h, w = template.shape[:2]
            return (loc[0], loc[1], score, w, h)
            
        except Exception as e:
            logging.error(f"Template matching error: {e}")
            return None
    
    def find_all_matches(self, template_path: str, screenshot: Union[str, Image.Image, np.ndarray],
                        confidence: float = 0.7, region: Optional[Tuple[int, int, int, int]] = None,
                        overlap_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Find all occurrences of a template in a screenshot.
        
        Args:
            template_path: Path to template image
            screenshot: Screenshot as path, PIL Image, or numpy array
            confidence: Minimum confidence threshold
            region: Optional region to search in
            overlap_threshold: Threshold for removing overlapping matches
            
        Returns:
            List of location tuples (x, y, width, height)
        """
        matches = []
        
        # Convert screenshot to numpy array if needed
        if isinstance(screenshot, str):
            screenshot_cv = cv2.imread(screenshot)
        elif isinstance(screenshot, Image.Image):
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        else:
            screenshot_cv = screenshot
            
        # Apply region if specified
        if region:
            x, y, w, h = region
            screenshot_cv = screenshot_cv[y:y+h, x:x+w]
            offset = (x, y)
        else:
            offset = (0, 0)
        
        template = cv2.imread(template_path)
        if template is None:
            return matches
        
        # Use normalized cross-correlation for finding multiple matches
        result = cv2.matchTemplate(screenshot_cv, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= confidence)
        
        h, w = template.shape[:2]
        
        # Collect all matches
        for pt in zip(*locations[::-1]):
            score = result[pt[1], pt[0]]
            match = (pt[0] + offset[0], pt[1] + offset[1], w, h)
            matches.append((match, score))
        
        # Remove overlapping matches
        matches = self._remove_overlapping_matches(matches, overlap_threshold)
        
        return [match[0] for match in matches]
    
    def _remove_overlapping_matches(self, matches: List[Tuple[Tuple[int, int, int, int], float]], 
                                   threshold: float) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Remove overlapping matches, keeping the ones with highest scores."""
        if len(matches) <= 1:
            return matches
        
        # Sort by score (descending)
        matches = sorted(matches, key=lambda x: x[1], reverse=True)
        
        filtered_matches = []
        for match, score in matches:
            x1, y1, w1, h1 = match
            
            # Check if this match overlaps with any already accepted match
            overlaps = False
            for accepted_match, _ in filtered_matches:
                x2, y2, w2, h2 = accepted_match
                
                # Calculate overlap
                overlap_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * \
                              max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                
                union_area = w1 * h1 + w2 * h2 - overlap_area
                
                if overlap_area / union_area > threshold:
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_matches.append((match, score))
        
        return filtered_matches