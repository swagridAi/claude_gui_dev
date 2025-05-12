#!/usr/bin/env python3
"""
Visual debugging utilities for template matching and recognition.
"""

import os
import cv2
import numpy as np
import logging
from datetime import datetime
from typing import Tuple, List, Optional, Union, Dict, Any
from PIL import Image
from .screenshot import save_screenshot


def visualize_match(image: Union[str, Image.Image, np.ndarray],
                   location: Tuple[int, int, int, int],
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2,
                   label: Optional[str] = None,
                   save_path: Optional[str] = None) -> Union[Image.Image, np.ndarray]:
    """
    Visualize a template match on an image.
    
    Args:
        image: Source image
        location: Match location (x, y, width, height)
        color: Rectangle color (BGR for numpy, RGB for PIL)
        thickness: Rectangle thickness
        label: Optional label text
        save_path: Optional path to save result
        
    Returns:
        Image with visualization
    """
    # Convert to numpy array for drawing
    if isinstance(image, str):
        img = cv2.imread(image)
    elif isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()
    
    x, y, w, h = location
    
    # Draw rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    
    # Add label if provided
    if label:
        # Calculate text position
        label_y = y - 10 if y > 20 else y + h + 25
        cv2.putText(img, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, img)
    
    return img


def visualize_all_matches(image: Union[str, Image.Image, np.ndarray],
                         matches: List[Tuple[int, int, int, int]],
                         colors: Optional[List[Tuple[int, int, int]]] = None,
                         labels: Optional[List[str]] = None,
                         save_path: Optional[str] = None) -> Union[Image.Image, np.ndarray]:
    """
    Visualize multiple matches on an image.
    
    Args:
        image: Source image
        matches: List of match locations
        colors: Optional list of colors for each match
        labels: Optional list of labels for each match
        save_path: Optional path to save result
        
    Returns:
        Image with all visualizations
    """
    # Convert to numpy array for drawing
    if isinstance(image, str):
        img = cv2.imread(image)
    elif isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()
    
    # Default colors if not provided
    if colors is None:
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    # Draw each match
    for i, match in enumerate(matches):
        color = colors[i % len(colors)]
        label = labels[i] if labels and i < len(labels) else f"Match {i+1}"
        
        img = visualize_match(img, match, color, label=label)
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, img)
    
    return img


def save_debug_image(image: Union[Image.Image, np.ndarray],
                    name: str,
                    directory: str = "logs/debug",
                    prefix: str = "debug",
                    timestamp: bool = True) -> Optional[str]:
    """
    Save an image for debugging purposes.
    
    Args:
        image: Image to save
        name: Base name for the file
        directory: Directory to save in
        prefix: Prefix for filename
        timestamp: Whether to add timestamp
        
    Returns:
        Path to saved file or None if failed
    """
    try:
        os.makedirs(directory, exist_ok=True)
        
        # Build filename
        filename_parts = [prefix, name]
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename_parts.append(timestamp_str)
        
        filename = "_".join(filename_parts) + ".png"
        filepath = os.path.join(directory, filename)
        
        # Save image
        if isinstance(image, Image.Image):
            image.save(filepath)
        else:
            cv2.imwrite(filepath, image)
        
        logging.debug(f"Debug image saved to {filepath}")
        return filepath
        
    except Exception as e:
        logging.error(f"Failed to save debug image: {e}")
        return None


def create_comparison_image(image1: Union[Image.Image, np.ndarray],
                           image2: Union[Image.Image, np.ndarray],
                           title1: str = "Image 1",
                           title2: str = "Image 2",
                           save_path: Optional[str] = None) -> np.ndarray:
    """
    Create a side-by-side comparison of two images.
    
    Args:
        image1: First image
        image2: Second image
        title1: Title for first image
        title2: Title for second image
        save_path: Optional path to save result
        
    Returns:
        Combined comparison image
    """
    # Convert to numpy arrays if needed
    if isinstance(image1, Image.Image):
        img1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
    else:
        img1 = image1
    
    if isinstance(image2, Image.Image):
        img2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
    else:
        img2 = image2
    
    # Resize images to same height
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if h1 != h2:
        if h1 > h2:
            img2 = cv2.resize(img2, (int(w2 * h1 / h2), h1))
        else:
            img1 = cv2.resize(img1, (int(w1 * h2 / h1), h2))
    
    # Add titles
    title_height = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (255, 255, 255)
    
    # Create title bars
    h, w = img1.shape[:2]
    title_bar1 = np.zeros((title_height, w, 3), dtype=np.uint8)
    cv2.putText(title_bar1, title1, (10, 30), font, font_scale, color, 2)
    
    h, w = img2.shape[:2]
    title_bar2 = np.zeros((title_height, w, 3), dtype=np.uint8)
    cv2.putText(title_bar2, title2, (10, 30), font, font_scale, color, 2)
    
    # Combine with titles
    img1_with_title = np.vstack([title_bar1, img1])
    img2_with_title = np.vstack([title_bar2, img2])
    
    # Combine horizontally
    comparison = np.hstack([img1_with_title, img2_with_title])
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, comparison)
    
    return comparison


def visualize_confidence_map(image: Union[str, Image.Image, np.ndarray],
                            confidence_map: np.ndarray,
                            threshold: float = 0.8,
                            save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize template matching confidence map.
    
    Args:
        image: Original image
        confidence_map: Confidence values from template matching
        threshold: Confidence threshold for visualization
        save_path: Optional path to save result
        
    Returns:
        Visualization of confidence map
    """
    # Load image if path provided
    if isinstance(image, str):
        img = cv2.imread(image)
    elif isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()
    
    # Normalize confidence map to 0-255 range
    confidence_norm = (confidence_map * 255).astype(np.uint8)
    
    # Create colored heatmap
    heatmap = cv2.applyColorMap(confidence_norm, cv2.COLORMAP_JET)
    
    # Resize heatmap to match image size
    h, w = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(img, 0.7, heatmap_resized, 0.3, 0)
    
    # Mark threshold areas
    locations = np.where(confidence_map >= threshold)
    for pt in zip(*locations[::-1]):
        cv2.circle(overlay, pt, 5, (255, 255, 255), 2)
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, overlay)
    
    return overlay


def create_debug_grid(images: List[Union[Image.Image, np.ndarray]],
                     labels: Optional[List[str]] = None,
                     cols: int = 3,
                     save_path: Optional[str] = None) -> np.ndarray:
    """
    Create a grid of images for debugging.
    
    Args:
        images: List of images to arrange
        labels: Optional labels for each image
        cols: Number of columns in grid
        save_path: Optional path to save result
        
    Returns:
        Grid image
    """
    if not images:
        return np.array([])
    
    # Convert all images to numpy arrays
    cv_images = []
    for img in images:
        if isinstance(img, Image.Image):
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            cv_img = img
        cv_images.append(cv_img)
    
    # Calculate grid dimensions
    num_images = len(cv_images)
    rows = (num_images + cols - 1) // cols
    
    # Find maximum dimensions
    max_height = max(img.shape[0] for img in cv_images)
    max_width = max(img.shape[1] for img in cv_images)
    
    # Create grid
    grid_rows = []
    for r in range(rows):
        row_images = []
        for c in range(cols):
            idx = r * cols + c
            if idx < num_images:
                # Resize image to match max dimensions
                img = cv2.resize(cv_images[idx], (max_width, max_height))
                
                # Add label if provided
                if labels and idx < len(labels):
                    cv2.putText(img, labels[idx], (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                row_images.append(img)
            else:
                # Fill empty space with black image
                row_images.append(np.zeros((max_height, max_width, 3), dtype=np.uint8))
        
        # Combine row
        if row_images:
            row = np.hstack(row_images)
            grid_rows.append(row)
    
    # Combine all rows
    grid = np.vstack(grid_rows)
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, grid)
    
    return grid


def annotate_click_location(image: Union[Image.Image, np.ndarray],
                           click_point: Tuple[int, int],
                           color: Tuple[int, int, int] = (255, 0, 0),
                           size: int = 20,
                           label: Optional[str] = None,
                           save_path: Optional[str] = None) -> np.ndarray:
    """
    Annotate a click location on an image.
    
    Args:
        image: Source image
        click_point: Click coordinates (x, y)
        color: Marker color
        size: Marker size
        label: Optional label text
        save_path: Optional path to save result
        
    Returns:
        Annotated image
    """
    # Convert to numpy array for drawing
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()
    
    x, y = click_point
    
    # Draw crosshair
    cv2.drawMarker(img, (x, y), color, markerType=cv2.MARKER_CROSS, 
                   markerSize=size, thickness=2)
    
    # Draw circle
    cv2.circle(img, (x, y), size, color, 2)
    
    # Add label if provided
    if label:
        cv2.putText(img, label, (x + size + 5, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, img)
    
    return img