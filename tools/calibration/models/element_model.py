#!/usr/bin/env python3
"""
Element model for the calibration tool that represents UI elements and reference images.
"""

import os
from typing import List, Tuple, Optional, Dict, Any, Union, Callable


class ReferenceImage:
    """Representation of a reference image with metadata."""
    
    def __init__(self, 
                 path: str, 
                 is_variant: bool = False, 
                 variant_type: Optional[str] = None, 
                 parent_path: Optional[str] = None):
        """
        Initialize a reference image.
        
        Args:
            path: Path to the reference image
            is_variant: Whether this is a processed variant of an original image
            variant_type: Type of variant ('gray', 'contrast', 'thresh', etc.)
            parent_path: Path to the parent image if this is a variant
        """
        self.path = path
        self.is_variant = is_variant
        self.variant_type = variant_type
        self.parent_path = parent_path
        self.creation_time = os.path.getctime(path) if os.path.exists(path) else None


class ElementModel:
    """Model representation of a UI element in the calibration tool."""
    
    def __init__(self, 
                 name: str, 
                 region: Optional[Tuple[int, int, int, int]] = None, 
                 reference_paths: Optional[List[str]] = None, 
                 confidence: float = 0.7, 
                 click_coordinates: Optional[Tuple[int, int]] = None, 
                 use_coordinates_first: bool = True):
        """
        Initialize an element model.
        
        Args:
            name: Element name
            region: Screen region (x, y, width, height)
            reference_paths: List of paths to reference images
            confidence: Confidence threshold for recognition
            click_coordinates: (x, y) tuple for direct clicking
            use_coordinates_first: Whether to prioritize coordinates over visual recognition
        """
        self._name = name
        self._region = region
        self._reference_paths = reference_paths or []
        self._confidence = confidence
        self._click_coordinates = click_coordinates
        self._use_coordinates_first = use_coordinates_first
        self._last_match_location = None
    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, value: str):
        if not value:
            raise ValueError("Element name cannot be empty")
        self._name = value
    
    @property
    def region(self) -> Optional[Tuple[int, int, int, int]]:
        return self._region
    
    @region.setter
    def region(self, value: Optional[Tuple[int, int, int, int]]):
        if value is not None and (not isinstance(value, tuple) or len(value) != 4):
            raise ValueError("Region must be a tuple of (x, y, width, height)")
        self._region = value
    
    @property
    def reference_paths(self) -> List[str]:
        return self._reference_paths
    
    @reference_paths.setter
    def reference_paths(self, value: List[str]):
        if not isinstance(value, list):
            raise ValueError("Reference paths must be a list")
        self._reference_paths = value
    
    @property
    def confidence(self) -> float:
        return self._confidence
    
    @confidence.setter
    def confidence(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        self._confidence = value
    
    @property
    def click_coordinates(self) -> Optional[Tuple[int, int]]:
        return self._click_coordinates
    
    @click_coordinates.setter
    def click_coordinates(self, value: Optional[Tuple[int, int]]):
        if value is not None and (not isinstance(value, tuple) or len(value) != 2):
            raise ValueError("Click coordinates must be a tuple of (x, y)")
        self._click_coordinates = value
    
    @property
    def use_coordinates_first(self) -> bool:
        return self._use_coordinates_first
    
    @use_coordinates_first.setter
    def use_coordinates_first(self, value: bool):
        self._use_coordinates_first = bool(value)
    
    @property
    def reference_count(self) -> int:
        return len(self._reference_paths)
    
    @property
    def last_match_location(self) -> Optional[Tuple[int, int, int, int]]:
        return self._last_match_location
    
    @last_match_location.setter
    def last_match_location(self, value: Optional[Tuple[int, int, int, int]]):
        if value is not None and (not isinstance(value, tuple) or len(value) != 4):
            raise ValueError("Match location must be a tuple of (x, y, width, height)")
        self._last_match_location = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the element model to a dictionary."""
        return {
            'name': self._name,
            'region': self._region,
            'reference_paths': self._reference_paths,
            'confidence': self._confidence,
            'click_coordinates': self._click_coordinates,
            'use_coordinates_first': self._use_coordinates_first
        }
    
    @classmethod
    def create_button(cls, name: str, region: Tuple[int, int, int, int], **kwargs) -> 'ElementModel':
        """Create a button element with defaults appropriate for buttons."""
        return cls(name, region, confidence=0.8, **kwargs)
    
    @classmethod
    def create_text_input(cls, name: str, region: Tuple[int, int, int, int], **kwargs) -> 'ElementModel':
        """Create a text input element with defaults appropriate for text inputs."""
        return cls(name, region, confidence=0.7, **kwargs)
    
    @classmethod
    def create_indicator(cls, name: str, region: Tuple[int, int, int, int], **kwargs) -> 'ElementModel':
        """Create an indicator element with defaults appropriate for indicators."""
        return cls(name, region, confidence=0.7, **kwargs)


class ElementCollection:
    """Manages a collection of ElementModel instances."""
    
    def __init__(self):
        """Initialize an empty element collection."""
        self.elements: Dict[str, ElementModel] = {}
    
    def add(self, element: ElementModel) -> None:
        """
        Add or update an element in the collection.
        
        Args:
            element: The element model to add or update
        """
        self.elements[element.name] = element
    
    def remove(self, element_name: str) -> None:
        """
        Remove an element from the collection.
        
        Args:
            element_name: Name of the element to remove
        """
        if element_name in self.elements:
            del self.elements[element_name]
    
    def get(self, element_name: str) -> Optional[ElementModel]:
        """
        Get an element by name.
        
        Args:
            element_name: Name of the element to retrieve
            
        Returns:
            The element model or None if not found
        """
        return self.elements.get(element_name)
    
    def get_all(self) -> Dict[str, ElementModel]:
        """
        Get all elements in the collection.
        
        Returns:
            Dictionary of all elements by name
        """
        return self.elements.copy()
    
    def clear(self) -> None:
        """Clear all elements from the collection."""
        self.elements.clear()
    
    def count(self) -> int:
        """
        Get the number of elements in the collection.
        
        Returns:
            Count of elements
        """
        return len(self.elements)


def create_element_from_dict(element_dict: Dict[str, Any], name: Optional[str] = None) -> ElementModel:
    """
    Convert a dictionary to an ElementModel instance.
    
    Args:
        element_dict: Dictionary containing element properties
        name: Optional name to use if not in the dictionary
        
    Returns:
        An ElementModel instance
    """
    element_name = element_dict.get('name', name)
    if not element_name:
        raise ValueError("Element must have a name")
    
    return ElementModel(
        name=element_name,
        region=element_dict.get('region'),
        reference_paths=element_dict.get('reference_paths', []),
        confidence=element_dict.get('confidence', 0.7),
        click_coordinates=element_dict.get('click_coordinates'),
        use_coordinates_first=element_dict.get('use_coordinates_first', True)
    )


def validate_element(element: ElementModel) -> Tuple[bool, str]:
    """
    Validate an ElementModel instance.
    
    Args:
        element: The element to validate
        
    Returns:
        A tuple of (is_valid, error_message)
    """
    if not element.name:
        return False, "Element must have a name"
    
    if element.region is not None:
        x, y, w, h = element.region
        if w <= 0 or h <= 0:
            return False, "Region width and height must be positive"
        if x < 0 or y < 0:
            return False, "Region x and y coordinates must be non-negative"
    
    if element.confidence < 0 or element.confidence > 1:
        return False, "Confidence must be between 0 and 1"
    
    if not element.reference_paths and not element.click_coordinates:
        return False, "Element must have either reference images or click coordinates"
    
    return True, ""


def convert_to_ui_element(element_model: ElementModel) -> Any:
    """
    Convert an ElementModel to a UIElement for the automation system.
    
    Args:
        element_model: The element model to convert
        
    Returns:
        A UIElement instance
    """
    try:
        from src.models.ui_element import UIElement
        
        return UIElement(
            name=element_model.name,
            reference_paths=element_model.reference_paths,
            region=element_model.region,
            confidence=element_model.confidence,
            click_coordinates=element_model.click_coordinates,
            use_coordinates_first=element_model.use_coordinates_first
        )
    except ImportError:
        # If UIElement is not available, return a dictionary representation
        return element_model.to_dict()


def filter_original_references(reference_paths: List[str]) -> List[str]:
    """
    Filter out variant reference images, returning only originals.
    
    Args:
        reference_paths: List of reference image paths
        
    Returns:
        List of original reference image paths
    """
    return [path for path in reference_paths if not any(
        x in path for x in ['_gray', '_contrast', '_thresh', '_scale'])]


def get_reference_metadata(reference_path: str) -> ReferenceImage:
    """
    Extract metadata from a reference image path.
    
    Args:
        reference_path: Path to a reference image
        
    Returns:
        A ReferenceImage instance with extracted metadata
    """
    basename = os.path.basename(reference_path)
    is_variant = any(x in basename for x in ['_gray', '_contrast', '_thresh', '_scale'])
    
    variant_type = None
    parent_path = None
    
    if is_variant:
        variant_parts = basename.split('_')
        if len(variant_parts) > 1:
            variant_type = variant_parts[-1].split('.')[0]
            # Reconstruct parent path
            parent_name = '_'.join(variant_parts[:-1]) + os.path.splitext(basename)[1]
            parent_path = os.path.join(os.path.dirname(reference_path), parent_name)
    
    return ReferenceImage(
        path=reference_path,
        is_variant=is_variant,
        variant_type=variant_type,
        parent_path=parent_path
    )