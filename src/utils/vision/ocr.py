#!/usr/bin/env python3
"""
OCR (Optical Character Recognition) utilities using Tesseract.
"""

import os
import platform
import logging
import tempfile
from typing import List, Dict, Any, Optional, Union, Tuple
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from .preprocessing import preprocess_image


class OCREngine:
    """Unified OCR engine with automatic Tesseract path detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OCR Engine with configuration.
        
        Args:
            config: OCR configuration dictionary
        """
        self.config = config or {}
        self._setup_tesseract()
        
        # Default OCR configuration
        self.default_config = self.config.get("config", "--psm 6")
        self.preprocess_options = {
            'preprocess': self.config.get("preprocess", True),
            'enhance_contrast': self.config.get("contrast_enhance", True),
            'denoise': self.config.get("denoise", True)
        }
    
    def _setup_tesseract(self):
        """Set up Tesseract command path."""
        # Use config path if provided
        if "tesseract_cmd" in self.config:
            if os.path.exists(self.config["tesseract_cmd"]):
                pytesseract.pytesseract.tesseract_cmd = self.config["tesseract_cmd"]
                return
        
        # Auto-detect Tesseract path
        self._set_default_tesseract_path()
        
        if hasattr(pytesseract.pytesseract, 'tesseract_cmd'):
            logging.info(f"OCR Engine initialized with Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
        else:
            logging.warning("Tesseract path not configured. OCR may not work.")
    
    def _set_default_tesseract_path(self):
        """Set the default Tesseract path based on operating system."""
        system = platform.system()
        
        if system == "Windows":
            paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
            ]
        elif system == "Darwin":  # macOS
            paths = [
                "/usr/local/bin/tesseract",
                "/opt/homebrew/bin/tesseract"
            ]
        else:  # Linux
            paths = [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract"
            ]
        
        # Find the first valid path
        for path in paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return
        
        logging.warning("Could not find Tesseract executable")
    
    def extract_text(self, image: Union[str, Image.Image, np.ndarray],
                    region: Optional[Tuple[int, int, int, int]] = None,
                    config: Optional[str] = None) -> str:
        """
        Extract text from an image.
        
        Args:
            image: Image as path, PIL Image, or numpy array
            region: Optional region to extract from (x, y, width, height)
            config: Tesseract configuration string
            
        Returns:
            Extracted text
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image, str):
                pil_image = Image.open(image)
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
            
            # Apply region if specified
            if region:
                x, y, w, h = region
                pil_image = pil_image.crop((x, y, x + w, y + h))
            
            # Preprocess image if enabled
            if self.preprocess_options.get('preprocess', True):
                pil_image = self._preprocess_for_ocr(pil_image)
            
            # Extract text
            text = pytesseract.image_to_string(
                pil_image,
                config=config or self.default_config
            )
            
            return text.strip()
            
        except Exception as e:
            logging.error(f"OCR extraction failed: {e}")
            return ""
    
    def extract_structured_data(self, image: Union[str, Image.Image, np.ndarray],
                               region: Optional[Tuple[int, int, int, int]] = None,
                               config: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract text with position information.
        
        Args:
            image: Image as path, PIL Image, or numpy array
            region: Optional region to extract from
            config: Tesseract configuration string
            
        Returns:
            List of dictionaries with text and position information
        """
        try:
            # Convert to PIL Image and apply region if needed
            pil_image = self._prepare_image(image, region)
            
            # Preprocess if enabled
            if self.preprocess_options.get('preprocess', True):
                pil_image = self._preprocess_for_ocr(pil_image)
            
            # Save to temporary file (required for image_to_data)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                temp_filename = temp.name
                pil_image.save(temp_filename)
            
            try:
                # Extract data with position information
                data = pytesseract.image_to_data(
                    temp_filename,
                    config=config or self.default_config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Format results
                results = []
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    if text:  # Only include non-empty text
                        result = {
                            'text': text,
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i],
                            'confidence': data['conf'][i]
                        }
                        
                        # Adjust coordinates if region was specified
                        if region:
                            result['x'] += region[0]
                            result['y'] += region[1]
                        
                        results.append(result)
                
                return results
                
            finally:
                # Clean up temp file
                os.unlink(temp_filename)
                
        except Exception as e:
            logging.error(f"Structured OCR extraction failed: {e}")
            return []
    
    def verify_text_presence(self, image: Union[str, Image.Image, np.ndarray],
                            expected_text: str,
                            region: Optional[Tuple[int, int, int, int]] = None,
                            case_sensitive: bool = False) -> bool:
        """
        Verify if specific text is present in an image.
        
        Args:
            image: Image to search in
            expected_text: Text to look for
            region: Optional region to search in
            case_sensitive: Whether to perform case-sensitive matching
            
        Returns:
            True if text is found
        """
        extracted_text = self.extract_text(image, region)
        
        if not case_sensitive:
            return expected_text.lower() in extracted_text.lower()
        else:
            return expected_text in extracted_text
    
    def _prepare_image(self, image: Union[str, Image.Image, np.ndarray],
                      region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        """Prepare image for OCR processing."""
        # Convert to PIL Image if needed
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        # Apply region if specified
        if region:
            x, y, w, h = region
            pil_image = pil_image.crop((x, y, x + w, y + h))
        
        return pil_image
    
    def _preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """Apply OCR-specific preprocessing."""
        # Prepare preprocessing operations
        operations = {}
        
        if self.preprocess_options.get('enhance_contrast', True):
            operations['enhance_contrast'] = {'factor': 2.0}
        
        if self.preprocess_options.get('denoise', True):
            operations['denoise'] = {'strength': 10}
        
        # Apply threshold for OCR
        operations['threshold'] = {'method': 'adaptive'}
        
        # Process image
        processed = preprocess_image(image, operations)
        
        # Convert back to PIL if needed
        if isinstance(processed, np.ndarray):
            return Image.fromarray(processed)
        
        return processed
    
    def get_languages(self) -> List[str]:
        """Get available OCR languages."""
        try:
            return pytesseract.get_languages()
        except Exception as e:
            logging.error(f"Error getting OCR languages: {e}")
            return ['eng']  # Default to English
    
    def set_language(self, lang: str) -> bool:
        """
        Set the OCR language.
        
        Args:
            lang: Language code (e.g., 'eng', 'fra', 'deu')
            
        Returns:
            True if language was set successfully
        """
        try:
            available_langs = self.get_languages()
            if lang in available_langs:
                self.default_config = f"-l {lang} {self.config.get('config', '--psm 6')}"
                return True
            else:
                logging.warning(f"Language '{lang}' not available. Available: {available_langs}")
                return False
        except Exception as e:
            logging.error(f"Error setting OCR language: {e}")
            return False