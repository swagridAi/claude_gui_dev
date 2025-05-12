#!/usr/bin/env python3
"""
Vision utilities package for Claude GUI automation.

This package consolidates all vision-related functionality including screenshot handling,
template matching, visual verification, and image preprocessing.
"""

# Export main interfaces
from .screenshot import take_screenshot, save_screenshot, ScreenshotManager
from .template import TemplateMatcher
from .verification import VisualVerifier
from .preprocessing import enhance_contrast, denoise, apply_threshold, preprocess_image
from .ocr import OCREngine
from .debugging import visualize_match, save_debug_image

__all__ = [
    'take_screenshot',
    'save_screenshot', 
    'ScreenshotManager',
    'TemplateMatcher',
    'VisualVerifier',
    'enhance_contrast',
    'denoise',
    'apply_threshold',
    'preprocess_image',
    'OCREngine',
    'visualize_match',
    'save_debug_image'
]