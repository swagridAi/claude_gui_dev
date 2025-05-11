import pytesseract
import cv2
import numpy as np
import pyautogui
import logging
import os
import tempfile
from PIL import Image, ImageEnhance

class OCREngine:
    """OCR Engine class that handles text extraction from images."""
    
    def __init__(self, config=None):
        """
        Initialize OCR Engine with configuration.
        
        Args:
            config: OCR configuration dictionary
        """
        self.config = config or {}
        
        # Set tesseract command if provided in config
        if "tesseract_cmd" in self.config:
            pytesseract.pytesseract.tesseract_cmd = self.config["tesseract_cmd"]
        
        # Default tesseract paths based on OS
        if not hasattr(pytesseract.pytesseract, "tesseract_cmd") or not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            self._set_default_tesseract_path()
        
        logging.info(f"OCR Engine initialized with Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
    
    def _set_default_tesseract_path(self):
        """Set the default Tesseract path based on operating system."""
        import platform
        
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
        
        logging.warning("Could not find Tesseract executable. Please install Tesseract or set path manually.")

def extract_text_from_region(region, config=None):
    """
    Extract text from a screen region using OCR.
    
    Args:
        region: Tuple (x, y, width, height) defining screen region
        config: Optional OCR configuration
    
    Returns:
        Extracted text as string
    """
    try:
        # Initialize OCR engine with config
        ocr_config = config.get("ocr", {}) if config else {}
        engine = OCREngine(ocr_config)
        
        # Take screenshot of region
        screenshot = pyautogui.screenshot(region=region)
        
        # Process image for better OCR results
        processed_image = preprocess_image(
            screenshot, 
            preprocess=ocr_config.get("preprocess", True),
            contrast_enhance=ocr_config.get("contrast_enhance", True),
            denoise=ocr_config.get("denoise", True)
        )
        
        # Get OCR configuration parameters
        tesseract_config = ocr_config.get("config", "--psm 6")
        
        # Perform OCR
        text = pytesseract.image_to_string(processed_image, config=tesseract_config)
        
        logging.debug(f"Extracted text from region {region}: {text[:50]}..." if len(text) > 50 else f"Extracted text: {text}")
        return text.strip()
        
    except Exception as e:
        logging.error(f"OCR extraction failed: {e}")
        return ""

def preprocess_image(image, preprocess=True, contrast_enhance=True, denoise=True):
    """
    Preprocess an image for better OCR accuracy.
    
    Args:
        image: PIL Image or numpy array
        preprocess: Whether to apply preprocessing
        contrast_enhance: Whether to enhance contrast
        denoise: Whether to apply denoising
    
    Returns:
        Processed PIL Image
    """
    if not preprocess:
        return image
    
    try:
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image
            
        # Convert to grayscale if image is color
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            
        # Apply contrast enhancement
        if contrast_enhance:
            # Convert back to PIL for easier contrast adjustment
            pil_img = Image.fromarray(gray)
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(2.0)  # Increase contrast
            gray = np.array(pil_img)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        # Apply denoising if enabled
        if denoise:
            denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
            return Image.fromarray(denoised)
        
        return Image.fromarray(thresh)
        
    except Exception as e:
        logging.warning(f"Image preprocessing failed: {e}, using original image")
        return image

def extract_text_from_file(image_path, config=None):
    """
    Extract text from an image file.
    
    Args:
        image_path: Path to image file
        config: Optional OCR configuration
    
    Returns:
        Extracted text as string
    """
    try:
        # Initialize OCR engine with config
        ocr_config = config.get("ocr", {}) if config else {}
        engine = OCREngine(ocr_config)
        
        # Read image
        image = Image.open(image_path)
        
        # Process image for better OCR results
        processed_image = preprocess_image(
            image, 
            preprocess=ocr_config.get("preprocess", True),
            contrast_enhance=ocr_config.get("contrast_enhance", True),
            denoise=ocr_config.get("denoise", True)
        )
        
        # Get OCR configuration parameters
        tesseract_config = ocr_config.get("config", "--psm 6")
        
        # Perform OCR
        text = pytesseract.image_to_string(processed_image, config=tesseract_config)
        
        logging.debug(f"Extracted text from {image_path}: {text[:50]}..." if len(text) > 50 else f"Extracted text: {text}")
        return text.strip()
        
    except Exception as e:
        logging.error(f"OCR extraction from file failed: {e}")
        return ""

def extract_structured_text(region, config=None):
    """
    Extract structured text with position information.
    
    Args:
        region: Tuple (x, y, width, height) defining screen region
        config: Optional OCR configuration
    
    Returns:
        List of dictionaries with text and position information
    """
    try:
        # Initialize OCR engine with config
        ocr_config = config.get("ocr", {}) if config else {}
        engine = OCREngine(ocr_config)
        
        # Take screenshot of region
        screenshot = pyautogui.screenshot(region=region)
        
        # Process image for better OCR results
        processed_image = preprocess_image(
            screenshot, 
            preprocess=ocr_config.get("preprocess", True),
            contrast_enhance=ocr_config.get("contrast_enhance", True),
            denoise=ocr_config.get("denoise", True)
        )
        
        # Save processed image to temp file (required for some pytesseract functions)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
            temp_filename = temp.name
            processed_image.save(temp_filename)
        
        # Get OCR configuration parameters
        tesseract_config = ocr_config.get("config", "--psm 6")
        
        # Extract data including bounding box info
        data = pytesseract.image_to_data(
            temp_filename,
            config=tesseract_config,
            output_type=pytesseract.Output.DICT
        )
        
        # Clean up temp file
        os.unlink(temp_filename)
        
        # Format results
        result = []
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                result.append({
                    'text': data['text'][i],
                    'x': data['left'][i] + region[0] if region else data['left'][i],
                    'y': data['top'][i] + region[1] if region else data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'conf': data['conf'][i]
                })
        
        return result
        
    except Exception as e:
        logging.error(f"Structured OCR extraction failed: {e}")
        return []

def verify_text_presence(region, expected_text, config=None):
    """
    Verify if specific text is present in a region.
    
    Args:
        region: Tuple (x, y, width, height) defining screen region
        expected_text: Text to look for
        config: Optional OCR configuration
    
    Returns:
        True if text is found, False otherwise
    """
    extracted_text = extract_text_from_region(region, config)
    return expected_text.lower() in extracted_text.lower()