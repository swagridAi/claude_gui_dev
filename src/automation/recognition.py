import pyautogui
import cv2
import numpy as np
import logging
import glob
import os
from src.models.ui_element import UIElement
def adaptive_confidence(ui_element, min_confidence=0.5, max_confidence=0.95, step=0.05, ui_elements=None, region_manager=None):
    """
    Adaptively adjust confidence threshold to find UI elements.
    
    Args:
        ui_element: UIElement object
        min_confidence: Minimum confidence threshold to try
        max_confidence: Maximum confidence threshold to use
        step: Step size for adjusting confidence
        ui_elements: Dictionary of all UI elements
        region_manager: RegionManager for handling relative regions
        
    Returns:
        Location object or None if not found
    """
    if region_manager and ui_element.relative_region:
        region = ui_element.get_effective_region(ui_elements, region_manager.screen_size)
    else:
        region = ui_element.region 

    # Try with configured confidence first
    location = find_element(ui_element)
    if location:
        return location
        
    # If not found, try with gradually reduced confidence
    current_confidence = ui_element.confidence - step
    while current_confidence >= min_confidence:
        logging.debug(f"Trying adaptive confidence: {current_confidence:.2f} for {ui_element.name}")
        location = find_element(ui_element, confidence_override=current_confidence)
        if location:
            logging.info(f"Found {ui_element.name} with adaptive confidence: {current_confidence:.2f}")
            # Update UI element's confidence for future searches
            ui_element.adaptive_confidence = current_confidence
            return location
        current_confidence -= step
        
    logging.warning(f"Element {ui_element.name} not found even with adaptive confidence")
    return None



def find_element_cv(ui_element, confidence=0.7):
    """
    Find a UI element using advanced computer vision techniques.
    
    Args:
        ui_element: UIElement object
        confidence: Confidence threshold
        
    Returns:
        Location object or None if not found
    """
    # Take screenshot of region or full screen
    if ui_element.region:
        screenshot = pyautogui.screenshot(region=ui_element.region)
        x_offset, y_offset = ui_element.region[0], ui_element.region[1]
    else:
        screenshot = pyautogui.screenshot()
        x_offset, y_offset = 0, 0
    
    # Convert screenshot to CV2 format
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    best_match = None
    best_score = 0
    
    # Try all reference images
    for reference_path in ui_element.reference_paths:
        if not os.path.exists(reference_path):
            continue
            
        template = cv2.imread(reference_path)
        if template is None:
            continue
        
        # Try multiple methods
        methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
        for method in methods:
            try:
                result = cv2.matchTemplate(screenshot_cv, template, method)
                _, score, _, location = cv2.minMaxLoc(result)
                
                if score > best_score and score >= confidence:
                    best_score = score
                    h, w = template.shape[:2]
                    best_match = (
                        location[0] + x_offset,
                        location[1] + y_offset,
                        w,
                        h
                    )
            except Exception as e:
                logging.warning(f"CV matching error: {e}")
    
    if best_match and best_score >= confidence:
        logging.debug(f"Found {ui_element.name} using CV (score: {best_score:.2f})")
        return best_match
    
    return None

def find_element(ui_element, confidence_override=None, use_advanced=True):
    """
    Enhanced version of find_element that finds all possible matches and
    selects the one with the highest confidence score.
    
    Args:
        ui_element: UIElement object
        confidence_override: Optional override for confidence threshold
        use_advanced: Whether to use advanced recognition techniques
        
    Returns:
        Location object or None if not found
    """
    from src.models.ui_element import UIElement
    import pyautogui
    import cv2
    import numpy as np
    import logging
    import glob
    import os
    from src.utils.logging_util import log_with_screenshot
    
    # Log beginning of search
    log_with_screenshot(
        f"Searching for element: {ui_element.name}", 
        stage_name=f"SEARCH_{ui_element.name}_START"
    )
    
    confidence = confidence_override or ui_element.confidence
    min_confidence = max(0.4, confidence - 0.2)  # Set a minimum confidence threshold
    region = ui_element.region
    
    # Get a screenshot for analysis
    if region:
        screenshot = pyautogui.screenshot(region=region)
        x_offset, y_offset = region[0], region[1]
    else:
        screenshot = pyautogui.screenshot()
        x_offset, y_offset = 0, 0
    
    # Convert screenshot to CV2 format
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    screenshot_gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)
    
    # Store all potential matches
    all_matches = []
    
    # Helper function to check if template is larger than screenshot
    def template_fits(template, img):
        h, w = template.shape[:2]
        img_h, img_w = img.shape[:2]
        return h <= img_h and w <= img_w
    
    # Try reference images
    for reference_path in ui_element.reference_paths:
        if not os.path.exists(reference_path):
            logging.warning(f"Reference image not found: {reference_path}")
            continue
        
        try:
            template = cv2.imread(reference_path)
            if template is None:
                logging.warning(f"Failed to load reference image: {reference_path}")
                continue
            
            # Skip if template is larger than screenshot
            if not template_fits(template, screenshot_cv):
                logging.warning(f"Template too large for region: {reference_path}")
                continue
            
            # Try standard PyAutoGUI method first - often fastest
            try:
                location = pyautogui.locate(
                    reference_path,
                    screenshot,
                    confidence=min_confidence
                )
                
                if location:
                    # Create a match_info without trying to access .confidence
                    # This is the key fix for the 'Box' object has no attribute 'confidence' error
                    match_info = {
                        'location': (
                            location.left + x_offset,
                            location.top + y_offset,
                            location.width,
                            location.height
                        ),
                        'score': min_confidence,  # Use the minimum confidence as fallback
                        'method': 'pyautogui',
                        'template': reference_path
                    }
                    all_matches.append(match_info)
            except Exception as e:
                logging.debug(f"PyAutoGUI locate failed: {e}")
            
            # If advanced methods enabled, try them too
            if use_advanced:
                # Convert template to grayscale
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                
                # Try multiple template matching methods
                methods = [
                    cv2.TM_CCOEFF_NORMED,
                    cv2.TM_CCORR_NORMED,
                    cv2.TM_SQDIFF_NORMED
                ]
                
                for method in methods:
                    try:
                        # Try original images
                        result = cv2.matchTemplate(screenshot_gray, template_gray, method)
                        
                        # Different handling based on method
                        if method == cv2.TM_SQDIFF_NORMED:
                            # For SQDIFF, smaller values are better matches
                            # Find all matches above threshold (below 1-threshold for SQDIFF)
                            threshold = 1.0 - min_confidence
                            loc = np.where(result <= threshold)
                            
                            # Convert similarity score (smaller is better for SQDIFF)
                            for pt in zip(*loc[::-1]):
                                score = 1.0 - result[pt[1], pt[0]]
                                if score >= min_confidence:
                                    h, w = template_gray.shape
                                    match_info = {
                                        'location': (
                                            pt[0] + x_offset,
                                            pt[1] + y_offset,
                                            w,
                                            h
                                        ),
                                        'score': score,
                                        'method': 'cv2.TM_SQDIFF_NORMED',
                                        'template': reference_path
                                    }
                                    all_matches.append(match_info)
                        else:
                            # For other methods, larger values are better matches
                            threshold = min_confidence
                            loc = np.where(result >= threshold)
                            
                            for pt in zip(*loc[::-1]):
                                score = result[pt[1], pt[0]]
                                if score >= min_confidence:
                                    h, w = template_gray.shape
                                    match_info = {
                                        'location': (
                                            pt[0] + x_offset,
                                            pt[1] + y_offset,
                                            w,
                                            h
                                        ),
                                        'score': score,
                                        'method': 'cv2.' + method.__str__().split('.')[-1],
                                        'template': reference_path
                                    }
                                    all_matches.append(match_info)
                        
                        # Try with multi-scale template matching
                        scales = [0.8, 0.9, 1.1, 1.2]
                        for scale in scales:
                            # Resize template
                            h, w = template_gray.shape
                            new_h, new_w = int(h * scale), int(w * scale)
                            
                            # Skip if resized template is too large
                            if new_h > screenshot_gray.shape[0] or new_w > screenshot_gray.shape[1]:
                                continue
                            
                            resized_template = cv2.resize(template_gray, (new_w, new_h))
                            
                            # Match with resized template
                            result = cv2.matchTemplate(screenshot_gray, resized_template, method)
                            
                            # Get results for this scale
                            if method == cv2.TM_SQDIFF_NORMED:
                                threshold = 1.0 - min_confidence
                                loc = np.where(result <= threshold)
                                
                                for pt in zip(*loc[::-1]):
                                    score = 1.0 - result[pt[1], pt[0]]
                                    if score >= min_confidence:
                                        match_info = {
                                            'location': (
                                                pt[0] + x_offset,
                                                pt[1] + y_offset,
                                                new_w,
                                                new_h
                                            ),
                                            'score': score,
                                            'method': f'cv2.TM_SQDIFF_NORMED (scale: {scale})',
                                            'template': reference_path
                                        }
                                        all_matches.append(match_info)
                            else:
                                threshold = min_confidence
                                loc = np.where(result >= threshold)
                                
                                for pt in zip(*loc[::-1]):
                                    score = result[pt[1], pt[0]]
                                    if score >= min_confidence:
                                        match_info = {
                                            'location': (
                                                pt[0] + x_offset,
                                                pt[1] + y_offset,
                                                new_w,
                                                new_h
                                            ),
                                            'score': score,
                                            'method': f'cv2.{method.__str__().split(".")[-1]} (scale: {scale})',
                                            'template': reference_path
                                        }
                                        all_matches.append(match_info)
                                        
                    except Exception as e:
                        logging.debug(f"Method {method} failed for {reference_path}: {e}")
                        
        except Exception as e:
            logging.warning(f"Error processing reference {reference_path}: {e}")
    
    if all_matches:
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        best_match = all_matches[0]
        log_with_screenshot(
            f"Found {ui_element.name} with score {best_match['score']:.2f}", 
            stage_name=f"FOUND_{ui_element.name}",
            region=best_match['location']
        )
        return best_match['location']
    else:
        log_with_screenshot(
            f"Element {ui_element.name} not found", 
            level=logging.WARNING,
            stage_name=f"NOT_FOUND_{ui_element.name}"
        )

    # If we found any matches, return the best one
    if all_matches:
        # Sort by score, highest first
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        best_match = all_matches[0]
        
        # Log all potential matches for debugging
        if len(all_matches) > 1:
            logging.debug(f"Found {len(all_matches)} potential matches for {ui_element.name}")
            for i, match in enumerate(all_matches[:min(5, len(all_matches))]):
                logging.debug(f"Match #{i+1}: score={match['score']:.2f}, method={match['method']}, location={match['location']}")
        
        # Debug visualization - save image showing what was found
        debug_dir = "logs/recognition_debug"
        os.makedirs(debug_dir, exist_ok=True)
        
        try:
            # Take a screenshot and mark all potential matches
            full_screenshot = pyautogui.screenshot()
            debug_img = cv2.cvtColor(np.array(full_screenshot), cv2.COLOR_RGB2BGR)
            
            # Draw rectangle for each potential match (up to 5)
            for i, match in enumerate(all_matches[:min(5, len(all_matches))]):
                x, y, w, h = match['location']
                
                # Use different colors based on rank
                colors = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (0, 0, 255), (128, 0, 255)]
                color = colors[i] if i < len(colors) else (200, 200, 200)
                
                # Draw rectangle
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
                
                # Add text with score
                cv2.putText(
                    debug_img,
                    f"#{i+1}: {match['score']:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
            
            # Save the debug image
            import time
            timestamp = int(time.time())
            debug_path = f"{debug_dir}/{ui_element.name}_matches_{timestamp}.png"
            cv2.imwrite(debug_path, debug_img)
            logging.debug(f"Saved matches debug image to {debug_path}")
        except Exception as e:
            logging.error(f"Error saving debug image: {e}")
        
        logging.debug(f"Best match for {ui_element.name}: score={best_match['score']:.2f}, "
                     f"method={best_match['method']}, location={best_match['location']}")
        
        return best_match['location']
    
    # If no match was found, return None
    return None

def wait_for_visual_change(region, timeout=60, check_interval=0.5, threshold=0.1):
    """
    Wait until the visual content in a region changes.
    
    Args:
        region: Screen region to monitor (x, y, width, height)
        timeout: Maximum wait time in seconds
        check_interval: Time between checks
        threshold: Difference threshold to detect change
        
    Returns:
        True if change detected, False on timeout
    """
    # Take initial screenshot
    initial = pyautogui.screenshot(region=region)
    initial_np = np.array(initial)
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Take current screenshot
        current = pyautogui.screenshot(region=region)
        current_np = np.array(current)
        
        # Compare images
        if initial_np.shape == current_np.shape:
            difference = np.sum(np.abs(initial_np - current_np)) / (initial_np.size * 255)
            
            if difference > threshold:
                logging.debug(f"Visual change detected (diff: {difference:.4f})")
                return True
        
        time.sleep(check_interval)
    
    return False