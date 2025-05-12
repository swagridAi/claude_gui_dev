#!/usr/bin/env python3
"""
Humanized input module for Claude GUI automation.

This module provides utilities for simulating human-like keyboard input,
with configurable typing speeds, natural pauses, and adaptive behavior.
"""

import random
import time
import logging
from typing import Dict, Any, Optional, Tuple, Union, List

# Import pyautogui for keyboard simulation
import pyautogui

# Try to import clipboard handling, provide fallback if not available
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False
    logging.debug("pyperclip not available, clipboard functionality will be limited")


class HumanizedKeyboard:
    """
    Simulates human-like keyboard input with configurable behavior profiles.
    
    This class provides methods for typing text with realistic timing patterns,
    including variable delays between keystrokes, occasional pauses, and
    clipboard-based input for efficiency with longer texts.
    """
    
    # Default typing profiles with configurable parameters
    TYPING_PROFILES = {
        'fast': {
            'base_delay': 0.01,     # Base delay between keystrokes
            'variance': 0.005,      # Random variance in timing
            'pause_chance': 0.02,   # Probability of pause between characters
            'pause_duration': (0.1, 0.3),  # Min/max pause duration
            'typo_chance': 0.0,     # Future: probability of making a typo
            'correction_chance': 1.0  # Future: probability of correcting a typo
        },
        'normal': {
            'base_delay': 0.03,
            'variance': 0.01,
            'pause_chance': 0.05,
            'pause_duration': (0.2, 0.5),
            'typo_chance': 0.0,
            'correction_chance': 1.0
        },
        'slow': {
            'base_delay': 0.05,
            'variance': 0.02,
            'pause_chance': 0.1,
            'pause_duration': (0.3, 0.7),
            'typo_chance': 0.0,
            'correction_chance': 1.0
        }
    }
    
    def __init__(self, profile: str = 'normal', custom_profile: Optional[Dict[str, Any]] = None):
        """
        Initialize the humanized keyboard with a typing profile.
        
        Args:
            profile: Name of predefined profile ('fast', 'normal', 'slow')
            custom_profile: Optional custom profile settings
        """
        self.settings = {}
        
        if custom_profile:
            # Use custom profile if provided
            self.settings = custom_profile
            self.profile = 'custom'
        else:
            # Otherwise use a predefined profile
            self.set_profile(profile)
    
    def set_profile(self, profile: str) -> None:
        """
        Set the typing profile to use.
        
        Args:
            profile: Name of predefined profile ('fast', 'normal', 'slow')
        """
        if profile not in self.TYPING_PROFILES:
            logging.warning(f"Unknown typing profile: {profile}, using 'normal'")
            profile = 'normal'
        
        self.profile = profile
        self.settings = self.TYPING_PROFILES[profile].copy()
        logging.debug(f"Set typing profile to '{profile}'")
    
    def get_delay_for_char(self, char: str) -> float:
        """
        Get an appropriate delay for a specific character.
        
        Some characters might take longer to type than others in real human typing.
        
        Args:
            char: The character to determine delay for
            
        Returns:
            Delay in seconds
        """
        base_delay = self.settings['base_delay']
        variance = self.settings['variance']
        
        # For now, just add random variance
        return base_delay + random.uniform(-variance, variance)
    
    def type_text(self, text: str, clear_existing: bool = True) -> bool:
        """
        Type text with human-like timing patterns.
        
        Args:
            text: Text to type
            clear_existing: Whether to clear existing text first (Ctrl+A, Delete)
            
        Returns:
            True if successful, False if error occurred
        """
        try:
            if clear_existing:
                # Clear any existing text with Ctrl+A and Delete
                pyautogui.hotkey('ctrl', 'a')
                time.sleep(0.5)
                pyautogui.press('delete')
                time.sleep(0.5)
            
            # Get text length for logging
            text_preview = text[:50] + "..." if len(text) > 50 else text
            logging.debug(f"Typing text ({len(text)} chars): {text_preview}")
            
            # Type each character with human-like delays
            for char in text:
                # Get appropriate delay for this character
                typing_delay = self.get_delay_for_char(char)
                
                # Type the character
                pyautogui.write(char)
                time.sleep(typing_delay)
                
                # Occasionally add a pause (simulating thinking)
                if random.random() < self.settings['pause_chance']:
                    min_pause, max_pause = self.settings['pause_duration']
                    pause_time = random.uniform(min_pause, max_pause)
                    time.sleep(pause_time)
            
            return True
            
        except Exception as e:
            logging.error(f"Error typing text: {e}")
            return False
    
    def type_with_clipboard(self, text: str, clear_existing: bool = True) -> bool:
        """
        Type using clipboard for efficiency with long text.
        
        Args:
            text: Text to type
            clear_existing: Whether to clear existing text first
            
        Returns:
            True if successful, False if error occurred
        """
        if not CLIPBOARD_AVAILABLE:
            logging.warning("Clipboard not available, falling back to regular typing")
            return self.type_text(text, clear_existing)
        
        try:
            # Log operation
            text_preview = text[:50] + "..." if len(text) > 50 else text
            logging.debug(f"Typing with clipboard ({len(text)} chars): {text_preview}")
            
            if clear_existing:
                # Clear any existing text with Ctrl+A and Delete
                pyautogui.hotkey('ctrl', 'a')
                time.sleep(0.5)
                pyautogui.press('delete')
                time.sleep(0.5)
            
            # Save original clipboard content
            try:
                original_clipboard = pyperclip.paste()
            except Exception:
                original_clipboard = None
            
            try:
                # Copy text to clipboard
                pyperclip.copy(text)
                
                # Paste with Ctrl+V
                time.sleep(0.5)
                pyautogui.hotkey('ctrl', 'v')
                time.sleep(0.5)
                
                return True
                
            finally:
                # Restore original clipboard if possible
                if original_clipboard is not None:
                    try:
                        pyperclip.copy(original_clipboard)
                    except Exception:
                        pass
                        
        except Exception as e:
            logging.error(f"Error typing with clipboard: {e}")
            return False
    
    def type_smart(self, text: str, clear_existing: bool = True, 
                  length_threshold: int = 100) -> bool:
        """
        Intelligently choose between regular typing and clipboard based on length.
        
        Args:
            text: Text to type
            clear_existing: Whether to clear existing text first
            length_threshold: Character threshold for using clipboard
            
        Returns:
            True if successful, False if error occurred
        """
        # Use clipboard for long text if available
        if len(text) > length_threshold and CLIPBOARD_AVAILABLE:
            return self.type_with_clipboard(text, clear_existing)
        else:
            return self.type_text(text, clear_existing)


class InputManager:
    """
    Manages keyboard and other input simulation for automation.
    
    This class serves as a facade for various input-related functions,
    providing a consistent interface and additional high-level features
    beyond basic keyboard input.
    """
    
    def __init__(self, typing_profile: str = 'normal'):
        """
        Initialize input manager with specified typing profile.
        
        Args:
            typing_profile: Typing profile name
        """
        self.keyboard = HumanizedKeyboard(typing_profile)
    
    def set_typing_profile(self, profile: str) -> None:
        """
        Set the typing profile to use.
        
        Args:
            profile: Profile name
        """
        self.keyboard.set_profile(profile)
    
    def type_text(self, text: str, clear_existing: bool = True, 
                 use_smart_method: bool = True) -> bool:
        """
        Type text using the appropriate method.
        
        Args:
            text: Text to type
            clear_existing: Whether to clear existing text
            use_smart_method: Whether to use smart typing method selection
            
        Returns:
            True if successful, False otherwise
        """
        if use_smart_method:
            return self.keyboard.type_smart(text, clear_existing)
        else:
            return self.keyboard.type_text(text, clear_existing)
    
    def press_key(self, key: str) -> bool:
        """
        Press a single key.
        
        Args:
            key: Key to press (e.g., 'enter', 'tab', 'esc')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pyautogui.press(key)
            logging.debug(f"Pressed key: {key}")
            return True
        except Exception as e:
            logging.error(f"Error pressing key {key}: {e}")
            return False
    
    def press_hotkey(self, *keys) -> bool:
        """
        Press a key combination.
        
        Args:
            *keys: Keys to press together (e.g., 'ctrl', 'c')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pyautogui.hotkey(*keys)
            logging.debug(f"Pressed hotkey: {'+'.join(keys)}")
            return True
        except Exception as e:
            logging.error(f"Error pressing hotkey {'+'.join(keys)}: {e}")
            return False
    
    def send_prompt(self, text: str, use_enter: bool = True, 
                   clear_existing: bool = True) -> bool:
        """
        Send a prompt by typing text and submitting.
        
        Args:
            text: Prompt text to type
            use_enter: Whether to use Enter key to submit
            clear_existing: Whether to clear existing text
            
        Returns:
            True if successful, False otherwise
        """
        if not self.type_text(text, clear_existing):
            return False
        
        # Add a small delay before submitting
        time.sleep(random.uniform(0.3, 0.7))
        
        if use_enter:
            return self.press_key('enter')
        
        return True


# Create global instances for easy access
default_keyboard = HumanizedKeyboard()
default_input = InputManager()


# Facade functions to maintain backward compatibility
def type_text(text: str, clear_existing: bool = True) -> bool:
    """Type text with human-like timing (facade function)."""
    return default_keyboard.type_smart(text, clear_existing)

def send_text(text: str, target=None, delay: float = 0.03) -> bool:
    """
    Legacy compatibility function for src/automation/interaction.py:send_text.
    
    Args:
        text: Text to type
        target: Ignored (for compatibility)
        delay: Typing delay (used to adjust profile)
        
    Returns:
        True if successful, False otherwise
    """
    # Adjust typing profile based on delay
    if delay <= 0.01:
        default_keyboard.set_profile('fast')
    elif delay >= 0.05:
        default_keyboard.set_profile('slow')
    else:
        default_keyboard.set_profile('normal')
        
    return default_keyboard.type_smart(text, True)

def press_key(key: str) -> bool:
    """Press a key (facade function)."""
    return default_input.press_key(key)

def press_hotkey(*keys) -> bool:
    """Press a key combination (facade function)."""
    return default_input.press_hotkey(*keys)