import yaml
import os
import logging
import copy
from pathlib import Path

from src.utils.config_manager import ConfigManager


class CalibrationConfig:
    """
    Manages loading, saving, and converting configuration data for the calibration tool.
    Provides a clean interface to the underlying ConfigManager implementation.
    """
    
    def __init__(self, config_path="config/user_config.yaml", default_config_path="config/default_config.yaml"):
        """
        Initialize configuration manager with specified paths.
        
        Args:
            config_path: Path to user configuration file
            default_config_path: Path to default configuration file
        """
        self.config_path = config_path
        self.default_config_path = default_config_path
        self.config_manager = ConfigManager(config_path, default_config_path)
        self._original_state = {}
        self._modified = False
    
    def load_from_file(self, custom_path=None):
        """
        Load configuration from specified file or default.
        
        Args:
            custom_path: Optional alternate path to load from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if custom_path and os.path.exists(custom_path):
                self.config_manager = ConfigManager(custom_path, self.default_config_path)
            
            # Store original state for change tracking
            self._original_state = copy.deepcopy(self.config_manager.get_all())
            self._modified = False
            
            logging.info(f"Configuration loaded from {custom_path or self.config_path}")
            return True
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            return False
    
    def save_to_file(self, custom_path=None, preserve_sessions=True):
        """
        Save configuration to file.
        
        Args:
            custom_path: Optional path to save to
            preserve_sessions: Whether to preserve session data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine save path
            save_path = custom_path or self.config_path
            
            # Handle session preservation
            if preserve_sessions:
                # Enter session mode if not already in it
                if not self.config_manager.is_in_session_mode():
                    self.config_manager.enter_session_mode()
                    result = self.config_manager.save(save_path)
                    self.config_manager.exit_session_mode()
                    self._modified = False
                    return result
                else:
                    # Already in session mode, use preserving save
                    result = self.config_manager.save_preserving_sessions(save_path)
                    self._modified = False
                    return result
            else:
                # Standard save without session preservation
                result = self.config_manager.save(save_path)
                self._modified = False
                return result
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            return False
    
    def get_ui_elements(self):
        """
        Return UI elements configuration in standardized format.
        
        Returns:
            Dictionary of UI element configurations
        """
        return self.config_manager.get("ui_elements", {})
    
    def update_ui_elements(self, ui_elements_dict):
        """
        Update UI elements in configuration.
        
        Args:
            ui_elements_dict: Dictionary of UI element configurations
            
        Returns:
            True if successful
        """
        self.config_manager.set("ui_elements", ui_elements_dict)
        self._modified = True
        return True
    
    def convert_to_element_models(self):
        """
        Convert configuration to ElementModel instances.
        
        This method would create ElementModel objects from the UI element configuration.
        The actual implementation depends on ElementModel class from element_model.py.
        
        Returns:
            Dictionary of element name to ElementModel instance
        """
        # This is a stub implementation that would be completed when element_model.py is created
        # It would convert configuration dictionary to ElementModel objects
        ui_elements = self.get_ui_elements()
        element_models = {}
        
        # In the actual implementation, we would import ElementModel and create instances
        # For now, we'll just return the raw configuration
        for name, config in ui_elements.items():
            element_models[name] = config  # Would be: ElementModel(name, **config)
            
        return element_models
    
    def update_from_element_models(self, element_models):
        """
        Update configuration from ElementModel instances.
        
        Args:
            element_models: Dictionary of element name to ElementModel instance
            
        Returns:
            True if successful
        """
        # This is a stub implementation that would be completed when element_model.py is created
        # It would convert ElementModel objects back to configuration dictionaries
        ui_elements = {}
        
        # In the actual implementation, we would convert ElementModel to dict
        # For now, we'll just update with the raw data
        for name, model in element_models.items():
            ui_elements[name] = model  # Would be: model.to_dict()
            
        self.update_ui_elements(ui_elements)
        return True
    
    def get_value(self, key, default=None):
        """
        Get configuration value with dot notation support.
        
        Args:
            key: Configuration key (can be nested using dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config_manager.get(key, default)
    
    def set_value(self, key, value):
        """
        Set configuration value with dot notation support.
        
        Args:
            key: Configuration key (can be nested using dot notation)
            value: Value to set
            
        Returns:
            True if successful
        """
        self.config_manager.set(key, value)
        self._modified = True
        return True
    
    def is_modified(self):
        """
        Check if configuration has been modified since last load/save.
        
        Returns:
            True if modified, False otherwise
        """
        return self._modified
    
    def reset(self):
        """
        Reset configuration to last saved state.
        
        Returns:
            True if successful
        """
        if self._original_state:
            self.config_manager.config = copy.deepcopy(self._original_state)
            self._modified = False
            return True
        return False
    
    def validate(self):
        """
        Validate the current configuration.
        
        Returns:
            Tuple of (valid, errors) where valid is a boolean and errors is a list of error messages
        """
        errors = []
        
        # Basic validation
        ui_elements = self.get_ui_elements()
        if not ui_elements:
            errors.append("No UI elements defined")
        
        for name, config in ui_elements.items():
            # Validate presence of required fields
            if not config.get("reference_paths"):
                errors.append(f"Element '{name}' has no reference images")
                
            # More validation logic would go here
        
        return (len(errors) == 0, errors)
    
    def create_temp_config(self):
        """
        Create a temporary configuration file for testing.
        
        Returns:
            Path to temporary configuration file
        """
        import tempfile
        
        # Create temporary file
        fd, path = tempfile.mkstemp(suffix='.yaml')
        os.close(fd)
        
        # Save current config to temp file
        self.save_to_file(path)
        
        return path