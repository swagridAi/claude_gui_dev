#!/usr/bin/env python3
"""
Base Configuration Management Module

Provides a standardized base class for configuration management
with dot notation access, YAML file handling, and consistent error management.
"""

import os
import logging
import copy
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple


class ConfigError(Exception):
    """Base exception for configuration-related errors."""
    pass


class ConfigLoadError(ConfigError):
    """Exception raised when configuration cannot be loaded."""
    pass


class ConfigSaveError(ConfigError):
    """Exception raised when configuration cannot be saved."""
    pass


class ConfigAccessError(ConfigError):
    """Exception raised when accessing invalid configuration."""
    pass


class ConfigBase:
    """
    Base class for configuration management with YAML files.
    
    Provides standardized methods for loading, accessing, and saving
    configuration with support for dot notation for nested access.
    """
    
    def __init__(self, config_path: str = "config/user_config.yaml", 
                 default_config_path: Optional[str] = None):
        """
        Initialize configuration with specified paths.
        
        Args:
            config_path: Path to the main configuration file
            default_config_path: Optional path to default configuration
        """
        self.config_path = config_path
        self.default_config_path = default_config_path
        self.config = {}
        self._original_config = {}  # For preserving original state
        self._in_session_mode = False  # For temporary modifications
        
        # Initialize configuration
        self._load_config()
    
    def _load_config(self) -> None:
        """
        Load configuration from file with error handling.
        
        Attempts to load default configuration first if available,
        then overlays user configuration.
        """
        try:
            # Load default configuration if specified
            if self.default_config_path and os.path.exists(self.default_config_path):
                self._load_yaml_file(self.default_config_path)
            
            # Load user configuration if it exists
            if os.path.exists(self.config_path):
                user_config = self._load_yaml_file(self.config_path)
                self._deep_update(self.config, user_config)
                
                # Store original config for preservation
                self._original_config = copy.deepcopy(self.config)
            else:
                # Create user config directory if it doesn't exist
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                logging.info(f"User configuration not found at {self.config_path}, using defaults")
                
                # Save current config as user config
                self.save()
                
                # Store as original
                self._original_config = copy.deepcopy(self.config)
                
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise ConfigLoadError(f"Failed to load configuration: {e}")
    
    def _load_yaml_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load a YAML file with error handling.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Dictionary with configuration data
            
        Raises:
            ConfigLoadError: If file cannot be loaded or parsed
        """
        try:
            import yaml  # Import here to avoid global dependency
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
                logging.debug(f"Loaded configuration from {file_path}")
                return config_data
        except ImportError:
            logging.error("PyYAML not installed. Cannot load configuration.")
            raise ConfigLoadError("PyYAML not installed. Please install with 'pip install pyyaml'")
        except yaml.YAMLError as e:
            logging.error(f"YAML parsing error in {file_path}: {e}")
            raise ConfigLoadError(f"Invalid YAML format in {file_path}: {e}")
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            raise ConfigLoadError(f"Failed to load {file_path}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with dot notation support.
        
        Args:
            key: Configuration key (can use dot notation for nested access)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        # This is the core duplicated function being centralized
        if '.' in key:
            parts = key.split('.')
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        
        # Simple key lookup
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value with dot notation support.
        
        Args:
            key: Configuration key (can use dot notation for nested access)
            value: Value to set
        """
        # Handle nested keys with dot notation
        if '.' in key:
            parts = key.split('.')
            config = self.config
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                elif not isinstance(config[part], dict):
                    config[part] = {}
                config = config[part]
            config[parts[-1]] = value
        else:
            # Simple key update
            self.config[key] = value
    
    def save(self, path: Optional[str] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            path: Optional path to save to (defaults to config_path)
            
        Returns:
            True if successful, False otherwise
        """
        save_path = path or self.config_path
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Handle session mode if active
            if self._in_session_mode:
                return self._save_preserving_original(save_path)
            
            # Standard save
            self._save_yaml_file(save_path, self.config)
            logging.debug(f"Configuration saved to {save_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            return False
    
    def _save_yaml_file(self, file_path: str, data: Dict[str, Any]) -> None:
        """
        Save data to a YAML file with error handling.
        
        Args:
            file_path: Path to save to
            data: Dictionary data to save
            
        Raises:
            ConfigSaveError: If file cannot be saved
        """
        try:
            import yaml  # Import here to avoid global dependency
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        except ImportError:
            logging.error("PyYAML not installed. Cannot save configuration.")
            raise ConfigSaveError("PyYAML not installed. Please install with 'pip install pyyaml'")
        except Exception as e:
            logging.error(f"Error saving to {file_path}: {e}")
            raise ConfigSaveError(f"Failed to save configuration to {file_path}: {e}")
    
    def _save_preserving_original(self, path: str) -> bool:
        """
        Save configuration while preserving original structure.
        
        Used in session mode to prevent overwriting certain sections.
        
        Args:
            path: Path to save to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            preserved_config = copy.deepcopy(self._original_config)
            self._update_preserving_structure(preserved_config, self.config)
            self._save_yaml_file(path, preserved_config)
            logging.debug(f"Configuration saved with preservation to {path}")
            return True
        except Exception as e:
            logging.error(f"Error saving preserved configuration: {e}")
            return False
    
    def _update_preserving_structure(self, target: Dict[str, Any], 
                                    source: Dict[str, Any]) -> None:
        """
        Update target dictionary from source while preserving structure.
        
        Default implementation does a simple deep update. Subclasses can
        override this to implement specific preservation logic.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with updates
        """
        self._deep_update(target, source)
    
    def _deep_update(self, target: Dict[str, Any], 
                    source: Dict[str, Any]) -> None:
        """
        Recursively update a nested dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with updates
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # Recursive update for nested dictionaries
                self._deep_update(target[key], value)
            else:
                # Simple update for non-dictionary values
                target[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration.
        
        Returns:
            A deep copy of the configuration dictionary
        """
        return copy.deepcopy(self.config)
    
    def reset_to_defaults(self) -> bool:
        """
        Reset configuration to defaults.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.default_config_path or not os.path.exists(self.default_config_path):
                logging.warning("No default configuration available for reset")
                return False
                
            # Load default configuration
            default_config = self._load_yaml_file(self.default_config_path)
            
            # Replace current configuration
            self.config = default_config
            logging.info("Configuration reset to defaults")
            return True
        except Exception as e:
            logging.error(f"Error resetting configuration: {e}")
            return False
    
    def restore_original_config(self) -> bool:
        """
        Restore the original configuration.
        
        Returns:
            True if successful, False otherwise
        """
        if self._original_config:
            self.config = copy.deepcopy(self._original_config)
            logging.info("Restored original configuration")
            return True
        else:
            logging.warning("No original configuration to restore")
            return False
    
    # Session mode methods for temporary changes
    
    def enter_session_mode(self) -> None:
        """Enter session mode for temporary configuration changes."""
        self._in_session_mode = True
        logging.debug("Entered session mode")
    
    def exit_session_mode(self) -> None:
        """Exit session mode, returning to normal operation."""
        self._in_session_mode = False
        logging.debug("Exited session mode")
    
    def is_in_session_mode(self) -> bool:
        """
        Check if currently in session mode.
        
        Returns:
            True if in session mode, False otherwise
        """
        return self._in_session_mode
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration format and required values.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        # Base implementation does no validation
        # Subclasses should override this for specific validation
        return True, []