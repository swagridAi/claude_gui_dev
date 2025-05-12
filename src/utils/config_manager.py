import yaml
import os
import logging
from pathlib import Path
import copy
import numpy as np

# Configuration constants
class Constants:
    """Configuration constants for the ConfigManager."""
    VERSION = "1.0"
    DEFAULT_CONFIG_DIR = "config"
    TEMP_CONFIG_DIR = "config/temp"
    SESSION_CACHE_TTL = 60  # seconds


# Centralized YAML handler management
class YAMLHandler:
    """Manages YAML custom representers and constructors."""
    
    @staticmethod
    def represent_tuple(dumper, data):
        """Custom representer for Python tuples in YAML."""
        return dumper.represent_sequence('tag:yaml.org,2002:seq', list(data))
    
    @staticmethod
    def construct_tuple(loader, node):
        """Custom constructor for Python tuples from YAML."""
        return tuple(loader.construct_sequence(node))
    
    @staticmethod
    def represent_numpy_scalar(dumper, data):
        """Convert numpy scalar to regular Python int/float."""
        return dumper.represent_scalar('tag:yaml.org,2002:int', str(int(data)))
    
    @staticmethod
    def construct_numpy_scalar(loader, node):
        """Construct a regular Python int from YAML."""
        value = loader.construct_scalar(node)
        return int(value)
    
    @classmethod
    def register_handlers(cls):
        """Register all custom YAML handlers."""
        yaml.add_representer(tuple, cls.represent_tuple)
        yaml.add_constructor('tag:yaml.org,2002:python/tuple', cls.construct_tuple)
        
        # Add numpy scalar handlers if numpy is available
        if hasattr(np, 'int64'):
            yaml.add_representer(np.int64, cls.represent_numpy_scalar)
            yaml.add_constructor('tag:yaml.org,2002:numpy.int64', cls.construct_numpy_scalar)


# Base configuration class
class ConfigBase:
    """Base class for configuration management with common patterns."""
    
    def __init__(self):
        """Initialize base configuration."""
        self.config = {}
    
    def get(self, key, default=None):
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (can be nested using dot notation)
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        if '.' in key:
            parts = key.split('.')
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        
        return self.config.get(key, default)
    
    def set(self, key, value):
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (can be nested using dot notation)
            value: Value to set
        """
        if '.' in key:
            parts = key.split('.')
            config = self.config
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
            config[parts[-1]] = value
        else:
            self.config[key] = value
    
    def _deep_update(self, source, update):
        """
        Deep update a nested dictionary.
        
        Args:
            source: Source dictionary to update
            update: Dictionary with updates to apply
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in source and isinstance(source[key], dict):
                self._deep_update(source[key], value)
            else:
                source[key] = value


# Main configuration manager
class ConfigManager(ConfigBase):
    """Configuration manager that handles loading and saving YAML config files."""
    
    def __init__(self, config_path="config/user_config.yaml", default_config_path="config/default_config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to user configuration file
            default_config_path: Path to default configuration file
        """
        super().__init__()
        self.config_path = config_path
        self.default_config_path = default_config_path
        self._original_config = {}
        self._in_session_mode = False
        self._original_sessions = None
        
        # Register YAML handlers
        YAMLHandler.register_handlers()
        
        # Load configurations
        self._load_configurations()
        
        # Store original config for restoration
        self._original_config = copy.deepcopy(self.config)
    
    def _load_config_file(self, file_path):
        """
        Load configuration from a single file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Loaded configuration dict or empty dict if error
        """
        if not os.path.exists(file_path):
            logging.warning(f"Configuration file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                logging.debug(f"Loaded configuration from {file_path}")
                return config
        except Exception as e:
            logging.error(f"Error loading configuration from {file_path}: {e}")
            return {}
    
    def _load_configurations(self):
        """Load default and user configurations."""
        # Load default configuration
        if os.path.exists(self.default_config_path):
            logging.debug(f"Loading default configuration from {self.default_config_path}")
            self.config = self._load_config_file(self.default_config_path)
        
        # Load user configuration and merge with defaults
        if os.path.exists(self.config_path):
            logging.debug(f"Loading user configuration from {self.config_path}")
            user_config = self._load_config_file(self.config_path)
            self._deep_update(self.config, user_config)
            # Store original sessions for preservation
            self._original_sessions = copy.deepcopy(self.config.get('sessions', {}))
        else:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            logging.info(f"User configuration not found, creating empty file at {self.config_path}")
            self.save()
    
    def save(self, path=None):
        """
        Save configuration to file.
        
        Args:
            path: Optional path to save to (defaults to config_path)
        
        Returns:
            True if successful, False otherwise
        """
        save_path = path or self.config_path
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Use session-preserving save if in session mode
            if self._in_session_mode:
                return self.save_preserving_sessions(save_path)
            
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            logging.debug(f"Configuration saved to {save_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            return False
    
    def save_preserving_sessions(self, path=None):
        """
        Save configuration while preserving the original sessions structure.
        
        Args:
            path: Optional path to save to (defaults to config_path)
        
        Returns:
            True if successful, False otherwise
        """
        save_path = path or self.config_path
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Create a preserved config with original sessions
            preserved_config = copy.deepcopy(self.config)
            if self._original_sessions is not None:
                preserved_config['sessions'] = self._original_sessions
            
            with open(save_path, 'w') as f:
                yaml.dump(preserved_config, f, default_flow_style=False)
            
            logging.debug(f"Configuration saved with preserved sessions to {save_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving configuration with preserved sessions: {e}")
            return False
    
    def get_all(self):
        """
        Get the entire configuration.
        
        Returns:
            A deep copy of the configuration dictionary
        """
        return copy.deepcopy(self.config)
    
    def reset_to_defaults(self):
        """
        Reset configuration to defaults.
        
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self.default_config_path):
            logging.error(f"Default configuration not found at {self.default_config_path}")
            return False
        
        self.config = self._load_config_file(self.default_config_path)
        logging.info("Configuration reset to defaults")
        return True
    
    def enter_session_mode(self, session_id=None):
        """
        Enter session mode - changes won't affect original sessions.
        
        Args:
            session_id: Optional session identifier to track
        """
        if not self._in_session_mode:
            # Store original sessions if not already stored
            if self._original_sessions is None:
                self._original_sessions = copy.deepcopy(self.config.get('sessions', {}))
            
            self._in_session_mode = True
            logging.debug(f"Entered session mode with session_id: {session_id}")
    
    def exit_session_mode(self):
        """
        Exit session mode and restore original sessions.
        
        Returns:
            True if successful, False otherwise
        """
        if self._in_session_mode:
            self._in_session_mode = False
            logging.debug("Exited session mode")
            return True
        return False
    
    def is_in_session_mode(self):
        """
        Check if the configuration manager is in session mode.
        
        Returns:
            True if in session mode, False otherwise
        """
        return self._in_session_mode
    
    def restore_original_config(self):
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
    
    def get_working_copy(self):
        """
        Get a working copy of the configuration manager.
        
        Returns:
            A new ConfigManager instance with the same settings
        """
        working_copy = ConfigManager(self.config_path, self.default_config_path)
        working_copy.config = copy.deepcopy(self.config)
        working_copy._in_session_mode = True
        working_copy._original_sessions = copy.deepcopy(self._original_sessions or self.config.get('sessions', {}))
        
        return working_copy
    
    def merge_session_config(self, session_id):
        """
        Merge a specific session configuration with global settings.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        sessions = self.get('sessions', {})
        
        if session_id not in sessions:
            logging.error(f"Session '{session_id}' not found")
            return False
        
        # Enter session mode
        self.enter_session_mode(session_id)
        
        # Get session configuration
        session_config = sessions[session_id]
        
        # Get global config without sessions
        global_config = self.get_all()
        if 'sessions' in global_config:
            del global_config['sessions']
        
        # Merge configurations (session overrides global)
        merged_config = {**global_config, **session_config}
        
        # Update configuration
        for key, value in merged_config.items():
            if key != 'sessions':
                self.set(key, value)
        
        logging.info(f"Merged configuration for session '{session_id}'")
        return True