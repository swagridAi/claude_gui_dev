"""
Application Controller for the Claude UI Calibration Tool.

This module contains the central controller that coordinates all components
of the calibration tool application.
"""

import logging
import tkinter as tk
from tkinter import ttk

from ..models.config_model import ConfigManager


class AppController:
    """Main application controller that coordinates all components of the calibration tool."""
    
    def __init__(self, root, config_manager=None):
        """
        Initialize the application controller.
        
        Args:
            root: Tkinter root window
            config_manager: Optional ConfigManager instance
        """
        self.root = root
        self.config_manager = config_manager or ConfigManager()
        self.tab_controllers = {}
        self.status_var = None
        self.notebook = None
        self._screenshot = None  # Shared screenshot across tabs
        self._tab_names = ["Setup", "Define", "Analyze", "Verify"]  # Tab names for reference
    
    def initialize_components(self):
        """Initialize all UI components and controllers."""
        # Create main UI structure
        self._create_main_ui()
        
        # Initialize controllers with dependencies
        self._initialize_controllers()
        
        # Set up event bindings
        self._setup_event_bindings()
    
    def _create_main_ui(self):
        """Create the main UI structure."""
        # Set window title and size
        self.root.title("Claude UI Unified Calibration Tool")
        self.root.state('zoomed')  # Maximize window
        
        # Create a tab control
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
    
    def _initialize_controllers(self):
        """Initialize all tab controllers with proper dependencies."""
        # Import controllers here to avoid circular imports
        from ..controllers.element_controller import ElementController
        from ..views.setup_tab import SetupTabView
        from ..views.define_tab import DefineTabView
        from ..views.analyze_tab import AnalyzeTabView
        from ..views.verify_tab import VerifyTabView
        
        # Create tabs and their frames
        setup_frame = ttk.Frame(self.notebook)
        define_frame = ttk.Frame(self.notebook)
        analyze_frame = ttk.Frame(self.notebook)
        verify_frame = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(setup_frame, text=self._tab_names[0])
        self.notebook.add(define_frame, text=self._tab_names[1])
        self.notebook.add(analyze_frame, text=self._tab_names[2])
        self.notebook.add(verify_frame, text=self._tab_names[3])
        
        # Create element controller (shared across tabs)
        element_controller = ElementController(self)
        
        # Create tab views
        setup_view = SetupTabView(setup_frame, self)
        define_view = DefineTabView(define_frame, self, element_controller)
        analyze_view = AnalyzeTabView(analyze_frame, self, element_controller)
        verify_view = VerifyTabView(verify_frame, self, element_controller)
        
        # Store tab controllers by index for later reference
        self.tab_controllers = {
            0: setup_view,
            1: define_view,
            2: analyze_view,
            3: verify_view
        }
        
        # Store element controller reference
        self.element_controller = element_controller
    
    def _setup_event_bindings(self):
        """Set up event listeners for application-wide events."""
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _on_tab_changed(self, event):
        """Handle tab change event."""
        tab_id = self.notebook.index("current")
        tab_name = self._get_tab_name(tab_id)
        
        # Update status
        self.show_status(f"Switched to {tab_name} tab")
        
        # Notify controllers of tab change
        if tab_id in self.tab_controllers:
            self.tab_controllers[tab_id].on_tab_activated()
    
    def _on_close(self):
        """Handle window close event."""
        self.shutdown()
        self.root.destroy()
    
    def _get_tab_name(self, tab_id):
        """Get the name of a tab by its index."""
        if 0 <= tab_id < len(self._tab_names):
            return self._tab_names[tab_id]
        return "Unknown"
    
    def show_status(self, message):
        """
        Update status bar message.
        
        Args:
            message: Message to display
        """
        if self.status_var:
            self.status_var.set(message)
            self.root.update_idletasks()
        
        # Log status messages as well
        logging.info(message)
    
    def navigate_to_tab(self, tab_index):
        """
        Navigate to the specified tab.
        
        Args:
            tab_index: Index of the tab to navigate to
        """
        if 0 <= tab_index < self.notebook.index("end"):
            self.notebook.select(tab_index)
            # Tab changed event will trigger _on_tab_changed
    
    def set_screenshot(self, screenshot):
        """
        Set the current screenshot and notify all tabs.
        
        Args:
            screenshot: PIL Image object
        """
        self._screenshot = screenshot
        
        # Notify all controllers of new screenshot
        for controller in self.tab_controllers.values():
            if hasattr(controller, 'update_screenshot'):
                controller.update_screenshot(screenshot)
    
    def get_screenshot(self):
        """
        Get the current screenshot.
        
        Returns:
            PIL Image object or None
        """
        return self._screenshot
    
    def start(self):
        """Start the application."""
        self.initialize_components()
        self.show_status("Application started")
    
    def shutdown(self):
        """Perform cleanup and shutdown operations."""
        # Notify all controllers
        for controller in self.tab_controllers.values():
            if hasattr(controller, 'on_shutdown'):
                controller.on_shutdown()
        
        # Save any pending changes
        if self.config_manager:
            self.config_manager.save()
        
        self.show_status("Application shutting down")
    
    def load_configuration(self):
        """Load configuration data and distribute to controllers."""
        config_data = self.config_manager.load_config()
        
        # Notify element controller of loaded configuration
        if hasattr(self.element_controller, 'set_elements'):
            ui_elements = config_data.get("ui_elements", {})
            self.element_controller.set_elements(ui_elements)
            
        self.show_status("Configuration loaded successfully")
        return config_data
    
    def save_configuration(self):
        """Save configuration data from controllers."""
        # Get elements from element controller
        ui_elements = {}
        if hasattr(self.element_controller, 'get_elements'):
            ui_elements = self.element_controller.get_elements()
        
        # Save to configuration manager
        success = self.config_manager.save_config({"ui_elements": ui_elements})
        
        if success:
            self.show_status("Configuration saved successfully")
        else:
            self.show_status("Failed to save configuration")
            
        return success