#!/usr/bin/env python3
"""
Claude UI Calibration Tool - Entry Point

A modular tool for calibrating UI element recognition for Claude GUI Automation.
This tool helps capture, configure, and test UI elements for automated interactions
with the Claude AI interface.
"""

import sys
import logging
import traceback
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# Add project root to path for relative imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Internal imports from new module structure
from tools.calibration.models.config_model import CalibrationConfig
from tools.calibration.models.element_model import ElementModel
from tools.calibration.controllers.app_controller import AppController
from tools.calibration.views.setup_tab import SetupTabView
from tools.calibration.views.define_tab import DefineTabView
from tools.calibration.views.analyze_tab import AnalyzeTabView
from tools.calibration.views.verify_tab import VerifyTabView


def setup_exception_handler(root):
    """
    Configure global exception handler to show error dialogs.
    
    Args:
        root: Tkinter root window
    """
    def show_error(exc_type, exc_value, exc_traceback):
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logging.error(f"Unhandled exception: {error_msg}")
        messagebox.showerror("Error", f"An unexpected error occurred:\n{exc_value}\n\nSee logs for details.")
        
    # Store original handler for restoration if needed
    original_handler = sys.excepthook
    sys.excepthook = show_error
    
    return original_handler


def create_main_window():
    """
    Create and configure the main application window.
    
    Returns:
        tk.Tk: Configured root window
    """
    root = tk.Tk()
    root.title("Claude UI Calibration Tool")
    root.state('zoomed')  # Maximize window
    
    # Configure window icon if available
    try:
        icon_path = Path(__file__).parent / "resources" / "icon.ico"
        if icon_path.exists():
            root.iconbitmap(str(icon_path))
    except Exception:
        pass  # Proceed without icon if not available
        
    # Configure window close handler
    root.protocol("WM_DELETE_WINDOW", lambda: root.destroy())
    
    return root


def initialize_components(root):
    """
    Initialize all application components and connect them.
    
    Args:
        root: Tkinter root window
        
    Returns:
        AppController: Main application controller
    """
    # Initialize configuration model
    config = CalibrationConfig()
    
    # Create notebook for tabs
    notebook = tk.ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Create status bar
    status_var = tk.StringVar(value="Ready")
    status_bar = tk.ttk.Label(
        root, 
        textvariable=status_var, 
        relief=tk.SUNKEN, 
        anchor=tk.W
    )
    status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5,0))
    
    # Initialize tab views
    setup_view = SetupTabView(notebook, status_var)
    define_view = DefineTabView(notebook, status_var)
    analyze_view = AnalyzeTabView(notebook, status_var)
    verify_view = VerifyTabView(notebook, status_var)
    
    # Add tabs to notebook
    notebook.add(setup_view.frame, text="1. Setup")
    notebook.add(define_view.frame, text="2. Define Elements")
    notebook.add(analyze_view.frame, text="3. Analyze")
    notebook.add(verify_view.frame, text="4. Verify")
    
    # Create application controller
    app_controller = AppController(
        root=root,
        notebook=notebook,
        status_var=status_var,
        config=config,
        views={
            'setup': setup_view,
            'define': define_view,
            'analyze': analyze_view,
            'verify': verify_view
        }
    )
    
    # Connect tab change event
    notebook.bind("<<NotebookTabChanged>>", app_controller.on_tab_changed)
    
    return app_controller


def run_application():
    """
    Run the calibration tool application.
    
    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        # Create and configure the main window
        root = create_main_window()
        
        # Set up exception handler
        original_handler = setup_exception_handler(root)
        
        # Initialize application components
        app_controller = initialize_components(root)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("calibration_tool.log"),
                logging.StreamHandler()
            ]
        )
        
        logging.info("Starting Claude UI Calibration Tool")
        
        # Start Tkinter main loop
        root.mainloop()
        
        # Restore original exception handler
        sys.excepthook = original_handler
        
        logging.info("Calibration Tool exited normally")
        return 0
        
    except Exception as e:
        logging.error(f"Error starting application: {e}")
        traceback.print_exc()
        return 1


def main():
    """Application entry point with error handling."""
    exit_code = run_application()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()