#!/usr/bin/env python3
"""
Unified Calibration Tool for Claude GUI Automation

This tool combines the functionality of simplified_calibration.py and debug_based_calibration.py
into a single comprehensive interface for configuring and optimizing UI element detection.
"""

import os
import sys
import time
import logging
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageEnhance
import pyautogui
import cv2
import numpy as np
import glob
import re
from pathlib import Path
from datetime import datetime
import yaml

# Add project root to path
sys.path.append('.')

# Import project modules
try:
    from src.utils.config_manager import ConfigManager
    from src.utils.reference_manager import ReferenceImageManager
    from src.utils.region_manager import RegionManager
    from src.models.ui_element import UIElement
    from src.automation.recognition import find_element
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class UnifiedCalibrationTool:

    # Add this in the __init__ method of the UnifiedCalibrationTool class
    # Right after initializing the managers:

    def __init__(self, root):
        self.root = root
        self.root.title("Claude UI Unified Calibration Tool")
        self.root.state('zoomed')  # Maximize window
        
        # Initialize managers and data
        self.config_manager = ConfigManager()
        self.reference_manager = ReferenceImageManager()
        self.region_manager = RegionManager()
        
        # Add this line to define the default_config_path
        self.default_config_path = "config/default_config.yaml"
        
        # UI-related variables
        self.screenshot = None
        self.screenshot_image = None
        self.current_element = None
        self.elements = {}  # Dictionary to store element configurations
        self.reference_count = {}  # Count of reference images per element
        self.scale = 1.0  # Scale factor for display
        self.selection_start = None
        self.selection_box = None
        self.capture_mode = False
        self.debug_images = {}  # Dictionary to store debug images by element
        
        # Set up the UI
        self.setup_ui()
        
        # Ensure directories exist
        self.setup_directories()
        
        # Load existing configuration
        self.load_configuration()
        
        # Show welcome message
        self.show_status("Welcome to the Unified Calibration Tool. Start by loading or capturing a screenshot.")

    def setup_directories(self):
        """Create necessary directories."""
        directories = [
            "assets/reference_images",
            "config",
            "logs/screenshots",
            "logs/recognition_debug",
            "logs/click_debug"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def setup_ui(self):
        """Set up the main user interface."""
        # Create main layout as a tab control
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.setup_tab = ttk.Frame(self.notebook)
        self.define_tab = ttk.Frame(self.notebook)
        self.analyze_tab = ttk.Frame(self.notebook)
        self.verify_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.setup_tab, text="1. Setup")
        self.notebook.add(self.define_tab, text="2. Define Elements")
        self.notebook.add(self.analyze_tab, text="3. Analyze")
        self.notebook.add(self.verify_tab, text="4. Verify")
        
        # Set up each tab
        self.setup_setup_tab()
        self.setup_define_tab()
        self.setup_analyze_tab()
        self.setup_verify_tab()
        
        # Create status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5,0))
        
        # Bind events
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def setup_setup_tab(self):
        """Set up the initial setup tab."""
        # Create a paned window for the setup tab
        pane = ttk.PanedWindow(self.setup_tab, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        left_frame = ttk.Frame(pane, padding=10)
        pane.add(left_frame, weight=1)
        
        # Right panel - Screenshot
        right_frame = ttk.LabelFrame(pane, text="Claude Interface Preview")
        pane.add(right_frame, weight=3)
        
        # Controls in left frame
        ttk.Label(left_frame, text="Initial Setup", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))
        ttk.Label(left_frame, text="Capture or load a screenshot of Claude:").pack(anchor="w", pady=(0, 5))
        
        # Buttons for screenshot actions
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="Take Screenshot", 
                command=self.take_screenshot).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Load Image", 
                command=self.load_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Load from Logs", 
                command=self.show_log_screenshots).pack(side=tk.LEFT)
        
        # Element management
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_frame, text="UI Elements", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        # New element input
        input_frame = ttk.Frame(left_frame)
        input_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.element_var = tk.StringVar()
        ttk.Label(input_frame, text="Element Name:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(input_frame, textvariable=self.element_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="Add", command=self.add_element).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Element list
        list_frame = ttk.LabelFrame(left_frame, text="Defined Elements")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        self.element_listbox = tk.Listbox(list_frame, height=10)
        self.element_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.element_listbox.bind('<<ListboxSelect>>', self.on_element_select)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.element_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.element_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Element actions
        action_frame = ttk.Frame(left_frame)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(action_frame, text="Delete Element", 
                command=self.delete_element).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(action_frame, text="Go to Define Tab", 
                command=lambda: self.notebook.select(1)).pack(side=tk.RIGHT)
        
        # Configuration actions
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        config_frame = ttk.Frame(left_frame)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(config_frame, text="Load Config", 
                command=self.load_configuration).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(config_frame, text="Save Config", 
                command=self.save_configuration).pack(side=tk.LEFT)
        
        # Canvas for screenshot display
        self.setup_canvas = tk.Canvas(right_frame, bg="gray90")
        self.setup_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Canvas bindings for setup tab
        self.setup_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.setup_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.setup_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)


    def scan_log_screenshots(self):
        """Scan for screenshots in the logs directory and organize them by run and element."""
        log_screenshots = {}
        
        # Find all run directories
        run_dirs = sorted(glob.glob("logs/run_*"), reverse=True)  # Sort newest first
        
        for run_dir in run_dirs:
            run_name = os.path.basename(run_dir)
            screenshot_dir = os.path.join(run_dir, "screenshots")
            
            if not os.path.exists(screenshot_dir):
                continue
                
            log_screenshots[run_name] = {
                "all": [],
                "elements": {}
            }
            
            # Scan all screenshots in this run directory
            screenshots = glob.glob(os.path.join(screenshot_dir, "*.png"))
            log_screenshots[run_name]["all"] = screenshots
            
            # Categorize by UI element
            for screenshot in screenshots:
                filename = os.path.basename(screenshot)
                
                # Look for element-specific screenshots based on filename patterns
                for element_name in self.elements.keys():
                    if element_name.lower() in filename.lower():
                        if element_name not in log_screenshots[run_name]["elements"]:
                            log_screenshots[run_name]["elements"][element_name] = []
                        log_screenshots[run_name]["elements"][element_name].append(screenshot)
                
                # Also check for generic patterns like SEARCH_element_START
                match = re.search(r"(?:SEARCH|FOUND|NOT_FOUND)_(\w+)", filename)
                if match:
                    element_name = match.group(1)
                    if element_name not in log_screenshots[run_name]["elements"]:
                        log_screenshots[run_name]["elements"][element_name] = []
                    log_screenshots[run_name]["elements"][element_name].append(screenshot)
        
        return log_screenshots
    
    def show_log_screenshots(self):
        """Show a dialog with log screenshots organized by run and element."""
        # Scan for log screenshots
        log_screenshots = self.scan_log_screenshots()
        
        if not log_screenshots:
            messagebox.showinfo("No Screenshots", "No log screenshots found.")
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Load Screenshot from Logs")
        dialog.geometry("800x600")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Create a paned window
        pane = ttk.PanedWindow(dialog, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Run and element selection
        left_frame = ttk.Frame(pane)
        pane.add(left_frame, weight=1)
        
        # Right panel - Screenshot preview
        right_frame = ttk.LabelFrame(pane, text="Screenshot Preview")
        pane.add(right_frame, weight=2)
        
        # Run selection
        ttk.Label(left_frame, text="Select Run:").pack(anchor="w", pady=(0, 5))
        
        run_frame = ttk.Frame(left_frame)
        run_frame.pack(fill=tk.X, pady=(0, 10))
        
        run_listbox = tk.Listbox(run_frame, height=5)
        run_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        run_scrollbar = ttk.Scrollbar(run_frame, orient=tk.VERTICAL, command=run_listbox.yview)
        run_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        run_listbox.configure(yscrollcommand=run_scrollbar.set)
        
        # Element selection
        ttk.Label(left_frame, text="Select Element:").pack(anchor="w", pady=(0, 5))
        
        element_frame = ttk.Frame(left_frame)
        element_frame.pack(fill=tk.X, pady=(0, 10))
        
        element_listbox = tk.Listbox(element_frame, height=5)
        element_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        element_scrollbar = ttk.Scrollbar(element_frame, orient=tk.VERTICAL, command=element_listbox.yview)
        element_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        element_listbox.configure(yscrollcommand=element_scrollbar.set)
        
        # Screenshot selection
        ttk.Label(left_frame, text="Select Screenshot:").pack(anchor="w", pady=(0, 5))
        
        screenshot_frame = ttk.Frame(left_frame)
        screenshot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        screenshot_listbox = tk.Listbox(screenshot_frame)
        screenshot_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        screenshot_scrollbar = ttk.Scrollbar(screenshot_frame, orient=tk.VERTICAL, command=screenshot_listbox.yview)
        screenshot_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        screenshot_listbox.configure(yscrollcommand=screenshot_scrollbar.set)
        
        # Preview canvas
        preview_canvas = tk.Canvas(right_frame, bg="gray90")
        preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Store preview image reference to prevent garbage collection
        preview_image_ref = None
        
        # Action buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(button_frame, text="Cancel", 
                command=dialog.destroy).pack(side=tk.RIGHT, padx=(5, 0))
        
        load_button = ttk.Button(button_frame, text="Load Selected", 
                                command=lambda: None)  # Will set actual command later
        load_button.pack(side=tk.RIGHT)
        load_button.config(state=tk.DISABLED)  # Disabled until selection made
        
        # Fill run listbox
        for run_name in log_screenshots.keys():
            run_listbox.insert(tk.END, run_name)
        
        # Select the first run if available
        if run_listbox.size() > 0:
            run_listbox.selection_set(0)
            current_run = run_listbox.get(0)
        else:
            current_run = None
        
        # Function to update element listbox based on selected run
        def update_element_list(*args):
            nonlocal current_run
            selection = run_listbox.curselection()
            if not selection:
                return
            
            current_run = run_listbox.get(selection[0])
            
            # Clear element listbox
            element_listbox.delete(0, tk.END)
            
            # Add "All Screenshots" option
            element_listbox.insert(tk.END, "All Screenshots")
            
            # Add elements with screenshots
            for element_name in sorted(log_screenshots[current_run]["elements"].keys()):
                element_listbox.insert(tk.END, element_name)
            
            # Select first element
            if element_listbox.size() > 0:
                element_listbox.selection_set(0)
                update_screenshot_list()
        
        # Dictionary to store path mapping
        path_mapping = {}
        
        # Function to update screenshot listbox based on selected element
        def update_screenshot_list(*args):
            nonlocal current_run, path_mapping
            if not current_run:
                return
                
            element_selection = element_listbox.curselection()
            if not element_selection:
                return
            
            selected_element = element_listbox.get(element_selection[0])
            
            # Clear screenshot listbox and path mapping
            screenshot_listbox.delete(0, tk.END)
            path_mapping.clear()
            
            # Add screenshots
            screenshots = []
            if selected_element == "All Screenshots":
                screenshots = log_screenshots[current_run]["all"]
            else:
                screenshots = log_screenshots[current_run]["elements"].get(selected_element, [])
            
            for screenshot in screenshots:
                # Extract just the filename for display
                filename = os.path.basename(screenshot)
                screenshot_listbox.insert(tk.END, filename)
                # Store path in our mapping dictionary
                path_mapping[filename] = screenshot
            
            # Clear preview
            preview_canvas.delete("all")
        
        # Function to show preview of selected screenshot
        def show_preview(*args):
            nonlocal preview_image_ref
            
            selection = screenshot_listbox.curselection()
            if not selection:
                load_button.config(state=tk.DISABLED)
                return
            
            # Get full path of selected screenshot
            index = selection[0]
            filename = screenshot_listbox.get(index)
            
            # Get the path from our mapping dictionary
            if filename in path_mapping:
                screenshot_path = path_mapping[filename]
            else:
                self.show_status(f"Path for {filename} not found")
                return
            
            # Load and display image
            try:
                img = Image.open(screenshot_path)
                
                # Get canvas dimensions
                canvas_width = preview_canvas.winfo_width()
                canvas_height = preview_canvas.winfo_height()
                
                # If canvas not yet sized, use defaults
                if canvas_width <= 1:
                    canvas_width = 400
                    canvas_height = 300
                
                # Resize the image to fit the canvas while maintaining aspect ratio
                img_width, img_height = img.size
                scale = min(canvas_width / img_width, canvas_height / img_height)
                
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                # Resize the image
                resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to Tkinter PhotoImage
                preview_image_ref = ImageTk.PhotoImage(resized)
                
                # Clear canvas
                preview_canvas.delete("all")
                
                # Draw the image
                preview_canvas.create_image(0, 0, anchor=tk.NW, image=preview_image_ref)
                preview_canvas.config(scrollregion=(0, 0, new_width, new_height))
                
                # Enable load button
                load_button.config(state=tk.NORMAL)
                
                # Update status
                self.show_status(f"Preview of {filename}")
                
            except Exception as e:
                messagebox.showerror("Preview Error", f"Error loading preview: {e}")
                self.show_status(f"Error loading preview: {e}")
        
        # Function to load the selected screenshot
        def load_selected_screenshot():
            selection = screenshot_listbox.curselection()
            if not selection:
                return
            
            # Get full path of selected screenshot
            index = selection[0]
            filename = screenshot_listbox.get(index)
            
            # Get the path from our mapping dictionary
            if filename in path_mapping:
                screenshot_path = path_mapping[filename]
            else:
                self.show_status(f"Path for {filename} not found")
                return
            
            # Load the screenshot
            try:
                self.screenshot = Image.open(screenshot_path)
                
                # Update canvases
                current_tab = self.notebook.index("current")
                if current_tab == 0:
                    self.update_canvas(self.setup_canvas)
                elif current_tab == 1:
                    self.update_canvas(self.define_canvas)
                
                dialog.destroy()
                
                self.show_status(f"Loaded screenshot from {os.path.basename(screenshot_path)}")
                
            except Exception as e:
                messagebox.showerror("Load Error", f"Error loading screenshot: {e}")
                self.show_status(f"Error loading screenshot: {e}")
        
        # Set the load button command
        load_button.config(command=load_selected_screenshot)
        
        # Bind events
        run_listbox.bind("<<ListboxSelect>>", update_element_list)
        element_listbox.bind("<<ListboxSelect>>", update_screenshot_list)
        screenshot_listbox.bind("<<ListboxSelect>>", show_preview)
        
        # Initialize lists
        update_element_list()

    def setup_define_tab(self):
        """Set up the element definition tab."""
        # Create a paned window
        pane = ttk.PanedWindow(self.define_tab, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Element properties
        left_frame = ttk.Frame(pane, padding=10)
        pane.add(left_frame, weight=1)
        
        # Right panel - Region definition
        right_frame = ttk.LabelFrame(pane, text="Element Region Definition")
        pane.add(right_frame, weight=3)
        
        # Element properties in left frame
        ttk.Label(left_frame, text="Element Properties", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Current element display
        self.element_info_frame = ttk.LabelFrame(left_frame, text="Current Element")
        self.element_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.element_info = tk.Text(self.element_info_frame, height=5, width=30, wrap=tk.WORD, state=tk.DISABLED)
        self.element_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Reference image management
        ttk.Label(left_frame, text="Reference Images", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        # Reference control buttons
        ref_button_frame = ttk.Frame(left_frame)
        ref_button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.capture_mode_var = tk.StringVar(value="Enter Capture Mode")
        self.capture_button = ttk.Button(
            ref_button_frame, 
            textvariable=self.capture_mode_var,
            command=self.toggle_capture_mode
        )
        self.capture_button.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(ref_button_frame, text="Capture Reference", 
                  command=self.capture_reference_from_selection).pack(side=tk.RIGHT)
        
        coord_button_frame = ttk.Frame(left_frame)
        coord_button_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            coord_button_frame, 
            text="Capture Coordinates", 
            command=self.capture_click_coordinates
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            coord_button_frame, 
            text="Toggle Coordinates First", 
            command=self.toggle_coordinate_preference
        ).pack(side=tk.LEFT, padx=(0, 5))

        # Reference image preview
        self.reference_frame = ttk.LabelFrame(left_frame, text="Reference Images")
        self.reference_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Region settings
        ttk.Label(left_frame, text="Region Settings", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        self.confidence_var = tk.DoubleVar(value=0.7)
        confidence_frame = ttk.Frame(left_frame)
        confidence_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(confidence_frame, text="Confidence:").pack(side=tk.LEFT)
        ttk.Scale(confidence_frame, from_=0.1, to=1.0, variable=self.confidence_var, 
                 orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(confidence_frame, textvariable=tk.StringVar(
            value=lambda: f"{self.confidence_var.get():.2f}")
        ).pack(side=tk.RIGHT)
        
        # Navigation buttons
        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(nav_frame, text="Back to Setup", 
                  command=lambda: self.notebook.select(0)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Go to Analyze", 
                  command=lambda: self.notebook.select(2)).pack(side=tk.RIGHT)
        
        # Canvas for region definition
        self.define_canvas = tk.Canvas(right_frame, bg="gray90")
        self.define_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Canvas bindings for define tab
        self.define_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.define_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.define_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
    
    def setup_analyze_tab(self):
        """Set up the analysis tab."""
        # Create a paned window
        pane = ttk.PanedWindow(self.analyze_tab, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Debug controls
        left_frame = ttk.Frame(pane, padding=10)
        pane.add(left_frame, weight=1)
        
        # Right panel - Debug visualization
        right_frame = ttk.LabelFrame(pane, text="Debug Visualization")
        pane.add(right_frame, weight=3)
        
        # Debug controls in left frame
        ttk.Label(left_frame, text="Debug Analysis", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Debug image selection
        ttk.Label(left_frame, text="Select Debug Image:").pack(anchor="w", pady=(0, 5))
        
        # Debug image listbox
        self.debug_listbox = tk.Listbox(left_frame, height=10)
        self.debug_listbox.pack(fill=tk.X, pady=(0, 10))
        self.debug_listbox.bind('<<ListboxSelect>>', self.on_debug_image_select)
        
        # Analysis controls
        ttk.Button(left_frame, text="Scan Debug Images", 
                  command=self.scan_debug_images).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(left_frame, text="Use Best Match Region", 
                  command=self.use_best_match_region).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(left_frame, text="Adjust Click Offset", 
                  command=self.adjust_click_offset).pack(fill=tk.X, pady=(0, 5))
        
        # Match information
        self.match_info_frame = ttk.LabelFrame(left_frame, text="Match Information")
        self.match_info_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        self.match_info = tk.Text(self.match_info_frame, height=10, width=30, wrap=tk.WORD, state=tk.DISABLED)
        self.match_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Navigation buttons
        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(nav_frame, text="Back to Define", 
                  command=lambda: self.notebook.select(1)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Go to Verify", 
                  command=lambda: self.notebook.select(3)).pack(side=tk.RIGHT)
        
        # Canvas for debug visualization
        self.analyze_canvas = tk.Canvas(right_frame, bg="gray90")
        self.analyze_canvas.pack(fill=tk.BOTH, expand=True)
    
    def setup_verify_tab(self):
        """Set up the verification tab."""
        # Create a paned window
        pane = ttk.PanedWindow(self.verify_tab, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Verification controls
        left_frame = ttk.Frame(pane, padding=10)
        pane.add(left_frame, weight=1)
        
        # Right panel - Verification results
        right_frame = ttk.LabelFrame(pane, text="Verification Results")
        pane.add(right_frame, weight=3)
        
        # Verification controls in left frame
        ttk.Label(left_frame, text="Verification", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Verification actions
        ttk.Button(left_frame, text="Test All Elements", 
                  command=self.test_all_elements).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(left_frame, text="Generate Calibration Report", 
                  command=self.generate_calibration_report).pack(fill=tk.X, pady=(0, 5))
        
        # Results display
        self.verify_results_frame = ttk.LabelFrame(left_frame, text="Test Results")
        self.verify_results_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        self.verify_results = tk.Text(self.verify_results_frame, height=15, width=30, wrap=tk.WORD, state=tk.DISABLED)
        self.verify_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Navigation buttons
        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(nav_frame, text="Back to Analyze", 
                  command=lambda: self.notebook.select(2)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Finish & Save", 
                  command=self.finalize_calibration).pack(side=tk.RIGHT)
        
        # Canvas for verification results
        self.verify_canvas = tk.Canvas(right_frame, bg="gray90")
        self.verify_canvas.pack(fill=tk.BOTH, expand=True)
    
    def on_tab_changed(self, event):
        """Handle tab change events."""
        tab_id = self.notebook.index("current")
        tab_name = ["Setup", "Define", "Analyze", "Verify"][tab_id]
        
        # Update status
        self.show_status(f"Switched to {tab_name} tab")
        
        # Perform tab-specific actions
        if tab_id == 0:  # Setup
            if self.screenshot:
                self.update_canvas(self.setup_canvas)
        elif tab_id == 1:  # Define
            if self.screenshot:
                self.update_canvas(self.define_canvas)
            if self.current_element:
                self.update_element_info()
                self.update_reference_preview()
        elif tab_id == 2:  # Analyze
            self.scan_debug_images()
        elif tab_id == 3:  # Verify
            if self.screenshot:
                self.update_canvas(self.verify_canvas)
    
    def show_status(self, message):
        """Update status bar message."""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def take_screenshot(self):
        """Take a screenshot of the screen."""
        self.root.iconify()  # Minimize window
        self.show_status("Taking screenshot in 2 seconds...")
        time.sleep(2)  # Wait for window to minimize
        
        try:
            self.screenshot = pyautogui.screenshot()
            self.root.deiconify()  # Restore window
            
            # Update canvas on current tab
            current_tab = self.notebook.index("current")
            if current_tab == 0:
                self.update_canvas(self.setup_canvas)
            elif current_tab == 1:
                self.update_canvas(self.define_canvas)
            
            self.show_status("Screenshot taken. Select regions by clicking and dragging.")
        except Exception as e:
            self.root.deiconify()  # Restore window
            messagebox.showerror("Screenshot Error", f"Error taking screenshot: {e}")
            self.show_status(f"Error taking screenshot: {e}")
    
    def load_image(self):
        """Load an image from file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        
        if file_path:
            try:
                self.screenshot = Image.open(file_path)
                
                # Update canvas on current tab
                current_tab = self.notebook.index("current")
                if current_tab == 0:
                    self.update_canvas(self.setup_canvas)
                elif current_tab == 1:
                    self.update_canvas(self.define_canvas)
                
                self.show_status(f"Loaded image from {file_path}")
            except Exception as e:
                messagebox.showerror("Image Load Error", f"Error loading image: {e}")
                self.show_status(f"Error loading image: {e}")
    
    def update_canvas(self, canvas):
        """Update the given canvas with the current screenshot."""
        if not self.screenshot:
            return
            
        # Get canvas dimensions
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # If canvas not yet sized, use defaults
        if canvas_width <= 1:
            canvas_width = 800
            canvas_height = 600
        
        # Resize the image to fit the canvas while maintaining aspect ratio
        img_width, img_height = self.screenshot.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize the image
        resized = self.screenshot.copy()
        resized = resized.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to Tkinter PhotoImage
        self.screenshot_image = ImageTk.PhotoImage(resized)
        
        # Clear canvas
        canvas.delete("all")
        
        # Draw the image
        canvas.create_image(0, 0, anchor=tk.NW, image=self.screenshot_image)
        canvas.config(scrollregion=(0, 0, new_width, new_height))
        
        # Store the scale factor for coordinate conversion
        self.scale = scale
        
        # Draw existing regions
        self.draw_regions(canvas)

        # Draw existing regions
        self.draw_regions(canvas)
        
        # Draw coordinate markers
        self.draw_coordinate_markers(canvas)
    
    def draw_regions(self, canvas):
        """Draw all defined regions on the given canvas."""
        if not self.screenshot:
            return
        
        # Clear previous regions
        canvas.delete("region")
        
        # Draw each element's region
        for name, config in self.elements.items():
            region = config.get("region")
            if not region:
                continue
                
            # Determine color (green for selected, blue for others)
            color = "green" if name == self.current_element else "blue"
            
            # Scale coordinates to canvas size
            x, y, w, h = region
            canvas_x = x * self.scale
            canvas_y = y * self.scale
            canvas_w = w * self.scale
            canvas_h = h * self.scale
            
            # Draw rectangle
            canvas.create_rectangle(
                canvas_x, canvas_y, 
                canvas_x + canvas_w, canvas_y + canvas_h,
                outline=color, width=2, tags="region"
            )
            
            # Draw label
            canvas.create_text(
                canvas_x + 5, canvas_y + 5,
                text=name, anchor=tk.NW,
                fill=color, tags="region"
            )
    
    def on_mouse_down(self, event):
        """Handle mouse button press."""
        if not self.screenshot:
            self.show_status("Please take or load a screenshot first.")
            return
            
        if not self.current_element and not self.capture_mode:
            self.show_status("Please select an element first.")
            return
            
        self.selection_start = (event.x, event.y)
        
        # Clear previous selection box
        current_tab = self.notebook.index("current")
        if current_tab == 0:
            self.setup_canvas.delete("selection")
        elif current_tab == 1:
            self.define_canvas.delete("selection")
    
    def on_mouse_drag(self, event):
        """Handle mouse drag."""
        if not self.selection_start:
            return
        
        # Get current canvas
        current_tab = self.notebook.index("current")
        if current_tab == 0:
            canvas = self.setup_canvas
        elif current_tab == 1:
            canvas = self.define_canvas
        else:
            return
        
        # Clear previous selection box
        canvas.delete("selection")
        
        # Draw new selection box
        selection_box = canvas.create_rectangle(
            self.selection_start[0], self.selection_start[1],
            event.x, event.y,
            outline="red" if self.capture_mode else "blue", 
            width=2,
            tags="selection"
        )
        
        # Update status with dimensions
        width = abs(event.x - self.selection_start[0])
        height = abs(event.y - self.selection_start[1])
        
        if self.capture_mode:
            self.show_status(f"Selection for reference capture: {width}x{height} pixels")
        else:
            self.show_status(f"Element region selection: {width}x{height} pixels")
    
    def on_mouse_up(self, event):
        """Handle mouse button release."""
        if not self.selection_start:
            return
            
        # Get current canvas
        current_tab = self.notebook.index("current")
        if current_tab == 0:
            canvas = self.setup_canvas
        elif current_tab == 1:
            canvas = self.define_canvas
        else:
            self.selection_start = None
            return
        
        # Calculate selection rectangle (ensure x1,y1 is top-left and x2,y2 is bottom-right)
        x1, y1 = self.selection_start
        x2, y2 = event.x, event.y
        
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        
        # Convert canvas coordinates to original image coordinates
        orig_left = int(left / self.scale)
        orig_top = int(top / self.scale)
        orig_width = int((right - left) / self.scale)
        orig_height = int((bottom - top) / self.scale)
        
        # Selection region in original image coordinates
        selection_region = (orig_left, orig_top, orig_width, orig_height)
        
        if self.capture_mode:
            # In capture mode, capture the selected region as a reference image
            self.capture_reference_image(selection_region)
        else:
            # In region definition mode, save the region for the element
            if self.current_element:
                if self.current_element not in self.elements:
                    self.elements[self.current_element] = {}
                    
                self.elements[self.current_element]["region"] = selection_region
                
                self.show_status(f"Region defined for {self.current_element}: {selection_region}")
                
                # Update element info
                self.update_element_info()
                
                # Redraw regions
                self.draw_regions(canvas)
        
        # Reset selection
        self.selection_start = None
    
    def add_element(self):
        """Add a new element."""
        name = self.element_var.get().strip()
        
        if not name:
            messagebox.showwarning("Input Error", "Element name cannot be empty.")
            return
        
        if name in self.elements:
            messagebox.showwarning("Duplicate Element", f"Element '{name}' already exists.")
            return
        
        # Add element with empty properties
        self.elements[name] = {
            "region": None,
            "reference_paths": [],
            "confidence": 0.7
        }
        
        # Add to listbox
        self.element_listbox.insert(tk.END, name)
        
        # Clear entry
        self.element_var.set("")
        
        # Set as current element
        self.current_element = name
        self.element_listbox.selection_clear(0, tk.END)
        index = list(self.elements.keys()).index(name)
        self.element_listbox.selection_set(index)
        
        # Initialize reference count
        self.reference_count[name] = 0
        
        # Create directory for reference images
        os.makedirs(f"assets/reference_images/{name}", exist_ok=True)
        
        # Update element info
        self.update_element_info()
        
        self.show_status(f"Added element '{name}'. Now select its region on the screenshot.")
    
    def on_element_select(self, event):
        """Handle element selection from listbox."""
        selection = self.element_listbox.curselection()
        if selection:
            index = selection[0]
            name = self.element_listbox.get(index)
            self.current_element = name
            
            # Update UI
            self.update_element_info()
            self.update_reference_preview()
            
            # Redraw regions on current canvas
            current_tab = self.notebook.index("current")
            if current_tab == 0:
                self.draw_regions(self.setup_canvas)
            elif current_tab == 1:
                self.draw_regions(self.define_canvas)
            
            self.show_status(f"Selected element: {name}")
    
    def delete_element(self):
        """Delete the selected element."""
        if not self.current_element:
            messagebox.showwarning("No Selection", "Please select an element to delete.")
            return
        
        # Confirm deletion
        if not messagebox.askyesno("Confirm Deletion", f"Delete element '{self.current_element}'?"):
            return
        
        # Remove from elements dictionary
        if self.current_element in self.elements:
            del self.elements[self.current_element]
        
        # Remove from listbox
        selection = self.element_listbox.curselection()
        if selection:
            self.element_listbox.delete(selection[0])
        
        # Reset current element
        self.current_element = None
        
        # Update UI
        self.update_element_info()
        
        # Clear reference preview
        for widget in self.reference_frame.winfo_children():
            widget.destroy()
        
        # Redraw regions on current canvas
        current_tab = self.notebook.index("current")
        if current_tab == 0:
            self.draw_regions(self.setup_canvas)
        elif current_tab == 1:
            self.draw_regions(self.define_canvas)
        
        self.show_status("Element deleted.")
    
    def update_element_info(self):
        """Update the element information display."""
        # Enable text widget for editing
        self.element_info.config(state=tk.NORMAL)
        
        # Clear current content
        self.element_info.delete(1.0, tk.END)
        
        if not self.current_element:
            self.element_info.insert(tk.END, "No element selected")
        elif self.current_element in self.elements:
            element_config = self.elements[self.current_element]
            
            # Format region information
            region_str = "Not defined"
            if "region" in element_config and element_config["region"]:
                x, y, w, h = element_config["region"]
                region_str = f"({x}, {y}, {w}, {h})"
            
            # Format reference image counts
            ref_count = len(element_config.get("reference_paths", []))
            
            # Format coordinates
            coordinates_str = "Not defined"
            if "click_coordinates" in element_config and element_config["click_coordinates"]:
                x, y = element_config["click_coordinates"]
                coordinates_str = f"({x}, {y})"
            
            # Build the info text
            info = f"Element: {self.current_element}\n"
            info += f"Region: {region_str}\n"
            info += f"References: {ref_count}\n"
            info += f"Confidence: {element_config.get('confidence', 0.7):.2f}\n"
            info += f"Click coords: {coordinates_str}\n"
            info += f"Use coords first: {element_config.get('use_coordinates_first', False)}"
            
            self.element_info.insert(tk.END, info)
        else:
            self.element_info.insert(tk.END, f"Element '{self.current_element}' not found in configuration.")
        
        # Disable editing
        self.element_info.config(state=tk.DISABLED)
    
    def toggle_capture_mode(self):
        """Toggle between region definition and reference capture modes."""
        if not self.current_element:
            messagebox.showwarning("No Selection", "Please select an element first.")
            return
        
        # Toggle capture mode
        self.capture_mode = not self.capture_mode
        
        if self.capture_mode:
            # Entering reference capture mode
            self.capture_mode_var.set("Exit Capture Mode")
            self.show_status(f"Reference capture mode active for '{self.current_element}'. Select area to capture.")
        else:
            # Exiting reference capture mode, back to region definition
            self.capture_mode_var.set("Enter Capture Mode")
            self.show_status("Region definition mode active. Select areas to define element regions.")
    
    def capture_reference_image(self, region=None):
        """Capture a reference image from the current screenshot."""
        if not self.current_element or not self.screenshot:
            return
            
        try:
            # Create directory if it doesn't exist
            reference_dir = f"assets/reference_images/{self.current_element}"
            os.makedirs(reference_dir, exist_ok=True)
            
            # Extract region from screenshot
            x, y, w, h = region
            cropped = self.screenshot.crop((x, y, x + w, y + h))
            
            # Save as reference image
            timestamp = int(time.time())
            filename = f"{reference_dir}/{self.current_element}_{timestamp}.png"
            cropped.save(filename)
            
            # Add to element's reference paths
            if self.current_element not in self.elements:
                self.elements[self.current_element] = {"region": None, "reference_paths": [], "confidence": 0.7}
                
            if "reference_paths" not in self.elements[self.current_element]:
                self.elements[self.current_element]["reference_paths"] = []
                
            self.elements[self.current_element]["reference_paths"].append(filename)
            
            # Update reference count
            ref_count = len(self.elements[self.current_element]["reference_paths"])
            self.reference_count[self.current_element] = ref_count
            
            # Update UI
            self.update_element_info()
            self.update_reference_preview()
            
            # Generate variants with ReferenceImageManager
            variants = self.reference_manager.preprocess_reference_images([filename])
            
            # Add variants to element's reference paths
            for variant in variants:
                if variant != filename and variant not in self.elements[self.current_element]["reference_paths"]:
                    self.elements[self.current_element]["reference_paths"].append(variant)
            
            # Update reference count again after adding variants
            ref_count = len(self.elements[self.current_element]["reference_paths"])
            self.reference_count[self.current_element] = ref_count
            
            self.show_status(f"Captured reference image for {self.current_element}. Created {len(variants)} variants.")
            
        except Exception as e:
            messagebox.showerror("Capture Error", f"Error capturing reference: {e}")
            self.show_status(f"Error capturing reference: {e}")
    
    def capture_reference_from_selection(self):
        """Capture a reference image from the current selection."""
        if not self.current_element:
            messagebox.showwarning("No Selection", "Please select an element first.")
            return
        
        # Get current canvas
        current_tab = self.notebook.index("current")
        if current_tab == 0:
            canvas = self.setup_canvas
        elif current_tab == 1:
            canvas = self.define_canvas
        else:
            return
        
        # Get the current selection
        selection_items = canvas.find_withtag("selection")
        if not selection_items:
            messagebox.showwarning("No Selection", "Please make a selection first.")
            return
        
        # Get the coordinates of the selection
        coords = canvas.coords(selection_items[0])
        if len(coords) != 4:
            return
            
        x1, y1, x2, y2 = coords
        
        # Ensure x1,y1 is top-left and x2,y2 is bottom-right
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        
        # Convert canvas coordinates to original image coordinates
        orig_left = int(left / self.scale)
        orig_top = int(top / self.scale)
        orig_width = int((right - left) / self.scale)
        orig_height = int((bottom - top) / self.scale)
        
        # Capture the reference image
        self.capture_reference_image((orig_left, orig_top, orig_width, orig_height))
    
    def update_reference_preview(self):
        """Update the reference image preview panel."""
        # Clear previous preview
        for widget in self.reference_frame.winfo_children():
            widget.destroy()
        
        if not self.current_element:
            ttk.Label(self.reference_frame, text="No element selected").pack(pady=10)
            return
        
        # Get reference images for this element
        if self.current_element not in self.elements:
            ttk.Label(self.reference_frame, text="Element not found in configuration").pack(pady=10)
            return
            
        reference_paths = self.elements[self.current_element].get("reference_paths", [])
        
        # Filter out variants to show only original images
        original_references = [path for path in reference_paths if not any(x in path for x in ['_gray', '_contrast', '_thresh', '_scale'])]
        
        if not original_references:
            ttk.Label(self.reference_frame, text="No reference images").pack(pady=10)
            return
        
        # Create a canvas for thumbnails
        canvas_frame = ttk.Frame(self.reference_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        preview_canvas = tk.Canvas(canvas_frame, bg="white")
        preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=preview_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        preview_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create a frame inside the canvas for the thumbnails
        frame = ttk.Frame(preview_canvas)
        preview_canvas.create_window((0, 0), window=frame, anchor=tk.NW)
        
        # Add thumbnails for each reference image
        self.reference_images = []  # Keep references to prevent garbage collection
        
        for i, path in enumerate(original_references):
            try:
                img = Image.open(path)
                
                # Create a small thumbnail
                img.thumbnail((100, 100))
                photo = ImageTk.PhotoImage(img)
                self.reference_images.append(photo)
                
                # Create a frame for this thumbnail
                thumbnail_frame = ttk.Frame(frame)
                thumbnail_frame.grid(row=i//2, column=i%2, padx=5, pady=5)
                
                # Add the thumbnail
                label = ttk.Label(thumbnail_frame, image=photo)
                label.pack()
                
                # Add the filename (shortened)
                basename = os.path.basename(path)
                short_name = basename[:15] + "..." if len(basename) > 18 else basename
                ttk.Label(thumbnail_frame, text=short_name).pack()
                
                # Add delete button
                ttk.Button(
                    thumbnail_frame, 
                    text="Delete", 
                    command=lambda p=path: self.delete_reference(p)
                ).pack(pady=(0, 5))
                
            except Exception as e:
                logging.error(f"Error loading thumbnail: {e}")
        
        # Update canvas scroll region
        frame.update_idletasks()
        preview_canvas.config(scrollregion=preview_canvas.bbox("all"))
    
    def delete_reference(self, path):
        """Delete a reference image."""
        if not self.current_element or self.current_element not in self.elements:
            return
            
        if path not in self.elements[self.current_element].get("reference_paths", []):
            return
            
        if messagebox.askyesno("Confirm Deletion", f"Delete reference image {os.path.basename(path)}?"):
            try:
                # Remove from reference paths
                self.elements[self.current_element]["reference_paths"].remove(path)
                
                # Delete file if it exists
                if os.path.exists(path):
                    os.remove(path)
                    
                # Delete related variants
                basename = os.path.splitext(path)[0]
                parent_dir = os.path.dirname(path)
                for variant in glob.glob(f"{basename}_*.png"):
                    if os.path.exists(variant):
                        os.remove(variant)
                        if variant in self.elements[self.current_element]["reference_paths"]:
                            self.elements[self.current_element]["reference_paths"].remove(variant)
                
                # Update reference count
                self.reference_count[self.current_element] = len(
                    self.elements[self.current_element].get("reference_paths", [])
                )
                
                # Update UI
                self.update_element_info()
                self.update_reference_preview()
                
                self.show_status(f"Deleted reference image {os.path.basename(path)}")
                
            except Exception as e:
                messagebox.showerror("Delete Error", f"Error deleting reference: {e}")
                self.show_status(f"Error deleting reference: {e}")
    
    def scan_debug_images(self):
        """Scan for debug images in the logs directory."""
        self.debug_images = {}
        
        # Find all run directories first
        run_dirs = glob.glob("logs/run_*")
        
        if not run_dirs:
            self.show_status("No run directories found in logs folder")
            return
            
        self.show_status(f"Found {len(run_dirs)} run directories")
        
        # Process each run directory
        for run_dir in run_dirs:
            screenshot_dir = os.path.join(run_dir, "screenshots")
            if not os.path.exists(screenshot_dir):
                continue
                
            # Scan all screenshots in this run directory
            for image_path in glob.glob(os.path.join(screenshot_dir, "*.png")):
                filename = os.path.basename(image_path)
                
                # Try to categorize screenshot based on filename patterns
                if "SEARCH_" in filename or "FOUND_" in filename:
                    # Recognition-related screenshot
                    # Extract element name from patterns like SEARCH_element_name_START or FOUND_element_name
                    match = re.search(r"(?:SEARCH|FOUND)_(\w+)", filename)
                    if match:
                        element_name = match.group(1)
                        if element_name not in self.debug_images:
                            self.debug_images[element_name] = {"recognition": [], "click": []}
                        self.debug_images[element_name]["recognition"].append(image_path)
                
                elif "AFTER_CLICKING" in filename or "click" in filename.lower():
                    # Click-related screenshot
                    match = re.search(r"(?:click|CLICK)_(\w+)", filename)
                    if match:
                        element_name = match.group(1)
                        if element_name not in self.debug_images:
                            self.debug_images[element_name] = {"recognition": [], "click": []}
                        self.debug_images[element_name]["click"].append(image_path)
                
                # Also look for screenshots related to specific elements
                for element_name in self.elements.keys():
                    if element_name.lower() in filename.lower():
                        if element_name not in self.debug_images:
                            self.debug_images[element_name] = {"recognition": [], "click": []}
                        # Categorize as recognition or click based on filename
                        if "click" in filename.lower():
                            self.debug_images[element_name]["click"].append(image_path)
                        else:
                            self.debug_images[element_name]["recognition"].append(image_path)
        
        # Update debug listbox
        self.debug_listbox.delete(0, tk.END)
        
        # Add entries for each element
        for element_name in sorted(self.debug_images.keys()):
            rec_count = len(self.debug_images[element_name]["recognition"])
            click_count = len(self.debug_images[element_name]["click"])
            
            self.debug_listbox.insert(tk.END, f"{element_name} (R:{rec_count}, C:{click_count})")
            
            # Add individual debug images with indent
            for i, path in enumerate(self.debug_images[element_name]["recognition"]):
                filename = os.path.basename(path)
                self.debug_listbox.insert(tk.END, f"  - R{i+1}: {filename}")
                
            for i, path in enumerate(self.debug_images[element_name]["click"]):
                filename = os.path.basename(path)
                self.debug_listbox.insert(tk.END, f"  - C{i+1}: {filename}")
        
        # Show status
        element_count = len(self.debug_images)
        if element_count > 0:
            self.show_status(f"Found debug images for {element_count} UI elements")
        else:
            self.show_status("No debug images found matching UI elements")
    
    def on_debug_image_select(self, event):
        """Handle selection of a debug image from the listbox."""
        selection = self.debug_listbox.curselection()
        if not selection:
            return
            
        # Get selected image entry
        selected_entry = self.debug_listbox.get(selection[0])
        
        # Check if it's an element or an image entry
        if selected_entry.startswith("  - "):
            # It's an image entry
            parts = selected_entry.strip()[4:].split(":", 1)  # Skip the "  - " prefix
            if len(parts) != 2:
                return
                
            image_ref = parts[0].strip()  # This should be something like "R1" or "C1"
            if not image_ref or len(image_ref) < 2:
                self.show_status(f"Invalid image reference: {image_ref}")
                return
                
            image_type = image_ref[0]  # R or C
            try:
                # Make sure there's actually a number after the R or C
                if len(image_ref) < 2 or not image_ref[1:].isdigit():
                    self.show_status(f"Invalid image index format: {image_ref}")
                    return
                    
                image_index = int(image_ref[1:]) - 1  # Convert to 0-based index
            except ValueError:
                self.show_status(f"Invalid image index: {image_ref[1:]}")
                return
            
            # Find the parent element entry
            parent_index = selection[0] - 1
            while parent_index >= 0:
                parent_entry = self.debug_listbox.get(parent_index)
                if not parent_entry.startswith("  - "):
                    break
                parent_index -= 1
            
            if parent_index < 0:
                self.show_status("Could not find parent element for this image")
                return
                
            # Extract element name from parent entry
            parent_parts = parent_entry.split(" ", 1)
            if len(parent_parts) != 2:
                self.show_status("Invalid parent element format")
                return
                
            element_name = parent_parts[0]
            
            # Get the image path
            if element_name not in self.debug_images:
                self.show_status(f"No debug images found for {element_name}")
                return
                
            if image_type == "R" and image_index < len(self.debug_images[element_name]["recognition"]):
                image_path = self.debug_images[element_name]["recognition"][image_index]
                self.display_debug_image(image_path, "recognition")
            elif image_type == "C" and image_index < len(self.debug_images[element_name]["click"]):
                image_path = self.debug_images[element_name]["click"][image_index]
                self.display_debug_image(image_path, "click")
            else:
                self.show_status(f"Image index out of range: {image_ref}")
        else:
            # It's an element entry - just show status
            self.show_status(f"Selected element: {selected_entry}")
    
    def display_debug_image(self, image_path, image_type):
        """Display a debug image on the analyze canvas."""
        try:
            # Load the image
            img = Image.open(image_path)
            
            # Get canvas dimensions
            canvas_width = self.analyze_canvas.winfo_width()
            canvas_height = self.analyze_canvas.winfo_height()
            
            # If canvas not yet sized, use defaults
            if canvas_width <= 1:
                canvas_width = 800
                canvas_height = 600
            
            # Keep aspect ratio
            img_width, img_height = img.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize the image
            resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.debug_image = ImageTk.PhotoImage(resized)
            
            # Clear canvas
            self.analyze_canvas.delete("all")
            
            # Draw the image
            self.analyze_canvas.create_image(0, 0, anchor=tk.NW, image=self.debug_image)
            self.analyze_canvas.config(scrollregion=(0, 0, new_width, new_height))
            
            # Store the scale factor for coordinate conversion
            self.debug_scale = scale
            
            # Extract information from image
            self.analyze_debug_image(image_path, image_type)
            
            # Update status
            self.show_status(f"Displaying debug image: {os.path.basename(image_path)}")
            
        except Exception as e:
            messagebox.showerror("Display Error", f"Error displaying debug image: {e}")
            self.show_status(f"Error displaying debug image: {e}")
    
    def analyze_debug_image(self, image_path, image_type):
        """Analyze a debug image and update match information."""
        # Enable text widget for editing
        self.match_info.config(state=tk.NORMAL)
        
        # Clear current content
        self.match_info.delete(1.0, tk.END)
        
        try:
            # Extract filename
            filename = os.path.basename(image_path)
            
            # Extract element name
            element_name = None
            if image_type == "recognition":
                match = re.match(r"(\w+)_matches_\d+\.png", filename)
                if match:
                    element_name = match.group(1)
            else:  # click
                match = re.match(r"click_(\w+)_\d+.*\.png", filename)
                if match:
                    element_name = match.group(1)
            
            # Build info text
            info = f"Debug Image: {filename}\n"
            info += f"Type: {image_type.capitalize()}\n"
            info += f"Element: {element_name}\n\n"
            
            # Add additional analysis based on image type
            if image_type == "recognition":
                # Analyze recognition debug image
                match_regions = self.analyze_recognition_debug(image_path)
                
                if match_regions:
                    info += f"Found {len(match_regions)} match regions:\n"
                    for i, region in enumerate(match_regions):
                        x, y, w, h = region
                        confidence = 0.0  # Would need to extract from image or filename
                        info += f"Match {i+1}: ({x}, {y}, {w}, {h}), conf: {confidence:.2f}\n"
                else:
                    info += "No match regions detected."
            
            elif image_type == "click":
                # Analyze click debug image
                click_points = self.analyze_click_debug(image_path)
                
                if click_points:
                    info += f"Found {len(click_points)} click points:\n"
                    for i, point in enumerate(click_points):
                        x, y = point
                        info += f"Click Point {i+1}: ({x}, {y})\n"
                else:
                    info += "No click points detected."
            
            # Insert analysis
            self.match_info.insert(tk.END, info)
            
        except Exception as e:
            self.match_info.insert(tk.END, f"Error analyzing debug image: {e}")
            
        # Disable editing
        self.match_info.config(state=tk.DISABLED)
    
    def analyze_recognition_debug(self, image_path):
        """Analyze a recognition debug image to extract match regions."""
        try:
            # Load the image using OpenCV to detect marked regions
            img = cv2.imread(image_path)
            
            if img is None:
                logging.error(f"Could not load image: {image_path}")
                return []
                
            # Convert to HSV for easier color detection
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Green mask (for found regions)
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract regions
            match_regions = []
            for contour in contours:
                # Get the bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                match_regions.append((x, y, w, h))
                
                # Draw rectangle on analyze canvas
                canvas_x = x * self.debug_scale
                canvas_y = y * self.debug_scale
                canvas_w = w * self.debug_scale
                canvas_h = h * self.debug_scale
                
                self.analyze_canvas.create_rectangle(
                    canvas_x, canvas_y, 
                    canvas_x + canvas_w, canvas_y + canvas_h,
                    outline="green", width=2, tags="match"
                )
                
                # Add label
                self.analyze_canvas.create_text(
                    canvas_x, canvas_y - 10,
                    text=f"Match: ({x}, {y}, {w}, {h})",
                    fill="green", anchor=tk.W, tags="match"
                )
            
            return match_regions
            
        except Exception as e:
            logging.error(f"Error analyzing recognition debug: {e}")
            return []
    
    def analyze_click_debug(self, image_path):
        """Analyze a click debug image to extract click locations."""
        try:
            # Load the image using OpenCV
            img = cv2.imread(image_path)
            
            if img is None:
                logging.error(f"Could not load image: {image_path}")
                return []
                
            # Look for the red circle/cross marking the click point
            # Extract red channel
            red_channel = img[:,:,2]
            _, thresh = cv2.threshold(red_channel, 200, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for circular contours
            click_points = []
            for contour in contours:
                # Calculate properties that can help identify the click marker
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # A circle has the maximum area for a given perimeter
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    
                    # Circles have circularity close to 1.0
                    if 0.5 < circularity < 1.2 and 100 < area < 5000:
                        # Get the center of the contour
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            click_points.append((cx, cy))
                            
                            # Draw marker on analyze canvas
                            canvas_x = cx * self.debug_scale
                            canvas_y = cy * self.debug_scale
                            
                            self.analyze_canvas.create_oval(
                                canvas_x - 10, canvas_y - 10, 
                                canvas_x + 10, canvas_y + 10,
                                outline="red", width=2, tags="click"
                            )
                            
                            # Add label
                            self.analyze_canvas.create_text(
                                canvas_x + 15, canvas_y,
                                text=f"Click: ({cx}, {cy})",
                                fill="red", anchor=tk.W, tags="click"
                            )
            
            return click_points
            
        except Exception as e:
            logging.error(f"Error analyzing click debug: {e}")
            return []
    
    def use_best_match_region(self):
        """Use the best match region from the debug image."""
        # Check if a debug image is displayed
        if not hasattr(self, 'debug_image') or not self.current_element:
            messagebox.showwarning("No Debug Image", "Please select a debug image first.")
            return
        
        # Look for match regions on canvas
        match_items = self.analyze_canvas.find_withtag("match")
        if not match_items:
            messagebox.showwarning("No Matches", "No match regions found in the current debug image.")
            return
        
        # Get the first (best) match region
        coords = self.analyze_canvas.coords(match_items[0])
        if len(coords) != 4:
            return
        
        # Convert to original image coordinates
        x1, y1, x2, y2 = coords
        w, h = x2 - x1, y2 - y1
        
        # Convert canvas coordinates to original image coordinates
        orig_x = int(x1 / self.debug_scale)
        orig_y = int(y1 / self.debug_scale)
        orig_w = int(w / self.debug_scale)
        orig_h = int(h / self.debug_scale)
        
        # Update element's region
        if self.current_element not in self.elements:
            self.elements[self.current_element] = {"region": None, "reference_paths": [], "confidence": 0.7}
        
        self.elements[self.current_element]["region"] = (orig_x, orig_y, orig_w, orig_h)
        
        # Update UI
        self.update_element_info()
        
        self.show_status(f"Updated region for {self.current_element} to best match: ({orig_x}, {orig_y}, {orig_w}, {orig_h})")
    
    def adjust_click_offset(self):
        """Adjust the click offset for the current element."""
        if not self.current_element:
            messagebox.showwarning("No Selection", "Please select an element first.")
            return
        
        # Create a dialog for offset input
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Adjust Click Offset - {self.current_element}")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Get current offset values
        current_offset = (0, 0)
        if self.current_element in self.elements and "click_offset" in self.elements[self.current_element]:
            current_offset = self.elements[self.current_element]["click_offset"]
        
        # Create input fields
        ttk.Label(dialog, text="X Offset (pixels):").pack(pady=(20, 5))
        x_var = tk.IntVar(value=current_offset[0])
        ttk.Entry(dialog, textvariable=x_var).pack(pady=(0, 10))
        
        ttk.Label(dialog, text="Y Offset (pixels):").pack(pady=(10, 5))
        y_var = tk.IntVar(value=current_offset[1])
        ttk.Entry(dialog, textvariable=y_var).pack(pady=(0, 10))
        
        # Save button
        def save_offset():
            offset = (x_var.get(), y_var.get())
            if self.current_element not in self.elements:
                self.elements[self.current_element] = {"region": None, "reference_paths": [], "confidence": 0.7}
            
            self.elements[self.current_element]["click_offset"] = offset
            self.update_element_info()
            dialog.destroy()
            
            self.show_status(f"Updated click offset for {self.current_element}: {offset}")
        
        ttk.Button(dialog, text="Save", command=save_offset).pack(pady=10)
    
    def test_all_elements(self):
        """Test the recognition of all elements."""
        if not self.screenshot:
            messagebox.showwarning("No Screenshot", "Please take or load a screenshot first.")
            return
        
        # Create UIElement objects for testing
        ui_elements = {}
        for name, config in self.elements.items():
            region = config.get("region")
            reference_paths = config.get("reference_paths", [])
            confidence = config.get("confidence", 0.7)
            
            if not region or not reference_paths:
                continue
                
            ui_elements[name] = UIElement(
                name=name,
                reference_paths=reference_paths,
                region=region,
                confidence=confidence
            )
        
        # Enable text widget for editing
        self.verify_results.config(state=tk.NORMAL)
        
        # Clear current content
        self.verify_results.delete(1.0, tk.END)
        
        # Test each element
        results = {}
        for name, element in ui_elements.items():
            self.show_status(f"Testing element: {name}")
            
            # Try to find the element
            location = find_element(element)
            
            results[name] = {
                "found": location is not None,
                "location": location
            }
            
            # Show result for this element
            status = "Found" if location else "Not found"
            self.verify_results.insert(tk.END, f"{name}: {status}\n")
            
            if location:
                x, y, w, h = location
                self.verify_results.insert(tk.END, f"  Location: ({x}, {y}, {w}, {h})\n\n")
                
                # Draw the match on the verify canvas
                canvas_x = x * self.scale
                canvas_y = y * self.scale
                canvas_w = w * self.scale
                canvas_h = h * self.scale
                
                self.verify_canvas.create_rectangle(
                    canvas_x, canvas_y, 
                    canvas_x + canvas_w, canvas_y + canvas_h,
                    outline="green", width=2, tags=name
                )
                
                self.verify_canvas.create_text(
                    canvas_x, canvas_y - 10,
                    text=name,
                    fill="green", anchor=tk.W, tags=name
                )
            else:
                self.verify_results.insert(tk.END, "  Not found! Check region and references.\n\n")
        
        # Add summary
        found_count = sum(1 for r in results.values() if r["found"])
        total_count = len(results)
        
        self.verify_results.insert(tk.END, f"\nSummary: Found {found_count} out of {total_count} elements.")
        
        # Disable editing
        self.verify_results.config(state=tk.DISABLED)
        
        self.show_status(f"Verification complete. Found {found_count} out of {total_count} elements.")
        
        # Show the screenshot on the verify canvas if not already visible
        self.update_canvas(self.verify_canvas)
    
    def generate_calibration_report(self):
        """Generate a detailed report of the calibration."""
        if not self.elements:
            messagebox.showwarning("No Elements", "No elements have been defined.")
            return
        
        # Create report directory
        report_dir = "logs/reports"
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create report filename
        report_file = os.path.join(report_dir, f"calibration_report_{timestamp}.html")
        
        # Create report content
        report = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Claude Calibration Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .success { color: green; }
                .failure { color: red; }
                .screenshot { max-width: 800px; margin: 20px 0; border: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <h1>Claude UI Calibration Report</h1>
            <p>Generated on: {}</p>
            
            <h2>Element Summary</h2>
            <table>
                <tr>
                    <th>Element</th>
                    <th>Region</th>
                    <th>References</th>
                    <th>Confidence</th>
                </tr>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Add each element
        for name, config in self.elements.items():
            region = config.get("region", "Not defined")
            if isinstance(region, tuple):
                region_str = f"({region[0]}, {region[1]}, {region[2]}, {region[3]})"
            else:
                region_str = str(region)
                
            ref_count = len(config.get("reference_paths", []))
            confidence = config.get("confidence", 0.7)
            
            report += f"""
                <tr>
                    <td>{name}</td>
                    <td>{region_str}</td>
                    <td>{ref_count}</td>
                    <td>{confidence:.2f}</td>
                </tr>
            """
        
        report += """
            </table>
            
            <h2>Reference Images</h2>
        """
        
        # Add reference images
        for name, config in self.elements.items():
            report += f"<h3>Element: {name}</h3>\n"
            
            reference_paths = config.get("reference_paths", [])
            if not reference_paths:
                report += "<p>No reference images defined.</p>\n"
                continue
            
            # Filter out variants
            original_references = [path for path in reference_paths if not any(x in path for x in ['_gray', '_contrast', '_thresh', '_scale'])]
            
            report += "<div style='display: flex; flex-wrap: wrap;'>\n"
            
            for path in original_references:
                # Create a copy of the image in the report directory
                try:
                    img = Image.open(path)
                    basename = os.path.basename(path)
                    copy_path = os.path.join(report_dir, basename)
                    img.save(copy_path)
                    
                    report += f"""
                        <div style='margin: 10px; text-align: center;'>
                            <img src="{basename}" style='max-width: 200px; max-height: 200px;'>
                            <div>{basename}</div>
                        </div>
                    """
                except Exception as e:
                    report += f"<p>Error including image {path}: {e}</p>\n"
            
            report += "</div>\n"
        
        # Finish the report
        report += """
            <h2>Configuration</h2>
            <pre>{}</pre>
            
        </body>
        </html>
        """.format(yaml.dump(self.elements, default_flow_style=False))
        
        # Write the report file
        with open(report_file, "w") as f:
            f.write(report)
        
        # Open the report in the default browser
        import webbrowser
        webbrowser.open(report_file)
        
        self.show_status(f"Calibration report generated: {report_file}")
    
    def finalize_calibration(self):
        """Finalize the calibration and save configuration."""
        if not self.elements:
            messagebox.showwarning("No Elements", "No elements have been defined.")
            return
        
        # Check for incomplete elements
        incomplete = [name for name, config in self.elements.items() 
                    if not config.get("region") or not config.get("reference_paths")]
        
        if incomplete:
            if not messagebox.askyesno("Incomplete Configuration", 
                                    f"The following elements are incomplete:\n{', '.join(incomplete)}\n\nSave anyway?"):
                return
        
        # Add option to preserve sessions
        preserve_sessions = messagebox.askyesno("Preserve Sessions", 
                                            "Do you want to preserve any existing session configurations?")
        
        if preserve_sessions:
            # Enter session mode if not already active
            if not self.config_manager.is_in_session_mode():
                self.config_manager.enter_session_mode()
                self.save_configuration()
                self.config_manager.exit_session_mode()
            else:
                # Already in session mode, use preserving save
                self.save_configuration()
        else:
            # Standard save without session preservation
            self.save_configuration()

    def load_configuration(self):
        """
        Load configuration from ConfigManager.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load configuration from config manager
            config = self.config_manager.get_all()
            
            # Extract UI elements configuration
            ui_elements = config.get("ui_elements", {})
            
            # Clear current elements
            self.elements = {}
            self.element_listbox.delete(0, tk.END)
            
            # Process each UI element
            for name, element_config in ui_elements.items():
                # Add to our elements dictionary
                self.elements[name] = element_config
                
                # Add to listbox
                self.element_listbox.insert(tk.END, name)
                
                # Count reference images
                ref_count = len(element_config.get("reference_paths", []))
                self.reference_count[name] = ref_count
            
            # Update UI if we have elements
            if self.elements:
                # Select the first element
                self.element_listbox.selection_set(0)
                self.current_element = self.element_listbox.get(0)
                self.update_element_info()
                self.update_reference_preview()
                
                # Update canvas if we have a screenshot
                if self.screenshot:
                    current_tab = self.notebook.index("current")
                    if current_tab == 0:
                        self.update_canvas(self.setup_canvas)
                    elif current_tab == 1:
                        self.update_canvas(self.define_canvas)
            
            self.show_status("Configuration loaded successfully")
            return True
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Error loading configuration: {e}")
            self.show_status(f"Error loading configuration: {e}")
            return False

    def _process_coordinate_properties(self):
        """
        Process coordinate-based properties for backward compatibility.
        Convert any list coordinates to tuples and ensure use_coordinates_first is set.
        """
        try:
            # Get UI elements from config
            ui_elements = self.config.get("ui_elements", {})
            
            for element_name, element_config in ui_elements.items():
                # Handle click_coordinates if present
                if "click_coordinates" in element_config:
                    coords = element_config["click_coordinates"]
                    
                    # Convert from list to tuple if needed
                    if isinstance(coords, list) and len(coords) == 2:
                        element_config["click_coordinates"] = tuple(coords)
                    
                    # Ensure use_coordinates_first is set if not present
                    if "use_coordinates_first" not in element_config:
                        # Use global default if present, otherwise true
                        global_preference = self.config.get("automation_settings", {}).get("prefer_coordinates", True)
                        element_config["use_coordinates_first"] = global_preference
                        
                        logging.debug(f"Set use_coordinates_first={global_preference} for {element_name}")
                
            # Ensure global automation_settings with coordinate preferences exist
            if "automation_settings" not in self.config:
                self.config["automation_settings"] = {}
                
            if "prefer_coordinates" not in self.config.get("automation_settings", {}):
                self.config["automation_settings"]["prefer_coordinates"] = True
                
            logging.debug("Processed coordinate properties for backward compatibility")
        except Exception as e:
            logging.error(f"Error processing coordinate properties: {e}")
    
    def save_configuration(self):
        """Save the current configuration."""
        try:
            # Build configuration
            config = self.config_manager.get_all()
            
            # Update UI elements
            config["ui_elements"] = self.elements
            
            # Check if we're in session mode
            in_session_mode = self.config_manager.is_in_session_mode()
            
            # Set values in config manager
            for key, value in config.items():
                self.config_manager.set(key, value)
            
            # Save configuration with appropriate method
            if in_session_mode:
                if self.config_manager.save_preserving_sessions():
                    messagebox.showinfo("Success", "Configuration saved successfully (with preserved sessions).")
                    self.show_status("Configuration saved successfully with preserved sessions.")
                else:
                    messagebox.showerror("Save Error", "Failed to save configuration.")
                    self.show_status("Failed to save configuration.")
            else:
                if self.config_manager.save():
                    messagebox.showinfo("Success", "Configuration saved successfully.")
                    self.show_status("Configuration saved successfully.")
                else:
                    messagebox.showerror("Save Error", "Failed to save configuration.")
                    self.show_status("Failed to save configuration.")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving configuration: {e}")
            self.show_status(f"Error saving configuration: {e}")

    def capture_click_coordinates(self):
        """Capture mouse coordinates for direct clicking."""
        if not self.current_element:
            messagebox.showwarning("No Selection", "Please select an element first.")
            return
            
        # Display instructions
        msg = f"Position your mouse where you want to click for '{self.current_element}'.\n\n"
        msg += "The window will minimize in 2 seconds. Position your mouse and remain still."
        messagebox.showinfo("Capture Coordinates", msg)
        
        self.root.iconify()  # Minimize window
        self.show_status(f"Position mouse for {self.current_element} click and wait 3 seconds...")
        time.sleep(3)  # Give user time to position mouse
        
        # Get mouse position
        x, y = pyautogui.position()
        
        # Update element configuration
        if self.current_element not in self.elements:
            self.elements[self.current_element] = {
                "region": None, 
                "reference_paths": [], 
                "confidence": 0.7
            }
        
        self.elements[self.current_element]["click_coordinates"] = [x, y]
        
        # Set use_coordinates_first to true if not already set
        if "use_coordinates_first" not in self.elements[self.current_element]:
            self.elements[self.current_element]["use_coordinates_first"] = True
        
        self.root.deiconify()  # Restore window
        self.update_element_info()  # Update UI to show new coordinates
        
        # Also update the canvas to show the coordinates
        current_tab = self.notebook.index("current")
        if current_tab == 1:  # Define tab
            self.draw_coordinate_markers(self.define_canvas)
        
        self.show_status(f"Captured click coordinates for {self.current_element}: ({x}, {y})")

    def toggle_coordinate_preference(self):
        """Toggle whether to use coordinates first or visual recognition first."""
        if not self.current_element or self.current_element not in self.elements:
            messagebox.showwarning("No Element", "Please select a valid element first.")
            return
        
        # Toggle the preference
        current = self.elements[self.current_element].get("use_coordinates_first", False)
        self.elements[self.current_element]["use_coordinates_first"] = not current
        
        # Update the UI
        self.update_element_info()
        self.show_status(f"Set {self.current_element} to {'use coordinates first' if not current else 'use visual recognition first'}")

    def draw_coordinate_markers(self, canvas):
        """Draw markers for click coordinates on the canvas."""
        if not self.screenshot:
            return
        
        # Clear previous markers
        canvas.delete("coordinate_marker")
        
        # Draw each element's coordinates
        for name, config in self.elements.items():
            if "click_coordinates" in config and config["click_coordinates"]:
                try:
                    x, y = config["click_coordinates"]
                    
                    # Scale to canvas size
                    canvas_x = x * self.scale
                    canvas_y = y * self.scale
                    
                    # Draw a crosshair
                    size = 10
                    canvas.create_line(
                        canvas_x - size, canvas_y, 
                        canvas_x + size, canvas_y, 
                        fill="red", width=2, tags="coordinate_marker"
                    )
                    canvas.create_line(
                        canvas_x, canvas_y - size, 
                        canvas_x, canvas_y + size, 
                        fill="red", width=2, tags="coordinate_marker"
                    )
                    
                    # Draw a circle
                    canvas.create_oval(
                        canvas_x - size, canvas_y - size,
                        canvas_x + size, canvas_y + size,
                        outline="red", width=2, tags="coordinate_marker"
                    )
                    
                    # Add label
                    canvas.create_text(
                        canvas_x + size + 5, canvas_y,
                        text=f"{name} ({x}, {y})",
                        fill="red", anchor=tk.W, tags="coordinate_marker"
                    )
                except Exception as e:
                    logging.error(f"Error drawing coordinate marker for {name}: {e}")


def main():
    """Main function to run the calibration tool."""
    try:
        root = tk.Tk()
        app = UnifiedCalibrationTool(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()