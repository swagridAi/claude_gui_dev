import tkinter as tk
from tkinter import ttk, messagebox
import os
import logging
from datetime import datetime
from PIL import Image, ImageTk

class VerifyTabView:
    """
    UI view for the verification tab - allows testing UI elements and generating reports.
    Represents the final stage in the calibration workflow.
    """
    
    def __init__(self, parent, app_controller, recognition_controller, element_controller, config_controller):
        """
        Initialize the verification tab view.
        
        Args:
            parent: Parent frame (ttk.Frame)
            app_controller: Main application controller
            recognition_controller: Controller for element recognition
            element_controller: Controller for element management
            config_controller: Controller for configuration management
        """
        self.parent = parent
        self.app_controller = app_controller
        self.recognition_controller = recognition_controller
        self.element_controller = element_controller
        self.config_controller = config_controller
        
        # UI components
        self.verify_canvas = None
        self.verify_results = None
        self.verify_results_frame = None
        self.report_button = None
        self.finalize_button = None
        
        # State tracking
        self.test_results = {}
        self.is_verified = False
        self.screenshot_image = None
        
        # Set up the UI components
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the verification tab user interface."""
        # Create a paned window
        pane = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Verification controls
        left_frame = ttk.Frame(pane, padding=10)
        pane.add(left_frame, weight=1)
        
        # Right panel - Verification results
        right_frame = ttk.LabelFrame(pane, text="Verification Results")
        pane.add(right_frame, weight=3)
        
        # Verification controls in left frame
        ttk.Label(left_frame, text="Verification", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Test controls
        ttk.Button(left_frame, text="Test All Elements", 
                  command=self.on_test_clicked).pack(fill=tk.X, pady=(0, 5))
        
        self.report_button = ttk.Button(left_frame, text="Generate Calibration Report", 
                  command=self.on_generate_report_clicked)
        self.report_button.pack(fill=tk.X, pady=(0, 5))
        
        # Results display
        self.verify_results_frame = ttk.LabelFrame(left_frame, text="Test Results")
        self.verify_results_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        self.verify_results = tk.Text(self.verify_results_frame, height=15, width=30, wrap=tk.WORD, state=tk.DISABLED)
        self.verify_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add a scrollbar to the results
        scrollbar = ttk.Scrollbar(self.verify_results_frame, orient=tk.VERTICAL, command=self.verify_results.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.verify_results.config(yscrollcommand=scrollbar.set)
        
        # Navigation buttons
        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(nav_frame, text="Back to Analyze", 
                  command=lambda: self.app_controller.select_tab(2)).pack(side=tk.LEFT, padx=(0, 5))
        
        self.finalize_button = ttk.Button(nav_frame, text="Finish & Save", 
                  command=self.on_finalize_clicked)
        self.finalize_button.pack(side=tk.RIGHT)
        
        # Canvas for verification results
        self.verify_canvas = tk.Canvas(right_frame, bg="gray90")
        self.verify_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initially disable report button until test is run
        self.enable_report_controls(False)
    
    def update_canvas(self, image=None):
        """
        Update the verification canvas with the current screenshot or provided image.
        
        Args:
            image: Optional PIL Image to display
        """
        if image is None and not hasattr(self.app_controller, 'screenshot'):
            return
            
        # Use provided image or get from controller
        img = image or self.app_controller.screenshot
        
        # Get canvas dimensions
        canvas_width = self.verify_canvas.winfo_width()
        canvas_height = self.verify_canvas.winfo_height()
        
        # If canvas not yet sized, use defaults
        if canvas_width <= 1:
            canvas_width = 800
            canvas_height = 600
        
        # Resize the image to fit the canvas while maintaining aspect ratio
        img_width, img_height = img.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize the image
        resized = img.copy()
        resized = resized.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to Tkinter PhotoImage
        self.screenshot_image = ImageTk.PhotoImage(resized)
        
        # Clear canvas
        self.verify_canvas.delete("all")
        
        # Draw the image
        self.verify_canvas.create_image(0, 0, anchor=tk.NW, image=self.screenshot_image)
        self.verify_canvas.config(scrollregion=(0, 0, new_width, new_height))
        
        # Store the scale factor for coordinate conversion
        self.scale = scale
    
    def draw_match_indicators(self, results):
        """
        Draw indicators on the canvas showing where elements were found.
        
        Args:
            results: Dictionary of element names to match information
        """
        self.verify_canvas.delete("match_indicator")
        
        for name, result in results.items():
            if result.get("found"):
                location = result.get("location")
                if location:
                    # Extract coordinates
                    x, y, w, h = location
                    
                    # Scale coordinates to canvas size
                    canvas_x = x * self.scale
                    canvas_y = y * self.scale
                    canvas_w = w * self.scale
                    canvas_h = h * self.scale
                    
                    # Draw rectangle for the element
                    self.verify_canvas.create_rectangle(
                        canvas_x, canvas_y, 
                        canvas_x + canvas_w, canvas_y + canvas_h,
                        outline="green", width=2, tags=("match_indicator", name)
                    )
                    
                    # Add element name label
                    self.verify_canvas.create_text(
                        canvas_x, canvas_y - 10,
                        text=name,
                        fill="green", anchor=tk.W, tags=("match_indicator", name)
                    )
    
    def update_results_display(self, results):
        """
        Update the test results text widget with verification results.
        
        Args:
            results: Dictionary of element names to match information
        """
        # Enable text widget for editing
        self.verify_results.config(state=tk.NORMAL)
        
        # Clear current content
        self.verify_results.delete(1.0, tk.END)
        
        # Add results for each element
        for name, result in results.items():
            status = "Found" if result.get("found") else "Not found"
            self.verify_results.insert(tk.END, f"{name}: {status}\n")
            
            if result.get("found"):
                # Show location info if found
                location = result.get("location")
                if location:
                    x, y, w, h = location
                    self.verify_results.insert(tk.END, f"  Location: ({x}, {y}, {w}, {h})\n\n")
            else:
                self.verify_results.insert(tk.END, "  Not found! Check region and references.\n\n")
        
        # Add summary
        found_count = sum(1 for r in results.values() if r.get("found"))
        total_count = len(results)
        
        self.verify_results.insert(tk.END, f"\nSummary: Found {found_count} out of {total_count} elements.")
        
        # Disable editing
        self.verify_results.config(state=tk.DISABLED)
        
        # Enable/disable report button based on results
        self.enable_report_controls(found_count > 0)
    
    def enable_report_controls(self, enabled=True):
        """
        Enable or disable the report generation controls.
        
        Args:
            enabled: Whether to enable the controls
        """
        state = tk.NORMAL if enabled else tk.DISABLED
        self.report_button.config(state=state)
    
    def set_verification_state(self, is_verified=True):
        """
        Set whether the verification has been performed successfully.
        
        Args:
            is_verified: Whether verification was successful
        """
        self.is_verified = is_verified
        # Could update UI to reflect verified state (e.g., change button color)
    
    def on_test_clicked(self):
        """Handle the Test All Elements button click."""
        if not hasattr(self.app_controller, 'screenshot') or self.app_controller.screenshot is None:
            messagebox.showwarning("No Screenshot", "Please take or load a screenshot first.")
            return
        
        # Update the canvas with current screenshot
        self.update_canvas()
        
        # Show status update
        self.app_controller.show_status("Testing all elements...")
        
        # Run the tests through the recognition controller
        self.test_results = self.recognition_controller.test_all_elements()
        
        # Update the results display
        self.update_results_display(self.test_results)
        
        # Draw match indicators on the canvas
        self.draw_match_indicators(self.test_results)
        
        # Set verification state based on results
        found_count = sum(1 for r in self.test_results.values() if r.get("found"))
        total_count = len(self.test_results)
        self.set_verification_state(found_count > 0)
        
        # Show status update
        self.app_controller.show_status(f"Verification complete. Found {found_count} out of {total_count} elements.")
    
    def on_generate_report_clicked(self):
        """Handle the Generate Calibration Report button click."""
        if not self.test_results:
            messagebox.showwarning("No Results", "Please run verification test first.")
            return
        
        # Delegate to recognition controller to generate report
        report_path = self.recognition_controller.generate_calibration_report(self.test_results)
        
        if report_path:
            self.app_controller.show_status(f"Calibration report generated: {report_path}")
        else:
            self.app_controller.show_status("Failed to generate calibration report")
    
    def on_finalize_clicked(self):
        """Handle the Finish & Save button click."""
        if not self.element_controller.get_elements():
            messagebox.showwarning("No Elements", "No elements have been defined.")
            return
        
        # Check for incomplete elements
        incomplete = self.element_controller.get_incomplete_elements()
        
        if incomplete:
            if not messagebox.askyesno("Incomplete Configuration", 
                                    f"The following elements are incomplete:\n{', '.join(incomplete)}\n\nSave anyway?"):
                return
        
        # Add option to preserve sessions
        preserve_sessions = messagebox.askyesno("Preserve Sessions", 
                                             "Do you want to preserve any existing session configurations?")
        
        # Save configuration through the config controller
        success = self.config_controller.save_configuration(preserve_sessions)
        
        if success:
            messagebox.showinfo("Success", "Configuration saved successfully.")
            self.app_controller.show_status("Configuration saved successfully.")
        else:
            messagebox.showerror("Save Error", "Failed to save configuration.")
            self.app_controller.show_status("Failed to save configuration.")