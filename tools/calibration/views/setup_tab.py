import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import logging


class SetupTabView:
    """
    View class for the Setup tab in the calibration tool.
    Handles UI components for taking screenshots and managing UI elements.
    """
    
    def __init__(self, parent, controller, image_processor):
        """
        Initialize the Setup tab view.
        
        Args:
            parent: Parent ttk.Frame container
            controller: Controller object handling business logic
            image_processor: Utility for processing images
        """
        self.parent = parent
        self.controller = controller
        self.image_processor = image_processor
        
        # UI state variables
        self.screenshot_image = None
        self.selection_start = None
        self.selection_box = None
        self.scale = 1.0
        
        # Initialize UI components
        self.setup_ui()
        
    def setup_ui(self):
        """Create all UI components for the Setup tab."""
        # Create a paned window for the setup tab
        main_pane = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)
        
        # Create left and right panels
        control_panel = self._create_left_panel(main_pane)
        preview_panel = self._create_right_panel(main_pane)
        
        main_pane.add(control_panel, weight=1)
        main_pane.add(preview_panel, weight=3)
        
    def _create_left_panel(self, parent_pane):
        """Create the control panel on the left side."""
        control_panel = ttk.Frame(parent_pane, padding=10)
        
        # Title
        ttk.Label(control_panel, text="Initial Setup", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Screenshot controls
        ttk.Label(control_panel, text="Capture or load a screenshot of Claude:").pack(anchor="w", pady=(0, 5))
        self._create_screenshot_controls(control_panel)
        
        # Element management
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(control_panel, text="UI Elements", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        self._create_element_management_section(control_panel)
        
        # Configuration controls
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        self._create_config_controls(control_panel)
        
        return control_panel
    
    def _create_right_panel(self, parent_pane):
        """Create the screenshot preview panel on the right side."""
        preview_panel = ttk.LabelFrame(parent_pane, text="Claude Interface Preview")
        
        # Canvas for screenshot display
        self.preview_canvas = tk.Canvas(preview_panel, bg="gray90")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Canvas bindings
        self.preview_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.preview_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        return preview_panel
    
    def _create_screenshot_controls(self, parent):
        """Create controls for taking/loading screenshots."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="Take Screenshot", 
                  command=self._on_take_screenshot).pack(side=tk.LEFT, padx=(0, 5))
                  
        ttk.Button(button_frame, text="Load Image", 
                  command=self._on_load_image).pack(side=tk.LEFT, padx=(0, 5))
                  
        ttk.Button(button_frame, text="Load from Logs", 
                  command=self._on_load_from_logs).pack(side=tk.LEFT)
    
    def _create_element_management_section(self, parent):
        """Create the UI element management section."""
        # Element input
        self.element_var = tk.StringVar()
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(input_frame, text="Element Name:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(input_frame, textvariable=self.element_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="Add", command=self._on_add_element).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Element list
        self._create_element_list(parent)
        
        # Element actions
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(action_frame, text="Delete Element", 
                  command=self._on_delete_element).pack(side=tk.LEFT, padx=(0, 5))
                  
        ttk.Button(action_frame, text="Go to Define Tab", 
                  command=self._on_go_to_define).pack(side=tk.RIGHT)
    
    def _create_element_list(self, parent):
        """Create the listbox for UI elements with scrollbar."""
        list_frame = ttk.LabelFrame(parent, text="Defined Elements")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        self.element_list_widget = tk.Listbox(list_frame, height=10)
        self.element_list_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.element_list_widget.bind('<<ListboxSelect>>', self.on_element_select)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.element_list_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.element_list_widget.configure(yscrollcommand=scrollbar.set)
    
    def _create_config_controls(self, parent):
        """Create buttons for configuration management."""
        config_frame = ttk.Frame(parent)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(config_frame, text="Load Config", 
                  command=self._on_load_config).pack(side=tk.LEFT, padx=(0, 5))
                  
        ttk.Button(config_frame, text="Save Config", 
                  command=self._on_save_config).pack(side=tk.LEFT)
    
    # Event handlers
    def on_element_select(self, event):
        """Handle selection from the element listbox."""
        selection = self.element_list_widget.curselection()
        if selection:
            element_name = self.element_list_widget.get(selection[0])
            self.controller.select_element(element_name)
    
    def on_mouse_down(self, event):
        """Handle mouse button press on canvas."""
        if not self.controller.has_screenshot():
            self.show_status("Please take or load a screenshot first.")
            return
            
        if not self.controller.has_current_element() and not self.controller.is_capture_mode():
            self.show_status("Please select an element first.")
            return
            
        self.selection_start = (event.x, event.y)
        self.preview_canvas.delete("selection")
    
    def on_mouse_drag(self, event):
        """Handle mouse drag on canvas."""
        if not self.selection_start:
            return
        
        # Clear previous selection box
        self.preview_canvas.delete("selection")
        
        # Draw new selection box
        selection_color = "red" if self.controller.is_capture_mode() else "blue"
        self.selection_box = self.preview_canvas.create_rectangle(
            self.selection_start[0], self.selection_start[1],
            event.x, event.y,
            outline=selection_color, 
            width=2,
            tags="selection"
        )
        
        # Update status with dimensions
        width = abs(event.x - self.selection_start[0])
        height = abs(event.y - self.selection_start[1])
        
        if self.controller.is_capture_mode():
            self.show_status(f"Selection for reference capture: {width}x{height} pixels")
        else:
            self.show_status(f"Element region selection: {width}x{height} pixels")
    
    def on_mouse_up(self, event):
        """Handle mouse button release on canvas."""
        if not self.selection_start:
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
        
        # Forward to controller
        self.controller.handle_selection(selection_region)
        
        # Reset selection
        self.selection_start = None
    
    # Button event handlers
    def _on_take_screenshot(self):
        """Handle take screenshot button click."""
        self.controller.take_screenshot()
    
    def _on_load_image(self):
        """Handle load image button click."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        
        if file_path:
            self.controller.load_image(file_path)
    
    def _on_load_from_logs(self):
        """Handle load from logs button click."""
        self.controller.show_log_screenshots()
    
    def _on_add_element(self):
        """Handle add element button click."""
        element_name = self.element_var.get().strip()
        if not element_name:
            messagebox.showwarning("Input Error", "Element name cannot be empty.")
            return
            
        self.controller.add_element(element_name)
        self.element_var.set("")  # Clear the entry
    
    def _on_delete_element(self):
        """Handle delete element button click."""
        self.controller.delete_element()
    
    def _on_go_to_define(self):
        """Handle go to define tab button click."""
        self.controller.switch_to_tab(1)  # Index 1 for Define tab
    
    def _on_load_config(self):
        """Handle load config button click."""
        self.controller.load_configuration()
    
    def _on_save_config(self):
        """Handle save config button click."""
        self.controller.save_configuration()
    
    # Public interface methods
    def update_canvas(self, screenshot):
        """
        Update the canvas with a new screenshot.
        
        Args:
            screenshot: PIL Image object
        """
        if not screenshot:
            return
            
        # Get canvas dimensions
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        # If canvas not yet sized, use defaults
        if canvas_width <= 1:
            canvas_width = 800
            canvas_height = 600
        
        # Resize the image to fit the canvas while maintaining aspect ratio
        img_width, img_height = screenshot.size
        self.scale = min(canvas_width / img_width, canvas_height / img_height)
        
        new_width = int(img_width * self.scale)
        new_height = int(img_height * self.scale)
        
        # Resize the image
        resized = screenshot.copy()
        resized = resized.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to Tkinter PhotoImage
        self.screenshot_image = ImageTk.PhotoImage(resized)
        
        # Clear canvas
        self.preview_canvas.delete("all")
        
        # Draw the image
        self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.screenshot_image)
        self.preview_canvas.config(scrollregion=(0, 0, new_width, new_height))
        
        # Draw existing regions
        self.draw_regions()
    
    def update_element_list(self, elements):
        """
        Update the element list with current elements.
        
        Args:
            elements: Dictionary of element objects
        """
        self.element_list_widget.delete(0, tk.END)
        
        for name in elements:
            self.element_list_widget.insert(tk.END, name)
    
    def draw_regions(self):
        """Draw all defined element regions on the canvas."""
        # This needs to be implemented based on the controller's element data
        self.preview_canvas.delete("region")
        
        elements = self.controller.get_elements()
        current_element = self.controller.get_current_element()
        
        for name, config in elements.items():
            region = config.get("region")
            if not region:
                continue
                
            # Determine color (green for selected, blue for others)
            color = "green" if name == current_element else "blue"
            
            # Scale coordinates to canvas size
            x, y, w, h = region
            canvas_x = x * self.scale
            canvas_y = y * self.scale
            canvas_w = w * self.scale
            canvas_h = h * self.scale
            
            # Draw rectangle
            self.preview_canvas.create_rectangle(
                canvas_x, canvas_y, 
                canvas_x + canvas_w, canvas_y + canvas_h,
                outline=color, width=2, tags="region"
            )
            
            # Draw label
            self.preview_canvas.create_text(
                canvas_x + 5, canvas_y + 5,
                text=name, anchor=tk.NW,
                fill=color, tags="region"
            )
    
    def set_selection_mode(self, is_capture_mode):
        """Set the current selection mode (region or capture)."""
        # This affects how selections are drawn and processed
        self.controller.set_capture_mode(is_capture_mode)
    
    def show_status(self, message):
        """Update the status message."""
        # Forward to controller for status bar update
        self.controller.show_status(message)