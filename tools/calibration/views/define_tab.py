import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import logging
import time

class DefineTabView:
    """
    Define Tab View - Manages UI for defining UI elements and capturing reference images.
    Handles region selection, reference image capture, and element property display.
    """
    def __init__(self, parent, controller, config=None):
        """
        Initialize the Define Tab View.
        
        Args:
            parent: Parent widget (notebook)
            controller: Controller managing tab interactions
            config: Configuration object
        """
        self.parent = parent
        self.controller = controller
        self.config = config
        
        # Main frame
        self.frame = ttk.Frame(parent)
        
        # Component references
        self.canvas = None
        self.element_info = None
        self.element_info_frame = None
        self.reference_frame = None
        self.capture_button = None
        self.confidence_var = None
        
        # State variables
        self.capture_mode = False
        self.selection_start = None
        self.selection_box = None
        self.scale = 1.0
        self.screenshot = None
        self.screenshot_image = None
        self.reference_images = []
        
        # Create UI
        self._create_ui()
        self._bind_events()
    
    def _create_ui(self):
        """Create all UI components for the Define tab."""
        # Create split pane layout
        self._create_paned_layout()
        
        # Create left panel components
        self._create_element_info_panel()
        self._create_reference_controls()
        self._create_coordinate_controls()
        self._create_region_settings()
        self._create_navigation_buttons()
        
        # Create canvas for region selection
        self._create_canvas()
    
    def _create_paned_layout(self):
        """Create paned window layout."""
        self.pane = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        self.pane.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Element properties
        self.left_frame = ttk.Frame(self.pane, padding=10)
        self.pane.add(self.left_frame, weight=1)
        
        # Right panel - Region definition
        self.right_frame = ttk.LabelFrame(self.pane, text="Element Region Definition")
        self.pane.add(self.right_frame, weight=3)
    
    def _create_element_info_panel(self):
        """Create element information panel."""
        ttk.Label(self.left_frame, text="Element Properties", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Current element display
        self.element_info_frame = ttk.LabelFrame(self.left_frame, text="Current Element")
        self.element_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.element_info = tk.Text(self.element_info_frame, height=5, width=30, wrap=tk.WORD, state=tk.DISABLED)
        self.element_info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _create_reference_controls(self):
        """Create reference image management controls."""
        ttk.Label(self.left_frame, text="Reference Images", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        # Reference control buttons
        ref_button_frame = ttk.Frame(self.left_frame)
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
        
        # Reference image preview
        self.reference_frame = ttk.LabelFrame(self.left_frame, text="Reference Images")
        self.reference_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
    
    def _create_coordinate_controls(self):
        """Create coordinate capture controls."""
        coord_button_frame = ttk.Frame(self.left_frame)
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
    
    def _create_region_settings(self):
        """Create region settings controls."""
        ttk.Label(self.left_frame, text="Region Settings", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        self.confidence_var = tk.DoubleVar(value=0.7)
        confidence_frame = ttk.Frame(self.left_frame)
        confidence_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(confidence_frame, text="Confidence:").pack(side=tk.LEFT)
        self.confidence_slider = ttk.Scale(
            confidence_frame, 
            from_=0.1, 
            to=1.0, 
            variable=self.confidence_var, 
            orient=tk.HORIZONTAL
        )
        self.confidence_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Create a StringVar that updates with the confidence value
        self.confidence_text = tk.StringVar(value="0.70")
        self.confidence_var.trace_add("write", self._update_confidence_text)
        ttk.Label(confidence_frame, textvariable=self.confidence_text).pack(side=tk.RIGHT)
    
    def _create_navigation_buttons(self):
        """Create navigation buttons for moving between tabs."""
        nav_frame = ttk.Frame(self.left_frame)
        nav_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(nav_frame, text="Back to Setup", 
                 command=lambda: self.controller.select_tab(0)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Go to Analyze", 
                 command=lambda: self.controller.select_tab(2)).pack(side=tk.RIGHT)
    
    def _create_canvas(self):
        """Create canvas for region definition."""
        self.canvas = tk.Canvas(self.right_frame, bg="gray90")
        self.canvas.pack(fill=tk.BOTH, expand=True)
    
    def _bind_events(self):
        """Bind events to their handlers."""
        # Canvas bindings for mouse events
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        
        # Confidence slider value change
        self.confidence_var.trace_add("write", self._on_confidence_changed)
    
    def _update_confidence_text(self, *args):
        """Update the confidence text label when the slider changes."""
        self.confidence_text.set(f"{self.confidence_var.get():.2f}")
    
    def _on_confidence_changed(self, *args):
        """Handle confidence slider changes."""
        if self.controller and hasattr(self.controller, 'update_element_confidence'):
            self.controller.update_element_confidence(self.confidence_var.get())
    
    def _on_mouse_down(self, event):
        """Handle mouse button press."""
        if not self.screenshot:
            self.controller.show_status("Please take or load a screenshot first.")
            return
            
        if not self.controller.current_element and not self.capture_mode:
            self.controller.show_status("Please select an element first.")
            return
            
        self.selection_start = (event.x, event.y)
        
        # Clear previous selection box
        self.canvas.delete("selection")
    
    def _on_mouse_drag(self, event):
        """Handle mouse drag."""
        if not self.selection_start:
            return
        
        # Clear previous selection box
        self.canvas.delete("selection")
        
        # Draw new selection box
        self.selection_box = self.canvas.create_rectangle(
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
            self.controller.show_status(f"Selection for reference capture: {width}x{height} pixels")
        else:
            self.controller.show_status(f"Element region selection: {width}x{height} pixels")
    
    def _on_mouse_up(self, event):
        """Handle mouse button release."""
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
        
        if self.capture_mode:
            # In capture mode, capture the selected region as a reference image
            self.capture_reference_image(selection_region)
        else:
            # In region definition mode, save the region for the element
            self.controller.set_element_region(selection_region)
        
        # Reset selection
        self.selection_start = None
    
    def canvas_to_image_coordinates(self, canvas_x, canvas_y):
        """Convert canvas coordinates to image coordinates."""
        image_x = int(canvas_x / self.scale)
        image_y = int(canvas_y / self.scale)
        return image_x, image_y
    
    def image_to_canvas_coordinates(self, image_x, image_y):
        """Convert image coordinates to canvas coordinates."""
        canvas_x = int(image_x * self.scale)
        canvas_y = int(image_y * self.scale)
        return canvas_x, canvas_y
    
    def get_scaled_region(self, x1, y1, x2, y2):
        """Create a properly scaled region from coordinates."""
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        
        orig_left = int(left / self.scale)
        orig_top = int(top / self.scale)
        orig_width = int((right - left) / self.scale)
        orig_height = int((bottom - top) / self.scale)
        
        return (orig_left, orig_top, orig_width, orig_height)
    
    def update_canvas(self, screenshot):
        """Update the canvas with the current screenshot."""
        if not screenshot:
            return
            
        self.screenshot = screenshot
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
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
        resized = resized.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to Tkinter PhotoImage
        self.screenshot_image = ImageTk.PhotoImage(resized)
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Draw the image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.screenshot_image)
        self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        
        # Draw existing regions and coordinate markers
        self.draw_regions()
        self.draw_coordinate_markers()
    
    def draw_regions(self):
        """Draw all defined regions on the canvas."""
        if not self.screenshot or not self.controller.elements:
            return
        
        # Clear previous regions
        self.canvas.delete("region")
        
        # Draw each element's region
        for name, config in self.controller.elements.items():
            region = config.get("region")
            if not region:
                continue
                
            # Determine color (green for selected, blue for others)
            color = "green" if name == self.controller.current_element else "blue"
            
            # Scale coordinates to canvas size
            x, y, w, h = region
            canvas_x = x * self.scale
            canvas_y = y * self.scale
            canvas_w = w * self.scale
            canvas_h = h * self.scale
            
            # Draw rectangle
            self.canvas.create_rectangle(
                canvas_x, canvas_y, 
                canvas_x + canvas_w, canvas_y + canvas_h,
                outline=color, width=2, tags="region"
            )
            
            # Draw label
            self.canvas.create_text(
                canvas_x + 5, canvas_y + 5,
                text=name, anchor=tk.NW,
                fill=color, tags="region"
            )
    
    def draw_coordinate_markers(self):
        """Draw markers for click coordinates on the canvas."""
        if not self.screenshot or not self.controller.elements:
            return
        
        # Clear previous markers
        self.canvas.delete("coordinate_marker")
        
        # Draw each element's coordinates
        for name, config in self.controller.elements.items():
            if "click_coordinates" in config and config["click_coordinates"]:
                try:
                    x, y = config["click_coordinates"]
                    
                    # Scale to canvas size
                    canvas_x = x * self.scale
                    canvas_y = y * self.scale
                    
                    # Draw a crosshair
                    size = 10
                    self.canvas.create_line(
                        canvas_x - size, canvas_y, 
                        canvas_x + size, canvas_y, 
                        fill="red", width=2, tags="coordinate_marker"
                    )
                    self.canvas.create_line(
                        canvas_x, canvas_y - size, 
                        canvas_x, canvas_y + size, 
                        fill="red", width=2, tags="coordinate_marker"
                    )
                    
                    # Draw a circle
                    self.canvas.create_oval(
                        canvas_x - size, canvas_y - size,
                        canvas_x + size, canvas_y + size,
                        outline="red", width=2, tags="coordinate_marker"
                    )
                    
                    # Add label
                    self.canvas.create_text(
                        canvas_x + size + 5, canvas_y,
                        text=f"{name} ({x}, {y})",
                        fill="red", anchor=tk.W, tags="coordinate_marker"
                    )
                except Exception as e:
                    logging.error(f"Error drawing coordinate marker for {name}: {e}")
    
    def toggle_capture_mode(self):
        """Toggle between region definition and reference capture modes."""
        if not self.controller.current_element:
            messagebox.showwarning("No Selection", "Please select an element first.")
            return
        
        # Toggle capture mode
        self.capture_mode = not self.capture_mode
        
        if self.capture_mode:
            # Entering reference capture mode
            self.capture_mode_var.set("Exit Capture Mode")
            self.controller.show_status(
                f"Reference capture mode active for '{self.controller.current_element}'. "
                "Select area to capture."
            )
        else:
            # Exiting reference capture mode, back to region definition
            self.capture_mode_var.set("Enter Capture Mode")
            self.controller.show_status("Region definition mode active. Select areas to define element regions.")
    
    def capture_reference_image(self, region=None):
        """Capture a reference image from the current screenshot."""
        if not self.controller.current_element or not self.screenshot:
            return
            
        try:
            # Let the controller handle the actual capturing
            self.controller.capture_reference_image(region)
            
            # Update the reference preview
            self.update_reference_preview()
            
        except Exception as e:
            messagebox.showerror("Capture Error", f"Error capturing reference: {e}")
            self.controller.show_status(f"Error capturing reference: {e}")
    
    def capture_reference_from_selection(self):
        """Capture a reference image from the current selection."""
        if not self.controller.current_element:
            messagebox.showwarning("No Selection", "Please select an element first.")
            return
        
        # Get the current selection
        selection_items = self.canvas.find_withtag("selection")
        if not selection_items:
            messagebox.showwarning("No Selection", "Please make a selection first.")
            return
        
        # Get the coordinates of the selection
        coords = self.canvas.coords(selection_items[0])
        if len(coords) != 4:
            return
            
        # Convert to a region
        region = self.get_scaled_region(coords[0], coords[1], coords[2], coords[3])
        
        # Capture the reference image
        self.capture_reference_image(region)
    
    def update_reference_preview(self):
        """Update the reference image preview panel."""
        # Clear previous preview
        for widget in self.reference_frame.winfo_children():
            widget.destroy()
        
        if not self.controller.current_element:
            ttk.Label(self.reference_frame, text="No element selected").pack(pady=10)
            return
        
        # Get reference images for this element
        element_name = self.controller.current_element
        if element_name not in self.controller.elements:
            ttk.Label(self.reference_frame, text="Element not found in configuration").pack(pady=10)
            return
            
        reference_paths = self.controller.elements[element_name].get("reference_paths", [])
        
        # Filter out variants to show only original images
        original_references = [path for path in reference_paths if not any(
            x in path for x in ['_gray', '_contrast', '_thresh', '_scale'])]
        
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
                    command=lambda p=path: self.controller.delete_reference(p)
                ).pack(pady=(0, 5))
                
            except Exception as e:
                logging.error(f"Error loading thumbnail: {e}")
        
        # Update canvas scroll region
        frame.update_idletasks()
        preview_canvas.config(scrollregion=preview_canvas.bbox("all"))
    
    def update_element_info(self):
        """Update the element information display."""
        # Enable text widget for editing
        self.element_info.config(state=tk.NORMAL)
        
        # Clear current content
        self.element_info.delete(1.0, tk.END)
        
        current_element = self.controller.current_element
        if not current_element:
            self.element_info.insert(tk.END, "No element selected")
        elif current_element in self.controller.elements:
            element_config = self.controller.elements[current_element]
            
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
            info = f"Element: {current_element}\n"
            info += f"Region: {region_str}\n"
            info += f"References: {ref_count}\n"
            info += f"Confidence: {element_config.get('confidence', 0.7):.2f}\n"
            info += f"Click coords: {coordinates_str}\n"
            info += f"Use coords first: {element_config.get('use_coordinates_first', False)}"
            
            self.element_info.insert(tk.END, info)
        else:
            self.element_info.insert(tk.END, f"Element '{current_element}' not found in configuration.")
        
        # Disable editing
        self.element_info.config(state=tk.DISABLED)
        
        # Update confidence slider to match current element
        if current_element and current_element in self.controller.elements:
            element_config = self.controller.elements[current_element]
            confidence = element_config.get('confidence', 0.7)
            self.confidence_var.set(confidence)
    
    def capture_click_coordinates(self):
        """Capture mouse coordinates for direct clicking."""
        if not self.controller.current_element:
            messagebox.showwarning("No Selection", "Please select an element first.")
            return
        
        self.controller.capture_click_coordinates()
    
    def toggle_coordinate_preference(self):
        """Toggle whether to use coordinates first or visual recognition first."""
        if not self.controller.current_element:
            messagebox.showwarning("No Element", "Please select a valid element first.")
            return
        
        self.controller.toggle_coordinate_preference()