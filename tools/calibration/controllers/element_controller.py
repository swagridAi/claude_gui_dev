"""
Element Controller Module

Handles element management operations for the Claude Automation calibration tool.
Responsible for creating, updating, deleting, and managing UI elements and their references.
"""

import os
import logging
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk

# Relative imports for the new package structure
from ..models.element_model import ElementModel
from ..utils.image_processor import capture_reference_image, preprocess_image


class ElementController:
    """
    Controls element management operations including creation, modification, and deletion.
    Acts as intermediary between UI views and element data models.
    """
    
    def __init__(self, app_controller, view, config_manager, reference_manager=None):
        """
        Initialize with required dependencies.
        
        Args:
            app_controller: Parent controller for app-wide operations
            view: View instance containing element UI components
            config_manager: Configuration manager for saving/loading elements
            reference_manager: Optional reference image manager
        """
        self.app_controller = app_controller
        self.view = view
        self.config_manager = config_manager
        self.reference_manager = reference_manager
        
        # Elements dictionary to store all registered elements
        self.elements = {}
        self.current_element = None
        self.reference_count = {}
        self.reference_images = []  # Store references to prevent garbage collection
        
        # Connect UI event handlers
        self._connect_event_handlers()
    
    def _connect_event_handlers(self):
        """Connect controller methods to view events."""
        # Connect element listbox selection event
        if hasattr(self.view, 'element_listbox'):
            self.view.element_listbox.bind('<<ListboxSelect>>', self._on_element_select_event)
    
    def load_elements(self):
        """Load elements from configuration."""
        ui_elements = self.config_manager.get("ui_elements", {})
        
        # Clear existing elements
        self.elements = {}
        if hasattr(self.view, 'element_listbox'):
            self.view.element_listbox.delete(0, tk.END)
        
        # Process each UI element
        for name, element_config in ui_elements.items():
            # Add to elements dictionary
            self.elements[name] = element_config
            
            # Add to listbox if available
            if hasattr(self.view, 'element_listbox'):
                self.view.element_listbox.insert(tk.END, name)
            
            # Count reference images
            ref_count = len(element_config.get("reference_paths", []))
            self.reference_count[name] = ref_count
        
        # Select the first element if any exist
        if self.elements and hasattr(self.view, 'element_listbox'):
            self.view.element_listbox.selection_set(0)
            self.current_element = self.view.element_listbox.get(0)
            self.update_element_info()
            self.update_reference_preview()
    
    def add_element(self, name=None, properties=None):
        """
        Add a new element with given name and properties.
        
        Args:
            name: Element name (if None, taken from view input)
            properties: Initial properties dictionary (optional)
        
        Returns:
            Boolean indicating success
        """
        try:
            # Get name from view if not provided
            if name is None and hasattr(self.view, 'element_var'):
                name = self.view.element_var.get().strip()
            
            # Validate element name
            if not self._validate_element_name(name):
                return False
            
            # Create default properties if not provided
            if properties is None:
                properties = {
                    "region": None,
                    "reference_paths": [],
                    "confidence": 0.7
                }
            
            # Add element to dictionary
            self.elements[name] = properties
            
            # Add to listbox if available
            if hasattr(self.view, 'element_listbox'):
                self.view.element_listbox.insert(tk.END, name)
                
                # Clear entry if available
                if hasattr(self.view, 'element_var'):
                    self.view.element_var.set("")
            
            # Set as current element
            self.select_element(name)
            
            # Initialize reference count
            self.reference_count[name] = 0
            
            # Create directory for reference images
            os.makedirs(f"assets/reference_images/{name}", exist_ok=True)
            
            # Notify status update
            self._notify_status(f"Added element '{name}'. Now select its region on the screenshot.")
            
            return True
            
        except Exception as e:
            self._handle_element_error("add", name, e)
            return False
    
    def delete_element(self, element_name=None):
        """
        Delete an element.
        
        Args:
            element_name: Name of element to delete (if None, uses current_element)
        
        Returns:
            Boolean indicating success
        """
        try:
            # Use current element if not specified
            if element_name is None:
                element_name = self.current_element
                
            if not element_name:
                self._notify_status("No element selected to delete.", "warning")
                return False
            
            # Confirm deletion via view if available
            if hasattr(self.view, 'messagebox'):
                if not self.view.messagebox.askyesno("Confirm Deletion", f"Delete element '{element_name}'?"):
                    return False
            
            # Remove from elements dictionary
            if element_name in self.elements:
                del self.elements[element_name]
            else:
                self._notify_status(f"Element '{element_name}' not found for deletion.", "warning")
                return False
            
            # Remove from listbox if available
            if hasattr(self.view, 'element_listbox'):
                selection = self.view.element_listbox.curselection()
                if selection:
                    self.view.element_listbox.delete(selection[0])
            
            # Reset current element
            self.current_element = None
            
            # Update UI
            self.update_element_info()
            self.clear_reference_preview()
            
            # Refresh canvas if needed
            self._notify_element_changed(element_name, "deleted")
            
            self._notify_status("Element deleted.")
            return True
            
        except Exception as e:
            self._handle_element_error("delete", element_name, e)
            return False
    
    def select_element(self, element_name):
        """
        Select an element as the current active element.
        
        Args:
            element_name: Name of element to select
        
        Returns:
            Boolean indicating success
        """
        try:
            if not element_name:
                return False
                
            if element_name not in self.elements:
                self._notify_status(f"Element '{element_name}' not found", "warning")
                return False
            
            # Update current element
            self.current_element = element_name
            
            # Update selection in listbox if available
            if hasattr(self.view, 'element_listbox'):
                try:
                    # Find index of element in listbox
                    items = self.view.element_listbox.get(0, tk.END)
                    if element_name in items:
                        index = list(items).index(element_name)
                        self.view.element_listbox.selection_clear(0, tk.END)
                        self.view.element_listbox.selection_set(index)
                except Exception as e:
                    logging.debug(f"Error updating listbox selection: {e}")
            
            # Update UI
            self.update_element_info()
            self.update_reference_preview()
            
            # Notify other components
            self._notify_element_changed(element_name, "selected")
            
            self._notify_status(f"Selected element: {element_name}")
            return True
            
        except Exception as e:
            self._handle_element_error("select", element_name, e)
            return False
    
    def update_element_property(self, element_name, property_name, value):
        """
        Update a single property of an element.
        
        Args:
            element_name: Name of the element to update
            property_name: Name of the property to change
            value: New value for the property
        
        Returns:
            Boolean indicating success
        """
        try:
            if element_name not in self.elements:
                self._notify_status(f"Element '{element_name}' not found", "warning")
                return False
            
            # Update the property
            self.elements[element_name][property_name] = value
            
            # Update UI if this is the current element
            if element_name == self.current_element:
                self.update_element_info()
            
            # Notify other components
            self._notify_element_changed(element_name, "updated")
            
            return True
            
        except Exception as e:
            self._handle_element_error("update_property", 
                                      f"{element_name}.{property_name}", e)
            return False
    
    def update_element_region(self, element_name, region):
        """
        Update the region of an element.
        
        Args:
            element_name: Name of the element
            region: (x, y, width, height) tuple
        
        Returns:
            Boolean indicating success
        """
        try:
            if element_name not in self.elements:
                self._notify_status(f"Element '{element_name}' not found", "warning")
                return False
            
            # Update region
            self.elements[element_name]["region"] = region
            
            # Update UI
            self.update_element_info()
            
            # Notify other components
            self._notify_element_changed(element_name, "updated")
            
            self._notify_status(f"Region defined for {element_name}: {region}")
            return True
            
        except Exception as e:
            self._handle_element_error("update_region", element_name, e)
            return False
    
    def update_element_info(self):
        """Update the element information display."""
        # Skip if no element info display
        if not hasattr(self.view, 'element_info'):
            return
        
        try:
            # Enable text widget for editing
            self.view.element_info.config(state=tk.NORMAL)
            
            # Clear current content
            self.view.element_info.delete(1.0, tk.END)
            
            if not self.current_element:
                self.view.element_info.insert(tk.END, "No element selected")
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
                
                self.view.element_info.insert(tk.END, info)
            else:
                self.view.element_info.insert(tk.END, 
                    f"Element '{self.current_element}' not found in configuration.")
            
            # Disable editing
            self.view.element_info.config(state=tk.DISABLED)
            
        except Exception as e:
            logging.error(f"Error updating element info: {e}")
    
    def update_reference_preview(self):
        """Update the reference image preview panel."""
        # Skip if no reference frame
        if not hasattr(self.view, 'reference_frame'):
            return
        
        try:
            # Clear previous preview
            for widget in self.view.reference_frame.winfo_children():
                widget.destroy()
            
            if not self.current_element:
                ttk_label = getattr(tk, 'ttk', tk).Label
                ttk_label(self.view.reference_frame, text="No element selected").pack(pady=10)
                return
            
            # Get reference images for this element
            if self.current_element not in self.elements:
                ttk_label = getattr(tk, 'ttk', tk).Label
                ttk_label(self.view.reference_frame, 
                         text="Element not found in configuration").pack(pady=10)
                return
                
            reference_paths = self.elements[self.current_element].get("reference_paths", [])
            
            # Filter out variants to show only original images
            original_references = [path for path in reference_paths 
                                  if not any(x in path for x in [
                                      '_gray', '_contrast', '_thresh', '_scale'
                                  ])]
            
            if not original_references:
                ttk_label = getattr(tk, 'ttk', tk).Label
                ttk_label(self.view.reference_frame, text="No reference images").pack(pady=10)
                return
            
            # Create a canvas for thumbnails
            canvas_frame = getattr(tk, 'ttk', tk).Frame(self.view.reference_frame)
            canvas_frame.pack(fill=tk.BOTH, expand=True)
            
            preview_canvas = tk.Canvas(canvas_frame, bg="white")
            preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Add a scrollbar
            scrollbar = getattr(tk, 'ttk', tk).Scrollbar(
                canvas_frame, orient=tk.VERTICAL, command=preview_canvas.yview
            )
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            preview_canvas.configure(yscrollcommand=scrollbar.set)
            
            # Create a frame inside the canvas for the thumbnails
            frame = getattr(tk, 'ttk', tk).Frame(preview_canvas)
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
                    thumbnail_frame = getattr(tk, 'ttk', tk).Frame(frame)
                    thumbnail_frame.grid(row=i//2, column=i%2, padx=5, pady=5)
                    
                    # Add the thumbnail
                    label = getattr(tk, 'ttk', tk).Label(thumbnail_frame, image=photo)
                    label.pack()
                    
                    # Add the filename (shortened)
                    basename = os.path.basename(path)
                    short_name = basename[:15] + "..." if len(basename) > 18 else basename
                    getattr(tk, 'ttk', tk).Label(thumbnail_frame, text=short_name).pack()
                    
                    # Add delete button
                    getattr(tk, 'ttk', tk).Button(
                        thumbnail_frame, 
                        text="Delete", 
                        command=lambda p=path: self.delete_reference(p)
                    ).pack(pady=(0, 5))
                    
                except Exception as e:
                    logging.error(f"Error loading thumbnail: {e}")
            
            # Update canvas scroll region
            frame.update_idletasks()
            preview_canvas.config(scrollregion=preview_canvas.bbox("all"))
            
        except Exception as e:
            logging.error(f"Error updating reference preview: {e}")
    
    def clear_reference_preview(self):
        """Clear the reference preview panel."""
        if hasattr(self.view, 'reference_frame'):
            for widget in self.view.reference_frame.winfo_children():
                widget.destroy()
    
    def add_reference_image(self, element_name=None, region=None):
        """
        Add a reference image to an element.
        
        Args:
            element_name: Element to add reference to (uses current if None)
            region: Screen region to capture
        
        Returns:
            Path to the new reference image or None on failure
        """
        try:
            if element_name is None:
                element_name = self.current_element
                
            if not element_name:
                self._notify_status("No element selected for reference capture.", "warning")
                return None
                
            if element_name not in self.elements:
                self._notify_status(f"Element '{element_name}' not found", "warning")
                return None
            
            # Create directory if it doesn't exist
            reference_dir = f"assets/reference_images/{element_name}"
            os.makedirs(reference_dir, exist_ok=True)
            
            # Capture or extract region
            if region is None and hasattr(self.view, 'screenshot'):
                self._notify_status("No region specified for capture.", "warning")
                return None
                
            # Extract region from screenshot
            if hasattr(self.view, 'screenshot'):
                x, y, w, h = region
                screenshot = self.view.screenshot
                cropped = screenshot.crop((x, y, x + w, y + h))
                
                # Save as reference image
                import time
                timestamp = int(time.time())
                filename = f"{reference_dir}/{element_name}_{timestamp}.png"
                cropped.save(filename)
                
                # Add to element's reference paths
                if "reference_paths" not in self.elements[element_name]:
                    self.elements[element_name]["reference_paths"] = []
                    
                self.elements[element_name]["reference_paths"].append(filename)
                
                # Update reference count
                ref_count = len(self.elements[element_name]["reference_paths"])
                self.reference_count[element_name] = ref_count
                
                # Update UI
                self.update_element_info()
                self.update_reference_preview()
                
                # Generate variants with ReferenceImageManager if available
                if self.reference_manager:
                    variants = self.reference_manager.preprocess_reference_images([filename])
                    
                    # Add variants to element's reference paths
                    for variant in variants:
                        if variant != filename and variant not in self.elements[element_name]["reference_paths"]:
                            self.elements[element_name]["reference_paths"].append(variant)
                    
                    # Update reference count again after adding variants
                    ref_count = len(self.elements[element_name]["reference_paths"])
                    self.reference_count[element_name] = ref_count
                    
                    variant_count = len(variants) if variants else 0
                    self._notify_status(
                        f"Captured reference image for {element_name}. "
                        f"Created {variant_count} variants."
                    )
                else:
                    self._notify_status(f"Captured reference image for {element_name}.")
                
                return filename
            else:
                self._notify_status("No screenshot available for reference capture.", "warning")
                return None
                
        except Exception as e:
            self._handle_element_error("add_reference", element_name, e)
            return None
    
    def delete_reference(self, path):
        """
        Delete a reference image.
        
        Args:
            path: Path to the reference image
            
        Returns:
            Boolean indicating success
        """
        try:
            if not self.current_element or self.current_element not in self.elements:
                return False
                
            if path not in self.elements[self.current_element].get("reference_paths", []):
                return False
                
            # Check with view if available
            if hasattr(self.view, 'messagebox'):
                if not self.view.messagebox.askyesno(
                    "Confirm Deletion", f"Delete reference image {os.path.basename(path)}?"
                ):
                    return False
            
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
            
            self._notify_status(f"Deleted reference image {os.path.basename(path)}")
            return True
            
        except Exception as e:
            self._handle_element_error("delete_reference", 
                                     f"{self.current_element}/{os.path.basename(path)}", e)
            return False
    
    def capture_click_coordinates(self, element_name=None):
        """
        Capture mouse coordinates for direct clicking.
        
        Args:
            element_name: Element to set coordinates for (uses current if None)
            
        Returns:
            Tuple of (x, y) coordinates or None on failure
        """
        try:
            if element_name is None:
                element_name = self.current_element
                
            if not element_name:
                self._notify_status("No element selected for coordinate capture.", "warning")
                return None
                
            if element_name not in self.elements:
                self._notify_status(f"Element '{element_name}' not found", "warning")
                return None
            
            # Display instructions via view if available
            if hasattr(self.view, 'messagebox'):
                msg = f"Position your mouse where you want to click for '{element_name}'.\n\n"
                msg += "The window will minimize in 2 seconds. Position your mouse and remain still."
                self.view.messagebox.showinfo("Capture Coordinates", msg)
            
            # Minimize window if available
            if hasattr(self.view, 'root') and hasattr(self.view.root, 'iconify'):
                self.view.root.iconify()
                
            self._notify_status(f"Position mouse for {element_name} click and wait 3 seconds...")
            import time
            time.sleep(3)  # Give user time to position mouse
            
            # Get mouse position
            import pyautogui
            x, y = pyautogui.position()
            
            # Update element configuration
            self.elements[element_name]["click_coordinates"] = [x, y]
            
            # Set use_coordinates_first to true if not already set
            if "use_coordinates_first" not in self.elements[element_name]:
                self.elements[element_name]["use_coordinates_first"] = True
            
            # Restore window if available
            if hasattr(self.view, 'root') and hasattr(self.view.root, 'deiconify'):
                self.view.root.deiconify()
                
            # Update UI
            self.update_element_info()
            
            # Notify change
            self._notify_element_changed(element_name, "updated")
            
            self._notify_status(f"Captured click coordinates for {element_name}: ({x}, {y})")
            return (x, y)
            
        except Exception as e:
            self._handle_element_error("capture_coordinates", element_name, e)
            return None
    
    def toggle_coordinate_preference(self, element_name=None):
        """
        Toggle whether to use coordinates first or visual recognition first.
        
        Args:
            element_name: Element to toggle preference for (uses current if None)
            
        Returns:
            New preference value or None on failure
        """
        try:
            if element_name is None:
                element_name = self.current_element
                
            if not element_name or element_name not in self.elements:
                self._notify_status("No valid element selected.", "warning")
                return None
            
            # Toggle the preference
            current = self.elements[element_name].get("use_coordinates_first", False)
            new_value = not current
            self.elements[element_name]["use_coordinates_first"] = new_value
            
            # Update the UI
            self.update_element_info()
            
            preference_text = "use coordinates first" if new_value else "use visual recognition first"
            self._notify_status(f"Set {element_name} to {preference_text}")
            
            # Notify change
            self._notify_element_changed(element_name, "updated")
            
            return new_value
            
        except Exception as e:
            self._handle_element_error("toggle_preference", element_name, e)
            return None
    
    def _on_element_select_event(self, event):
        """Handle element selection from listbox."""
        if not hasattr(self.view, 'element_listbox'):
            return
            
        selection = self.view.element_listbox.curselection()
        if selection:
            index = selection[0]
            name = self.view.element_listbox.get(index)
            self.select_element(name)
    
    def _validate_element_name(self, name):
        """
        Validate element name.
        
        Args:
            name: Element name to validate
            
        Returns:
            Boolean indicating if name is valid
        """
        if not name:
            self._notify_status("Element name cannot be empty.", "warning")
            return False
        
        if name in self.elements:
            self._notify_status(f"Element '{name}' already exists.", "warning")
            return False
            
        return True
    
    def _handle_element_error(self, operation, element_name, error):
        """
        Standardized error handling for element operations.
        
        Args:
            operation: String describing the operation that failed
            element_name: Name of the element involved
            error: Exception or error message
        """
        error_msg = f"Error during {operation} operation for '{element_name}': {error}"
        logging.error(error_msg)
        self._notify_status(error_msg, "error")
    
    def _notify_status(self, message, level="info"):
        """
        Update status message via app controller.
        
        Args:
            message: Status message to display
            level: Message level (info, warning, error)
        """
        if hasattr(self.app_controller, 'show_status'):
            self.app_controller.show_status(message)
        else:
            # Fallback to logging
            if level == "warning":
                logging.warning(message)
            elif level == "error":
                logging.error(message)
            else:
                logging.info(message)
    
    def _notify_element_changed(self, element_name, change_type):
        """
        Notify app controller of element changes.
        
        Args:
            element_name: Name of the changed element
            change_type: Type of change (added, updated, deleted, selected)
        """
        if hasattr(self.app_controller, 'on_element_changed'):
            self.app_controller.on_element_changed(element_name, change_type)
            
        # Update canvases or other UI elements if needed
        if change_type in ["added", "updated", "deleted", "selected"]:
            self._refresh_canvases()
    
    def _refresh_canvases(self):
        """Refresh canvases to display updated element information."""
        if hasattr(self.app_controller, 'refresh_canvases'):
            self.app_controller.refresh_canvases()