import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import logging

class BaseView:
    """Base class for all tab views with common functionality."""
    
    def __init__(self, parent, controller=None):
        """
        Initialize base view.
        
        Args:
            parent: Parent widget (typically a ttk.Notebook or Frame)
            controller: Reference to controller handling business logic
        """
        self.parent = parent
        self.controller = controller
        self.frame = ttk.Frame(parent, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Image references to prevent garbage collection
        self.image_references = []
        
        # Scale factor for image display
        self.scale = 1.0
    
    def setup_ui(self):
        """Create the basic UI structure - to be overridden by subclasses."""
        pass
        
    def update(self, data=None):
        """
        Update view with current data.
        
        Args:
            data: Data to display in the view
        """
        pass
        
    def clear(self):
        """Clear all dynamic content."""
        # Clear image references to prevent memory leaks
        self.image_references.clear()
        
    def show_status(self, message, level="info"):
        """
        Show status message if controller provides the capability.
        
        Args:
            message: Status message to display
            level: Message level (info, warning, error)
        """
        if hasattr(self.controller, 'show_status'):
            self.controller.show_status(message)
        else:
            # Fallback logging if no controller status method
            log_level = getattr(logging, level.upper(), logging.INFO)
            logging.log(log_level, message)


class StatusBar:
    """Status bar component for displaying messages."""
    
    def __init__(self, parent):
        """
        Initialize status bar.
        
        Args:
            parent: Parent widget
        """
        self.parent = parent
        self.status_var = tk.StringVar(value="Ready")
        
        self.status_label = ttk.Label(
            parent, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            padding=(5, 2)
        )
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        
        # Default styling - can be overridden
        self.colors = {
            "info": "#000000",     # Black
            "success": "#008000",  # Green
            "warning": "#FFA500",  # Orange
            "error": "#FF0000"     # Red
        }
        
    def set_status(self, message, level="info"):
        """
        Set status message with optional level.
        
        Args:
            message: Status message to display
            level: Message level (info, success, warning, error)
        """
        self.status_var.set(message)
        
        # Apply color based on level
        color = self.colors.get(level.lower(), self.colors["info"])
        self.status_label.configure(foreground=color)
        
        # Update UI immediately
        self.parent.update_idletasks()


def create_labeled_entry(parent, label_text, var=None, width=20):
    """
    Create a labeled entry field with consistent styling.
    
    Args:
        parent: Parent widget
        label_text: Text for the label
        var: Optional StringVar to bind to entry
        width: Width of entry field
        
    Returns:
        frame: Container frame
        entry: Entry widget
    """
    frame = ttk.Frame(parent)
    
    label = ttk.Label(frame, text=label_text)
    label.pack(side=tk.LEFT, padx=(0, 5))
    
    if var is None:
        var = tk.StringVar()
        
    entry = ttk.Entry(frame, textvariable=var, width=width)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    return frame, entry, var


def create_labeled_combobox(parent, label_text, values=None, width=20):
    """
    Create a labeled combobox with consistent styling.
    
    Args:
        parent: Parent widget
        label_text: Text for the label
        values: List of values for combobox
        width: Width of combobox
        
    Returns:
        frame: Container frame
        combobox: Combobox widget
        var: StringVar bound to combobox
    """
    frame = ttk.Frame(parent)
    
    label = ttk.Label(frame, text=label_text)
    label.pack(side=tk.LEFT, padx=(0, 5))
    
    var = tk.StringVar()
    combobox = ttk.Combobox(frame, textvariable=var, values=values or [], width=width)
    combobox.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    return frame, combobox, var


def create_scrollable_frame(parent, **kwargs):
    """
    Create a scrollable frame container.
    
    Args:
        parent: Parent widget
        **kwargs: Additional arguments for frame
        
    Returns:
        outer_frame: Outer container frame
        inner_frame: Scrollable inner frame
    """
    # Create container frame
    outer_frame = ttk.Frame(parent)
    
    # Create canvas for scrolling
    canvas = tk.Canvas(outer_frame, **kwargs)
    scrollbar = ttk.Scrollbar(outer_frame, orient=tk.VERTICAL, command=canvas.yview)
    
    # Configure canvas and scrollbar
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Create inner frame for content
    inner_frame = ttk.Frame(canvas)
    
    # Add inner frame to canvas
    canvas_frame = canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)
    
    # Configure canvas scrolling based on frame size
    def configure_scroll(event):
        # Update scroll region when inner frame size changes
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        # Make inner frame expand to fill canvas width
        canvas.itemconfig(canvas_frame, width=canvas.winfo_width())
        
    inner_frame.bind("<Configure>", configure_scroll)
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_frame, width=canvas.winfo_width()))
    
    # Enable mouse wheel scrolling
    def on_mousewheel(event):
        # Platform-specific scaling for mouse wheel
        if event.num == 5 or event.delta < 0:
            canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            canvas.yview_scroll(-1, "units")
            
    # Bind mouse wheel events
    canvas.bind_all("<MouseWheel>", on_mousewheel)  # Windows
    canvas.bind_all("<Button-4>", on_mousewheel)    # Linux scroll up
    canvas.bind_all("<Button-5>", on_mousewheel)    # Linux scroll down
    
    return outer_frame, inner_frame


def create_canvas_with_scrollbars(parent, **kwargs):
    """
    Create a canvas with scrollbars for displaying images.
    
    Args:
        parent: Parent widget
        **kwargs: Additional arguments for canvas
        
    Returns:
        frame: Container frame
        canvas: Canvas with scrollbars
    """
    # Create container frame
    frame = ttk.Frame(parent)
    
    # Create scrollbars
    h_scrollbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
    v_scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
    
    # Create canvas
    canvas = tk.Canvas(
        frame, 
        xscrollcommand=h_scrollbar.set,
        yscrollcommand=v_scrollbar.set,
        **kwargs
    )
    
    # Configure scrollbars
    h_scrollbar.config(command=canvas.xview)
    v_scrollbar.config(command=canvas.yview)
    
    # Position widgets
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    return frame, canvas


def setup_image_display(canvas, image, scale=1.0):
    """
    Configure canvas for displaying an image with proper scaling.
    
    Args:
        canvas: Canvas widget
        image: PIL.Image object
        scale: Scale factor for display
        
    Returns:
        photo_image: PhotoImage for Tkinter display
        image_item: Canvas item ID for the image
    """
    # Clear canvas
    canvas.delete("all")
    
    # Resize the image based on scale
    if scale != 1.0:
        width, height = image.size
        new_width, new_height = int(width * scale), int(height * scale)
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        resized = image
        
    # Convert to PhotoImage for display
    photo_image = ImageTk.PhotoImage(resized)
    
    # Add to canvas
    image_item = canvas.create_image(0, 0, anchor=tk.NW, image=photo_image)
    
    # Update canvas scroll region
    canvas.config(scrollregion=canvas.bbox("all"))
    
    return photo_image, image_item