Introduction
Claude GUI Automation is an advanced screen automation tool designed to interact with Claude AI through its web interface. Unlike API-based approaches, this tool simulates human-like interactions through mouse movements, keyboard input, and visual recognition, allowing it to bypass CAPTCHA and login requirements that might block other automation methods.
This implementation uses an image recognition-based approach rather than fixed coordinates, making it significantly more robust against UI changes and screen resolution differences.
Project Overview
Key Features

Image Recognition: Uses template matching to locate UI elements rather than fixed coordinates
Self-Calibration: Automatically detects UI elements and adjusts to different screen sizes
Simplified State Machine: Streamlined automation flow focused on reliable prompt delivery
Smart Retry Mechanism: Intelligent error recovery with exponential backoff, jitter, and failure-specific strategies
Visual Debugging: Comprehensive logging with screenshots for troubleshooting
Reliable Browser Management: Ensures proper browser startup and shutdown

Use Cases

Batch processing of multiple prompts
Automated testing of Claude's responses
Research workflows that require consistent question patterns
Data collection tasks requiring repeated inputs

Installation & Setup
Prerequisites

Python 3.8+
Tesseract OCR installed and in your PATH
Chrome browser
Active Claude account

Installation

Clone the repository:
bashgit clone https://github.com/yourusername/claude-automation.git
cd claude-automation

Install dependencies:
bashpip install -r requirements.txt

Install Tesseract OCR:

Windows: choco install tesseract
macOS: brew install tesseract
Linux: sudo apt install tesseract-ocr


Create required directories:
bashmkdir -p assets/reference_images/{prompt_box,send_button,thinking_indicator}
mkdir -p logs/screenshots


Usage Guide
Quick Start

Run the calibration tool to detect UI elements:
bashpython -m tools.calibrate_ui
This will guide you through capturing reference images for Claude's UI elements, which is a necessary first step.
Edit the prompts in config/user_config.yaml:
yamlprompts:
  - "Summarize the latest AI trends."
  - "Explain how reinforcement learning works."
  - "Write a Python function that reverses a string."

Run the automation:
bashpython python -m src.main


The system will:

Launch Chrome and navigate to Claude
Wait for you to complete login if needed
Automatically send your prompts with configured delays between them
Handle any errors encountered and retry as needed
Close the browser when complete

Command-Line Arguments
The following command-line arguments are available:
bashpython src/main.py [OPTIONS]

Options:
  --config PATH         Path to config file (default: config/user_config.yaml)
  --debug               Enable debug mode with verbose logging
  --calibrate           Run calibration before starting
  --max-retries NUMBER  Override maximum retry attempts
  --retry-delay NUMBER  Override initial delay between retries (seconds)
Interactive Calibration Process
The calibration process is critical for the tool to work correctly:

Run the calibration tool:
bashpython -m tools.calibrate_ui

Follow the on-screen instructions:

You'll be asked to position your mouse over the prompt box, send button, and thinking indicator
For each element, you'll get 3 seconds to switch back to Chrome and position your mouse
The tool will capture reference images automatically


After capturing reference images, the tool will run the calibration process:

It will detect UI elements based on the reference images
It will save the calibrated elements to your configuration file
A calibration results image will be saved to the logs directory


Verify the calibration by checking the generated image in the logs directory:

The image will show rectangles around the detected UI elements
Each element should be correctly positioned on your screen



If calibration fails or elements aren't detected correctly, you can retry with:
bashpython src/main.py --calibrate
Configuration Options
Edit config/user_config.yaml to customize:

Browser profile location
Prompt list
Timeout settings
Retry mechanism settings
Debug settings

Example configuration:
yaml# Browser settings
claude_url: "https://claude.ai"
browser_profile: "C:\\Temp\\ClaudeProfile"

# Automation settings
prompts:
  - "Summarize the latest AI trends."
  - "Explain how reinforcement learning works."
  - "Write a Python function that reverses a string."

# Runtime settings
max_retries: 3
delay_between_prompts: 3
debug: false

# Retry settings
retry_delay: 2          # Initial delay between retries in seconds
max_retry_delay: 30     # Maximum delay between retries in seconds
retry_jitter: 0.5       # Random jitter factor (0.5 means ±50%)
retry_backoff: 1.5      # Exponential backoff multiplier
Retry Mechanism
The system includes a sophisticated retry mechanism that improves reliability when sending prompts:
Failure Types
The system detects and handles different types of failures:

UI Not Found: When elements like prompt box can't be located
Network Error: Connection issues with Claude's website
Browser Error: Chrome crashes or freezes
Unknown Error: Fallback category for other issues

Recovery Strategies
Each failure type triggers a specific recovery approach:

UI issues: Refreshes the page and checks login status
Network issues: Waits longer before refreshing
Browser crashes: Completely restarts the browser
Unknown issues: Starts with simple fixes, escalates to restart if needed

Exponential Backoff with Jitter
The system uses smart timing between retries:

Each retry waits progressively longer (exponential backoff)
Random variation in timing prevents synchronization issues
Maximum delay cap prevents excessive waits

Recommended Settings

For stable connections: Default settings (3 retries, 2-second initial delay)
For unreliable networks: Increase max_retries to 5 and retry_delay to 5
For complex prompts: Increase delay_between_prompts to 5 or higher

Troubleshooting
Common Issues
Configuration File Errors
If you see errors like:
Error loading user configuration: could not determine a constructor for the tag 'tag:yaml.org,2002:python/tuple'
The tool will still work using default settings, but to fix this:

Delete your config/user_config.yaml file
Run the calibration process again:
bashpython -m tools.calibrate_ui

This will generate a new, properly formatted configuration file

Reference Image Not Found
If logs show messages like:
WARNING - Reference image not found: assets/reference_images/prompt_box/prompt_box_1.png
Run the capture reference tool:
bashpython -m tools.capture_reference
Recognition Failures
If the tool fails to recognize UI elements:

Run the visual debugger to see what's being detected:
bashpython -m tools.visual_debugger

Capture new reference images that better match your current UI:
bashpython -m tools.capture_reference

Decrease the confidence threshold in your config file:
yamlui_elements:
  prompt_box:
    confidence: 0.6  # Lower value = more lenient matching


Needle Dimension Errors
If you see errors like:
Error finding prompt_box: needle dimension(s) exceed the haystack image or region dimensions
This means your reference images are larger than the search region:

Run calibration again with smaller reference images:
bashpython -m tools.calibrate_ui

When prompted to position your mouse, try to be more precise

Browser Launch Failures
If the browser doesn't launch correctly:

Verify the Chrome path in automation/browser.py
Ensure you have a valid Chrome profile directory
Try running with debug mode enabled:
bashpython src/main.py --debug


Analyzing Log Files
The system creates detailed logs for each run:

Check the main log file in logs/run_TIMESTAMP/automation.log
Look for screenshots in logs/screenshots/ that show what happened during execution
Check calibration result images in logs/calibration_results_TIMESTAMP.png

Advanced Usage
Running Headless Mode
For server environments, you can adapt the tool to use a virtual display:
pythonfrom pyvirtualdisplay import Display

display = Display(visible=0, size=(1920, 1080))
display.start()

# Run your automation
# ...

display.stop()
Extending with Custom States
To add new functionality, extend the AutomationState enum and add corresponding handler methods:
python# In state_machine.py
class AutomationState(Enum):
    # Existing states...
    CUSTOM_ACTION = auto()

# Add handler method
def _handle_custom_action(self):
    # Implementation
    self.state = AutomationState.NEXT_STATE
Scheduled Execution
Use cron (Linux/Mac) or Task Scheduler (Windows) to run the tool on a schedule:
bash# Example cron entry (runs daily at 9 AM)
0 9 * * * cd /path/to/claude-automation && python src/main.py
Best Practices

Always respect Claude's terms of service and use this tool responsibly.
Avoid excessive automation that might trigger anti-bot measures.
Keep reference images up to date as Claude's UI evolves.
Run in debug mode when setting up to catch issues early.
Use reasonable delays between prompts (3-5 seconds minimum recommended).
Adjust retry settings based on your network stability and use case.
Back up your config files before making changes.
Regularly check log files to ensure the automation is working as expected.
Recalibrate after UI changes to maintain accuracy.

Contributing Guidelines
Contributions to improve the tool are welcome! Please follow these guidelines:

Create a fork of the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request