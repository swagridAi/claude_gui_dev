#!/usr/bin/env python3
"""
File operations utilities for the Claude GUI Automation calibration tools.
Provides standardized interfaces for file manipulation, configuration management,
log processing, and report generation.
"""

import os
import glob
import time
import logging
import yaml
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple


class FileOperationError(Exception):
    """Base exception for file operation errors."""
    pass


class ConfigurationError(FileOperationError):
    """Exception raised for configuration-related errors."""
    pass


class ReportGenerationError(FileOperationError):
    """Exception raised for errors during report generation."""
    pass


class DirectoryManager:
    """Handles directory creation and validation."""
    
    @staticmethod
    def setup_directories(config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Create required directory structure for the application.
        
        Args:
            config: Optional configuration with custom paths
            
        Returns:
            Dict mapping directory names to their created paths
            
        Raises:
            PermissionError: If directories cannot be created due to permissions
        """
        directories = {
            "assets": "assets/reference_images",
            "config": "config",
            "logs": "logs/screenshots",
            "debug": "logs/recognition_debug",
            "click_debug": "logs/click_debug",
            "reports": "logs/reports"
        }
        
        # Override with config values if provided
        if config and "directories" in config:
            directories.update(config["directories"])
        
        created_dirs = {}
        
        for name, path in directories.items():
            try:
                # Create nested directories with parents
                os.makedirs(path, exist_ok=True)
                created_dirs[name] = path
                logging.debug(f"Created directory: {path}")
            except PermissionError as e:
                logging.error(f"Permission error creating directory {path}: {e}")
                raise PermissionError(f"Cannot create directory {path}. Check permissions.") from e
            except Exception as e:
                logging.error(f"Error creating directory {path}: {e}")
                raise FileOperationError(f"Failed to create directory {path}: {e}") from e
        
        # Create element subdirectories
        standard_elements = ["prompt_box", "send_button", "thinking_indicator", "response_area"]
        for element in standard_elements:
            element_dir = os.path.join(directories["assets"], element)
            try:
                os.makedirs(element_dir, exist_ok=True)
                created_dirs[f"element_{element}"] = element_dir
            except Exception as e:
                logging.warning(f"Error creating element directory {element_dir}: {e}")
        
        return created_dirs


class FileUtils:
    """Common file utility operations."""
    
    @staticmethod
    def ensure_directory_exists(path: str) -> bool:
        """
        Ensure directory exists, creating it if necessary.
        
        Args:
            path: Directory path to ensure exists
            
        Returns:
            True if directory exists or was created, False on failure
        """
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            logging.error(f"Error ensuring directory exists {path}: {e}")
            return False
    
    @staticmethod
    def get_timestamped_filename(base_name: str, extension: str) -> str:
        """
        Generate filename with timestamp to avoid collisions.
        
        Args:
            base_name: Base filename without extension
            extension: File extension (without dot)
            
        Returns:
            Timestamped filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.{extension}"
    
    @staticmethod
    def find_files_by_pattern(directory: str, pattern: str, recursive: bool = True) -> List[str]:
        """
        Find files matching pattern in directory.
        
        Args:
            directory: Directory to search
            pattern: Glob pattern to match
            recursive: Whether to search subdirectories
            
        Returns:
            List of matching file paths
        """
        if recursive:
            return glob.glob(os.path.join(directory, "**", pattern), recursive=True)
        return glob.glob(os.path.join(directory, pattern))

    @staticmethod
    def backup_file(file_path: str, max_backups: int = 5) -> Optional[str]:
        """
        Create a backup of a file before modifying it.
        
        Args:
            file_path: Path to file to backup
            max_backups: Maximum number of backups to keep
            
        Returns:
            Path to backup file or None if backup failed
        """
        if not os.path.exists(file_path):
            return None
            
        try:
            backup_dir = os.path.join(os.path.dirname(file_path), "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create backup filename with timestamp
            filename = os.path.basename(file_path)
            base_name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"{base_name}_{timestamp}{ext}")
            
            # Copy file to backup
            shutil.copy2(file_path, backup_path)
            
            # Clean up old backups if needed
            backups = sorted(
                [f for f in os.listdir(backup_dir) if f.startswith(base_name)],
                reverse=True
            )
            
            for old_backup in backups[max_backups:]:
                os.remove(os.path.join(backup_dir, old_backup))
                
            return backup_path
            
        except Exception as e:
            logging.error(f"Error backing up file {file_path}: {e}")
            return None


class ConfigFileManager:
    """Manages configuration file operations."""
    
    @staticmethod
    def load_configuration(config_path: str, default_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file with fallback to defaults.
        
        Args:
            config_path: Path to configuration file
            default_path: Optional path to default configuration
            
        Returns:
            Dict containing loaded configuration
            
        Raises:
            ConfigurationError: If configuration cannot be loaded
        """
        config = {}
        
        # Try to load default configuration if provided
        if default_path and os.path.exists(default_path):
            try:
                with open(default_path, 'r') as f:
                    default_config = yaml.safe_load(f) or {}
                    config.update(default_config)
                logging.debug(f"Loaded default configuration from {default_path}")
            except Exception as e:
                logging.warning(f"Error loading default configuration: {e}")
        
        # Load user configuration if it exists
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f) or {}
                    config.update(user_config)
                logging.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                error_msg = f"Error loading configuration from {config_path}: {e}"
                logging.error(error_msg)
                raise ConfigurationError(error_msg) from e
        else:
            logging.info(f"Configuration file not found at {config_path}, using defaults")
            return config
    
    @staticmethod
    def save_configuration(config_data: Dict[str, Any], config_path: str, create_backup: bool = True) -> bool:
        """
        Save configuration to YAML file.
        
        Args:
            config_data: Configuration data to save
            config_path: Path to save configuration
            create_backup: Whether to backup existing file before overwriting
            
        Returns:
            True if configuration was saved successfully
            
        Raises:
            ConfigurationError: If configuration cannot be saved
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Create backup if requested
            if create_backup and os.path.exists(config_path):
                FileUtils.backup_file(config_path)
            
            # Save configuration
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            logging.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            error_msg = f"Error saving configuration to {config_path}: {e}"
            logging.error(error_msg)
            raise ConfigurationError(error_msg) from e

    @staticmethod
    def convert_to_ui_element_format(elements_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert element data to format expected by the main application.
        
        Args:
            elements_data: Raw element data from configuration
            
        Returns:
            Formatted element data compatible with application
        """
        formatted_elements = {}
        
        for name, data in elements_data.items():
            # Ensure required fields exist
            if "region" not in data:
                data["region"] = None
                
            if "reference_paths" not in data:
                data["reference_paths"] = []
                
            if "confidence" not in data:
                data["confidence"] = 0.7
            
            # Format click coordinates consistently
            if "click_coordinates" in data and isinstance(data["click_coordinates"], list):
                data["click_coordinates"] = tuple(data["click_coordinates"])
                
            # Ensure boolean fields are proper booleans
            if "use_coordinates_first" in data:
                data["use_coordinates_first"] = bool(data["use_coordinates_first"])
                
            formatted_elements[name] = data
            
        return formatted_elements


class LogFileManager:
    """Handles operations related to logs and screenshot files."""
    
    @staticmethod
    def scan_log_screenshots(base_dir: str = "logs", days_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Scan log directories for screenshots organized by run and element.
        
        Args:
            base_dir: Base directory containing log folders
            days_limit: Optional limit to only include logs from past N days
            
        Returns:
            Dict with organized structure of screenshots by run and element
        """
        log_screenshots = {}
        
        # Find all run directories
        run_dirs = sorted(glob.glob(os.path.join(base_dir, "run_*")), reverse=True)
        
        # Filter by date if days_limit is specified
        if days_limit is not None:
            cutoff_time = time.time() - (days_limit * 24 * 60 * 60)
            run_dirs = [d for d in run_dirs if os.path.getmtime(d) >= cutoff_time]
        
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
                LogFileManager._categorize_screenshot(filename, screenshot, log_screenshots[run_name])
        
        return log_screenshots
    
    @staticmethod
    def _categorize_screenshot(filename: str, path: str, run_data: Dict[str, Any]) -> None:
        """
        Categorize a screenshot based on filename patterns.
        
        Args:
            filename: Screenshot filename
            path: Full path to screenshot
            run_data: Run data dictionary to update
        """
        # Look for recognition-related screenshots
        if "SEARCH_" in filename or "FOUND_" in filename or "NOT_FOUND_" in filename:
            match = re.search(r"(?:SEARCH|FOUND|NOT_FOUND)_(\w+)", filename)
            if match:
                element_name = match.group(1)
                if element_name not in run_data["elements"]:
                    run_data["elements"][element_name] = []
                run_data["elements"][element_name].append(path)
                return
        
        # Look for click-related screenshots
        if "AFTER_CLICKING" in filename or "click" in filename.lower():
            match = re.search(r"(?:click|CLICK)_(\w+)", filename)
            if match:
                element_name = match.group(1)
                if element_name not in run_data["elements"]:
                    run_data["elements"][element_name] = []
                run_data["elements"][element_name].append(path)
                return
        
        # Look for element name in filename as fallback
        for element_name in ["prompt_box", "send_button", "thinking_indicator", "response_area", 
                           "limit_reached", "extended_thought"]:
            if element_name.lower() in filename.lower():
                if element_name not in run_data["elements"]:
                    run_data["elements"][element_name] = []
                run_data["elements"][element_name].append(path)
                return

    @staticmethod
    def get_recent_logs(days: int = 7) -> List[str]:
        """
        Get list of recent log files.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of log file paths
        """
        log_files = []
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        for log_dir in glob.glob("logs/run_*"):
            if os.path.getmtime(log_dir) >= cutoff_time:
                log_file = os.path.join(log_dir, "automation.log")
                if os.path.exists(log_file):
                    log_files.append(log_file)
        
        return sorted(log_files, key=os.path.getmtime, reverse=True)


class ReportGenerator:
    """Generates reports from calibration data."""
    
    # HTML template for the calibration report
    _REPORT_TEMPLATE = """
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
            .image-container { display: flex; flex-wrap: wrap; }
            .image-item { margin: 10px; text-align: center; }
            .image-item img { max-width: 200px; max-height: 200px; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>Claude UI Calibration Report</h1>
        <p>Generated on: {timestamp}</p>
        
        {content}
    </body>
    </html>
    """
    
    @staticmethod
    def generate_calibration_report(
        elements: Dict[str, Any], 
        test_results: Optional[Dict[str, Any]] = None, 
        output_path: Optional[str] = None,
        include_images: bool = True,
        copy_images: bool = True
    ) -> str:
        """
        Generate a detailed HTML report of calibration results.
        
        Args:
            elements: Dictionary of UI element definitions
            test_results: Test outcome data for verification
            output_path: Path to save report (default: logs/reports/[timestamp].html)
            include_images: Whether to include reference images
            copy_images: Whether to copy images to report directory
            
        Returns:
            Path to the generated report file
        """
        try:
            # Set default output path if not provided
            if not output_path:
                report_dir = "logs/reports"
                FileUtils.ensure_directory_exists(report_dir)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(report_dir, f"calibration_report_{timestamp}.html")
            
            # Generate report content
            content = []
            
            # Element summary table
            content.append("<h2>Element Summary</h2>")
            content.append("<table>")
            content.append("<tr><th>Element</th><th>Region</th><th>References</th><th>Confidence</th></tr>")
            
            for name, config in elements.items():
                region = config.get("region", "Not defined")
                if isinstance(region, (tuple, list)):
                    region_str = f"({region[0]}, {region[1]}, {region[2]}, {region[3]})"
                else:
                    region_str = str(region)
                    
                ref_count = len(config.get("reference_paths", []))
                confidence = config.get("confidence", 0.7)
                
                content.append(f"""
                    <tr>
                        <td>{name}</td>
                        <td>{region_str}</td>
                        <td>{ref_count}</td>
                        <td>{confidence:.2f}</td>
                    </tr>
                """)
            
            content.append("</table>")
            
            # Test results if provided
            if test_results:
                content.append("<h2>Test Results</h2>")
                content.append("<table>")
                content.append("<tr><th>Element</th><th>Status</th><th>Location</th></tr>")
                
                for name, result in test_results.items():
                    found = result.get("found", False)
                    status_class = "success" if found else "failure"
                    status_text = "Found" if found else "Not found"
                    
                    row = f"""
                        <tr>
                            <td>{name}</td>
                            <td class="{status_class}">{status_text}</td>
                    """
                    
                    if found and "location" in result:
                        loc = result["location"]
                        row += f"<td>({loc[0]}, {loc[1]}, {loc[2]}, {loc[3]})</td>"
                    else:
                        row += "<td>-</td>"
                        
                    row += "</tr>"
                    content.append(row)
                
                content.append("</table>")
            
            # Reference images
            if include_images:
                content.append("<h2>Reference Images</h2>")
                
                for name, config in elements.items():
                    content.append(f"<h3>Element: {name}</h3>")
                    
                    reference_paths = config.get("reference_paths", [])
                    if not reference_paths:
                        content.append("<p>No reference images defined.</p>")
                        continue
                    
                    # Filter out variants to show only original images
                    original_references = [
                        path for path in reference_paths 
                        if not any(x in path for x in ['_gray', '_contrast', '_thresh', '_scale'])
                    ]
                    
                    if not original_references:
                        content.append("<p>No original reference images found (only variants exist).</p>")
                        continue
                    
                    content.append('<div class="image-container">')
                    
                    for path in original_references:
                        try:
                            basename = os.path.basename(path)
                            image_path = basename
                            
                            # Copy image if requested
                            if copy_images and os.path.exists(path):
                                report_dir = os.path.dirname(output_path)
                                dest_path = os.path.join(report_dir, basename)
                                shutil.copy2(path, dest_path)
                            
                            content.append(f"""
                                <div class="image-item">
                                    <img src="{image_path}" alt="{basename}">
                                    <div>{basename}</div>
                                </div>
                            """)
                        except Exception as e:
                            content.append(f"<p>Error including image {path}: {e}</p>")
                    
                    content.append("</div>")
            
            # Configuration dump
            content.append("<h2>Configuration</h2>")
            config_yaml = yaml.dump(elements, default_flow_style=False)
            content.append(f"<pre>{config_yaml}</pre>")
            
            # Assemble report
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            html_content = ReportGenerator._REPORT_TEMPLATE.format(
                timestamp=timestamp,
                content="\n".join(content)
            )
            
            # Write to file
            with open(output_path, "w") as f:
                f.write(html_content)
            
            logging.info(f"Calibration report generated: {output_path}")
            return output_path
            
        except Exception as e:
            error_msg = f"Error generating calibration report: {e}"
            logging.error(error_msg)
            raise ReportGenerationError(error_msg) from e