"""
Display utilities for welcome messages and status display.
Provides various display components for user interface and status reporting.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager

from colorama import Fore, Style, init


# Initialize colorama for colored output
init(autoreset=True)


class DisplayTheme(Enum):
    """Available display themes for consistent styling."""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    COLORFUL = "colorful"
    MINIMAL = "minimal"


@dataclass
class DisplayConfig:
    """Configuration for display settings and styling."""
    theme: DisplayTheme = DisplayTheme.COLORFUL
    show_emoji: bool = True
    show_timestamp: bool = True
    max_line_length: int = 80
    use_unicode: bool = True
    animation_speed: float = 0.1


class ColorManager:
    """
    Manages color schemes and styling for different display themes.
    """
    
    _theme_colors = {
        DisplayTheme.DEFAULT: {
            "primary": Fore.WHITE,
            "secondary": Fore.CYAN,
            "success": Fore.GREEN,
            "warning": Fore.YELLOW,
            "error": Fore.RED,
            "info": Fore.BLUE,
            "border": Fore.WHITE
        },
        DisplayTheme.COLORFUL: {
            "primary": Fore.MAGENTA,
            "secondary": Fore.CYAN,
            "success": Fore.GREEN,
            "warning": Fore.YELLOW,
            "error": Fore.RED,
            "info": Fore.BLUE,
            "border": Fore.BLUE
        },
        DisplayTheme.DARK: {
            "primary": Fore.WHITE,
            "secondary": Fore.LIGHTBLACK_EX,
            "success": Fore.LIGHTGREEN_EX,
            "warning": Fore.LIGHTYELLOW_EX,
            "error": Fore.LIGHTRED_EX,
            "info": Fore.LIGHTBLUE_EX,
            "border": Fore.LIGHTBLACK_EX
        },
        DisplayTheme.LIGHT: {
            "primary": Fore.BLACK,
            "secondary": Fore.BLUE,
            "success": Fore.GREEN,
            "warning": Fore.YELLOW,
            "error": Fore.RED,
            "info": Fore.CYAN,
            "border": Fore.BLACK
        },
        DisplayTheme.MINIMAL: {
            "primary": Fore.WHITE,
            "secondary": Fore.WHITE,
            "success": Fore.WHITE,
            "warning": Fore.WHITE,
            "error": Fore.WHITE,
            "info": Fore.WHITE,
            "border": Fore.WHITE
        }
    }
    
    @classmethod
    def get_color(cls, theme: DisplayTheme, color_type: str) -> str:
        """
        Get color for a specific theme and color type.
        
        Args:
            theme: Display theme
            color_type: Type of color (primary, secondary, etc.)
            
        Returns:
            Color code string
        """
        return cls._theme_colors.get(theme, cls._theme_colors[DisplayTheme.DEFAULT])[color_type]





class StatusDisplay:
    """Status displayer for processing stages and system messages."""
    
    @staticmethod
    def show_processing_status(
        stage: str, 
        progress: Optional[float] = None, 
        details: Optional[str] = None,
        config: DisplayConfig = None
    ):
        """
        Display processing status with optional progress bar.
        
        Args:
            stage: Current processing stage
            progress: Progress percentage (0-100)
            details: Additional details to display
            config: Display configuration
        """
        if config is None:
            config = DisplayConfig()
        
        emoji = "🔄" if progress is None else "⚡"
        stage_text = f"{emoji} {stage}"
        
        if progress is not None:
            progress_bar = StatusDisplay._create_progress_bar(progress, config)
            stage_text += f" {progress_bar} {progress:.1f}%"
        
        warning_color = ColorManager.get_color(config.theme, "warning")
        print(f"{warning_color}{stage_text}{Style.RESET_ALL}")
        
        if details:
            primary_color = ColorManager.get_color(config.theme, "primary")
            print(f"{primary_color}  └─ {details}{Style.RESET_ALL}")
    
    @staticmethod
    def show_success(message: str, details: Optional[str] = None, config: DisplayConfig = None):
        """
        Display success message.
        
        Args:
            message: Success message
            details: Additional details
            config: Display configuration
        """
        if config is None:
            config = DisplayConfig()
        
        success_color = ColorManager.get_color(config.theme, "success")
        print(f"{success_color}✅ {message}{Style.RESET_ALL}")
        
        if details:
            primary_color = ColorManager.get_color(config.theme, "primary")
            print(f"{primary_color}  └─ {details}{Style.RESET_ALL}")
    
    @staticmethod
    def show_error(message: str, details: Optional[str] = None, config: DisplayConfig = None):
        """
        Display error message.
        
        Args:
            message: Error message
            details: Additional details
            config: Display configuration
        """
        if config is None:
            config = DisplayConfig()
        
        error_color = ColorManager.get_color(config.theme, "error")
        print(f"{error_color}❌ {message}{Style.RESET_ALL}")
        
        if details:
            primary_color = ColorManager.get_color(config.theme, "primary")
            print(f"{primary_color}  └─ {details}{Style.RESET_ALL}")
    
    @staticmethod
    def show_warning(message: str, details: Optional[str] = None, config: DisplayConfig = None):
        """
        Display warning message.
        
        Args:
            message: Warning message
            details: Additional details
            config: Display configuration
        """
        if config is None:
            config = DisplayConfig()
        
        warning_color = ColorManager.get_color(config.theme, "warning")
        print(f"{warning_color}⚠️  {message}{Style.RESET_ALL}")
        
        if details:
            primary_color = ColorManager.get_color(config.theme, "primary")
            print(f"{primary_color}  └─ {details}{Style.RESET_ALL}")
    
    @staticmethod
    def show_info(message: str, details: Optional[str] = None, config: DisplayConfig = None):
        """
        Display information message.
        
        Args:
            message: Information message
            details: Additional details
            config: Display configuration
        """
        if config is None:
            config = DisplayConfig()
        
        info_color = ColorManager.get_color(config.theme, "info")
        print(f"{info_color}ℹ️  {message}{Style.RESET_ALL}")
        
        if details:
            primary_color = ColorManager.get_color(config.theme, "primary")
            print(f"{primary_color}  └─ {details}{Style.RESET_ALL}")
    
    @staticmethod
    def _create_progress_bar(progress: float, config: DisplayConfig, width: int = 20) -> str:
        """
        Create a visual progress bar.
        
        Args:
            progress: Progress percentage (0-100)
            config: Display configuration
            width: Width of the progress bar
            
        Returns:
            Progress bar string
        """
        filled = int(width * progress / 100)
        if config.use_unicode:
            bar = "█" * filled + "░" * (width - filled)
        else:
            bar = "=" * filled + "-" * (width - filled)
        return f"[{bar}]"


class MetricsDisplay:
    """Metrics displayer for performance and component information."""
    
    @staticmethod
    def show_performance_metrics(metrics: Dict[str, Any], config: DisplayConfig = None):
        """
        Display performance metrics in a formatted table.
        
        Args:
            metrics: Dictionary of metrics to display
            config: Display configuration
        """
        if config is None:
            config = DisplayConfig()
        
        info_color = ColorManager.get_color(config.theme, "info")
        primary_color = ColorManager.get_color(config.theme, "primary")
        success_color = ColorManager.get_color(config.theme, "success")
        warning_color = ColorManager.get_color(config.theme, "warning")
        
        print(f"{info_color}📊 Performance Metrics{Style.RESET_ALL}")
        print(f"{info_color}{'='*50}{Style.RESET_ALL}")
        
        for key, value in metrics.items():
            if hasattr(value, 'total_time'):  # TimingStats object
                print(f"{primary_color}{key:20}: {success_color}Total={value.total_time:.3f}s, Count={value.count}, Avg={value.average_time:.3f}s{Style.RESET_ALL}")
            elif isinstance(value, (int, float)):
                print(f"{primary_color}{key:20}: {success_color}{value}{Style.RESET_ALL}")
            else:
                print(f"{primary_color}{key:20}: {warning_color}{value}{Style.RESET_ALL}")
    
    @staticmethod
    def show_component_info(components: Dict[str, Any], config: DisplayConfig = None):
        """
        Display component information with status indicators.
        
        Args:
            components: Dictionary of component information
            config: Display configuration
        """
        if config is None:
            config = DisplayConfig()
        
        info_color = ColorManager.get_color(config.theme, "info")
        primary_color = ColorManager.get_color(config.theme, "primary")
        success_color = ColorManager.get_color(config.theme, "success")
        warning_color = ColorManager.get_color(config.theme, "warning")
        
        print(f"{info_color}🔧 Component Information{Style.RESET_ALL}")
        print(f"{info_color}{'='*50}{Style.RESET_ALL}")
        
        for name, info in components.items():
            status = "✅" if info.get("exists", False) else "❌"
            status_color = success_color if info.get("exists", False) else warning_color
            print(f"{status} {primary_color}{name:20}: {status_color}{info.get('type', 'unknown')}{Style.RESET_ALL}")


class TableDisplay:
    """Table displayer for structured data presentation."""
    
    @staticmethod
    def show_table(
        headers: List[str], 
        rows: List[List[Any]], 
        title: Optional[str] = None,
        config: DisplayConfig = None
    ):
        """
        Display data in a formatted table.
        
        Args:
            headers: Table headers
            rows: Table data rows
            title: Optional table title
            config: Display configuration
        """
        if config is None:
            config = DisplayConfig()
        
        info_color = ColorManager.get_color(config.theme, "info")
        primary_color = ColorManager.get_color(config.theme, "primary")
        border_color = ColorManager.get_color(config.theme, "border")
        
        if title:
            print(f"{info_color}{title}{Style.RESET_ALL}")
            print(f"{info_color}{'='*50}{Style.RESET_ALL}")
        
        # Calculate column widths
        col_widths = []
        for i, header in enumerate(headers):
            max_width = len(header)
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(max_width + 2)
        
        # Unicode table characters
        if config.use_unicode:
            top_left, top_right, bottom_left, bottom_right = "┌", "┐", "└", "┘"
            horizontal, vertical, cross = "─", "│", "┼"
            top_cross, bottom_cross = "┬", "┴"
        else:
            top_left, top_right, bottom_left, bottom_right = "+", "+", "+", "+"
            horizontal, vertical, cross = "-", "|", "+"
            top_cross, bottom_cross = "+", "+"
        
        # Print header
        header_line = vertical
        separator_line = top_left
        for i, header in enumerate(headers):
            header_line += f" {header.ljust(col_widths[i])} {vertical}"
            separator_line += horizontal * (col_widths[i] + 2) + (top_cross if i < len(headers) - 1 else top_right)
        
        print(f"{border_color}{header_line}{Style.RESET_ALL}")
        print(f"{border_color}{separator_line}{Style.RESET_ALL}")
        
        # Print data rows
        for row in rows:
            row_line = vertical
            for i, cell in enumerate(row):
                cell_str = str(cell) if cell is not None else ""
                row_line += f" {cell_str.ljust(col_widths[i])} {vertical}"
            print(f"{primary_color}{row_line}{Style.RESET_ALL}")


class AnimationDisplay:
    """Animation displayer for loading and processing indicators."""
    
    def __init__(self, config: DisplayConfig = None):
        """
        Initialize animation displayer with spinner characters.
        
        Args:
            config: Display configuration
        """
        if config is None:
            config = DisplayConfig()
        
        self.config = config
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_index = 0
    
    def show_spinner(self, message: str):
        """
        Display spinning animation with message.
        
        Args:
            message: Message to display with spinner
        """
        spinner = self.spinner_chars[self.spinner_index]
        warning_color = ColorManager.get_color(self.config.theme, "warning")
        print(f"\r{warning_color}{spinner} {message}{Style.RESET_ALL}", end="", flush=True)
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)
    
    def clear_spinner(self):
        """Clear the spinning animation from the display."""
        print("\r" + " " * 80 + "\r", end="", flush=True)
    
    @contextmanager
    def spinner_context(self, message: str):
        """
        Context manager for spinner animation.
        
        Args:
            message: Message to display with spinner
            
        Yields:
            Self for chaining
        """
        try:
            self.show_spinner(message)
            yield self
        finally:
            self.clear_spinner()


class ProgressDisplay:
    """Progress displayer for long-running operations."""
    
    @staticmethod
    def show_progress(
        current: int, 
        total: int, 
        description: str = "Processing",
        config: DisplayConfig = None
    ):
        """
        Display progress bar with current/total counts.
        
        Args:
            current: Current progress count
            total: Total count
            description: Description of the operation
            config: Display configuration
        """
        if config is None:
            config = DisplayConfig()
        
        progress = (current / total) * 100 if total > 0 else 0
        progress_bar = StatusDisplay._create_progress_bar(progress, config)
        warning_color = ColorManager.get_color(config.theme, "warning")
        
        print(f"\r{warning_color}{description}: {progress_bar} {current}/{total} ({progress:.1f}%){Style.RESET_ALL}", 
              end="", flush=True)
        
        if current >= total:
            print()  # New line when complete


class DisplayManager:
    """
    Central display manager for coordinating all display operations.
    """
    
    def __init__(self, config: DisplayConfig = None):
        """
        Initialize the display manager.
        
        Args:
            config: Display configuration
        """
        self.config = config or DisplayConfig()
        self.animation = AnimationDisplay(self.config)
    

    
    def show_status(self, stage: str, progress: Optional[float] = None, details: Optional[str] = None):
        """Show processing status."""
        StatusDisplay.show_processing_status(stage, progress, details, self.config)
    
    def show_success(self, message: str, details: Optional[str] = None):
        """Show success message."""
        StatusDisplay.show_success(message, details, self.config)
    
    def show_error(self, message: str, details: Optional[str] = None):
        """Show error message."""
        StatusDisplay.show_error(message, details, self.config)
    
    def show_warning(self, message: str, details: Optional[str] = None):
        """Show warning message."""
        StatusDisplay.show_warning(message, details, self.config)
    
    def show_info(self, message: str, details: Optional[str] = None):
        """Show info message."""
        StatusDisplay.show_info(message, details, self.config)
    
    def show_metrics(self, metrics: Dict[str, Any]):
        """Show performance metrics."""
        MetricsDisplay.show_performance_metrics(metrics, self.config)
    
    def show_components(self, components: Dict[str, Any]):
        """Show component information."""
        MetricsDisplay.show_component_info(components, self.config)
    
    def show_table(self, headers: List[str], rows: List[List[Any]], title: Optional[str] = None):
        """Show data table."""
        TableDisplay.show_table(headers, rows, title, self.config)
    
    def show_progress(self, current: int, total: int, description: str = "Processing"):
        """Show progress bar."""
        ProgressDisplay.show_progress(current, total, description, self.config)
    
    def get_spinner(self):
        """Get spinner animation instance."""
        return self.animation 