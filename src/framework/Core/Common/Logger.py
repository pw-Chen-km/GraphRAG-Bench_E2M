"""
Logger Module - Logging configuration and utilities
Provides centralized logging setup and management for the GraphRAG system
"""

import sys
import os
from datetime import datetime
from typing import Optional, Callable, Any

from loguru import logger as _logger
# Import default_config as type hint only to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Option.merged_config import default_config


class LogManager:
    """Manages logging configuration and setup"""
    
    def __init__(self):
        self._print_level = "INFO"
        self._llm_stream_log_func = self._default_llm_stream_log
        
    def configure_logging(
        self, 
        print_level: str = "INFO", 
        logfile_level: str = "DEBUG", 
        name: Optional[str] = None
    ) -> None:
        """
        Configure logging with specified levels and output locations.
        
        Args:
            print_level: Log level for console output
            logfile_level: Log level for file output
            name: Optional name for log directory
        """
        self._print_level = print_level
        
        # Generate log file path
        current_date = datetime.now().strftime("%Y%m%d%H%M%S")
        
        if name:
            log_dir = os.path.join(name, "Logs")
            try:
                os.makedirs(log_dir, exist_ok=True)
                log_name = os.path.join(log_dir, f"{current_date}.log")
            except (PermissionError, OSError):
                # 如果无法创建指定目录，使用当前目录
                log_name = f"Logs/{current_date}.log"
                os.makedirs("Logs", exist_ok=True)
        else:
            log_name = f"Logs/{current_date}.log"
            os.makedirs("Logs", exist_ok=True)

        # Remove existing handlers and add new ones
        _logger.remove()
        _logger.add(sys.stderr, level=print_level)
        _logger.add(log_name, level=logfile_level)
    
    def log_llm_stream(self, message: str) -> None:
        """
        Log LLM stream messages.
        
        Args:
            message: Message to log
        """
        self._llm_stream_log_func(message)
    
    def set_llm_stream_log_function(self, func: Callable[[str], None]) -> None:
        """
        Set custom function for LLM stream logging.
        
        Args:
            func: Function to handle LLM stream logging
        """
        self._llm_stream_log_func = func
    
    def _default_llm_stream_log(self, message: str) -> None:
        """
        Default LLM stream logging function.
        
        Args:
            message: Message to log
        """
        if self._print_level == "INFO":
            print(message, end="")
    
    def get_log_level(self) -> str:
        """
        Get current print log level.
        
        Returns:
            Current log level
        """
        return self._print_level


# Global log manager instance
_log_manager = LogManager()


def define_log_level(
    print_level: str = "INFO", 
    logfile_level: str = "DEBUG", 
    name: Optional[str] = None
) -> Any:
    """
    Configure logging levels and return logger instance.
    
    Args:
        print_level: Log level for console output
        logfile_level: Log level for file output
        name: Optional name for log directory
        
    Returns:
        Configured logger instance
    """
    _log_manager.configure_logging(print_level, logfile_level, name)
    return _logger


def log_llm_stream(message: str) -> None:
    """
    Log LLM stream message using global log manager.
    
    Args:
        message: Message to log
    """
    _log_manager.log_llm_stream(message)


def set_llm_stream_logfunc(func: Callable[[str], None]) -> None:
    """
    Set custom LLM stream logging function.
    
    Args:
        func: Function to handle LLM stream logging
    """
    _log_manager.set_llm_stream_log_function(func)


def _llm_stream_log(message: str) -> None:
    """
    Legacy LLM stream logging function for backward compatibility.
    
    Args:
        message: Message to log
    """
    _log_manager.log_llm_stream(message)


def _get_default_log_name():
    from Option.merged_config import default_config
    # 检查路径是否存在且有权限，如果不存在则使用当前目录
    try:
        log_path = os.path.join(default_config.working_dir, default_config.exp_name)
        # 尝试创建目录来测试权限
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        return log_path
    except (PermissionError, OSError):
        # 如果无法创建目录，使用当前目录
        return os.path.join(".", default_config.exp_name)

# Initialize default logger
logger = define_log_level(
    name=_get_default_log_name()
)