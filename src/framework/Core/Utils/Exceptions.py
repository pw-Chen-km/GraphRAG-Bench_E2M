"""
Exception handling utilities for GraphRAG system.
Provides custom exceptions and exception handling decorators for robust error management.
"""

import asyncio
import functools
import traceback
from typing import Any, Callable, Tuple, Type, TypeVar, Union, Optional
from enum import Enum

from Core.Common.Logger import logger

ReturnType = TypeVar("ReturnType")


class ErrorSeverity(Enum):
    """Error severity levels for exception handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GraphRAGException(Exception):
    """
    Base exception class for GraphRAG system.
    
    Provides a common interface for all GraphRAG-related exceptions
    with additional context and severity information.
    """
    
    def __init__(
        self, 
        message: str, 
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[dict[str, Any]] = None
    ):
        """
        Initialize the GraphRAG exception.
        
        Args:
            message: Error message
            severity: Error severity level
            context: Additional context information
        """
        self.message = message
        self.severity = severity
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        return f"[{self.severity.value.upper()}] {self.message}"
    
    def get_context(self) -> dict[str, Any]:
        """Get the exception context."""
        return self.context.copy()


class InvalidStorageError(GraphRAGException):
    """Exception raised for errors in storage operations."""
    
    def __init__(self, message: str = "Invalid storage operation", context: Optional[dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.HIGH, context)


class ComponentError(GraphRAGException):
    """Exception raised for component-related errors."""
    
    def __init__(self, message: str, component_name: str, context: Optional[dict[str, Any]] = None):
        context = context or {}
        context["component_name"] = component_name
        super().__init__(message, ErrorSeverity.MEDIUM, context)


class ConfigurationError(GraphRAGException):
    """Exception raised for configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, context: Optional[dict[str, Any]] = None):
        context = context or {}
        if config_key:
            context["config_key"] = config_key
        super().__init__(message, ErrorSeverity.HIGH, context)


class ValidationError(GraphRAGException):
    """Exception raised for validation errors."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, context: Optional[dict[str, Any]] = None):
        context = context or {}
        if field_name:
            context["field_name"] = field_name
        super().__init__(message, ErrorSeverity.MEDIUM, context)


class NetworkError(GraphRAGException):
    """Exception raised for network-related errors."""
    
    def __init__(self, message: str, url: Optional[str] = None, status_code: Optional[int] = None, context: Optional[dict[str, Any]] = None):
        context = context or {}
        if url:
            context["url"] = url
        if status_code:
            context["status_code"] = status_code
        super().__init__(message, ErrorSeverity.MEDIUM, context)


class TimeoutError(GraphRAGException):
    """Exception raised for timeout errors."""
    
    def __init__(self, message: str, timeout_duration: Optional[float] = None, context: Optional[dict[str, Any]] = None):
        context = context or {}
        if timeout_duration:
            context["timeout_duration"] = timeout_duration
        super().__init__(message, ErrorSeverity.MEDIUM, context)


class ResourceError(GraphRAGException):
    """Exception raised for resource-related errors."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, context: Optional[dict[str, Any]] = None):
        context = context or {}
        if resource_type:
            context["resource_type"] = resource_type
        super().__init__(message, ErrorSeverity.HIGH, context)


class ExceptionHandler:
    """
    Centralized exception handler for managing and logging exceptions.
    
    Provides methods for handling different types of exceptions with
    appropriate logging and recovery strategies.
    """
    
    @staticmethod
    def log_exception(
        exception: Exception, 
        function_name: str, 
        args: Tuple[Any, ...], 
        kwargs: dict[str, Any],
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> None:
        """
        Log exception details with context information.
        
        Args:
            exception: The exception that occurred
            function_name: Name of the function where exception occurred
            args: Function arguments
            kwargs: Function keyword arguments
            severity: Error severity level
        """
        error_context = {
            "function_name": function_name,
            "args": str(args),
            "kwargs": str(kwargs),
            "exception_type": type(exception).__name__,
            "severity": severity.value
        }
        
        if severity == ErrorSeverity.CRITICAL:
            logger.opt(depth=1).critical(
                f"Critical error in {function_name}: {exception}\n"
                f"Context: {error_context}\n"
                f"Stack trace: {traceback.format_exc()}"
            )
        elif severity == ErrorSeverity.HIGH:
            logger.opt(depth=1).error(
                f"High severity error in {function_name}: {exception}\n"
                f"Context: {error_context}\n"
                f"Stack trace: {traceback.format_exc()}"
            )
        elif severity == ErrorSeverity.MEDIUM:
            logger.opt(depth=1).warning(
                f"Medium severity error in {function_name}: {exception}\n"
                f"Context: {error_context}"
            )
        else:  # LOW
            logger.opt(depth=1).info(
                f"Low severity error in {function_name}: {exception}\n"
                f"Context: {error_context}"
            )
    
    @staticmethod
    def get_exception_severity(exception: Exception) -> ErrorSeverity:
        """
        Determine the severity of an exception based on its type.
        
        Args:
            exception: The exception to analyze
            
        Returns:
            Error severity level
        """
        if isinstance(exception, (InvalidStorageError, ConfigurationError, ResourceError)):
            return ErrorSeverity.HIGH
        elif isinstance(exception, (ComponentError, ValidationError, NetworkError, TimeoutError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(exception, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
    
    @staticmethod
    def should_retry(exception: Exception, retry_count: int, max_retries: int) -> bool:
        """
        Determine if an operation should be retried based on the exception.
        
        Args:
            exception: The exception that occurred
            retry_count: Current retry attempt number
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if operation should be retried, False otherwise
        """
        if retry_count >= max_retries:
            return False
        
        # Retry on network and timeout errors
        if isinstance(exception, (NetworkError, TimeoutError)):
            return True
        
        # Don't retry on configuration or validation errors
        if isinstance(exception, (ConfigurationError, ValidationError)):
            return False
        
        # Don't retry on critical resource errors
        if isinstance(exception, ResourceError):
            return False
        
        return True


def handle_exception(
    _func: Callable[..., ReturnType] = None,
    *,
    exception_type: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    exception_msg: str = "",
    default_return: Any = None,
    max_retries: int = 0,
    retry_delay: float = 1.0,
    log_errors: bool = True
) -> Callable[..., ReturnType]:
    """
    Decorator for handling exceptions with optional retry logic.
    
    Args:
        _func: Function to decorate (used for decorator without parameters)
        exception_type: Type(s) of exceptions to handle
        exception_msg: Additional error message
        default_return: Default value to return on exception
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retry attempts in seconds
        log_errors: Whether to log errors
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> ReturnType:
            retry_count = 0
            
            while True:
                try:
                    return await func(*args, **kwargs)
                except exception_type as e:
                    retry_count += 1
                    
                    if log_errors:
                        severity = ExceptionHandler.get_exception_severity(e)
                        ExceptionHandler.log_exception(e, func.__name__, args, kwargs, severity)
                    
                    if not ExceptionHandler.should_retry(e, retry_count, max_retries):
                        break
                    
                    if retry_count <= max_retries:
                        await asyncio.sleep(retry_delay * retry_count)
                        continue
                    
                    break
            
            return default_return

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> ReturnType:
            retry_count = 0
            
            while True:
                try:
                    return func(*args, **kwargs)
                except exception_type as e:
                    retry_count += 1
                    
                    if log_errors:
                        severity = ExceptionHandler.get_exception_severity(e)
                        ExceptionHandler.log_exception(e, func.__name__, args, kwargs, severity)
                    
                    if not ExceptionHandler.should_retry(e, retry_count, max_retries):
                        break
                    
                    if retry_count <= max_retries:
                        import time
                        time.sleep(retry_delay * retry_count)
                        continue
                    
                    break
            
            return default_return

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def safe_execute(
    func: Callable[..., ReturnType],
    *args: Any,
    default_return: Any = None,
    exception_types: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    **kwargs: Any
) -> ReturnType:
    """
    Safely execute a function with exception handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Default value to return on exception
        exception_types: Types of exceptions to handle
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return on exception
    """
    try:
        return func(*args, **kwargs)
    except exception_types as e:
        severity = ExceptionHandler.get_exception_severity(e)
        ExceptionHandler.log_exception(e, func.__name__, args, kwargs, severity)
        return default_return


async def safe_execute_async(
    func: Callable[..., ReturnType],
    *args: Any,
    default_return: Any = None,
    exception_types: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    **kwargs: Any
) -> ReturnType:
    """
    Safely execute an async function with exception handling.
    
    Args:
        func: Async function to execute
        *args: Function arguments
        default_return: Default value to return on exception
        exception_types: Types of exceptions to handle
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return on exception
    """
    try:
        return await func(*args, **kwargs)
    except exception_types as e:
        severity = ExceptionHandler.get_exception_severity(e)
        ExceptionHandler.log_exception(e, func.__name__, args, kwargs, severity)
        return default_return


class ExceptionContext:
    """
    Context manager for exception handling with custom error recovery.
    """
    
    def __init__(
        self,
        exception_types: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
        default_return: Any = None,
        log_errors: bool = True
    ):
        """
        Initialize the exception context.
        
        Args:
            exception_types: Types of exceptions to handle
            default_return: Default value to return on exception
            log_errors: Whether to log errors
        """
        self.exception_types = exception_types
        self.default_return = default_return
        self.log_errors = log_errors
        self.exception = None
    
    def __enter__(self):
        """Enter the context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context with exception handling."""
        if exc_type is not None and issubclass(exc_type, self.exception_types):
            self.exception = exc_val
            
            if self.log_errors:
                severity = ExceptionHandler.get_exception_severity(exc_val)
                ExceptionHandler.log_exception(
                    exc_val, "context_manager", (), {}, severity
                )
            
            return True  # Suppress the exception
        
        return False
    
    def get_exception(self) -> Optional[Exception]:
        """Get the exception that occurred, if any."""
        return self.exception