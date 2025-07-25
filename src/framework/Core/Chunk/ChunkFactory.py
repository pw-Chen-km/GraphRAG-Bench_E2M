"""
Chunking Factory for managing document chunking methods.

This module provides a factory pattern implementation for registering and
retrieving different document chunking strategies. It allows for dynamic
registration of chunking methods and provides a centralized registry.
"""

from typing import Any, Callable, Optional
from collections import defaultdict
from Core.Common.Utils import mdhash_id
from Core.Schema.ChunkSchema import TextChunk


class ChunkingMethodRegistry:
    """
    Registry for managing chunking methods.
    
    This class provides functionality to register, store, and retrieve
    different chunking methods for document processing.
    """
    
    def __init__(self):
        """Initialize the chunking methods registry."""
        self._methods: dict[str, Any] = defaultdict(Any)
    
    def register_method(
        self,
        method_name: str,
        method_func: Optional[Callable] = None
    ) -> None:
        """
        Register a new chunking method.
        
        Args:
            method_name: The unique identifier for the chunking method
            method_func: The function or class implementing the chunking logic
        """
        if not self.has_method(method_name):
            self._methods[method_name] = method_func
    
    def has_method(self, method_name: str) -> bool:
        """
        Check if a chunking method is registered.
        
        Args:
            method_name: The name of the method to check
            
        Returns:
            True if the method exists, False otherwise
        """
        return method_name in self._methods
    
    def get_method(self, method_name: str) -> Optional[Any]:
        """
        Retrieve a registered chunking method.
        
        Args:
            method_name: The name of the method to retrieve
            
        Returns:
            The registered method or None if not found
        """
        return self._methods.get(method_name)
    
    def list_methods(self) -> list[str]:
        """
        Get a list of all registered method names.
        
        Returns:
            List of registered method names
        """
        return list(self._methods.keys())


# Global registry instance
CHUNKING_REGISTRY = ChunkingMethodRegistry()


def register_chunking_method(method_name: str) -> Callable:
    """
    Decorator for registering chunking methods.
    
    This decorator simplifies the registration of new chunking methods
    by automatically adding them to the global registry.
    
    Args:
        method_name: The unique identifier for the chunking method
        
    Returns:
        Decorator function that registers the method
    """
    def decorator(func: Callable) -> Callable:
        """Register the decorated function as a chunking method."""
        CHUNKING_REGISTRY.register_method(method_name, func)
        return func
    return decorator


def create_chunk_method(method_name: str) -> Optional[Any]:
    """
    Create and return a chunking method by name.
    
    Args:
        method_name: The name of the method to create
        
    Returns:
        The chunking method function or None if not found
    """
    return CHUNKING_REGISTRY.get_method(method_name)


def get_available_chunk_methods() -> list[str]:
    """
    Get a list of all available chunking methods.
    
    Returns:
        List of available chunking method names
    """
    return CHUNKING_REGISTRY.list_methods()
