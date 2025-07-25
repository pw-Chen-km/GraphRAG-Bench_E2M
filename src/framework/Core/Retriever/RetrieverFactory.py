"""
Retriever factory for managing different retrieval methods.
Provides a registry system for registering and accessing various retrieval strategies.
"""

from collections import defaultdict
from typing import Any, Callable


class RetrieverFactory:
    """
    Factory class for managing different retrieval methods.
    
    This factory provides a registry system for registering and accessing
    various retrieval strategies based on type and method name.
    """

    def __init__(self):
        """Initialize the factory with an empty registry."""
        self.retriever_methods: dict = defaultdict(dict)

    def register_retriever_method(
        self,
        retriever_type: str,
        method_name: str,
        method_func: Callable = None
    ):
        """
        Register a new retrieval method.
        
        Args:
            retriever_type: Type of retriever (e.g., 'chunk', 'entity', 'community')
            method_name: Name of the retrieval method
            method_func: Function or class implementing the retrieval method
        """
        if self.has_retriever_method(retriever_type, method_name):
            return

        self.retriever_methods[retriever_type][method_name] = method_func

    def has_retriever_method(self, retriever_type: str, method_name: str) -> bool:
        """
        Check if a retrieval method exists.
        
        Args:
            retriever_type: Type of retriever
            method_name: Name of the retrieval method
            
        Returns:
            True if the method exists, False otherwise
        """
        return (retriever_type in self.retriever_methods and 
                method_name in self.retriever_methods[retriever_type])

    def get_method(self, retriever_type: str, method_name: str) -> Any:
        """
        Get a registered retrieval method.
        
        Args:
            retriever_type: Type of retriever
            method_name: Name of the retrieval method
            
        Returns:
            The registered method function or class
        """
        return self.retriever_methods.get(retriever_type, {}).get(method_name)


# Global registry instance
RETRIEVER_REGISTRY = RetrieverFactory()


def register_retriever_method(retriever_type: str, method_name: str):
    """
    Decorator for registering a new retrieval method.
    
    This decorator can be used to register a new retrieval method.
    The method will be stored in the registry for later use.
    
    Args:
        retriever_type: Type of retriever (e.g., 'chunk', 'entity', 'community')
        method_name: Name of the retrieval method
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable):
        """Register the decorated function as a retrieval method."""
        RETRIEVER_REGISTRY.register_retriever_method(retriever_type, method_name, func)
        return func
    return decorator


def get_retriever_operator(retriever_type: str, method_name: str) -> Any:
    """
    Get a registered retrieval operator.
    
    Args:
        retriever_type: Type of retriever
        method_name: Name of the retrieval method
        
    Returns:
        The registered retrieval operator
    """
    return RETRIEVER_REGISTRY.get_method(retriever_type, method_name)
