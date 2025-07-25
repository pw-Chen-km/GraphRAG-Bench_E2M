"""
Base Factory Module - Generic factory pattern implementation
Provides flexible object creation mechanisms for the GraphRAG system
"""

from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
from abc import ABC, abstractmethod

T = TypeVar('T')


class BaseFactory(ABC):
    """Abstract base class for factory implementations"""
    
    @abstractmethod
    def get_instance(self, key: Any, **kwargs) -> Any:
        """Get instance by key"""
        pass
    
    @abstractmethod
    def _raise_for_key(self, key: Any) -> None:
        """Raise appropriate exception for missing key"""
        pass


class GenericFactory(BaseFactory):
    """
    Generic factory for creating objects based on any keys.
    Supports dynamic registration of creator functions.
    """

    def __init__(self, creators: Optional[Dict[Any, Callable]] = None):
        """
        Initialize factory with optional creator functions.
        
        Args:
            creators: Dictionary mapping keys to creator functions
        """
        self._creators = creators or {}

    def register_creator(self, key: Any, creator: Callable) -> None:
        """
        Register a new creator function for a key.
        
        Args:
            key: Identifier for the creator
            creator: Function that creates objects
        """
        self._creators[key] = creator

    def unregister_creator(self, key: Any) -> None:
        """
        Remove a creator function for a key.
        
        Args:
            key: Identifier to remove
        """
        self._creators.pop(key, None)

    def get_instances(self, keys: List[Any], **kwargs) -> List[Any]:
        """
        Get multiple instances by keys.
        
        Args:
            keys: List of keys to create instances for
            **kwargs: Additional arguments for creator functions
            
        Returns:
            List of created instances
        """
        return [self.get_instance(key, **kwargs) for key in keys]

    def get_instance(self, key: Any, **kwargs) -> Any:
        """
        Get instance by key.
        
        Args:
            key: Identifier for the creator
            **kwargs: Arguments for the creator function
            
        Returns:
            Created instance
            
        Raises:
            ValueError: If key is not registered
        """
        creator = self._creators.get(key)
        if creator:
            return creator(**kwargs)
        self._raise_for_key(key)

    def _raise_for_key(self, key: Any) -> None:
        """Raise ValueError for missing key"""
        raise ValueError(f"Creator not registered for key: {key}")

    def has_creator(self, key: Any) -> bool:
        """
        Check if a creator is registered for the key.
        
        Args:
            key: Key to check
            
        Returns:
            True if creator exists, False otherwise
        """
        return key in self._creators

    def get_registered_keys(self) -> List[Any]:
        """
        Get all registered keys.
        
        Returns:
            List of registered keys
        """
        return list(self._creators.keys())


class ConfigBasedFactory(GenericFactory):
    """
    Factory for creating objects based on configuration object types.
    Automatically selects creator based on the type of the config object.
    """

    def get_instance(self, key: Any, **kwargs) -> Any:
        """
        Get instance by the type of key.
        
        Args:
            key: Configuration object (e.g., Pydantic model)
            **kwargs: Additional arguments for creator function
            
        Returns:
            Created instance
            
        Raises:
            ValueError: If config type is not registered
        """
        creator = self._creators.get(type(key))
        if creator:
            return creator(key, **kwargs)
        self._raise_for_key(key)

    def _raise_for_key(self, key: Any) -> None:
        """Raise ValueError for unknown config type"""
        raise ValueError(f"Unknown config: `{type(key)}`, {key}")

    @staticmethod
    def extract_value_from_config_or_kwargs(
        key: str, 
        config: Optional[object] = None, 
        **kwargs
    ) -> Any:
        """
        Extract value from config object or kwargs with priority.
        
        Priority: config attribute > kwargs > None
        
        Args:
            key: Key to extract
            config: Configuration object
            **kwargs: Keyword arguments
            
        Returns:
            Extracted value or None if not found
        """
        # Try config object first
        if config is not None and hasattr(config, key):
            value = getattr(config, key)
            if value is not None:
                return value

        # Fall back to kwargs
        return kwargs.get(key, None)

    def register_for_type(self, config_type: Type, creator: Callable) -> None:
        """
        Register creator for a specific config type.
        
        Args:
            config_type: Type of configuration object
            creator: Function that creates objects from config
        """
        self._creators[config_type] = creator