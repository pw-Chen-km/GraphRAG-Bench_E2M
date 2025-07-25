"""
Base Factory Module

This module provides base factory classes for creating objects based on
configuration or keys. It includes both generic and configuration-based
factory patterns for flexible object instantiation.
"""

from typing import Any, Callable, Optional


class GenericFactory:
    """
    Generic factory designed to create objects based on any keys.
    
    This factory provides a flexible way to instantiate objects using
    a dictionary mapping of keys to creator functions.
    """

    def __init__(self, creators: Optional[dict[Any, Callable]] = None):
        """
        Initialize the factory with creator functions.
        
        Args:
            creators: Dictionary mapping keys to creator functions.
                     Keys are identifiers, and values are the associated
                     creator functions that create objects.
        """
        self._creators = creators or {}

    def get_instances(self, keys: list[Any], **kwargs) -> list[Any]:
        """
        Get multiple instances by keys.
        
        Args:
            keys: List of keys to create instances for
            **kwargs: Additional arguments to pass to creator functions
            
        Returns:
            List of created instances
        """
        return [self.get_instance(key, **kwargs) for key in keys]

    def get_instance(self, key: Any, **kwargs) -> Any:
        """
        Get a single instance by key.
        
        Args:
            key: Key identifier for the object to create
            **kwargs: Additional arguments to pass to the creator function
            
        Returns:
            Created object instance
            
        Raises:
            ValueError: If the key is not found in the creators dictionary
        """
        creator = self._creators.get(key)
        if creator:
            return creator(**kwargs)

        self._raise_for_key(key)

    def _raise_for_key(self, key: Any) -> None:
        """
        Raise an error for an unknown key.
        
        Args:
            key: Unknown key that caused the error
            
        Raises:
            ValueError: With descriptive error message
        """
        raise ValueError(f"Creator not registered for key: {key}")


class ConfigBasedFactory(GenericFactory):
    """
    Configuration-based factory designed to create objects based on object type.
    
    This factory extends GenericFactory to work with configuration objects,
    using the type of the configuration object as the key for object creation.
    """

    def get_instance(self, key: Any, **kwargs) -> Any:
        """
        Get instance by the type of the configuration object.
        
        The key is a config object (such as a pydantic model). The factory
        calls the appropriate creator function based on the type of the key,
        and passes the key to the creator function.
        
        Args:
            key: Configuration object
            **kwargs: Additional arguments to pass to the creator function
            
        Returns:
            Created object instance
            
        Raises:
            ValueError: If the configuration type is not supported
        """
        creator = self._creators.get(type(key))
        if creator:
            return creator(key, **kwargs)

        self._raise_for_key(key)

    def _raise_for_key(self, key: Any) -> None:
        """
        Raise an error for an unknown configuration type.
        
        Args:
            key: Unknown configuration object that caused the error
            
        Raises:
            ValueError: With descriptive error message
        """
        raise ValueError(f"Unknown config: `{type(key)}`, {key}")

    @staticmethod
    def _val_from_config_or_kwargs(key: str, config: Optional[object] = None, **kwargs) -> Any:
        """
        Extract value from configuration object or kwargs with priority.
        
        This method prioritizes the configuration object's value unless it is None,
        in which case it looks into kwargs. Returns None if not found.
        
        Args:
            key: Key to look for in config or kwargs
            config: Configuration object to search in
            **kwargs: Keyword arguments to search in
            
        Returns:
            Value from config or kwargs, or None if not found
        """
        if config is not None and hasattr(config, key):
            val = getattr(config, key)
            if val is not None:
                return val

        if key in kwargs:
            return kwargs[key]

        return None