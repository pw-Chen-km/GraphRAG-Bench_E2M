"""
Base storage abstraction layer.

Provides the foundational interface and common functionality for all storage implementations
in the GraphRAG system. This module defines the core storage abstractions that enable
consistent behavior across different storage backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field

from Core.Storage.NameSpace import Namespace


class StorageProtocol(Protocol):
    """Protocol defining the interface for storage implementations."""
    
    async def save(self, key: str, data: Any) -> bool:
        """Save data to storage."""
        ...
    
    async def load(self, key: str) -> Optional[Any]:
        """Load data from storage."""
        ...
    
    async def delete(self, key: str) -> bool:
        """Delete data from storage."""
        ...
    
    async def exists(self, key: str) -> bool:
        """Check if data exists in storage."""
        ...


class BaseStorage(ABC, BaseModel):
    """
    Abstract base class for all storage implementations.
    
    Provides a unified interface and common functionality for different storage backends.
    This class serves as the foundation for all storage operations in the GraphRAG system.
    
    Attributes:
        config: Configuration object containing storage parameters
        namespace: Namespace manager for organizing storage resources
    """
    
    config: Optional[Any] = Field(default=None, description="Storage configuration object")
    namespace: Optional[Namespace] = Field(default=None, description="Namespace for resource organization")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate storage configuration."""
        # Allow config to be None for some storage implementations
        # if self.config is None:
        #     raise ValueError("Storage configuration is required")
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the storage backend."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Clean up storage resources."""
        pass
    
    def get_storage_info(self) -> dict[str, Any]:
        """Get information about the storage implementation."""
        return {
            "type": self.__class__.__name__,
            "config": self.config,
            "namespace": self.namespace.namespace if self.namespace else None
        }
