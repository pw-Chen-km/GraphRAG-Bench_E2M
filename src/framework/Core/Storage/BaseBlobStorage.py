"""
Base blob storage abstraction.

Provides the foundational interface for blob storage implementations that handle
large binary objects and serialized data structures in the GraphRAG system.
"""

from abc import abstractmethod
from typing import Any, Optional

from pydantic import Field

from Core.Storage.BaseStorage import BaseStorage


class BaseBlobStorage(BaseStorage):
    """
    Abstract base class for blob storage implementations.
    
    Blob storage is designed to handle large binary objects, serialized data structures,
    and other non-textual data that doesn't fit well into key-value storage patterns.
    
    Attributes:
        compression_enabled: Whether compression is enabled for blob storage
        max_blob_size: Maximum size limit for individual blobs in bytes
    """
    
    compression_enabled: bool = Field(default=False, description="Enable compression for blobs")
    max_blob_size: int = Field(default=100 * 1024 * 1024, description="Maximum blob size in bytes")  # 100MB
    
    @abstractmethod
    async def get(self, key: Optional[str] = None) -> Any:
        """
        Retrieve blob data from storage.
        
        Args:
            key: Optional key identifier for the blob
            
        Returns:
            The blob data or None if not found
        """
        raise NotImplementedError
    
    @abstractmethod
    async def set(self, blob: Any, key: Optional[str] = None) -> bool:
        """
        Store blob data in storage.
        
        Args:
            blob: The blob data to store
            key: Optional key identifier for the blob
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    async def exists(self, key: Optional[str] = None) -> bool:
        """
        Check if blob exists in storage.
        
        Args:
            key: Optional key identifier for the blob
            
        Returns:
            True if blob exists, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    async def delete(self, key: Optional[str] = None) -> bool:
        """
        Delete blob from storage.
        
        Args:
            key: Optional key identifier for the blob
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    async def load(self, force: bool = False) -> bool:
        """
        Load blob data from persistent storage.
        
        Args:
            force: Force reload even if data is already loaded
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    async def persist(self) -> bool:
        """
        Persist blob data to storage.
        
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError
    
    def validate_blob_size(self, blob: Any) -> bool:
        """
        Validate if blob size is within limits.
        
        Args:
            blob: The blob data to validate
            
        Returns:
            True if blob size is acceptable, False otherwise
        """
        if hasattr(blob, '__sizeof__'):
            size = blob.__sizeof__()
            return size <= self.max_blob_size
        return True
    
    def get_blob_info(self) -> dict[str, Any]:
        """Get information about the blob storage configuration."""
        return {
            **self.get_storage_info(),
            "compression_enabled": self.compression_enabled,
            "max_blob_size": self.max_blob_size
        }
