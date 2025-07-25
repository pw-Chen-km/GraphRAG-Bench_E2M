"""
Base key-value storage abstraction.

Provides the foundational interface for key-value storage implementations that handle
structured data with efficient lookup and retrieval capabilities in the GraphRAG system.
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar, Union

from pydantic import Field

from Core.Storage.BaseStorage import BaseStorage

T = TypeVar("T")


class BaseKVStorage(Generic[T], BaseStorage):
    """
    Abstract base class for key-value storage implementations.
    
    Key-value storage provides efficient storage and retrieval of structured data
    using string keys. This abstraction supports various data types through generics
    and provides common operations like batch retrieval and filtering.
    
    Attributes:
        cache_enabled: Whether in-memory caching is enabled
        cache_size: Maximum number of items in cache
        batch_size: Default batch size for bulk operations
    """
    
    cache_enabled: bool = Field(default=True, description="Enable in-memory caching")
    cache_size: int = Field(default=1000, description="Maximum cache size")
    batch_size: int = Field(default=100, description="Default batch size for operations")
    
    @abstractmethod
    async def all_keys(self) -> List[str]:
        """
        Get all keys in the storage.
        
        Returns:
            List of all keys in the storage
        """
        raise NotImplementedError
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[T]:
        """
        Retrieve a single item by its key.
        
        Args:
            id: The key to look up
            
        Returns:
            The item if found, None otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    async def get_by_ids(
        self, 
        ids: List[str], 
        fields: Optional[Set[str]] = None
    ) -> List[Optional[T]]:
        """
        Retrieve multiple items by their keys.
        
        Args:
            ids: List of keys to look up
            fields: Optional set of fields to include in the result
            
        Returns:
            List of items in the same order as input keys
        """
        raise NotImplementedError
    
    @abstractmethod
    async def filter_keys(self, data: List[str]) -> Set[str]:
        """
        Filter out keys that don't exist in storage.
        
        Args:
            data: List of keys to check
            
        Returns:
            Set of keys that don't exist in storage
        """
        raise NotImplementedError
    
    @abstractmethod
    async def upsert(self, data: Dict[str, T]) -> bool:
        """
        Insert or update multiple items.
        
        Args:
            data: Dictionary mapping keys to values
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete a single item by key.
        
        Args:
            key: The key to delete
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    async def drop(self) -> bool:
        """
        Clear all data from storage.
        
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    async def count(self) -> int:
        """
        Get the total number of items in storage.
        
        Returns:
            Number of items in storage
        """
        raise NotImplementedError
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in storage.
        
        Args:
            key: The key to check
            
        Returns:
            True if key exists, False otherwise
        """
        raise NotImplementedError
    
    async def get_batch(self, keys: List[str], batch_size: Optional[int] = None) -> List[Optional[T]]:
        """
        Retrieve items in batches for better performance.
        
        Args:
            keys: List of keys to retrieve
            batch_size: Size of each batch (uses default if None)
            
        Returns:
            List of items in the same order as input keys
        """
        batch_size = batch_size or self.batch_size
        results = []
        
        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i:i + batch_size]
            batch_results = await self.get_by_ids(batch_keys)
            results.extend(batch_results)
        
        return results
    
    async def upsert_batch(self, data: Dict[str, T], batch_size: Optional[int] = None) -> bool:
        """
        Insert or update items in batches for better performance.
        
        Args:
            data: Dictionary mapping keys to values
            batch_size: Size of each batch (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        batch_size = batch_size or self.batch_size
        items = list(data.items())
        
        for i in range(0, len(items), batch_size):
            batch_items = dict(items[i:i + batch_size])
            success = await self.upsert(batch_items)
            if not success:
                return False
        
        return True
    
    def get_kv_info(self) -> Dict[str, Any]:
        """Get information about the key-value storage configuration."""
        return {
            **self.get_storage_info(),
            "cache_enabled": self.cache_enabled,
            "cache_size": self.cache_size,
            "batch_size": self.batch_size
        }