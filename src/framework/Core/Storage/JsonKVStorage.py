"""
JSON-based key-value storage implementation.

Provides efficient storage and retrieval of structured data using JSON format.
This implementation is optimized for human-readable data and cross-platform compatibility.
"""

from typing import Any, Dict, List, Optional, Set

from pydantic import Field

from Core.Common.Logger import logger
from Core.Common.Utils import load_json, write_json
from Core.Storage.BaseKVStorage import BaseKVStorage


class JsonKVStorage(BaseKVStorage):
    """
    JSON-based key-value storage implementation.
    
    This storage backend uses JSON format for storing key-value pairs. It provides
    human-readable data storage with good cross-platform compatibility and is
    suitable for configuration data, metadata, and other structured information.
    
    Attributes:
        name: JSON filename for storage
        data: Internal dictionary for storing key-value pairs
    """
    
    name: str = Field(description="JSON filename for storage")
    data: Dict[str, Any] = Field(default_factory=dict, description="Internal data storage")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.name.endswith('.json'):
            self.name = f"{self.name}.json"
    
    async def all_keys(self) -> List[str]:
        """
        Get all keys in the storage.
        
        Returns:
            List of all keys in the storage
        """
        return list(self.data.keys())
    
    async def get_by_id(self, id: str) -> Optional[Any]:
        """
        Retrieve a single item by its key.
        
        Args:
            id: The key to look up
            
        Returns:
            The item if found, None otherwise
        """
        return self.data.get(id)
    
    async def get_by_ids(
        self, 
        ids: List[str], 
        fields: Optional[Set[str]] = None
    ) -> List[Optional[Any]]:
        """
        Retrieve multiple items by their keys.
        
        Args:
            ids: List of keys to look up
            fields: Optional set of fields to include in the result
            
        Returns:
            List of items in the same order as input keys
        """
        if fields is None:
            return [self.data.get(id) for id in ids]
        
        results = []
        for id in ids:
            item = self.data.get(id)
            if item is not None and isinstance(item, dict):
                filtered_item = {k: v for k, v in item.items() if k in fields}
                results.append(filtered_item)
            else:
                results.append(item)
        
        return results
    
    async def filter_keys(self, data: List[str]) -> Set[str]:
        """
        Filter out keys that don't exist in storage.
        
        Args:
            data: List of keys to check
            
        Returns:
            Set of keys that don't exist in storage
        """
        return set(key for key in data if key not in self.data)
    
    async def upsert(self, data: Dict[str, Any]) -> bool:
        """
        Insert or update multiple items.
        
        Args:
            data: Dictionary mapping keys to values
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.data.update(data)
            return True
        except Exception as e:
            logger.error(f"Failed to upsert data: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete a single item by key.
        
        Args:
            key: The key to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if key in self.data:
                del self.data[key]
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete key '{key}': {e}")
            return False
    
    async def drop(self) -> bool:
        """
        Clear all data from storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.data.clear()
            return True
        except Exception as e:
            logger.error(f"Failed to drop all data: {e}")
            return False
    
    async def count(self) -> int:
        """
        Get the total number of items in storage.
        
        Returns:
            Number of items in storage
        """
        return len(self.data)
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in storage.
        
        Args:
            key: The key to check
            
        Returns:
            True if key exists, False otherwise
        """
        return key in self.data
    
    async def persist(self) -> bool:
        """
        Persist data to JSON file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            write_json(self.data, self._file_path)
            logger.info(f"Successfully wrote {len(self.data)} items to: {self._file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to persist data to {self._file_path}: {e}")
            return False
    
    async def load(self) -> bool:
        """
        Load data from JSON file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.data = load_json(self._file_path) or {}
            logger.info(f"Successfully loaded {len(self.data)} items from: {self._file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load data from {self._file_path}: {e}")
            self.data = {}
            return False
    
    async def initialize(self) -> bool:
        """
        Initialize the JSON storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return await self.load()
        except Exception as e:
            logger.error(f"Failed to initialize JSON storage: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """
        Clean up JSON storage resources.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.data.clear()
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup JSON storage: {e}")
            return False
    
    @property
    def json_data(self) -> Dict[str, Any]:
        """
        Get the raw JSON data dictionary.
        
        Returns:
            The internal data dictionary
        """
        return self.data
    
    @property
    def _file_path(self) -> str:
        """
        Get the file path for JSON storage.
        
        Returns:
            File path for JSON storage
        """
        if self.namespace is None:
            raise ValueError("Namespace must be configured for JSON storage")
        return self.namespace.get_save_path(self.name)
    
    async def is_empty(self) -> bool:
        """
        Check if storage is empty.
        
        Returns:
            True if storage is empty, False otherwise
        """
        return len(self.data) == 0
    
    def get_json_info(self) -> Dict[str, Any]:
        """Get information about the JSON storage configuration."""
        return {
            **self.get_kv_info(),
            "filename": self.name,
            "file_path": self._file_path,
            "item_count": len(self.data)
        }
