"""
Pickle-based blob storage implementation.

Provides efficient storage and retrieval of Python objects using pickle serialization.
This implementation is optimized for storing complex data structures and large objects.
"""

import pickle
from typing import Any, Optional, Dict

from pydantic import Field

from Core.Common.Logger import logger
from Core.Storage.BaseBlobStorage import BaseBlobStorage


class PickleBlobStorage(BaseBlobStorage):
    """
    Pickle-based blob storage implementation.
    
    This storage backend uses Python's pickle module to serialize and deserialize
    Python objects. It's suitable for storing complex data structures, numpy arrays,
    and other Python objects that can be pickled.
    
    Attributes:
        RESOURCE_NAME: Default filename for blob data
        _data: Internal storage for blob data
    """
    
    RESOURCE_NAME: str = Field(default="blob_data.pkl", description="Default blob data filename")
    data: Optional[Any] = Field(default=None, description="Internal blob data storage")
    
    async def get(self, key: Optional[str] = None) -> Any:
        """
        Retrieve blob data from storage.
        
        Args:
            key: Ignored for pickle storage (uses single blob)
            
        Returns:
            The stored blob data or None if not found
        """
        return self.data
    
    async def set(self, blob: Any, key: Optional[str] = None) -> bool:
        """
        Store blob data in memory.
        
        Args:
            blob: The blob data to store
            key: Ignored for pickle storage (uses single blob)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.validate_blob_size(blob):
                logger.warning(f"Blob size exceeds maximum limit of {self.max_blob_size} bytes")
                return False
            
            self.data = blob
            return True
        except Exception as e:
            logger.error(f"Failed to set blob data: {e}")
            return False
    
    async def exists(self, key: Optional[str] = None) -> bool:
        """
        Check if blob data exists in storage.
        
        Args:
            key: Ignored for pickle storage (uses single blob)
            
        Returns:
            True if blob data exists, False otherwise
        """
        return self.data is not None
    
    async def delete(self, key: Optional[str] = None) -> bool:
        """
        Delete blob data from storage.
        
        Args:
            key: Ignored for pickle storage (uses single blob)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._data = None
            return True
        except Exception as e:
            logger.error(f"Failed to delete blob data: {e}")
            return False
    
    async def load(self, force: bool = False) -> bool:
        """
        Load blob data from persistent storage.
        
        Args:
            force: Force reload even if data is already loaded
            
        Returns:
            True if successful, False otherwise
        """
        if force:
            logger.info(f"Forcing rebuild of mapping for: {self._get_data_file_path()}")
            self.data = None
            return False
        
        if self.namespace is None:
            self.data = None
            logger.info("Creating new volatile blob storage (no namespace)")
            return False
        
        data_file_path = self._get_data_file_path()
        if not data_file_path:
            logger.info("No data file found, loading empty storage")
            self.data = None
            return False
        
        try:
            with open(data_file_path, "rb") as f:
                self.data = pickle.load(f)
            logger.info(f"Successfully loaded blob data from: {data_file_path}")
            return True
        except FileNotFoundError:
            logger.info(f"No data file found at: {data_file_path}")
            self.data = None
            return False
        except Exception as e:
            logger.error(f"Error loading blob data from {data_file_path}: {e}")
            self.data = None
            return False
    
    async def persist(self) -> bool:
        """
        Persist blob data to storage.
        
        Returns:
            True if successful, False otherwise
        """
        if self.namespace is None:
            logger.warning("Cannot persist blob data: no namespace configured")
            return False
        
        data_file_path = self._get_save_file_path()
        try:
            with open(data_file_path, "wb") as f:
                pickle.dump(self.data, f)
            logger.info(f"Successfully saved blob data to: {data_file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving blob data to {data_file_path}: {e}")
            return False
    
    async def initialize(self) -> bool:
        """
        Initialize the pickle blob storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return await self.load()
        except Exception as e:
            logger.error(f"Failed to initialize pickle blob storage: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """
        Clean up pickle blob storage resources.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.data = None
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup pickle blob storage: {e}")
            return False
    
    def _get_data_file_path(self) -> Optional[str]:
        """
        Get the file path for loading blob data.
        
        Returns:
            File path or None if namespace is not configured
        """
        if self.namespace is None:
            return None
        return self.namespace.get_load_path(self.RESOURCE_NAME)
    
    def _get_save_file_path(self) -> str:
        """
        Get the file path for saving blob data.
        
        Returns:
            File path for saving
        """
        if self.namespace is None:
            raise ValueError("Namespace must be configured for persistence")
        return self.namespace.get_save_path(self.RESOURCE_NAME)
    
    def get_pickle_info(self) -> Dict[str, Any]:
        """Get information about the pickle blob storage configuration."""
        return {
            **self.get_blob_info(),
            "resource_name": self.RESOURCE_NAME,
            "data_loaded": self.data is not None,
            "data_file_path": self._get_data_file_path()
        }
