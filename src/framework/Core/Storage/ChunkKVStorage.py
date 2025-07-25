"""
Chunk-based key-value storage implementation.

Provides efficient storage and retrieval of document chunks with indexing capabilities.
This implementation is optimized for storing and retrieving text chunks with metadata.
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pydantic import Field

from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Logger import logger
from Core.Common.Utils import split_string_by_multiple_delimiters
from Core.Schema.ChunkSchema import TextChunk
from Core.Storage.BaseKVStorage import BaseKVStorage


class ChunkKVStorage(BaseKVStorage):
    """
    Chunk-based key-value storage implementation.
    
    This storage backend is specifically designed for storing document chunks with
    efficient indexing and retrieval capabilities. It maintains both key-based and
    index-based access patterns for optimal performance.
    
    Attributes:
        data_name: Filename for indexed chunk data
        chunk_name: Filename for key-based chunk data
        data: Index-based chunk storage
        chunk: Key-based chunk storage
        key_to_index: Mapping from keys to indices
        np_keys: Numpy array of keys for efficient operations
    """
    
    data_name: str = Field(default="chunk_data_idx.pkl", description="Indexed chunk data filename")
    chunk_name: str = Field(default="chunk_data_key.pkl", description="Key-based chunk data filename")
    data: Dict[int, TextChunk] = Field(default_factory=dict, description="Index-based chunk storage")
    chunk: Dict[str, TextChunk] = Field(default_factory=dict, description="Key-based chunk storage")
    key_to_index: Dict[str, int] = Field(default_factory=dict, description="Key to index mapping")
    np_keys: Optional[npt.NDArray[np.object_]] = Field(default=None, description="Numpy array of keys")
    
    class Config:
        arbitrary_types_allowed = True
    
    async def size(self) -> int:
        """
        Get the total number of chunks in storage.
        
        Returns:
            Number of chunks in storage
        """
        return len(self.data)
    
    async def get_by_key(self, key: str) -> Optional[TextChunk]:
        """
        Retrieve a chunk by its key.
        
        Args:
            key: The chunk key
            
        Returns:
            The chunk if found, None otherwise
        """
        index = self.key_to_index.get(key)
        if index is not None:
            return self.data.get(index)
        return None
    
    async def get_data_by_index(self, index: int) -> Optional[TextChunk]:
        """
        Retrieve a chunk by its index.
        
        Args:
            index: The chunk index
            
        Returns:
            The chunk if found, None otherwise
        """
        return self.data.get(index)
    
    async def get_index_by_merge_key(self, merge_chunk_id: str) -> List[int]:
        """
        Get indices for a merged chunk ID.
        
        Args:
            merge_chunk_id: Merged chunk identifier containing multiple chunk IDs
            
        Returns:
            List of chunk indices
        """
        key_list = split_string_by_multiple_delimiters(merge_chunk_id, [GRAPH_FIELD_SEP])
        index_list = [self.key_to_index.get(chunk_id) for chunk_id in key_list]
        return [idx for idx in index_list if idx is not None]
    
    async def get_index_by_key(self, key: str) -> Optional[int]:
        """
        Get the index for a given key.
        
        Args:
            key: The chunk key
            
        Returns:
            The chunk index or None if not found
        """
        return self.key_to_index.get(key)
    
    async def upsert_batch(self, keys: List[str], values: List[TextChunk]) -> bool:
        """
        Insert or update multiple chunks in batch.
        
        Args:
            keys: List of chunk keys
            values: List of chunk values
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for key, value in zip(keys, values):
                await self.upsert(key, value)
            return True
        except Exception as e:
            logger.error(f"Failed to upsert batch: {e}")
            return False
    
    async def upsert(self, key: str, value: TextChunk) -> bool:
        """
        Insert or update a single chunk.
        
        Args:
            key: The chunk key
            value: The chunk value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.chunk[key] = value
            index = self.key_to_index.get(key)
            
            if index is None:
                index = value.index
                self.key_to_index[key] = index
            
            self.data[index] = value
            return True
        except Exception as e:
            logger.error(f"Failed to upsert chunk '{key}': {e}")
            return False
    
    async def delete_by_key(self, key: str) -> bool:
        """
        Delete a chunk by its key.
        
        Args:
            key: The chunk key to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            index = self.key_to_index.pop(key, None)
            if index is not None:
                self.data.pop(index, None)
                self.chunk.pop(key, None)
                return True
            else:
                logger.warning(f"Key '{key}' not found in chunk storage")
                return False
        except Exception as e:
            logger.error(f"Failed to delete chunk '{key}': {e}")
            return False
    
    async def chunk_datas(self) -> List[Tuple[str, TextChunk]]:
        """
        Get all chunk data as key-value pairs.
        
        Returns:
            List of (key, chunk) tuples
        """
        return list(self.chunk.items())
    
    async def get_chunks(self) -> List[Tuple[str, TextChunk]]:
        """
        Get all chunks in storage.
        
        Returns:
            List of (key, chunk) tuples
        """
        return list(self.chunk.items())
    
    async def all_keys(self) -> List[str]:
        """
        Get all keys in the storage.
        
        Returns:
            List of all keys
        """
        return list(self.chunk.keys())
    
    async def get_by_id(self, id: str) -> Optional[TextChunk]:
        """
        Retrieve a chunk by its ID (key).
        
        Args:
            id: The chunk ID/key
            
        Returns:
            The chunk if found, None otherwise
        """
        return await self.get_by_key(id)
    
    async def get_by_ids(
        self, 
        ids: List[str], 
        fields: Optional[set] = None
    ) -> List[Optional[TextChunk]]:
        """
        Retrieve multiple chunks by their IDs.
        
        Args:
            ids: List of chunk IDs
            fields: Ignored for chunk storage
            
        Returns:
            List of chunks in the same order as input IDs
        """
        chunk_list = []
        for id in ids:
            chunk = await self.get_by_key(id)
            chunk_list.append(chunk)
        return chunk_list
    
    async def filter_keys(self, data: List[str]) -> set:
        """
        Filter out keys that don't exist in storage.
        
        Args:
            data: List of keys to check
            
        Returns:
            Set of keys that don't exist in storage
        """
        return set(key for key in data if key not in self.chunk)
    
    async def drop(self) -> bool:
        """
        Clear all data from storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.data.clear()
            self.chunk.clear()
            self.key_to_index.clear()
            self.np_keys = None
            return True
        except Exception as e:
            logger.error(f"Failed to drop all data: {e}")
            return False
    
    async def count(self) -> int:
        """
        Get the total number of chunks in storage.
        
        Returns:
            Number of chunks
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
        return key in self.chunk
    
    async def load_chunk(self) -> bool:
        """
        Load chunk data from persistent storage.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Attempting to load chunk data from: {self._get_data_idx_path()} and {self._get_data_key_path()}")
        
        if not (os.path.exists(self._get_data_idx_path()) and os.path.exists(self._get_data_key_path())):
            logger.info("Chunk data files do not exist, need to chunk documents from scratch")
            return False
        
        try:
            with open(self._get_data_idx_path(), "rb") as file:
                self.data = pickle.load(file)
            with open(self._get_data_key_path(), "rb") as file:
                self.chunk = pickle.load(file)
            
            self.key_to_index = {key: value.index for key, value in self.chunk.items()}
            logger.info(f"Successfully loaded chunk data with {len(self.data)} indexed chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to load chunk data: {e}")
            return False
    
    async def persist(self) -> bool:
        """
        Persist chunk data to storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Writing chunk data to {self._get_data_idx_path()} and {self._get_data_key_path()}")
            await self._persist()
            return True
        except Exception as e:
            logger.error(f"Failed to persist chunk data: {e}")
            return False
    
    async def _persist(self) -> None:
        """Internal method to persist chunk data."""
        self._write_chunk_data(self.data, self._get_data_idx_path())
        self._write_chunk_data(self.chunk, self._get_data_key_path())
    
    @staticmethod
    def _write_chunk_data(data: Any, pkl_file: str) -> None:
        """
        Write chunk data to pickle file.
        
        Args:
            data: Data to write
            pkl_file: Pickle file path
        """
        with open(pkl_file, "wb") as file:
            pickle.dump(data, file)
    
    async def initialize(self) -> bool:
        """
        Initialize the chunk storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return await self.load_chunk()
        except Exception as e:
            logger.error(f"Failed to initialize chunk storage: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """
        Clean up chunk storage resources.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.data.clear()
            self.chunk.clear()
            self.key_to_index.clear()
            self.np_keys = None
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup chunk storage: {e}")
            return False
    
    def _get_data_idx_path(self) -> str:
        """Get the path for indexed chunk data file."""
        if self.namespace is None:
            raise ValueError("Namespace must be configured for chunk storage")
        return self.namespace.get_save_path(self.data_name)
    
    def _get_data_key_path(self) -> str:
        """Get the path for key-based chunk data file."""
        if self.namespace is None:
            raise ValueError("Namespace must be configured for chunk storage")
        return self.namespace.get_save_path(self.chunk_name)
    
    def get_chunk_info(self) -> Dict[str, Any]:
        """Get information about the chunk storage configuration."""
        return {
            **self.get_kv_info(),
            "data_name": self.data_name,
            "chunk_name": self.chunk_name,
            "indexed_chunks": len(self.data),
            "keyed_chunks": len(self.chunk),
            "mappings": len(self.key_to_index)
        }