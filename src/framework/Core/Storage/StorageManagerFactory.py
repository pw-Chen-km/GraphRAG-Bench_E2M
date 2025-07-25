"""
Storage manager factory for managing different storage backends.
Provides a unified interface for various storage strategies using factory and strategy patterns.
"""

import asyncio
import json
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from Core.Common.Logger import logger
from Core.Utils.Display import StatusDisplay


class StorageType(Enum):
    """Supported storage types."""
    NETWORKX = "networkx"
    PICKLE = "pickle"
    JSON = "json"
    CHUNK_KV = "chunk_kv"
    TREE = "tree"


@dataclass
class StorageConfig:
    """Configuration for storage managers."""
    storage_type: StorageType = StorageType.NETWORKX
    base_path: str = ""
    namespace: str = ""
    compression: bool = False
    backup_enabled: bool = True
    max_file_size: int = 100 * 1024 * 1024  # 100MB


class StorageManager(ABC):
    """
    Abstract base class for storage managers.
    
    Provides a unified interface for different storage backends
    with common functionality like backup and restore operations.
    """
    
    def __init__(self, config: StorageConfig, context: Any):
        """Initialize storage manager with configuration."""
        self.config = config
        self.context = context
        self.storage_path = os.path.join(config.base_path, config.namespace)
        self._ensure_storage_directory()
    
    @abstractmethod
    async def save(self, key: str, data: Any) -> bool:
        """Save data to storage."""
        pass
    
    @abstractmethod
    async def load(self, key: str) -> Optional[Any]:
        """Load data from storage."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data from storage."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if data exists in storage."""
        pass
    
    @abstractmethod
    async def list_keys(self) -> List[str]:
        """List all keys in storage."""
        pass
    
    def _ensure_storage_directory(self):
        """Ensure storage directory exists."""
        os.makedirs(self.storage_path, exist_ok=True)
    
    def _get_file_path(self, key: str) -> str:
        """Get file path for a given key."""
        return os.path.join(self.storage_path, f"{key}")
    
    async def backup(self, key: str) -> bool:
        """Create backup of data."""
        if not self.config.backup_enabled:
            return True
        
        try:
            if await self.exists(key):
                backup_path = self._get_file_path(f"{key}.backup")
                data = await self.load(key)
                await self._save_to_file(backup_path, data)
                return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
        
        return False
    
    async def restore(self, key: str) -> bool:
        """Restore data from backup."""
        try:
            backup_path = self._get_file_path(f"{key}.backup")
            if os.path.exists(backup_path):
                data = await self._load_from_file(backup_path)
                await self.save(key, data)
                return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
        
        return False
    
    async def _save_to_file(self, file_path: str, data: Any) -> None:
        """Save data to file (to be implemented by subclasses)."""
        pass
    
    async def _load_from_file(self, file_path: str) -> Any:
        """Load data from file (to be implemented by subclasses)."""
        pass


class NetworkXStorageManager(StorageManager):
    """Storage manager for NetworkX graphs."""
    
    async def save(self, key: str, data: Any) -> bool:
        """Save NetworkX graph to storage."""
        try:
            import networkx as nx
            
            file_path = self._get_file_path(f"{key}.gpickle")
            nx.write_gpickle(data, file_path)
            
            # Create backup
            await self.backup(key)
            
            StatusDisplay.show_success(f"Saved NetworkX graph: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to save NetworkX graph: {e}")
            return False
    
    async def load(self, key: str) -> Optional[Any]:
        """Load NetworkX graph from storage."""
        try:
            import networkx as nx
            
            file_path = self._get_file_path(f"{key}.gpickle")
            if os.path.exists(file_path):
                graph = nx.read_gpickle(file_path)
                StatusDisplay.show_success(f"Loaded NetworkX graph: {key}")
                return graph
            else:
                logger.warning(f"NetworkX graph file not found: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to load NetworkX graph: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete NetworkX graph from storage."""
        try:
            file_path = self._get_file_path(f"{key}.gpickle")
            if os.path.exists(file_path):
                os.remove(file_path)
                StatusDisplay.show_success(f"Deleted NetworkX graph: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete NetworkX graph: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if NetworkX graph exists in storage."""
        file_path = self._get_file_path(f"{key}.gpickle")
        return os.path.exists(file_path)
    
    async def list_keys(self) -> List[str]:
        """List all NetworkX graph keys."""
        keys = []
        for file in os.listdir(self.storage_path):
            if file.endswith('.gpickle') and not file.endswith('.backup'):
                key = file[:-8]  # Remove .gpickle extension
                keys.append(key)
        return keys
    
    async def _save_to_file(self, file_path: str, data: Any) -> None:
        """Save NetworkX graph to file."""
        import networkx as nx
        nx.write_gpickle(data, file_path)
    
    async def _load_from_file(self, file_path: str) -> Any:
        """Load NetworkX graph from file."""
        import networkx as nx
        return nx.read_gpickle(file_path)


class PickleStorageManager(StorageManager):
    """Storage manager for pickle data."""
    
    async def save(self, key: str, data: Any) -> bool:
        """Save pickle data to storage."""
        try:
            file_path = self._get_file_path(f"{key}.pkl")
            await self._save_to_file(file_path, data)
            
            # Create backup
            await self.backup(key)
            
            StatusDisplay.show_success(f"Saved pickle data: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to save pickle data: {e}")
            return False
    
    async def load(self, key: str) -> Optional[Any]:
        """Load pickle data from storage."""
        try:
            file_path = self._get_file_path(f"{key}.pkl")
            if os.path.exists(file_path):
                data = await self._load_from_file(file_path)
                StatusDisplay.show_success(f"Loaded pickle data: {key}")
                return data
            else:
                logger.warning(f"Pickle file not found: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to load pickle data: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete pickle data from storage."""
        try:
            file_path = self._get_file_path(f"{key}.pkl")
            if os.path.exists(file_path):
                os.remove(file_path)
                StatusDisplay.show_success(f"Deleted pickle data: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete pickle data: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if pickle data exists in storage."""
        file_path = self._get_file_path(f"{key}.pkl")
        return os.path.exists(file_path)
    
    async def list_keys(self) -> List[str]:
        """List all pickle data keys."""
        keys = []
        for file in os.listdir(self.storage_path):
            if file.endswith('.pkl') and not file.endswith('.backup'):
                key = file[:-4]  # Remove .pkl extension
                keys.append(key)
        return keys
    
    async def _save_to_file(self, file_path: str, data: Any) -> None:
        """Save data to pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    async def _load_from_file(self, file_path: str) -> Any:
        """Load data from pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)


class JsonStorageManager(StorageManager):
    """Storage manager for JSON data."""
    
    async def save(self, key: str, data: Any) -> bool:
        """Save JSON data to storage."""
        try:
            file_path = self._get_file_path(f"{key}.json")
            await self._save_to_file(file_path, data)
            
            # Create backup
            await self.backup(key)
            
            StatusDisplay.show_success(f"Saved JSON data: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON data: {e}")
            return False
    
    async def load(self, key: str) -> Optional[Any]:
        """Load JSON data from storage."""
        try:
            file_path = self._get_file_path(f"{key}.json")
            if os.path.exists(file_path):
                data = await self._load_from_file(file_path)
                StatusDisplay.show_success(f"Loaded JSON data: {key}")
                return data
            else:
                logger.warning(f"JSON file not found: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to load JSON data: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete JSON data from storage."""
        try:
            file_path = self._get_file_path(f"{key}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                StatusDisplay.show_success(f"Deleted JSON data: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete JSON data: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if JSON data exists in storage."""
        file_path = self._get_file_path(f"{key}.json")
        return os.path.exists(file_path)
    
    async def list_keys(self) -> List[str]:
        """List all JSON data keys."""
        keys = []
        for file in os.listdir(self.storage_path):
            if file.endswith('.json') and not file.endswith('.backup'):
                key = file[:-5]  # Remove .json extension
                keys.append(key)
        return keys
    
    async def _save_to_file(self, file_path: str, data: Any) -> None:
        """Save data to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    async def _load_from_file(self, file_path: str) -> Any:
        """Load data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)


class ChunkKVStorageManager(StorageManager):
    """Storage manager for chunked key-value data."""
    
    def __init__(self, config: StorageConfig, context: Any):
        """Initialize chunked storage manager."""
        super().__init__(config, context)
        self.chunk_size = 1024 * 1024  # 1MB per chunk
    
    async def save(self, key: str, data: Any) -> bool:
        """Save chunked data to storage."""
        try:
            # Serialize data
            serialized_data = pickle.dumps(data)
            
            # Split into chunks
            chunks = [serialized_data[i:i+self.chunk_size] 
                     for i in range(0, len(serialized_data), self.chunk_size)]
            
            # Save chunk information
            chunk_info = {
                "total_chunks": len(chunks),
                "total_size": len(serialized_data),
                "chunk_size": self.chunk_size
            }
            
            info_path = self._get_file_path(f"{key}.info")
            with open(info_path, 'w') as f:
                json.dump(chunk_info, f)
            
            # Save each chunk
            for i, chunk in enumerate(chunks):
                chunk_path = self._get_file_path(f"{key}.chunk_{i}")
                with open(chunk_path, 'wb') as f:
                    f.write(chunk)
            
            # Create backup
            await self.backup(key)
            
            StatusDisplay.show_success(f"Saved chunked data: {key} ({len(chunks)} chunks)")
            return True
        except Exception as e:
            logger.error(f"Failed to save chunked data: {e}")
            return False
    
    async def load(self, key: str) -> Optional[Any]:
        """Load chunked data from storage."""
        try:
            # Load chunk information
            info_path = self._get_file_path(f"{key}.info")
            if not os.path.exists(info_path):
                logger.warning(f"Chunk info file not found: {info_path}")
                return None
            
            with open(info_path, 'r') as f:
                chunk_info = json.load(f)
            
            # Load all chunks
            chunks = []
            for i in range(chunk_info["total_chunks"]):
                chunk_path = self._get_file_path(f"{key}.chunk_{i}")
                with open(chunk_path, 'rb') as f:
                    chunks.append(f.read())
            
            # Merge chunks
            serialized_data = b''.join(chunks)
            data = pickle.loads(serialized_data)
            
            StatusDisplay.show_success(f"Loaded chunked data: {key}")
            return data
        except Exception as e:
            logger.error(f"Failed to load chunked data: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete chunked data from storage."""
        try:
            # Delete chunk information
            info_path = self._get_file_path(f"{key}.info")
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    chunk_info = json.load(f)
                
                # Delete all chunks
                for i in range(chunk_info["total_chunks"]):
                    chunk_path = self._get_file_path(f"{key}.chunk_{i}")
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                
                # Delete info file
                os.remove(info_path)
                
                StatusDisplay.show_success(f"Deleted chunked data: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete chunked data: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if chunked data exists in storage."""
        info_path = self._get_file_path(f"{key}.info")
        return os.path.exists(info_path)
    
    async def list_keys(self) -> List[str]:
        """List all chunked data keys."""
        keys = set()
        for file in os.listdir(self.storage_path):
            if file.endswith('.info'):
                key = file[:-5]  # Remove .info extension
                keys.add(key)
        return list(keys)


class TreeStorageManager(StorageManager):
    """Storage manager for tree data structures."""
    
    async def save(self, key: str, data: Any) -> bool:
        """Save tree data to storage."""
        try:
            # Convert tree data to serializable format
            tree_data = self._serialize_tree(data)
            
            file_path = self._get_file_path(f"{key}.tree")
            await self._save_to_file(file_path, tree_data)
            
            # Create backup
            await self.backup(key)
            
            StatusDisplay.show_success(f"Saved tree data: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to save tree data: {e}")
            return False
    
    async def load(self, key: str) -> Optional[Any]:
        """Load tree data from storage."""
        try:
            file_path = self._get_file_path(f"{key}.tree")
            if os.path.exists(file_path):
                tree_data = await self._load_from_file(file_path)
                data = self._deserialize_tree(tree_data)
                StatusDisplay.show_success(f"Loaded tree data: {key}")
                return data
            else:
                logger.warning(f"Tree data file not found: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to load tree data: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete tree data from storage."""
        try:
            file_path = self._get_file_path(f"{key}.tree")
            if os.path.exists(file_path):
                os.remove(file_path)
                StatusDisplay.show_success(f"Deleted tree data: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete tree data: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if tree data exists in storage."""
        file_path = self._get_file_path(f"{key}.tree")
        return os.path.exists(file_path)
    
    async def list_keys(self) -> List[str]:
        """List all tree data keys."""
        keys = []
        for file in os.listdir(self.storage_path):
            if file.endswith('.tree') and not file.endswith('.backup'):
                key = file[:-5]  # Remove .tree extension
                keys.append(key)
        return keys
    
    def _serialize_tree(self, tree_data: Any) -> Dict:
        """Serialize tree data structure."""
        # Implement tree serialization logic based on specific tree structure
        return {"tree_data": tree_data}
    
    def _deserialize_tree(self, tree_data: Dict) -> Any:
        """Deserialize tree data structure."""
        # Implement tree deserialization logic based on specific tree structure
        return tree_data.get("tree_data")
    
    async def _save_to_file(self, file_path: str, data: Any) -> None:
        """Save data to pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    async def _load_from_file(self, file_path: str) -> Any:
        """Load data from pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)


class StorageManagerFactory:
    """
    Factory class for creating storage managers.
    
    Provides a centralized way to instantiate different storage managers
    based on configuration parameters.
    """
    
    _managers = {
        StorageType.NETWORKX: NetworkXStorageManager,
        StorageType.PICKLE: PickleStorageManager,
        StorageType.JSON: JsonStorageManager,
        StorageType.CHUNK_KV: ChunkKVStorageManager,
        StorageType.TREE: TreeStorageManager
    }
    
    @classmethod
    def create_manager(cls, config: Any, context: Any) -> StorageManager:
        """
        Create a storage manager based on configuration.
        
        Args:
            config: Configuration object containing storage parameters
            context: Context object for the storage manager
            
        Returns:
            StorageManager: Instance of the appropriate storage manager
        """
        # Extract storage configuration from config
        storage_config = StorageConfig(
            storage_type=StorageType.NETWORKX,  # Default to NetworkX
            base_path=config.working_dir if hasattr(config, 'working_dir') else "",
            namespace=config.index_name if hasattr(config, 'index_name') else "default",
            compression=False,
            backup_enabled=True,
            max_file_size=100 * 1024 * 1024
        )
        
        manager_class = cls._managers.get(storage_config.storage_type, NetworkXStorageManager)
        return manager_class(storage_config, context)
    
    @classmethod
    def register_manager(cls, storage_type: StorageType, manager_class: type):
        """
        Register a new storage manager.
        
        Args:
            storage_type: Type of storage to register
            manager_class: Class implementing the storage manager
        """
        cls._managers[storage_type] = manager_class
        logger.info(f"Registered new storage manager: {storage_type.value}")
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available storage types."""
        return [storage_type.value for storage_type in cls._managers.keys()] 