"""
Index Configuration Factory Module

This module provides a factory for creating different types of index configurations.
It centralizes the creation of configuration objects for various index types
including vector, ColBERT, and FAISS indices.
"""

from Core.Index import get_rag_embedding
from Core.Index.Schema import (
    VectorIndexConfig,
    ColBertIndexConfig,
    FAISSIndexConfig
)


class IndexConfigFactory:
    """
    Factory for creating index configuration objects.
    
    This factory provides a centralized way to instantiate various index
    configuration objects based on the vector database type specified
    in the configuration.
    """

    def __init__(self):
        """
        Initialize the factory with supported index type creators.
        """
        self.creators = {
            "vector": self._create_vector_config,
            "colbert": self._create_colbert_config,
            "faiss": self._create_faiss_config,
        }

    def get_config(self, config, persist_path: str):
        """
        Get an index configuration object based on the vector database type.
        
        Args:
            config: Configuration object containing index parameters
            persist_path: Path where the index will be persisted
            
        Returns:
            Index configuration object of the specified type
            
        Raises:
            KeyError: If the vector database type is not supported
        """
        vdb_type = config.vdb_type
        if vdb_type not in self.creators:
            raise KeyError(f"Unsupported vector database type: {vdb_type}")
            
        return self.creators[vdb_type](config, persist_path)

    @staticmethod
    def _create_vector_config(config, persist_path: str) -> VectorIndexConfig:
        """
        Create a vector index configuration.
        
        Args:
            config: Configuration object containing vector index parameters
            persist_path: Path where the index will be persisted
            
        Returns:
            VectorIndexConfig instance
        """
        return VectorIndexConfig(
            persist_path=persist_path,
            embed_model=get_rag_embedding(config.embedding.api_type, config)
        )

    @staticmethod
    def _create_faiss_config(config, persist_path: str) -> FAISSIndexConfig:
        """
        Create a FAISS index configuration.
        
        Args:
            config: Configuration object containing FAISS index parameters
            persist_path: Path where the index will be persisted
            
        Returns:
            FAISSIndexConfig instance
        """
        return FAISSIndexConfig(
            persist_path=persist_path,
            embed_model=get_rag_embedding(config.embedding.api_type, config)
        )

    @staticmethod
    def _create_colbert_config(config, persist_path: str) -> ColBertIndexConfig:
        """
        Create a ColBERT index configuration.
        
        Args:
            config: Configuration object containing ColBERT index parameters
            persist_path: Path where the index will be persisted
            
        Returns:
            ColBertIndexConfig instance
        """
        return ColBertIndexConfig(
            persist_path=persist_path, 
            index_name="nbits_2",
            model_name=config.colbert_checkpoint_path, 
            nbits=2
        )


# Global factory instance for convenience
get_index_config = IndexConfigFactory().get_config
