"""
Index Factory Module

This module provides a factory for creating different types of vector indices.
It offers a unified interface for instantiating various indexing strategies
including vector, ColBERT, and FAISS implementations.
"""

import os

import faiss
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage

from Core.Common.BaseFactory import ConfigBasedFactory
from Core.Index.ColBertIndex import ColBertIndex
from Core.Index.FaissIndex import FaissIndex
from Core.Index.Schema import (
    BaseIndexConfig,
    ColBertIndexConfig,
    FAISSIndexConfig,
    VectorIndexConfig,
)
from Core.Index.VectorIndex import VectorIndex


class RAGIndexFactory(ConfigBasedFactory):
    """
    Factory class for creating different types of vector indices.
    
    This factory provides a centralized way to instantiate various indexing
    strategies based on configuration parameters, supporting different vector
    storage and retrieval approaches.
    """

    def __init__(self):
        """
        Initialize the factory with index type mappings.
        """
        creators = {
            VectorIndexConfig: self._create_vector_index,
            ColBertIndexConfig: self._create_colbert_index,
            FAISSIndexConfig: self._create_faiss_index,
        }
        super().__init__(creators)

    def get_index(self, config: BaseIndexConfig):
        """
        Create an index instance based on configuration.
        
        Args:
            config: Configuration object containing index parameters
            
        Returns:
            Base index instance of the specified type
            
        Raises:
            ValueError: If the index type is not supported
        """
        return super().get_instance(config)

    @classmethod
    def _create_vector_index(cls, config: VectorIndexConfig) -> VectorIndex:
        """
        Create a vector index instance.
        
        Args:
            config: Vector index configuration
            
        Returns:
            VectorIndex instance
        """
        return VectorIndex(config)

    @classmethod
    def _create_colbert_index(cls, config: ColBertIndexConfig) -> ColBertIndex:
        """
        Create a ColBERT index instance.
        
        Args:
            config: ColBERT index configuration
            
        Returns:
            ColBertIndex instance
        """
        return ColBertIndex(config)

    def _create_faiss_index(self, config: FAISSIndexConfig) -> FaissIndex:
        """
        Create a FAISS index instance.
        
        Args:
            config: FAISS index configuration
            
        Returns:
            FaissIndex instance
        """
        return FaissIndex(config)


# Global factory instance for convenience
get_index = RAGIndexFactory().get_index
