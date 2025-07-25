"""
RAG Index Module

This module provides a comprehensive set of index implementations for
Retrieval-Augmented Generation (RAG) systems, including vector-based,
neural, and traditional text-based retrieval methods.
"""

from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Index.IndexFactory import get_index
from Core.Index.IndexConfigFactory import get_index_config

# Export base classes for extensibility
from Core.Index.BaseIndex import BaseIndex
from Core.Index.VectorIndexBase import VectorIndexBase

# Export concrete implementations
from Core.Index.VectorIndex import VectorIndex
from Core.Index.FaissIndex import FaissIndex
from Core.Index.ColBertIndex import ColBertIndex
from Core.Index.TFIDFStore import TFIDFIndex
from Core.Index.ColBertStore import ColbertIndex

# Export configuration schemas
from Core.Index.Schema import (
    BaseIndexConfig,
    VectorIndexConfig,
    ColBertIndexConfig,
    FAISSIndexConfig
)

__all__ = [
    # Factory functions
    "get_rag_embedding",
    "get_index", 
    "get_index_config",
    
    # Base classes
    "BaseIndex",
    "VectorIndexBase",
    
    # Concrete implementations
    "VectorIndex",
    "FaissIndex", 
    "ColBertIndex",
    "TFIDFIndex",
    "ColbertIndex",
    
    # Configuration schemas
    "BaseIndexConfig",
    "VectorIndexConfig",
    "ColBertIndexConfig", 
    "FAISSIndexConfig"
]