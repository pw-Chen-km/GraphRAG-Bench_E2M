"""
Index Schema Module

This module provides Pydantic schema definitions for various index configurations.
It defines the structure and validation rules for different types of vector
and neural retrieval indices.
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Union

from llama_index.core.embeddings import BaseEmbedding
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class BaseIndexConfig(BaseModel):
    """
    Base configuration class for all index types.
    
    This class provides common configuration parameters that are shared
    across different index implementations. When adding new subconfigs,
    it is necessary to add the corresponding instance implementation
    in the index factories.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_config['protected_namespaces'] = ()
    
    persist_path: Union[str, Path] = Field(
        description="The directory path where index data will be saved."
    )


class VectorIndexConfig(BaseIndexConfig):
    """
    Configuration for vector-based index implementations.
    
    This configuration is used for traditional vector similarity search
    methods that rely on dense vector embeddings.
    """

    embed_model: BaseEmbedding = Field(
        default=None, 
        description="Embedding model for generating vector representations."
    )


class ColBertIndexConfig(BaseIndexConfig):
    """
    Configuration for ColBERT-based index implementations.
    
    ColBERT is a neural retrieval method that uses token-level encodings
    for zero-shot retrieval on out-of-domain datasets.
    """
    
    index_name: str = Field(
        default="", 
        description="The name identifier for the ColBERT index."
    )
    
    model_name: str = Field(
        default="colbert-ir/colbertv2.0", 
        description="The name of the ColBERT model to use."
    )
    
    nbits: int = Field(
        default=2, 
        description="Number of bits for vector quantization."
    )
    
    gpus: int = Field(
        default=0, 
        description="Number of GPUs to use for indexing operations."
    )
    
    ranks: int = Field(
        default=1, 
        description="Number of ranks for distributed indexing."
    )
    
    doc_maxlen: int = Field(
        default=120, 
        description="Maximum length of documents in tokens."
    )
    
    query_maxlen: int = Field(
        default=60, 
        description="Maximum length of queries in tokens."
    )
    
    kmeans_niters: int = Field(
        default=4, 
        description="Number of iterations for K-means clustering."
    )


class FAISSIndexConfig(VectorIndexConfig):
    """
    Configuration for FAISS-based index implementations.
    
    FAISS (Facebook AI Similarity Search) is a library for efficient
    similarity search and clustering of dense vectors.
    """
    
    # Inherits all fields from VectorIndexConfig
    # Additional FAISS-specific configuration can be added here
    pass
