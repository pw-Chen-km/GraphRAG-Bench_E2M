"""
Configuration for embedding models and API settings.
Defines parameters for different embedding providers and models.
"""

from enum import Enum
from typing import Optional

from pydantic import field_validator

from Core.Utils.YamlModel import YamlModel


class EmbeddingType(Enum):
    """Supported embedding API providers."""
    OPENAI = "openai"
    HF = "hf"
    OLLAMA = "ollama"


class EmbeddingConfig(YamlModel):
    """
    Configuration for embedding model settings and API parameters.
    
    This class manages all embedding-related configuration including
    API credentials, model specifications, and processing parameters.
    
    Attributes:
        api_type: Type of embedding API provider
        api_key: API key for authentication
        base_url: Base URL for API requests
        api_version: API version to use
        model: Name of the embedding model
        cache_folder: Directory for caching embeddings
        embed_batch_size: Batch size for embedding generation
        dimensions: Output dimension of the embedding model
    """

    api_type: Optional[EmbeddingType] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    model: Optional[str] = None
    cache_dir: Optional[str] = None
    cache_folder: Optional[str] = None
    embed_batch_size: Optional[int] = None
    dimensions: Optional[int] = None
    embedding_func_max_async: Optional[int] = None

    @field_validator("api_type", mode="before")
    @classmethod
    def validate_api_type(cls, value):
        """Validate and normalize API type value."""
        if value == "":
            return None
        return value
