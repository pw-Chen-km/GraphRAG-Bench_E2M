"""
RAG Embedding Factory Module

This module provides a factory for creating different types of embedding models
for RAG (Retrieval-Augmented Generation) systems. It supports various embedding
providers including OpenAI, Ollama, and HuggingFace models.

Reference: https://github.com/geekan/MetaGPT/blob/main/metagpt/rag/factories/embedding.py
"""

from __future__ import annotations

from typing import Any, Optional

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    print("HuggingFaceEmbedding not found, please install it by `pip install huggingface_hub`")
    HuggingFaceEmbedding = None

from Config.EmbConfig import EmbeddingType
from Config.LLMConfig import LLMType
from Core.Common.BaseFactory import GenericFactory
from Option.merged_config import MergedConfig as Config


class RAGEmbeddingFactory(GenericFactory):
    """
    Factory for creating LlamaIndex embedding models with MetaGPT's embedding configuration.
    
    This factory provides a centralized way to instantiate various embedding models
    based on configuration parameters, supporting different embedding providers
    and model types.
    """

    def __init__(self):
        """
        Initialize the embedding factory with supported embedding types.
        """
        creators = {
            EmbeddingType.OPENAI: self._create_openai,
            EmbeddingType.OLLAMA: self._create_ollama,
            EmbeddingType.HF: self._create_hf,
        }
        super().__init__(creators)

    def get_rag_embedding(self, key: Optional[EmbeddingType] = None, config: Optional[Config] = None) -> BaseEmbedding:
        """
        Get an embedding model instance based on configuration.
        
        Args:
            key: Embedding type identifier
            config: Configuration object containing embedding parameters
            
        Returns:
            BaseEmbedding instance of the specified type
            
        Raises:
            ValueError: If the embedding type is not supported
        """
        embedding_type = key or self._resolve_embedding_type(config)
        return super().get_instance(embedding_type, config=config)

    @staticmethod
    def _resolve_embedding_type(config: Config) -> EmbeddingType | LLMType:
        """
        Resolve the embedding type from configuration.
        
        If the embedding type is not specified, for backward compatibility,
        it checks if the LLM API type is either OPENAI or AZURE.
        
        Args:
            config: Configuration object
            
        Returns:
            Resolved embedding type
            
        Raises:
            TypeError: If embedding type cannot be resolved
        """
        if config.embedding.api_type:
            return config.embedding.api_type

        raise TypeError("To use RAG, please set your embedding in merged_config.yaml.")

    def _create_openai(self, config: Config) -> OpenAIEmbedding:
        """
        Create an OpenAI embedding model instance.

        Args:
            config: Configuration object containing OpenAI parameters

        Returns:
            OpenAIEmbedding instance
        """
        openai_compatible_llms = {
            LLMType.OPENAI,
            LLMType.FIREWORKS,
            LLMType.OPEN_LLM,
        }

        shared_api_key = None
        shared_base_url = None

        if config.llm.api_type in openai_compatible_llms:
            shared_api_key = config.llm.api_key
            shared_base_url = config.llm.base_url

        api_key = config.embedding.api_key or shared_api_key
        api_base = (
            config.embedding.base_url
            or shared_base_url
            or "https://api.openai.com/v1"
        )

        if not api_key:
            raise ValueError(
                "OpenAI embeddings require an API key. "
                "Set embedding.api_key or use an OpenAI-compatible LLM configuration."
            )

        params = dict(
            api_key=api_key,
            api_base=api_base,
        )

        self._try_set_model_and_batch_size(params, config)
        return OpenAIEmbedding(**params)

    def _create_ollama(self, config: Config) -> OllamaEmbedding:
        """
        Create an Ollama embedding model instance.
        
        Args:
            config: Configuration object containing Ollama parameters
            
        Returns:
            OllamaEmbedding instance
        """
        params = dict(
            base_url=config.embedding.base_url,
        )

        self._try_set_model_and_batch_size(params, config)
        return OllamaEmbedding(**params)

    def _create_hf(self, config: Config) -> HuggingFaceEmbedding:
        """
        Create a HuggingFace embedding model instance.
        
        Args:
            config: Configuration object containing HuggingFace parameters
            
        Returns:
            HuggingFaceEmbedding instance
            
        Raises:
            ImportError: If HuggingFaceEmbedding is not available
        """
        if HuggingFaceEmbedding is None:
            raise ImportError("HuggingFaceEmbedding not available. Please install huggingface_hub.")
        
        # For huggingface-hub embedding model, we only need to set the model_name
        params = dict(
            model_name=config.embedding.model,
            cache_folder=config.embedding.cache_folder,
            device="cuda",
            target_devices=["cuda:7"],
            embed_batch_size=128,
        )
        
        if config.embedding.cache_folder == "":
            del params["cache_folder"]
            
        return HuggingFaceEmbedding(**params)

    @staticmethod
    def _try_set_model_and_batch_size(params: dict, config: Config) -> None:
        """
        Set model name and batch size parameters if specified in config.
        
        Args:
            params: Dictionary of parameters to update
            config: Configuration object containing model parameters
        """
        if config.embedding.model:
            params["model_name"] = config.embedding.model

        if config.embedding.embed_batch_size:
            params["embed_batch_size"] = config.embedding.embed_batch_size

        if config.embedding.dimensions:
            params["dimensions"] = config.embedding.dimensions

    def _raise_for_key(self, key: Any) -> None:
        """
        Raise an error for unsupported embedding types.
        
        Args:
            key: Unsupported embedding type
            
        Raises:
            ValueError: With descriptive error message
        """
        raise ValueError(f"The embedding type is currently not supported: `{type(key)}`, {key}")


# Global factory instance for convenience
get_rag_embedding = RAGEmbeddingFactory().get_rag_embedding