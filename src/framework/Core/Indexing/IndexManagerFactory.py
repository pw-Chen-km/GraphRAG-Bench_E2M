"""
Index manager factory for managing different index building strategies.
Provides a unified interface for various indexing approaches using factory and strategy patterns.
"""

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

from Core.Common.Logger import logger
from Core.Schema.VdbResult import VdbResult
from Core.Utils.Display import StatusDisplay


class IndexType(Enum):
    """Supported index types."""
    VECTOR = "vector"
    FAISS = "faiss"
    COLBERT = "colbert"
    TFIDF = "tfidf"


@dataclass
class IndexConfig:
    """Configuration for index building."""
    index_type: IndexType = IndexType.VECTOR
    persist_path: str = ""
    embed_model: Any = None
    dimension: int = 1536
    similarity_metric: str = "cosine"
    top_k: int = 10
    force_rebuild: bool = False


class IndexManager(ABC):
    """
    Abstract base class for index managers.
    
    Provides a unified interface for different index building strategies
    with common functionality for loading, saving, and searching.
    """
    
    def __init__(self, config: IndexConfig, context: Any):
        """Initialize index manager with configuration."""
        self.config = config
        self.context = context
        self.index = None
        self.embed_model = config.embed_model
    
    @abstractmethod
    async def execute(self, data: List[Any], metadata: List[Dict], force_rebuild: bool = False) -> Any:
        """Execute index building process."""
        pass
    
    @abstractmethod
    async def _build_index(self, data: List[Any], metadata: List[Dict]) -> Any:
        """Build the index from data and metadata."""
        pass
    
    @abstractmethod
    async def _load_index(self) -> bool:
        """Load existing index from storage."""
        pass
    
    @abstractmethod
    async def _save_index(self) -> None:
        """Save index to storage."""
        pass
    
    @abstractmethod
    async def search(self, query: str, top_k: int = None) -> List[VdbResult]:
        """Search the index with a query."""
        pass
    
    def get_index(self) -> Any:
        """Get the current index instance."""
        return self.index
    
    def exists(self) -> bool:
        """Check if index exists in storage."""
        import os
        return os.path.exists(self.config.persist_path)
    
    def get_entity_index(self) -> Any:
        """Get entity index for graph augmentation."""
        # Return the index itself for now
        # In practice, this might need to return a specific entity index
        return self.index


class VectorIndexManager(IndexManager):
    """Index manager for vector-based indexing."""
    
    async def execute(self, data: List[Any], metadata: List[Dict], force_rebuild: bool = False) -> Any:
        """Execute vector index building."""
        StatusDisplay.show_processing_status("Index building", details="Vector index")
        
        # Check if rebuild is needed
        if not force_rebuild and self.exists():
            StatusDisplay.show_info("Loading existing index")
            success = await self._load_index()
            if success:
                return self.index
        
        # Build new index
        self.index = await self._build_index(data, metadata)
        await self._save_index()
        
        StatusDisplay.show_success(f"Vector index building completed, contains {len(data)} entries")
        return self.index
    
    async def _build_index(self, data: List[Any], metadata: List[Dict]) -> Any:
        """Build vector index."""
        from llama_index.core import VectorStoreIndex, Document
        
        # Create document objects
        documents = []
        for i, (content, meta) in enumerate(zip(data, metadata)):
            if isinstance(content, str):
                doc = Document(text=content, metadata=meta)
            else:
                doc = Document(text=str(content), metadata=meta)
            documents.append(doc)
        
        # Create vector index
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=self.embed_model
        )
        
        return index
    
    async def _load_index(self) -> bool:
        """Load index from storage."""
        try:
            from llama_index.core import load_index_from_storage
            from llama_index.core import StorageContext
            
            storage_context = StorageContext.from_defaults(persist_dir=self.config.persist_path)
            self.index = load_index_from_storage(storage_context)
            return True
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            return False
    
    async def _save_index(self) -> None:
        """Save index to storage."""
        if self.index and self.config.persist_path:
            import os
            # Create directory if it doesn't exist
            os.makedirs(self.config.persist_path, exist_ok=True)
            self.index.storage_context.persist(persist_dir=self.config.persist_path)
    
    async def search(self, query: str, top_k: int = None) -> List[VdbResult]:
        """Search the index."""
        if not self.index:
            return []
        
        top_k = top_k or self.config.top_k
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        
        from llama_index.core.schema import QueryBundle
        query_bundle = QueryBundle(query_str=query)
        
        results = await retriever.aretrieve(query_bundle)
        
        # Convert to VdbResult format
        vdb_results = []
        for result in results:
            vdb_result = VdbResult(
                content=result.text,
                metadata=result.metadata,
                score=result.score if hasattr(result, 'score') else 0.0
            )
            vdb_results.append(vdb_result)
        
        return vdb_results


class FaissIndexManager(IndexManager):
    """Index manager for FAISS-based indexing."""
    
    async def execute(self, data: List[Any], metadata: List[Dict], force_rebuild: bool = False) -> Any:
        """Execute FAISS index building."""
        StatusDisplay.show_processing_status("Index building", details="FAISS index")
        
        # Check if rebuild is needed
        if not force_rebuild and self.exists():
            StatusDisplay.show_info("Loading existing index")
            success = await self._load_index()
            if success:
                return self.index
        
        # Build new index
        self.index = await self._build_index(data, metadata)
        await self._save_index()
        
        StatusDisplay.show_success(f"FAISS index building completed, contains {len(data)} entries")
        return self.index
    
    async def _build_index(self, data: List[Any], metadata: List[Dict]) -> Any:
        """Build FAISS index."""
        import faiss
        from llama_index.vector_stores.faiss import FaissVectorStore
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.core.schema import Document, TextNode
        from llama_index.core import Settings
        
        # Create document objects
        documents = []
        for i, (content, meta) in enumerate(zip(data, metadata)):
            if isinstance(content, str):
                doc = Document(text=content, metadata=meta)
            else:
                doc = Document(text=str(content), metadata=meta)
            documents.append(doc)
        
        # Handle case where embedding model is None (embeddings disabled)
        if self.embed_model is None:
            # Create a simple mock embedding model
            class MockEmbedding:
                def __init__(self, embed_dim=1024):
                    self.embed_dim = embed_dim
                
                def get_text_embedding(self, text):
                    import numpy as np
                    # Generate a simple random embedding
                    return np.random.rand(self.embed_dim).tolist()
            
            embed_model = MockEmbedding(embed_dim=1024)
            Settings.embed_model = embed_model
        else:
            Settings.embed_model = self.embed_model
        
        # Generate embeddings
        texts = [doc.text for doc in documents]
        if self.embed_model is None:
            # Use our simple mock embedding
            class MockEmbedding:
                def __init__(self, embed_dim=1024):
                    self.embed_dim = embed_dim
                
                def get_text_embedding(self, text):
                    import numpy as np
                    # Generate a simple random embedding
                    return np.random.rand(self.embed_dim).tolist()
            
            embed_model = MockEmbedding(embed_dim=1024)
            text_embeddings = [embed_model.get_text_embedding(text) for text in texts]
        else:
            text_embeddings = self.embed_model._get_text_embeddings(texts)
        
        # Create FAISS vector store with HNSW index
        vector_store = FaissVectorStore(faiss_index=faiss.IndexHNSWFlat(1024, 32))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create vector index
        index = VectorStoreIndex(
            [], 
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        
        # Insert nodes with pre-computed embeddings
        nodes = []
        for doc, embedding in zip(documents, text_embeddings):
            node = TextNode(text=doc.text, embedding=embedding, metadata=doc.metadata)
            nodes.append(node)
        index.insert_nodes(nodes)
        
        return index
    
    async def _load_index(self) -> bool:
        """Load index from storage."""
        try:
            from llama_index.core import load_index_from_storage
            from llama_index.core import StorageContext
            
            storage_context = StorageContext.from_defaults(persist_dir=self.config.persist_path)
            self.index = load_index_from_storage(storage_context)
            return True
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            return False
    
    async def _save_index(self) -> None:
        """Save index to storage."""
        if self.index and self.config.persist_path:
            import os
            # Create directory if it doesn't exist
            os.makedirs(self.config.persist_path, exist_ok=True)
            self.index.storage_context.persist(persist_dir=self.config.persist_path)
    
    async def search(self, query: str, top_k: int = None) -> List[VdbResult]:
        """Search the index."""
        if not self.index:
            return []
        
        top_k = top_k or self.config.top_k
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        
        from llama_index.core.schema import QueryBundle
        query_bundle = QueryBundle(query_str=query)
        
        results = await retriever.aretrieve(query_bundle)
        
        # Convert to VdbResult format
        vdb_results = []
        for result in results:
            vdb_result = VdbResult(
                content=result.text,
                metadata=result.metadata,
                score=result.score if hasattr(result, 'score') else 0.0
            )
            vdb_results.append(vdb_result)
        
        return vdb_results


class ColBertIndexManager(IndexManager):
    """Index manager for ColBERT-based indexing."""
    
    async def execute(self, data: List[Any], metadata: List[Dict], force_rebuild: bool = False) -> Any:
        """Execute ColBERT index building."""
        StatusDisplay.show_processing_status("Index building", details="ColBERT index")
        
        # Check if rebuild is needed
        if not force_rebuild and self.exists():
            StatusDisplay.show_info("Loading existing index")
            success = await self._load_index()
            if success:
                return self.index
        
        # Build new index
        self.index = await self._build_index(data, metadata)
        await self._save_index()
        
        StatusDisplay.show_success(f"ColBERT index building completed, contains {len(data)} entries")
        return self.index
    
    async def _build_index(self, data: List[Any], metadata: List[Dict]) -> Any:
        """Build ColBERT index."""
        # Implement ColBERT index building logic here
        # Can reference the original ColBertIndex implementation
        from Core.Index.ColBertIndex import ColBertIndex
        
        colbert_index = ColBertIndex(self.config)
        await colbert_index.build_index(data, metadata, force=True)
        
        return colbert_index
    
    async def _load_index(self) -> bool:
        """Load index from storage."""
        try:
            from Core.Index.ColBertIndex import ColBertIndex
            
            colbert_index = ColBertIndex(self.config)
            success = await colbert_index._load_index()
            if success:
                self.index = colbert_index
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            return False
    
    async def _save_index(self) -> None:
        """Save index to storage."""
        if self.index:
            self.index._storage_index()
    
    async def search(self, query: str, top_k: int = None) -> List[VdbResult]:
        """Search the index."""
        if not self.index:
            return []
        
        top_k = top_k or self.config.top_k
        results = await self.index.retrieval(query, top_k)
        
        # Convert to VdbResult format
        # ColBERT returns (node_idxs, ranks, scores) tuple
        vdb_results = []
        if results and len(results) >= 3:
            node_idxs, ranks, scores = results
            for i, node_idx in enumerate(node_idxs):
                # For ColBERT, we need to get the actual content from the index
                # This is a simplified conversion - in practice, you might need to
                # get the actual content from the ColBERT collection
                vdb_result = VdbResult(
                    content=f"node_{node_idx}",  # Placeholder content
                    metadata={"node_idx": node_idx, "rank": ranks[i] if i < len(ranks) else 0},
                    score=scores[i] if i < len(scores) else 0.0
                )
                vdb_results.append(vdb_result)
        
        return vdb_results


class IndexManagerFactory:
    """
    Factory class for creating index managers.
    
    Provides a centralized way to instantiate different index building
    strategies based on configuration parameters.
    """
    
    _managers = {
        IndexType.VECTOR: VectorIndexManager,
        IndexType.FAISS: FaissIndexManager,
        IndexType.COLBERT: ColBertIndexManager
    }
    
    @classmethod
    def create_manager(cls, config: Any, context: Any) -> IndexManager:
        """
        Create an index manager based on configuration.
        
        Args:
            config: Configuration object containing index parameters
            context: Context object for the manager
            
        Returns:
            IndexManager: Instance of the appropriate index manager
        """
        # Create embedding model using embedding factory
        embed_model = None
        try:
            from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
            embedding_factory = RAGEmbeddingFactory()
            embed_model = embedding_factory.get_rag_embedding(config=config)
        except Exception as e:
            logger.warning(f"Failed to create embedding model: {e}")
            # If embedding creation fails, embed_model will remain None
            # and the index manager will use mock embeddings
        
        # Create persist path based on working directory and index type
        persist_path = ""
        if hasattr(config, 'working_dir') and config.working_dir:
            import os
            persist_path = os.path.join(config.working_dir, "index", config.vdb_type)
        
        # Extract index configuration from config
        index_config = IndexConfig(
            index_type=IndexType(config.vdb_type),
            persist_path=persist_path,
            embed_model=embed_model,
            dimension=config.embedding.dimensions if hasattr(config, 'embedding') and hasattr(config.embedding, 'dimensions') else 1536,
            similarity_metric=config.similarity_metric if hasattr(config, 'similarity_metric') else "cosine",
            top_k=config.top_k if hasattr(config, 'top_k') else 10,
            force_rebuild=config.force_rebuild if hasattr(config, 'force_rebuild') else False
        )
        
        manager_class = cls._managers.get(index_config.index_type, VectorIndexManager)
        return manager_class(index_config, context)
    
    @classmethod
    def register_manager(cls, index_type: IndexType, manager_class: type):
        """
        Register a new index manager.
        
        Args:
            index_type: Index type to register
            manager_class: Class implementing the index manager
        """
        cls._managers[index_type] = manager_class
        logger.info(f"Registered new index manager: {index_type.value}")
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available index types."""
        return [index_type.value for index_type in cls._managers.keys()] 