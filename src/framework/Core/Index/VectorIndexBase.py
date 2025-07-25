"""
Vector Index Base Module

This module provides a common base class for vector-based index implementations
that share similar functionality like document processing, embedding generation,
and retrieval operations.
"""

import asyncio
from typing import Any, List, Optional, Tuple
from abc import abstractmethod

from llama_index.core.schema import Document, QueryBundle
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser

from Core.Common.Utils import mdhash_id
from Core.Common.Logger import logger
from Core.Index.BaseIndex import BaseIndex, VectorIndexNodeResult, VectorIndexEdgeResult


class VectorIndexBase(BaseIndex):
    """
    Base class for vector-based index implementations.
    
    This class provides common functionality for vector indices including
    document processing, embedding generation, and retrieval operations.
    """
    
    def __init__(self, config):
        """
        Initialize the vector index base with configuration.
        
        Args:
            config: Configuration object containing index parameters
        """
        super().__init__(config)
        self.embedding_model = config.embed_model

    async def retrieval(self, query: str, top_k: Optional[int] = None) -> Any:
        """
        Retrieve documents from the vector index.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            Search results
        """
        if top_k is None:
            top_k = self._get_retrieve_top_k()
            
        retriever = self._index.as_retriever(
            similarity_top_k=top_k, 
            embed_model=self.config.embed_model
        )
        query_bundle = QueryBundle(query_str=query)
        
        return await retriever.aretrieve(query_bundle)

    async def retrieval_nodes(self, query: str, top_k: Optional[int], graph: Any, 
                            need_score: bool = False, tree_node: bool = False) -> Tuple[List[dict], Optional[List[float]]]:
        """
        Retrieve nodes from the graph based on vector similarity.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            graph: Graph instance to search in
            need_score: Whether to return similarity scores
            tree_node: Whether to return tree node format
            
        Returns:
            Tuple of (nodes, scores) where scores is optional
        """
        results = await self.retrieval(query, top_k)
        result = VectorIndexNodeResult(results)
        
        if tree_node:
            return await result.get_tree_node_data(graph, need_score)
        else:
            return await result.get_node_data(graph, need_score)

    async def retrieval_edges(self, query: str, top_k: Optional[int], graph: Any, 
                            need_score: bool = False) -> Tuple[List[dict], Optional[List[float]]]:
        """
        Retrieve edges from the graph based on vector similarity.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            graph: Graph instance to search in
            need_score: Whether to return similarity scores
            
        Returns:
            Tuple of (edges, scores) where scores is optional
        """
        results = await self.retrieval(query, top_k)
        result = VectorIndexEdgeResult(results)
        
        return await result.get_edge_data(graph, need_score)

    async def _update_index(self, datas: List[dict], meta_data: List[str]) -> None:
        """
        Update the vector index with new documents.
        
        Args:
            datas: List of data dictionaries
            meta_data: List of metadata keys to include
        """
        async def process_document(data: dict) -> Document:
            """Process a single data item into a Document."""
            return Document(
                doc_id=mdhash_id(data["content"]),
                text=data["content"],
                metadata={key: data[key] for key in meta_data},
                excluded_embed_metadata_keys=meta_data,
            )

        documents = await asyncio.gather(*[process_document(data) for data in datas])
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)
        
        self._index = VectorStoreIndex(nodes)
        logger.info(f"Updated index with {len(nodes)} nodes")

    async def _load_index(self) -> bool:
        """
        Load the vector index from storage.
        
        Returns:
            True if loading successful, False otherwise
        """
        try:
            Settings.embed_model = self.config.embed_model
            storage_context = StorageContext.from_defaults(persist_dir=self.config.persist_path)
            self._index = self._load_index_from_storage(storage_context)
            return True
        except Exception as e:
            logger.error(f"Loading index error: {e}")
            return False

    @abstractmethod
    def _load_index_from_storage(self, storage_context: StorageContext) -> Any:
        """
        Load index from storage context.
        
        Args:
            storage_context: Storage context containing the index
            
        Returns:
            Loaded index instance
        """
        pass

    def _get_retrieve_top_k(self) -> int:
        """
        Get the default number of top results to retrieve.
        
        Returns:
            Default top_k value
        """
        return getattr(self.config, 'retrieve_top_k', 10)

    def _storage_index(self) -> None:
        """Persist the vector index to storage."""
        self._index.storage_context.persist(persist_dir=self.config.persist_path)

    async def _similarity_score(self, object_q: Any, object_d: Any) -> float:
        """
        Calculate similarity score between two objects.
        
        For llama_index based vector databases, this is typically not needed.
        
        Args:
            object_q: Query object
            object_d: Document object
            
        Returns:
            Similarity score (default implementation returns 0)
        """
        # For llama_index based vector database, we do not need this now!
        return 0.0

    async def retrieval_batch(self, queries: List[str], top_k: Optional[int] = None) -> List[Any]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of top results to return per query
            
        Returns:
            List of search results for each query
        """
        # Default implementation - can be overridden by subclasses
        return await asyncio.gather(*[self.retrieval(query, top_k) for query in queries])

    async def upsert(self, data: dict[str, Any]) -> None:
        """
        Upsert a single document into the index.
        
        Args:
            data: Document data to upsert
        """
        # Default implementation - can be overridden by subclasses
        pass

    async def _update_index_from_documents(self, docs: List[Document]) -> None:
        """
        Update index from existing documents.
        
        Args:
            docs: List of documents to update
        """
        refreshed_docs = self._index.refresh_ref_docs(docs)
        refreshed_count = len([True for doc in refreshed_docs if doc])
        logger.info(f"Refreshed {refreshed_count} documents in index")

    @abstractmethod
    def _get_index(self) -> Any:
        """
        Get the underlying vector index instance.
        
        Returns:
            Vector index instance
        """
        pass 