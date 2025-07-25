"""
Base Index Module

This module provides the abstract base class for all index implementations.
It defines the common interface and shared functionality for vector database operations.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
import asyncio
import numpy as np

from Core.Common.Utils import clean_storage
from Core.Common.Logger import logger
from Core.Schema.VdbResult import *


class BaseIndex(ABC):
    """
    Abstract base class for all index implementations.
    
    This class provides a unified interface for different vector database
    implementations, including common operations like building, loading,
    and retrieving from indices.
    """
    
    def __init__(self, config):
        """
        Initialize the base index with configuration.
        
        Args:
            config: Configuration object containing index parameters
        """
        self.config = config
        self._index = None

    async def build_index(self, elements: List[dict], meta_data: List[str], force: bool = False) -> None:
        """
        Build or load the index from the given elements.
        
        Args:
            elements: List of data elements to index
            meta_data: List of metadata keys to include
            force: Whether to force rebuild the index
        """
        logger.info("Starting to build index for the given graph elements")
 
        from_load = False
        if self.exist_index() and not force:
            logger.info(f"Loading index from file: {self.config.persist_path}")
            from_load = await self._load_index()
        else:
            self._index = self._get_index()
            
        if not from_load:
            # Note: When successfully loading from file, rebuilding is not needed
            await self.clean_index()
            logger.info("Building index for input elements")
            await self._update_index(elements, meta_data)
            self._storage_index()
            logger.info("Index successfully built and stored.")
            
        logger.info("✅ Finished building index for graph elements")

    def exist_index(self) -> bool:
        """
        Check if the index already exists on disk.
        
        Returns:
            True if index exists, False otherwise
        """
        return os.path.exists(self.config.persist_path)

    @abstractmethod
    async def retrieval(self, query: str, top_k: Optional[int] = None) -> Any:
        """
        Retrieve documents from the index based on query.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            Search results
        """
        pass

    @abstractmethod
    async def retrieval_batch(self, queries: List[str], top_k: Optional[int] = None) -> List[Any]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of top results to return per query
            
        Returns:
            List of search results for each query
        """
        pass

    @abstractmethod
    def _get_index(self) -> Any:
        """
        Get the underlying index instance.
        
        Returns:
            Index instance
        """
        pass

    @abstractmethod
    async def _update_index(self, elements: List[dict], meta_data: List[str]) -> None:
        """
        Update the index with new elements.
        
        Args:
            elements: List of data elements to index
            meta_data: List of metadata keys to include
        """
        pass

    @abstractmethod
    def _get_retrieve_top_k(self) -> int:
        """
        Get the default number of top results to retrieve.
        
        Returns:
            Default top_k value
        """
        return 10

    @abstractmethod
    def _storage_index(self) -> None:
        """Persist the index to storage."""
        pass

    @abstractmethod
    async def _load_index(self) -> bool:
        """
        Load the index from storage.
        
        Returns:
            True if loading successful, False otherwise
        """
        pass

    @abstractmethod
    async def retrieval_nodes(self, query: str, top_k: Optional[int], graph: Any, 
                            need_score: bool = False, tree_node: bool = False) -> Tuple[List[dict], Optional[List[float]]]:
        """
        Retrieve nodes from the graph based on query.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            graph: Graph instance to search in
            need_score: Whether to return similarity scores
            tree_node: Whether to return tree node format
            
        Returns:
            Tuple of (nodes, scores) where scores is optional
        """
        pass

    async def similarity_score(self, object_q: Any, object_d: Any) -> float:
        """
        Calculate similarity score between two objects.
        
        Args:
            object_q: Query object
            object_d: Document object
            
        Returns:
            Similarity score
        """
        return await self._similarity_score(object_q, object_d)

    @abstractmethod
    async def _similarity_score(self, object_q: Any, object_d: Any) -> float:
        """
        Abstract method for calculating similarity score.
        
        Args:
            object_q: Query object
            object_d: Document object
            
        Returns:
            Similarity score
        """
        pass

    async def get_max_score(self, query: Any) -> float:
        """
        Get the maximum possible score for a query.
        
        Args:
            query: Query object
            
        Returns:
            Maximum score
        """
        pass

    async def clean_index(self) -> None:
        """Clean the index storage directory."""
        clean_storage(self.config.persist_path)

    async def retrieval_nodes_with_score_matrix(self, query_list: List[str], top_k: int, graph: Any) -> np.ndarray:
        """
        Retrieve nodes and return a score matrix for multiple queries.
        
        Args:
            query_list: List of search queries
            top_k: Number of top results to return per query
            graph: Graph instance to search in
            
        Returns:
            Score matrix where each row represents scores for one query
        """
        if isinstance(query_list, str):
            query_list = [query_list]
            
        results = await asyncio.gather(
            *[self.retrieval_nodes(query, top_k, graph, need_score=True) for query in query_list]
        )
        
        reset_prob_matrix = np.zeros((len(query_list), graph.node_num))
        entity_indices = []
        scores = []

        async def set_idx_score(idx: int, res: Tuple[List[dict], List[float]]) -> None:
            """Set index and score for a single result."""
            for entity, score in zip(res[0], res[1]):
                entity_indices.append(await graph.get_node_index(entity["entity_name"]))
                scores.append(score)

        await asyncio.gather(*[set_idx_score(idx, res) for idx, res in enumerate(results)])
        reset_prob_matrix[np.arange(len(query_list)).reshape(-1, 1), entity_indices] = scores
        all_entity_weights = reset_prob_matrix.max(axis=0)  # (1, #all_entities)

        # Normalize the scores
        all_entity_weights /= all_entity_weights.sum()
        return all_entity_weights