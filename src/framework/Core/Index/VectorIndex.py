"""
Vector Index Module

This module provides a simple vector index implementation using LlamaIndex.
It is designed to be lightweight and easy-to-use for ANN search operations.
"""

import asyncio
from typing import Any, List, Optional, Tuple
import numpy as np

from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, Settings
from llama_index.core.schema import QueryBundle

from Core.Index.VectorIndexBase import VectorIndexBase
from Core.Common.Logger import logger


class VectorIndex(VectorIndexBase):
    """
    Simple vector index implementation using LlamaIndex.
    
    This class provides a lightweight and straightforward vector database
    for approximate nearest neighbor (ANN) search operations.
    """

    def __init__(self, config):
        """
        Initialize the vector index with configuration.
        
        Args:
            config: Configuration object containing index parameters
        """
        super().__init__(config)

    def _get_index(self) -> VectorStoreIndex:
        """
        Get a new VectorStoreIndex instance.
        
        Returns:
            Empty VectorStoreIndex instance
        """
        Settings.embed_model = self.config.embed_model
        return VectorStoreIndex([])

    def _load_index_from_storage(self, storage_context: StorageContext) -> VectorStoreIndex:
        """
        Load VectorStoreIndex from storage context.
        
        Args:
            storage_context: Storage context containing the index
            
        Returns:
            Loaded VectorStoreIndex instance
        """
        return load_index_from_storage(storage_context)

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
