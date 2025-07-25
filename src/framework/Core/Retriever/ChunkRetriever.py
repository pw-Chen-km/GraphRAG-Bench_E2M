"""
Chunk retriever for retrieving relevant text chunks from the knowledge graph.
Supports various retrieval methods including entity occurrence, PPR, and relationship-based retrieval.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from Core.Common.Logger import logger
from Core.Common.Utils import min_max_normalize, to_str_by_maxtokens
from Core.Common.Constants import TOKEN_TO_CHAR_RATIO
from Core.Retriever.BaseRetriever import BaseRetriever
from Core.Retriever.RetrieverMixin import RetrieverMixin
from Core.Retriever.RetrieverFactory import register_retriever_method


class ChunkRetriever(BaseRetriever, RetrieverMixin):
    """
    Retriever for text chunks with support for multiple retrieval strategies.
    
    Implements various methods to find relevant text chunks based on entities,
    relationships, and graph structure.
    """

    def __init__(self, **kwargs):
        """
        Initialize the chunk retriever.
        
        Args:
            **kwargs: Configuration and dependencies
        """
        config = kwargs.pop("config")
        super().__init__(config)
        self._mode_list = ["entity_occurrence", "ppr", "from_relation", "aug_ppr"]
        self._type = "chunk"
        
        # Set additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def mode_list(self) -> List[str]:
        """List of supported retrieval modes."""
        return self._mode_list

    @property
    def type(self) -> str:
        """Type identifier for this retriever."""
        return self._type

    @register_retriever_method(retriever_type="chunk", method_name="entity_occurrence")
    async def _find_relevant_chunks_from_entity_occurrence(self, node_datas: List[Dict]) -> Optional[List[str]]:
        """
        Find relevant chunks based on entity occurrence patterns.
        
        Args:
            node_datas: List of entity data dictionaries
            
        Returns:
            List of relevant text chunks or None if no chunks found
        """
        if not self._validate_input(node_datas, "node_datas"):
            return None
            
        # Extract text units from entities
        text_units = await self._process_text_units_from_entities(node_datas)
        
        # Get one-hop neighbors and their text units
        all_one_hop_nodes, all_one_hop_text_units_lookup = await self._get_one_hop_neighbors(node_datas)
        
        # Calculate relation counts for text units
        edges = await self._safe_async_gather(
            [self.graph.get_node_edges(dp["entity_name"]) for dp in node_datas],
            "Failed to get node edges"
        )
        
        text_units_lookup = await self._calculate_text_unit_relation_counts(
            text_units, edges, all_one_hop_text_units_lookup
        )
        
        # Process and rank text units
        return await self._process_and_rank_text_units(
            text_units_lookup, 
            self.config.local_max_token_for_text_unit,
            sort_key="relation_counts"
        )

    @register_retriever_method(retriever_type="chunk", method_name="from_relation")
    async def _find_relevant_chunks_from_relationships(self, seed: List[Dict]) -> Optional[List[str]]:
        """
        Find relevant chunks based on relationship data.
        
        Args:
            seed: List of relationship data dictionaries
            
        Returns:
            List of relevant text chunks or None if no chunks found
        """
        if not self._validate_input(seed, "seed"):
            return None
            
        # Extract text units from relationship data
        text_units = await self._process_text_units_from_entities(seed)
        
        # Build text units lookup
        all_text_units_lookup = {}
        for index, unit_list in enumerate(text_units):
            for c_id in unit_list:
                if c_id not in all_text_units_lookup:
                    all_text_units_lookup[c_id] = {
                        "data": await self.doc_chunk.get_data_by_key(c_id),
                        "order": index,
                    }
        
        # Process and rank text units
        return await self._process_and_rank_text_units(
            all_text_units_lookup,
            self.config.local_max_token_for_text_unit,
            sort_key="order"
        )

    @register_retriever_method(retriever_type="chunk", method_name="ppr")
    async def _find_relevant_chunks_by_ppr(self, query: str, seed_entities: List[Dict], link_entity: bool = False) -> Optional[Tuple[List, np.ndarray]]:
        """
        Find relevant chunks using Personalized PageRank.
        
        Args:
            query: Input query string
            seed_entities: List of seed entities
            link_entity: Whether to link entities first
            
        Returns:
            Tuple of (chunks, scores) or None if retrieval fails
        """
        # Get PPR scores
        node_ppr_matrix = await self._process_ppr_results(query, seed_entities, link_entity)
        
        # Get matrices for chunk probability calculation
        if not hasattr(self, 'entities_to_relationships') or self.entities_to_relationships is None:
            logger.warning("entities_to_relationships not available for PPR chunk retrieval")
            return None
            
        if not hasattr(self, 'relationships_to_chunks') or self.relationships_to_chunks is None:
            logger.warning("relationships_to_chunks not available for PPR chunk retrieval")
            return None
            
        entity_to_edge_mat = await self.entities_to_relationships.get()
        relationship_to_chunk_mat = await self.relationships_to_chunks.get()
        
        if entity_to_edge_mat is None or relationship_to_chunk_mat is None:
            logger.warning("Entity-relationship matrices not available for PPR chunk retrieval")
            return None
        
        # Calculate chunk probabilities
        edge_prob = entity_to_edge_mat.T.dot(node_ppr_matrix)
        ppr_chunk_prob = relationship_to_chunk_mat.T.dot(edge_prob)
        ppr_chunk_prob = min_max_normalize(ppr_chunk_prob)
        
        # Get top-k chunks
        return await self._get_top_k_results(
            ppr_chunk_prob, 
            self.config.top_k, 
            lambda indices: self.doc_chunk.get_data_by_indices(indices)
        )

    @register_retriever_method(retriever_type="chunk", method_name="aug_ppr")
    async def _find_relevant_chunks_by_aug_ppr(self, query: str, seed_entities: List[Dict]) -> Optional[Dict[str, List]]:
        """
        Find relevant chunks, entities, and relationships using augmented PPR.
        
        Args:
            query: Input query string
            seed_entities: List of seed entities
            
        Returns:
            Dictionary containing chunks, entities, and relationships or None if retrieval fails
        """
        # Get PPR scores
        node_ppr_matrix = await self._process_ppr_results(query, seed_entities)
        
        # Get matrices for probability calculation
        if not hasattr(self, 'entities_to_relationships') or self.entities_to_relationships is None:
            logger.warning("entities_to_relationships not available for augmented PPR retrieval")
            return None
            
        if not hasattr(self, 'relationships_to_chunks') or self.relationships_to_chunks is None:
            logger.warning("relationships_to_chunks not available for augmented PPR retrieval")
            return None
            
        entity_to_edge_mat = await self.entities_to_relationships.get()
        relationship_to_chunk_mat = await self.relationships_to_chunks.get()
        
        if entity_to_edge_mat is None or relationship_to_chunk_mat is None:
            logger.warning("Entity-relationship matrices not available for augmented PPR retrieval")
            return None
        
        # Calculate probabilities for different types
        edge_prob = entity_to_edge_mat.T.dot(node_ppr_matrix)
        ppr_chunk_prob = relationship_to_chunk_mat.T.dot(edge_prob)
        
        # Get sorted indices for all types
        sorted_doc_ids = np.argsort(ppr_chunk_prob, kind='mergesort')[::-1]
        sorted_entity_ids = np.argsort(node_ppr_matrix, kind='mergesort')[::-1]
        sorted_relationship_ids = np.argsort(edge_prob, kind='mergesort')[::-1]
        
        # Retrieve data for all types
        sorted_docs = await self.doc_chunk.get_data_by_indices(sorted_doc_ids)
        sorted_entities = await self.graph.get_node_by_indices(sorted_entity_ids)
        sorted_relationships = await self.graph.get_edge_by_indices(sorted_relationship_ids)
        
        # Return formatted results
        return to_str_by_maxtokens(max_chars={
            "entities": self.config.entities_max_tokens * TOKEN_TO_CHAR_RATIO,
            "relationships": self.config.relationships_max_tokens * TOKEN_TO_CHAR_RATIO,
            "chunks": self.config.local_max_token_for_text_unit * TOKEN_TO_CHAR_RATIO,
        }, entities=sorted_entities, relationships=sorted_relationships, chunks=sorted_docs)
