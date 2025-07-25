"""
RetrieverMixin provides common functionality shared across different retriever types.
This mixin reduces code duplication by centralizing common operations and utilities.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import re
from collections import defaultdict

from Core.Common.Utils import truncate_list_by_token_size, split_string_by_multiple_delimiters
from Core.Common.Logger import logger
from Core.Common.Constants import GRAPH_FIELD_SEP


class RetrieverMixin:
    """
    Mixin class providing common functionality for retriever implementations.
    
    Contains shared methods for data processing, validation, and common
    retrieval operations used across different retriever types.
    """

    async def _process_text_units_from_entities(self, node_datas: List[Dict]) -> List[str]:
        """
        Extract and process text units from entity data.
        
        Args:
            node_datas: List of entity data dictionaries
            
        Returns:
            List of processed text units
        """
        if not self._validate_input(node_datas, "node_datas"):
            return []
            
        text_units = [
            split_string_by_multiple_delimiters(dp["source_id"], [GRAPH_FIELD_SEP])
            for dp in node_datas
        ]
        
        return text_units

    async def _get_one_hop_neighbors(self, node_datas: List[Dict]) -> Tuple[List[str], Dict[str, set]]:
        """
        Get one-hop neighbors for given nodes and create text unit lookup.
        
        Args:
            node_datas: List of entity data dictionaries
            
        Returns:
            Tuple of (neighbor nodes, text unit lookup dictionary)
        """
        edges = await self._safe_async_gather(
            [self.graph.get_node_edges(dp["entity_name"]) for dp in node_datas],
            "Failed to get node edges"
        )
        
        all_one_hop_nodes = set()
        for this_edges in edges:
            if not this_edges:
                continue
            all_one_hop_nodes.update([e[1] for e in this_edges])
            
        all_one_hop_nodes = list(all_one_hop_nodes)
        all_one_hop_nodes_data = await self._safe_async_gather(
            [self.graph.get_node(e) for e in all_one_hop_nodes],
            "Failed to get neighbor node data"
        )
        
        all_one_hop_text_units_lookup = {
            k: set(split_string_by_multiple_delimiters(v["source_id"], [GRAPH_FIELD_SEP]))
            for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
            if v is not None
        }
        
        return all_one_hop_nodes, all_one_hop_text_units_lookup

    async def _calculate_text_unit_relation_counts(
        self, 
        text_units: List[List[str]], 
        edges: List[List], 
        all_one_hop_text_units_lookup: Dict[str, set]
    ) -> Dict[str, Dict]:
        """
        Calculate relation counts for text units based on edge connections.
        
        Args:
            text_units: List of text unit lists for each entity
            edges: List of edge lists for each entity
            all_one_hop_text_units_lookup: Lookup dictionary for neighbor text units
            
        Returns:
            Dictionary mapping text unit IDs to their data and relation counts
        """
        all_text_units_lookup = {}
        
        for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
            for c_id in this_text_units:
                if c_id in all_text_units_lookup:
                    continue
                    
                relation_counts = 0
                for e in this_edges:
                    if (e[1] in all_one_hop_text_units_lookup and 
                        c_id in all_one_hop_text_units_lookup[e[1]]):
                        relation_counts += 1
                        
                all_text_units_lookup[c_id] = {
                    "data": await self.doc_chunk.get_data_by_key(c_id),
                    "order": index,
                    "relation_counts": relation_counts,
                }
                
        return all_text_units_lookup

    async def _process_and_rank_text_units(
        self, 
        text_units_lookup: Dict[str, Dict], 
        max_token_size: int,
        sort_key: str = "order"
    ) -> List[str]:
        """
        Process and rank text units based on specified criteria.
        
        Args:
            text_units_lookup: Dictionary of text unit data
            max_token_size: Maximum token size for truncation
            sort_key: Key to use for sorting ("order" or "relation_counts")
            
        Returns:
            List of processed text unit data
        """
        if any([v is None for v in text_units_lookup.values()]):
            logger.warning("Some text chunks are missing, storage may be damaged")
            
        all_text_units = [
            {"id": k, **v} for k, v in text_units_lookup.items() if v is not None
        ]
        
        if sort_key == "relation_counts":
            all_text_units = sorted(
                all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
            )
        else:
            all_text_units = sorted(all_text_units, key=lambda x: x["order"])
            
        all_text_units = truncate_list_by_token_size(
            all_text_units,
            key=lambda x: x["data"],
            max_token_size=max_token_size,
        )
        
        return [t["data"] for t in all_text_units]

    async def _process_ppr_results(
        self, 
        query: str, 
        seed_entities: List[Dict], 
        link_entity: bool = False
    ) -> np.ndarray:
        """
        Process Personalized PageRank results for entity ranking.
        
        Args:
            query: Input query string
            seed_entities: List of seed entities
            link_entity: Whether to link entities first
            
        Returns:
            Personalized PageRank scores
        """
        if link_entity:
            seed_entities = await self.link_query_entities(seed_entities)
            
        if len(seed_entities) == 0:
            return np.ones(self.graph.node_num) / self.graph.node_num
        else:
            return await self._run_personalized_pagerank(query, seed_entities)

    async def _get_top_k_results(
        self, 
        scores: np.ndarray, 
        top_k: int, 
        get_func: callable
    ) -> Tuple[List, np.ndarray]:
        """
        Get top-k results based on scores.
        
        Args:
            scores: Score array
            top_k: Number of top results to retrieve
            get_func: Function to retrieve data by indices
            
        Returns:
            Tuple of (top-k results, top-k scores)
        """
        sorted_indices = np.argsort(scores, kind='mergesort')[::-1]
        sorted_scores = scores[sorted_indices]
        top_k_results = await get_func(sorted_indices[:top_k])
        return top_k_results, sorted_scores[:top_k]

    async def _process_entity_data_with_degrees(self, node_datas: List[Dict]) -> List[Dict]:
        """
        Process entity data by adding degree information.
        
        Args:
            node_datas: List of entity data dictionaries
            
        Returns:
            Processed entity data with degree information
        """
        if not self._validate_input(node_datas, "node_datas"):
            return []
            
        node_degrees = await self._safe_async_gather(
            [self.graph.node_degree(node["entity_name"]) for node in node_datas],
            "Failed to get node degrees"
        )
        
        processed_data = []
        for n, d in zip(node_datas, node_degrees):
            if n is not None:
                # Ensure source_id is always present
                if 'source_id' not in n or not n['source_id']:
                    n['source_id'] = n.get('entity_name', 'unknown')
                
                processed_data.append({**n, "entity_name": n["entity_name"], "rank": d})
        
        return processed_data

    async def _extract_entities_from_relationships(self, seed: List[Dict]) -> List[str]:
        """
        Extract unique entity names from relationship data.
        
        Args:
            seed: List of relationship data dictionaries
            
        Returns:
            List of unique entity names
        """
        entity_names = set()
        for e in seed:
            entity_names.add(e["src_id"])
            entity_names.add(e["tgt_id"])
        return list(entity_names)

    def _parse_agent_scores(self, result: str, candidate_list: List[str]) -> List[float]:
        """
        Parse scores from agent response using regex.
        
        Args:
            result: Agent response string
            candidate_list: List of candidates to score
            
        Returns:
            List of parsed scores
        """
        scores = re.findall(r'\d+\.\d+', result)
        scores = [float(number) for number in scores]
        
        if len(scores) != len(candidate_list):
            logger.info("All entities are created with equal scores.")
            scores = [1 / len(candidate_list)] * len(candidate_list)
            
        return scores

    def _validate_and_log_result(self, result: Any, operation_name: str) -> bool:
        """
        Validate operation result and log appropriate messages.
        
        Args:
            result: Operation result to validate
            operation_name: Name of the operation for logging
            
        Returns:
            True if result is valid, False otherwise
        """
        if result is None:
            logger.warning(f"{operation_name} returned None")
            return False
        if isinstance(result, (list, tuple)) and len(result) == 0:
            logger.warning(f"{operation_name} returned empty result")
            return False
        return True

    async def _safe_operation(self, operation: callable, *args, error_msg: str = "Operation failed", **kwargs):
        """
        Safely execute an operation with error handling.
        
        Args:
            operation: Function to execute
            *args: Positional arguments for the operation
            error_msg: Error message for logging
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Operation result or None if failed
        """
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            logger.exception(f"{error_msg}: {e}")
            return None 