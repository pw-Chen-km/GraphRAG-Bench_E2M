"""
Entity retriever for retrieving relevant entities from the knowledge graph.
Supports various retrieval methods including PPR, vector database, TF-IDF, and agent-based retrieval.
"""

import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from Core.Common.Logger import logger
from Core.Common.Utils import truncate_list_by_token_size
from Core.Index.TFIDFStore import TFIDFIndex
from Core.Retriever.BaseRetriever import BaseRetriever
from Core.Retriever.RetrieverMixin import RetrieverMixin
from Core.Retriever.RetrieverFactory import register_retriever_method


class EntityRetriever(BaseRetriever, RetrieverMixin):
    """
    Retriever for entities with support for multiple retrieval strategies.
    
    Implements various methods to find relevant entities based on queries,
    relationships, and graph structure.
    """

    def __init__(self, **kwargs):
        """
        Initialize the entity retriever.
        
        Args:
            **kwargs: Configuration and dependencies
        """
        config = kwargs.pop("config")
        super().__init__(config)
        self._mode_list = ["ppr", "vdb", "from_relation", "tf_df", "all", "by_neighbors", "link_entity", "get_all", "from_relation_by_agent"]
        self._type = "entity"
        
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

    @register_retriever_method(retriever_type="entity", method_name="ppr")
    async def _find_relevant_entities_by_ppr(self, query: str, seed_entities: List[Dict], link_entity: bool = False, top_k: Optional[int] = None, **kwargs) -> Optional[List[Dict]]:
        """
        Find relevant entities using Personalized PageRank.
        
        Args:
            query: Input query string
            seed_entities: List of seed entities
            link_entity: Whether to link entities first
            
        Returns:
            List of entities with PPR scores or None if retrieval fails
        """
        if not self._validate_input(seed_entities, "seed_entities"):
            return None
            
        # Get PPR scores
        ppr_node_matrix = await self._process_ppr_results(query, seed_entities, link_entity)
        
        # Get top-k entities
        if top_k is None:
            top_k = self.config.top_k
            
        entities, scores = await self._get_top_k_results(
            ppr_node_matrix, 
            top_k, 
            lambda indices: self.graph.get_node_by_indices(indices)
        )
        
        # Format results with PPR scores
        if entities and scores is not None:
            return [
                {**entity, "ppr_score": float(score)} 
                for entity, score in zip(entities, scores)
                if entity is not None
            ]
        
        return []

    @register_retriever_method(retriever_type="entity", method_name="vdb")
    async def _find_relevant_entities_vdb(self, seed: str, tree_node: bool = False, top_k: Optional[int] = None) -> Optional[List[Dict]]:
        """
        Find relevant entities using vector database.
        
        Args:
            seed: Query string or entity name
            tree_node: Whether to return tree node format
            top_k: Number of top results to retrieve
            
        Returns:
            List of relevant entities or None if retrieval fails
        """
        try:
            if top_k is None:
                top_k = self.config.top_k
                
            node_datas = await self.entities_vdb.retrieval_nodes(
                query=seed, top_k=top_k, graph=self.graph, tree_node=tree_node
            )

            if not self._validate_and_log_result(node_datas, "Vector database retrieval"):
                return None
                
            if tree_node:
                try:
                    node_datas = [node.text for node in node_datas]
                    return node_datas
                except:
                    logger.warning("Only tree graph supports this! Please check the 'graph_type' item in your config file")
                    return None
                    
            # Process entity data with degree information
            return await self._process_entity_data_with_degrees(node_datas)
            
        except Exception as e:
            logger.exception(f"Failed to find relevant entities via vector database: {e}")
            return None

    @register_retriever_method(retriever_type="entity", method_name="tf_df")
    async def _find_relevant_entities_tf_df(self, seed: str, corpus: Optional[Dict] = None, top_k: Optional[int] = None, candidates_idx: Optional[List] = None) -> Optional[Tuple[List, List]]:
        """
        Find relevant entities using TF-IDF scoring.
        
        Args:
            seed: Query string
            corpus: Corpus dictionary (optional)
            top_k: Number of top results to retrieve
            candidates_idx: Candidate indices (optional)
            
        Returns:
            Tuple of (contexts, indices) or None if retrieval fails
        """
        try:
            # Build corpus from graph nodes if not provided
            if corpus is None or candidates_idx is None:
                graph_nodes = list(self.graph.nodes())
                node_data = await self._safe_async_gather(
                    [self.graph.get_node(id) for id in graph_nodes],
                    "Failed to get node data"
                )
                corpus = {id: node['description'] for id, node in zip(graph_nodes, node_data) if node is not None}
                candidates_idx = list(id for id in graph_nodes)
                
            # Build TF-IDF index and query
            index = TFIDFIndex()
            index._build_index_from_list([corpus[_] for _ in candidates_idx])
            idxs = index.query(query_str=seed, top_k=top_k)

            # Return results
            new_candidates_idx = [candidates_idx[_] for _ in idxs]
            cur_contexts = [corpus[_] for _ in new_candidates_idx]
            return cur_contexts, new_candidates_idx

        except Exception as e:
            logger.exception(f"Failed to find relevant entities via TF-IDF: {e}")
            return None

    @register_retriever_method(retriever_type="entity", method_name="all")
    async def _find_relevant_entities_all(self, key: str) -> Tuple[Dict, List]:
        """
        Get all entities with specified key.
        
        Args:
            key: Key to extract from entity data
            
        Returns:
            Tuple of (corpus, candidate indices)
        """
        graph_nodes = list(self.graph.nodes())
        node_data = await self._safe_async_gather(
            [self.graph.get_node(id) for id in graph_nodes],
            "Failed to get node data"
        )
        corpus = {id: node[key] for id, node in zip(graph_nodes, node_data) if node is not None}
        candidates_idx = list(id for id in graph_nodes)
        return corpus, candidates_idx

    @register_retriever_method(retriever_type="entity", method_name="link_entity")
    async def _link_entities(self, query_entities: List[str]) -> List[Dict]:
        """
        Link query entities to graph entities using vector similarity.
        
        Args:
            query_entities: List of query entity strings
            
        Returns:
            List of linked entities
        """
        entities = await self._safe_async_gather(
            [self.entities_vdb.retrieval_nodes(query_entity, top_k=1, graph=self.graph) for query_entity in query_entities],
            "Failed to link entities"
        )
        return list(map(lambda x: x[0], entities))

    @register_retriever_method(retriever_type="entity", method_name="get_all")
    async def _get_all_entities(self) -> List[Dict]:
        """
        Get all entities from the graph.
        
        Returns:
            List of all entity data
        """
        # GraphWrapper doesn't have nodes_data method, use nodes() instead
        nodes = self.graph.nodes()
        node_data = await self._safe_async_gather(
            [self.graph.get_node(node_id) for node_id in nodes],
            "Failed to get node data"
        )
        return node_data

    @register_retriever_method(retriever_type="entity", method_name="from_relation_by_agent")
    async def _find_relevant_entities_by_relationships_agent(
        self, 
        query: str, 
        total_entity_relation_list: List[Dict],
        total_relations_dict: defaultdict,
        width: int = 3
    ) -> Tuple[bool, List]:
        """
        Use agent to select top-K relations based on query and entities.
        
        Args:
            query: Query string to process
            total_entity_relation_list: List of entity-relation pairs with scores
            total_relations_dict: Dictionary mapping (src, rel) to targets
            width: Search width for agent
            
        Returns:
            Tuple of (success_flag, filtered_results)
        """
        try:
            from Core.Prompt.TogPrompt import score_entity_candidates_prompt
            
            total_candidates = []
            total_scores = []
            total_relations = []
            total_topic_entities = []
            total_head = []

            # Process each entity-relation pair
            for index, entity in enumerate(total_entity_relation_list):
                candidate_list = total_relations_dict[(entity["entity"], entity["relation"])]

                # Score candidate entities
                if len(candidate_list) == 1:
                    scores = [entity["score"]]
                elif len(candidate_list) == 0:
                    scores = [0.0]
                else:
                    # Use agent to score candidates
                    prompt = score_entity_candidates_prompt.format(query, entity["relation"]) + '; '.join(candidate_list) + ';' + '\nScore: '
                    result = await self.llm.aask(msg=[{"role": "user", "content": prompt}])
                    scores = self._parse_agent_scores(result, candidate_list)

                # Update collections
                if len(candidate_list) == 0:
                    candidate_list.append("[FINISH]")
                    
                candidates_relation = [entity['relation']] * len(candidate_list)
                topic_entities = [entity['entity']] * len(candidate_list)
                head_num = [entity['head']] * len(candidate_list)
                
                total_candidates.extend(candidate_list)
                total_scores.extend(scores)
                total_relations.extend(candidates_relation)
                total_topic_entities.extend(topic_entities)
                total_head.extend(head_num)

            # Prune entities according to width
            zipped = list(zip(total_relations, total_candidates, total_topic_entities, total_head, total_scores))
            sorted_zipped = sorted(zipped, key=lambda x: x[4], reverse=True)
            
            relations = list(map(lambda x: x[0], sorted_zipped))[:width]
            candidates = list(map(lambda x: x[1], sorted_zipped))[:width]
            topics = list(map(lambda x: x[2], sorted_zipped))[:width]
            heads = list(map(lambda x: x[3], sorted_zipped))[:width]
            scores = list(map(lambda x: x[4], sorted_zipped))[:width]

            # Merge and filter results
            merged_list = list(zip(relations, candidates, topics, heads, scores))
            filtered_list = [(rel, ent, top, hea, score) for rel, ent, top, hea, score in merged_list if score != 0]
            
            return len(filtered_list) > 0, filtered_list
            
        except Exception as e:
            logger.exception(f"Failed to find relevant entities by relation agent: {e}")
            return False, []

    @register_retriever_method(retriever_type="entity", method_name="from_relation")
    async def _find_relevant_entities_by_relationships(self, seed: List[Dict]) -> Optional[List[Dict]]:
        """
        Find relevant entities from relationship data.
        
        Args:
            seed: List of relationship data dictionaries
            
        Returns:
            List of relevant entities or None if retrieval fails
        """
        # Extract unique entity names
        entity_names = await self._extract_entities_from_relationships(seed)
        
        # Get entity data and degrees
        node_datas = await self._safe_async_gather(
            [self.graph.get_node(entity_name) for entity_name in entity_names],
            "Failed to get entity data"
        )
        
        node_degrees = await self._safe_async_gather(
            [self.graph.node_degree(entity_name) for entity_name in entity_names],
            "Failed to get entity degrees"
        )
        
        # Process entity data
        processed_data = []
        for k, n, d in zip(entity_names, node_datas, node_degrees):
            if n is not None:
                if "description" not in n:
                    n['description'] = ""
                # Ensure source_id is always present
                if 'source_id' not in n or not n['source_id']:
                    n['source_id'] = k
                processed_data.append({**n, "entity_name": k, "rank": d})
            else:
                # Create fallback node if graph.get_node returns None
                fallback_node = {
                    'entity_name': k,
                    'source_id': k,
                    'entity_type': '',
                    'description': '',
                    'rank': d
                }
                processed_data.append(fallback_node)
        
        # Truncate and return
        return truncate_list_by_token_size(
            processed_data,
            key=lambda x: x["description"],
            max_token_size=self.config.max_token_for_local_context,
        )

    @register_retriever_method(retriever_type="entity", method_name="by_neighbors")
    async def _find_relevant_entities_by_neighbor(self, seed: str) -> List[str]:
        """
        Find entities by getting neighbors of a seed entity.
        
        Args:
            seed: Seed entity name
            
        Returns:
            List of neighbor entities
        """
        return list(await self.graph.get_neighbors(seed))