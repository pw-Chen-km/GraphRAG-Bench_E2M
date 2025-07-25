"""
Relationship retriever for retrieving relevant relationships from the knowledge graph.
Supports various retrieval methods including PPR, vector database, and agent-based retrieval.
"""

import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from Core.Common.Logger import logger
from Core.Common.Utils import truncate_list_by_token_size
from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Retriever.BaseRetriever import BaseRetriever
from Core.Retriever.RetrieverMixin import RetrieverMixin
from Core.Retriever.RetrieverFactory import register_retriever_method


class RelationshipRetriever(BaseRetriever, RetrieverMixin):
    """
    Retriever for relationships with support for multiple retrieval strategies.
    
    Implements various methods to find relevant relationships based on entities,
    queries, and graph structure.
    """

    def __init__(self, **kwargs):
        """
        Initialize the relationship retriever.
        
        Args:
            **kwargs: Configuration and dependencies
        """
        config = kwargs.pop("config")
        super().__init__(config)
        self._mode_list = ["entity_occurrence", "from_entity", "ppr", "vdb", "from_entity_by_agent", "get_all", "by_source&target"]
        self._type = "relationship"
        
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

    @register_retriever_method(retriever_type="relationship", method_name="ppr")
    async def _find_relevant_relationships_by_ppr(self, query: str, seed_entities: List[Dict], node_ppr_matrix: Optional[np.ndarray] = None) -> Optional[List[Dict]]:
        """
        Find relevant relationships using Personalized PageRank.
        
        Args:
            query: Input query string
            seed_entities: List of seed entities
            node_ppr_matrix: Pre-computed PPR matrix (optional)
            
        Returns:
            List of relevant relationships or None if retrieval fails
        """
        if not hasattr(self, '_entities_to_relationships') or self._entities_to_relationships is None:
            logger.warning("_entities_to_relationships not available for PPR relationship retrieval")
            return None
            
        entity_to_edge_mat = await self._entities_to_relationships.get()
        
        if entity_to_edge_mat is None:
            logger.warning("Entity-relationship matrix not available for PPR relationship retrieval")
            return None
        
        if node_ppr_matrix is None:
            node_ppr_matrix = await self._process_ppr_results(query, seed_entities)
            
        edge_prob_matrix = entity_to_edge_mat.T.dot(node_ppr_matrix)
        
        # Get top-k relationships
        return await self._get_top_k_results(
            edge_prob_matrix, 
            self.config.top_k, 
            lambda indices: self.graph.get_edge_by_indices(indices)
        )

    @register_retriever_method(retriever_type="relationship", method_name="vdb")
    async def _find_relevant_relations_vdb(self, seed: str, need_score: bool = False, need_context: bool = True, top_k: Optional[int] = None) -> Optional[List[Dict]]:
        """
        Find relevant relationships using vector database.
        
        Args:
            seed: Query string
            need_score: Whether to include scores in results
            need_context: Whether to construct relationship context
            top_k: Number of top results to retrieve
            
        Returns:
            List of relevant relationships or None if retrieval fails
        """
        try:
            if not self._validate_input(seed, "seed"):
                return None
                
            if top_k is None:
                top_k = self.config.top_k

            edge_datas = await self.relations_vdb.retrieval_edges(
                query=seed, top_k=top_k, graph=self.graph, need_score=need_score
            )

            if not self._validate_and_log_result(edge_datas, "Vector database retrieval"):
                return None

            if need_context:
                edge_datas = await self._construct_relationship_context(edge_datas)
                
            return edge_datas
            
        except Exception as e:
            logger.exception(f"Failed to find relevant relationships via vector database: {e}")
            return None

    @register_retriever_method(retriever_type="relationship", method_name="from_entity")
    async def _find_relevant_relationships_from_entities(self, seed: List[Dict]) -> Optional[List[Dict]]:
        """
        Find relevant relationships from entity data.
        
        Args:
            seed: List of entity data dictionaries
            
        Returns:
            List of relevant relationships or None if retrieval fails
        """
        if not self._validate_input(seed, "seed"):
            return None
            
        # Get all related edges for the entities
        all_related_edges = await self._safe_async_gather(
            [self.graph.get_node_edges(node["entity_name"]) for node in seed],
            "Failed to get node edges"
        )
        
        # Collect unique edges
        all_edges = set()
        for this_edges in all_related_edges:
            all_edges.update([tuple(sorted(e)) for e in this_edges])
        all_edges = list(all_edges)
        
        # Get edge data and degrees
        all_edges_pack = await self._safe_async_gather(
            [self.graph.get_edge(e[0], e[1]) for e in all_edges],
            "Failed to get edge data"
        )
        
        all_edges_degree = await self._safe_async_gather(
            [self.graph.edge_degree(e[0], e[1]) for e in all_edges],
            "Failed to get edge degrees"
        )
        
        # Process edge data
        all_edges_data = [
            {"src_tgt": k, "rank": d, **v}
            for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
            if v is not None
        ]
        
        # Sort and truncate
        all_edges_data = sorted(
            all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
        )
        
        return truncate_list_by_token_size(
            all_edges_data,
            key=lambda x: x["description"],
            max_token_size=self.config.max_token_for_local_context,
        )

    @register_retriever_method(retriever_type="relationship", method_name="from_entity_by_agent")
    async def _find_relevant_relations_by_entity_agent(
        self, 
        query: str, 
        entity: str, 
        pre_relations_name: Optional[List[str]] = None,
        pre_head: Optional[bool] = None, 
        width: int = 3
    ) -> Tuple[List[Dict], defaultdict]:
        """
        Use agent to select top-K relations based on query and entity.
        
        Args:
            query: Query string to process
            entity: Entity seed
            pre_relations_name: List of existing relation names
            pre_head: Whether pre-head relations exist
            width: Search width for agent
            
        Returns:
            Tuple of (relations, relations_dict)
        """
        try:
            from Core.Prompt.TogPrompt import extract_relation_prompt

            # Get relations from graph
            edges = await self.graph.get_node_edges(source_node_id=entity)
            relations_name_super_edge = await self.graph.get_edge_relation_name_batch(edges=edges)
            relations_name = list(map(lambda x: x.split(GRAPH_FIELD_SEP), relations_name_super_edge))

            # Build relations dictionary
            relations_dict = defaultdict(list)
            for index, edge in enumerate(edges):
                src, tar = edge[0], edge[1]
                for rel in relations_name[index]:
                    relations_dict[(src, rel)].append(tar)

            # Separate head and tail relations
            tail_relations = []
            head_relations = []
            for index, rels in enumerate(relations_name):
                if edges[index][0] == entity:
                    head_relations.extend(rels)  # head
                else:
                    tail_relations.extend(rels)  # tail

            # Filter out pre-existing relations
            if pre_relations_name:
                if pre_head:
                    tail_relations = list(set(tail_relations) - set(pre_relations_name))
                else:
                    head_relations = list(set(head_relations) - set(pre_relations_name))

            head_relations = list(set(head_relations))
            tail_relations = list(set(tail_relations))
            total_relations = head_relations + tail_relations
            total_relations.sort()  # Ensure consistent order in prompt

            head_relations = set(head_relations)

            # Use agent to select relations
            prompt = extract_relation_prompt % (str(width), str(width), str(width)) + query + '\nTopic Entity: ' + entity + '\nRelations: There are %s relations provided in total, seperated by ;.' % str(len(total_relations)) + '; '.join(total_relations) + ';' + "\nA: "

            result = await self.llm.aask(msg=[
                {"role": "system", "content": "You are an AI assistant that helps people find information."},
                {"role": "user", "content": prompt}
            ])

            # Parse agent response
            relations = self._parse_relation_scores(result, head_relations, entity)

            flag = len(relations) > 0
            if not flag:
                logger.info(f"No relations found by entity: {entity} and query: {query}")

            return (relations, relations_dict) if flag else ([], relations_dict)
            
        except Exception as e:
            logger.exception(f"Failed to find relevant relations by entity agent: {e}")
            return [], defaultdict(list)

    @register_retriever_method(retriever_type="relationship", method_name="get_all")
    async def _get_all_relationships(self) -> List[Dict]:
        """
        Get all relationships from the graph.
        
        Returns:
            List of all relationship data
        """
        return await self.graph.edges_data()

    @register_retriever_method(retriever_type="relationship", method_name="by_source&target")
    async def _get_relationships_by_source_target(self, seed: List[Tuple[str, str]]) -> List[str]:
        """
        Get relationship names by source and target pairs.
        
        Args:
            seed: List of (source, target) pairs
            
        Returns:
            List of relationship names
        """
        return await self.graph.get_edge_relation_name_batch(edges=seed)

    def _parse_relation_scores(self, result: str, head_relations: set, entity: str) -> List[Dict]:
        """
        Parse relation scores from agent response.
        
        Args:
            result: Agent response string
            head_relations: Set of head relations
            entity: Entity name
            
        Returns:
            List of parsed relations with scores
        """
        pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
        relations = []
        
        for match in re.finditer(pattern, result):
            relation = match.group("relation").strip()
            if ';' in relation:
                continue
                
            score = match.group("score")
            if not relation or not score:
                continue
                
            try:
                score = float(score)
            except ValueError:
                continue
                
            if relation in head_relations:
                relations.append({"entity": entity, "relation": relation, "score": score, "head": True})
            else:
                relations.append({"entity": entity, "relation": relation, "score": score, "head": False})
                
        return relations