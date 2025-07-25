"""
Subgraph retriever for retrieving relevant subgraphs from the knowledge graph.
Supports various retrieval methods including k-hop neighbors, induced subgraphs, and paths.
"""

from typing import List, Dict, Any, Optional, Set

from Core.Common.Logger import logger
from Core.Prompt import QueryPrompt
from Core.Retriever.BaseRetriever import BaseRetriever
from Core.Retriever.RetrieverMixin import RetrieverMixin
from Core.Retriever.RetrieverFactory import register_retriever_method


class SubgraphRetriever(BaseRetriever, RetrieverMixin):
    """
    Retriever for subgraphs with support for multiple retrieval strategies.
    
    Implements various methods to find relevant subgraphs based on nodes,
    paths, and graph structure.
    """

    def __init__(self, **kwargs):
        """
        Initialize the subgraph retriever.
        
        Args:
            **kwargs: Configuration and dependencies
        """
        config = kwargs.pop("config")
        super().__init__(config)
        self._mode_list = ["concatenate_information_return_list", "induced_subgraph_return_networkx", "k_hop_return_set", "paths_return_list", "neighbors_return_list"]
        self._type = "subgraph"
        
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

    @register_retriever_method(retriever_type="subgraph", method_name="concatenate_information_return_list")
    async def _find_relevant_subgraph_by_concatenate_information(self, seed: str) -> Optional[List[Dict]]:
        """
        Find relevant subgraphs using vector database concatenation.
        
        Args:
            seed: Query string
            
        Returns:
            List of relevant subgraphs or None if retrieval fails
        """
        try:
            if not self._validate_input(seed, "seed"):
                return None
                
            assert self.config.use_subgraphs_vdb
            
            subgraph_datas = await self.subgraphs_vdb.retrieval_subgraphs(
                seed, top_k=self.config.top_k, need_score=False
            )
            
            if not self._validate_and_log_result(subgraph_datas, "Subgraph retrieval"):
                return None

            return subgraph_datas
            
        except Exception as e:
            logger.exception(f"Failed to find relevant subgraph: {e}")
            return None

    @register_retriever_method(retriever_type="subgraph", method_name="k_hop_return_set")
    async def _find_subgraph_by_k_hop(self, seed: List[str], k: int) -> Optional[Set]:
        """
        Find subgraph using k-hop neighbors.
        
        Args:
            seed: List of seed nodes
            k: Number of hops
            
        Returns:
            Set of nodes in k-hop subgraph or None if retrieval fails
        """
        try:
            if not self._validate_input(seed, "seed"):
                return None
                
            subgraph_datas = await self.graph.find_k_hop_neighbors_batch(start_nodes=seed, k=k)
            
            if not self._validate_and_log_result(subgraph_datas, "K-hop subgraph retrieval"):
                return None

            return subgraph_datas
            
        except Exception as e:
            logger.exception(f"Failed to find relevant subgraph: {e}")
            return None

    @register_retriever_method(retriever_type="subgraph", method_name="induced_subgraph_return_networkx")
    def _find_subgraph_by_networkx(self, seed: List[str]) -> Optional[Any]:
        """
        Find induced subgraph using NetworkX.
        
        Args:
            seed: List of seed nodes
            
        Returns:
            NetworkX subgraph or None if retrieval fails
        """
        try:
            if not self._validate_input(seed, "seed"):
                return None
                
            subgraph_datas = self.graph.get_induced_subgraph(nodes=seed)
            return subgraph_datas
            
        except Exception as e:
            logger.exception(f"Failed to find relevant subgraph: {e}")
            return None

    @register_retriever_method(retriever_type="subgraph", method_name="paths_return_list")
    async def _find_subgraph_by_paths(self, seed: List[str], cutoff: int = 5) -> Optional[List]:
        """
        Find subgraph using paths from source nodes.
        
        Args:
            seed: List of source nodes
            cutoff: Maximum path length
            
        Returns:
            List of paths or None if retrieval fails
        """
        try:
            if not self._validate_input(seed, "seed"):
                return None
                
            path_datas = await self.graph.get_paths_from_sources(start_nodes=seed, cutoff=cutoff)
            return path_datas
            
        except Exception as e:
            logger.exception(f"Failed to find relevant paths: {e}")
            return None

    @register_retriever_method(retriever_type="subgraph", method_name="neighbors_return_list")
    async def _find_subgraph_by_neighbors(self, seed: List[str]) -> Optional[List]:
        """
        Find subgraph using neighbors of source nodes.
        
        Args:
            seed: List of source nodes
            
        Returns:
            List of neighbors or None if retrieval fails
        """
        try:
            if not self._validate_input(seed, "seed"):
                return None
                
            nei_datas = await self.graph.get_neighbors_from_sources(start_nodes=seed)
            return nei_datas
            
        except Exception as e:
            logger.exception(f"Failed to find relevant neighbors: {e}")
            return None