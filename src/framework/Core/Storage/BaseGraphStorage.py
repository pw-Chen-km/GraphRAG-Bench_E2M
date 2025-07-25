"""
Base graph storage abstraction.

Provides the foundational interface for graph storage implementations that handle
complex graph structures with nodes, edges, and graph algorithms in the GraphRAG system.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import Field

from Core.Storage.BaseStorage import BaseStorage


class BaseGraphStorage(BaseStorage):
    """
    Abstract base class for graph storage implementations.
    
    Graph storage provides efficient storage and retrieval of graph structures
    including nodes, edges, and graph algorithms. This abstraction supports
    various graph operations like traversal, clustering, and embedding.
    
    Attributes:
        directed: Whether the graph is directed
        weighted: Whether the graph supports edge weights
        allow_self_loops: Whether self-loops are allowed
        max_nodes: Maximum number of nodes allowed
        max_edges: Maximum number of edges allowed
    """
    
    directed: bool = Field(default=False, description="Whether the graph is directed")
    weighted: bool = Field(default=True, description="Whether the graph supports edge weights")
    allow_self_loops: bool = Field(default=False, description="Whether self-loops are allowed")
    max_nodes: int = Field(default=1000000, description="Maximum number of nodes allowed")
    max_edges: int = Field(default=10000000, description="Maximum number of edges allowed")
    
    @abstractmethod
    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node exists in the graph.
        
        Args:
            node_id: The node identifier
            
        Returns:
            True if node exists, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Check if an edge exists between two nodes.
        
        Args:
            source_node_id: Source node identifier
            target_node_id: Target node identifier
            
        Returns:
            True if edge exists, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    async def node_degree(self, node_id: str) -> int:
        """
        Get the degree of a node.
        
        Args:
            node_id: The node identifier
            
        Returns:
            The degree of the node
        """
        raise NotImplementedError
    
    @abstractmethod
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """
        Get the weight or multiplicity of an edge.
        
        Args:
            src_id: Source node identifier
            tgt_id: Target node identifier
            
        Returns:
            The edge weight or multiplicity
        """
        raise NotImplementedError
    
    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve node data by identifier.
        
        Args:
            node_id: The node identifier
            
        Returns:
            Node data dictionary or None if not found
        """
        raise NotImplementedError
    
    @abstractmethod
    async def get_edge(
        self, 
        source_node_id: str, 
        target_node_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve edge data between two nodes.
        
        Args:
            source_node_id: Source node identifier
            target_node_id: Target node identifier
            
        Returns:
            Edge data dictionary or None if not found
        """
        raise NotImplementedError
    
    @abstractmethod
    async def get_node_edges(
        self, 
        source_node_id: str
    ) -> Optional[List[Tuple[str, str]]]:
        """
        Get all edges from a source node.
        
        Args:
            source_node_id: Source node identifier
            
        Returns:
            List of (source, target) tuples or None if node not found
        """
        raise NotImplementedError
    
    @abstractmethod
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]) -> bool:
        """
        Insert or update a node in the graph.
        
        Args:
            node_id: The node identifier
            node_data: Node data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    async def upsert_edge(
        self, 
        source_node_id: str, 
        target_node_id: str, 
        edge_data: Dict[str, Any]
    ) -> bool:
        """
        Insert or update an edge in the graph.
        
        Args:
            source_node_id: Source node identifier
            target_node_id: Target node identifier
            edge_data: Edge data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    async def clustering(self, algorithm: str) -> Dict[str, Any]:
        """
        Perform graph clustering using specified algorithm.
        
        Args:
            algorithm: The clustering algorithm to use
            
        Returns:
            Clustering results dictionary
        """
        raise NotImplementedError
    
    @abstractmethod
    async def embed_nodes(self, algorithm: str) -> Tuple[np.ndarray, List[str]]:
        """
        Generate node embeddings using specified algorithm.
        
        Args:
            algorithm: The embedding algorithm to use
            
        Returns:
            Tuple of (embeddings array, node identifiers)
        """
        raise NotImplementedError("Node embedding is not implemented in this storage backend.")
    
    @abstractmethod
    async def persist(self, force: bool = False) -> bool:
        """
        Persist graph data to storage.
        
        Args:
            force: Force persistence even if data hasn't changed
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    async def load_graph(self, force: bool = False) -> bool:
        """
        Load graph data from storage.
        
        Args:
            force: Force reload even if data is already loaded
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    async def get_nodes_data(self) -> List[Dict[str, Any]]:
        """
        Get data for all nodes in the graph.
        
        Returns:
            List of node data dictionaries
        """
        raise NotImplementedError
    
    @abstractmethod
    async def get_node_metadata(self) -> List[str]:
        """
        Get available metadata fields for nodes.
        
        Returns:
            List of metadata field names
        """
        raise NotImplementedError
    
    @abstractmethod
    async def get_subgraph_metadata(self) -> Optional[List[str]]:
        """
        Get available metadata fields for subgraphs.
        
        Returns:
            List of metadata field names or None if not supported
        """
        raise NotImplementedError
    
    def validate_graph_constraints(self, num_nodes: int, num_edges: int) -> bool:
        """
        Validate graph size constraints.
        
        Args:
            num_nodes: Number of nodes
            num_edges: Number of edges
            
        Returns:
            True if constraints are satisfied, False otherwise
        """
        return num_nodes <= self.max_nodes and num_edges <= self.max_edges
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the graph storage configuration."""
        return {
            **self.get_storage_info(),
            "directed": self.directed,
            "weighted": self.weighted,
            "allow_self_loops": self.allow_self_loops,
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges
        }
