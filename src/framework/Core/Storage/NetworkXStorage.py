"""
NetworkX-based graph storage implementation.

Provides efficient storage and retrieval of graph structures using NetworkX library.
This implementation supports complex graph operations, community detection, and
graph algorithms commonly used in knowledge graph applications.
"""

import html
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import networkx as nx
import numpy as np
from pydantic import Field, model_validator

from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Logger import logger
from Core.Schema.CommunitySchema import LeidenInfo
from Core.Storage.BaseGraphStorage import BaseGraphStorage


class NetworkXStorage(BaseGraphStorage):
    """
    NetworkX-based graph storage implementation.
    
    This storage backend uses NetworkX library for efficient graph operations including
    node/edge management, graph algorithms, community detection, and path finding.
    It supports both directed and undirected graphs with weighted edges.
    
    Attributes:
        name: GraphML filename for storage
        graph: Internal NetworkX graph instance
        node_embed_algorithms: Available node embedding algorithms
    """
    
    name: str = Field(default="nx_data.graphml", description="GraphML filename for storage")
    graph: nx.Graph = Field(default_factory=nx.Graph, description="Internal NetworkX graph")
    node_embed_algorithms: Dict[str, Any] = Field(default_factory=dict, description="Node embedding algorithms")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        self._register_embedding_algorithms()
    
    def _register_embedding_algorithms(self):
        """Register available node embedding algorithms."""
        self.node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }
    
    async def load_nx_graph(self) -> bool:
        """
        Load graph data from GraphML file.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Attempting to load graph from: {self._get_graphml_file_path()}")
        
        if not os.path.exists(self._get_graphml_file_path()):
            logger.info("GraphML file does not exist, need to build graph from scratch")
            return False
        
        try:
            self.graph = nx.read_graphml(self._get_graphml_file_path())
            logger.info(f"Successfully loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return True
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            return False
    
    @staticmethod
    def _write_nx_graph(graph: nx.Graph, file_path: str) -> None:
        """
        Write graph to GraphML file.
        
        Args:
            graph: NetworkX graph to write
            file_path: Output file path
        """
        logger.info(f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        nx.write_graphml(graph, file_path)
    
    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """
        Stabilize graph for consistent representation.
        
        Ensures an undirected graph with the same relationships will always be read
        the same way by sorting nodes and edges consistently.
        
        Args:
            graph: Input graph to stabilize
            
        Returns:
            Stabilized graph
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
        
        # Sort nodes consistently
        sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: x[0])
        fixed_graph.add_nodes_from(sorted_nodes)
        
        # Sort edges consistently
        edges = list(graph.edges(data=True))
        
        if not graph.is_directed():
            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    source, target = target, source
                return source, target, edge_data
            
            edges = [_sort_source_target(edge) for edge in edges]
        
        def _get_edge_key(edge):
            source, target, _ = edge
            return f"{source} -> {target}"
        
        edges = sorted(edges, key=_get_edge_key)
        fixed_graph.add_edges_from(edges)
        
        return fixed_graph
    
    async def load_graph(self, force: bool = False) -> bool:
        """
        Load graph data from storage.
        
        Args:
            force: Force reload even if data is already loaded
            
        Returns:
            True if successful, False otherwise
        """
        if force:
            logger.info("Force rebuilding the graph")
            return False
        return self.load_nx_graph()
    
    @property
    def graph_instance(self) -> nx.Graph:
        """Get the NetworkX graph instance."""
        return self.graph
    
    async def _persist(self, force: bool = False) -> bool:
        """
        Internal method to persist graph data.
        
        Args:
            force: Force persistence even if file exists
            
        Returns:
            True if successful, False otherwise
        """
        if os.path.exists(self._get_graphml_file_path()) and not force:
            return True
        
        try:
            logger.info(f"Writing graph to {self._get_graphml_file_path()}")
            self._write_nx_graph(self.graph, self._get_graphml_file_path())
            return True
        except Exception as e:
            logger.error(f"Failed to persist graph: {e}")
            return False
    
    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node exists in the graph.
        
        Args:
            node_id: The node identifier
            
        Returns:
            True if node exists, False otherwise
        """
        return self.graph.has_node(node_id)
    
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Check if an edge exists between two nodes.
        
        Args:
            source_node_id: Source node identifier
            target_node_id: Target node identifier
            
        Returns:
            True if edge exists, False otherwise
        """
        return self.graph.has_edge(source_node_id, target_node_id)
    
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve node data by identifier.
        
        Args:
            node_id: The node identifier
            
        Returns:
            Node data dictionary or None if not found
        """
        return self.graph.nodes.get(node_id)
    
    async def node_degree(self, node_id: str) -> int:
        """
        Get the degree of a node.
        
        Args:
            node_id: The node identifier
            
        Returns:
            The degree of the node
        """
        return self.graph.degree(node_id) if self.graph.has_node(node_id) else 0
    
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """
        Get the combined degree of two connected nodes.
        
        Args:
            src_id: Source node identifier
            tgt_id: Target node identifier
            
        Returns:
            Combined degree of both nodes
        """
        src_degree = self.graph.degree(src_id) if self.graph.has_node(src_id) else 0
        tgt_degree = self.graph.degree(tgt_id) if self.graph.has_node(tgt_id) else 0
        return src_degree + tgt_degree
    
    async def get_edge_weight(self, source_node_id: str, target_node_id: str) -> Optional[float]:
        """
        Get the weight of an edge.
        
        Args:
            source_node_id: Source node identifier
            target_node_id: Target node identifier
            
        Returns:
            Edge weight or None if edge doesn't exist
        """
        edge_data = self.graph.edges.get((source_node_id, target_node_id))
        return edge_data.get("weight") if edge_data is not None else None
    
    async def get_edge(self, source_node_id: str, target_node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve edge data between two nodes.
        
        Args:
            source_node_id: Source node identifier
            target_node_id: Target node identifier
            
        Returns:
            Edge data dictionary or None if not found
        """
        return self.graph.edges.get((source_node_id, target_node_id))
    
    async def get_node_edges(self, source_node_id: str) -> Optional[List[Tuple[str, str]]]:
        """
        Get all edges from a source node.
        
        Args:
            source_node_id: Source node identifier
            
        Returns:
            List of (source, target) tuples or None if node not found
        """
        if self.graph.has_node(source_node_id):
            return list(self.graph.edges(source_node_id))
        return None
    
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]) -> bool:
        """
        Insert or update a node in the graph.
        
        Args:
            node_id: The node identifier
            node_data: Node data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.graph.add_node(node_id, **node_data)
            return True
        except Exception as e:
            logger.error(f"Failed to upsert node {node_id}: {e}")
            return False
    
    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]) -> bool:
        """
        Insert or update an edge in the graph.
        
        Args:
            source_node_id: Source node identifier
            target_node_id: Target node identifier
            edge_data: Edge data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.graph.add_edge(source_node_id, target_node_id, **edge_data)
            return True
        except Exception as e:
            logger.error(f"Failed to upsert edge: {e}")
            return False
    
    async def _cluster_data_to_subgraphs(self, cluster_data: Dict[str, List[Dict[str, str]]]) -> bool:
        """
        Add cluster data to graph nodes.
        
        Args:
            cluster_data: Cluster data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for node_id, clusters in cluster_data.items():
                self.graph.nodes[node_id]["clusters"] = json.dumps(clusters)
            logger.info("Added cluster data to graph nodes")
            await self._persist(force=True)
            return True
        except Exception as e:
            logger.error(f"Failed to add cluster data: {e}")
            return False
    
    async def embed_nodes(self, algorithm: str) -> Tuple[np.ndarray, List[str]]:
        """
        Generate node embeddings using specified algorithm.
        
        Args:
            algorithm: The embedding algorithm to use
            
        Returns:
            Tuple of (embeddings array, node identifiers)
        """
        if algorithm not in self.node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self.node_embed_algorithms[algorithm]()
    
    async def _node2vec_embed(self) -> Tuple[np.ndarray, List[str]]:
        """
        Generate node2vec embeddings.
        
        Returns:
            Tuple of (embeddings array, node identifiers)
        """
        # Implementation would go here
        raise NotImplementedError("Node2vec embedding not implemented")
    
    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """
        Get the largest connected component with stable node ordering.
        
        Args:
            graph: Input graph
            
        Returns:
            Largest connected component graph
        """
        try:
            from graspologic.utils import largest_connected_component
            graph = graph.copy()
            graph = cast(nx.Graph, largest_connected_component(graph))
            node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}
            graph = nx.relabel_nodes(graph, node_mapping)
            return NetworkXStorage._stabilize_graph(graph)
        except ImportError:
            logger.warning("graspologic not available, using NetworkX largest_connected_components")
            largest_cc = max(nx.connected_components(graph), key=len)
            return graph.subgraph(largest_cc).copy()
    
    async def persist(self, force: bool = False) -> bool:
        """
        Persist graph data to storage.
        
        Args:
            force: Force persistence even if data hasn't changed
            
        Returns:
            True if successful, False otherwise
        """
        return await self._persist(force)
    
    async def get_nodes_data(self) -> List[Dict[str, Any]]:
        """
        Get data for all nodes in the graph.
        
        Returns:
            List of node data dictionaries
        """
        node_list = list(self.graph.nodes())
        
        async def get_node_data(node_id):
            node_data = await self.get_node(node_id) or {}
            node_data.setdefault("description", "")
            node_data.setdefault("entity_type", "")
            
            content_parts = []
            if "entity_name" in node_data:
                content_parts.append(node_data["entity_name"])
            
            if node_data.get("entity_type"):
                content_parts.append(f"({node_data['entity_type']})")
            
            if node_data.get("description"):
                content_parts.append(node_data["description"])
            
            return {
                "content": " ".join(content_parts),
                "index": node_id,
                **node_data
            }
        
        node_data_list = []
        for node_id in node_list:
            node_data = await get_node_data(node_id)
            node_data_list.append(node_data)
        return node_data_list
    
    async def get_edges_data(self, need_content: bool = True) -> List[Dict[str, Any]]:
        """
        Get data for all edges in the graph.
        
        Args:
            need_content: Whether to include content in edge data
            
        Returns:
            List of edge data dictionaries
        """
        edge_list = list(self.graph.edges())
        
        async def get_edge_data(edge):
            source, target = edge
            edge_data = await self.get_edge(source, target) or {}
            
            if need_content:
                source_node = await self.get_node(source) or {}
                target_node = await self.get_node(target) or {}
                
                content_parts = []
                if "relation_name" in edge_data:
                    content_parts.append(edge_data["relation_name"])
                
                source_name = source_node.get("entity_name", source)
                target_name = target_node.get("entity_name", target)
                content_parts.append(f"{source_name} -> {target_name}")
                
                edge_data["content"] = " ".join(content_parts)
            
            return {
                "source": source,
                "target": target,
                **edge_data
            }
        
        edge_data_list = []
        for edge in edge_list:
            edge_data = await get_edge_data(edge)
            edge_data_list.append(edge_data)
        return edge_data_list
    
    async def get_community_schema(self) -> Optional[LeidenInfo]:
        """
        Get community detection schema.
        
        Returns:
            Community schema information or None if not available
        """
        try:
            # Implementation would extract community information from graph
            return None
        except Exception as e:
            logger.error(f"Failed to get community schema: {e}")
            return None
    
    async def get_node_metadata(self) -> List[str]:
        """
        Get available metadata fields for nodes.
        
        Returns:
            List of metadata field names
        """
        if not self.graph.nodes():
            return []
        
        sample_node = next(iter(self.graph.nodes(data=True)))[1]
        return list(sample_node.keys())
    
    async def get_edge_metadata(self) -> List[str]:
        """
        Get available metadata fields for edges.
        
        Returns:
            List of metadata field names
        """
        if not self.graph.edges():
            return []
        
        sample_edge = next(iter(self.graph.edges(data=True)))[2]
        return list(sample_edge.keys())
    
    async def get_subgraph_metadata(self) -> List[str]:
        """
        Get available metadata fields for subgraphs.
        
        Returns:
            List of metadata field names
        """
        return ["clusters", "community_id"]
    
    def get_node_num(self) -> int:
        """Get the total number of nodes in the graph."""
        return self.graph.number_of_nodes()
    
    def get_edge_num(self) -> int:
        """Get the total number of edges in the graph."""
        return self.graph.number_of_edges()
    
    async def nodes(self) -> List[str]:
        """Get all node identifiers in the graph."""
        return list(self.graph.nodes())
    
    async def edges(self) -> List[Tuple[str, str]]:
        """Get all edge tuples in the graph."""
        return list(self.graph.edges())
    
    async def neighbors(self, node_id: str) -> List[str]:
        """
        Get neighboring nodes for a given node.
        
        Args:
            node_id: The node identifier
            
        Returns:
            List of neighboring node identifiers
        """
        return list(self.graph.neighbors(node_id))
    
    async def clustering(self, algorithm: str) -> Dict[str, Any]:
        """
        Perform graph clustering using specified algorithm.
        
        Args:
            algorithm: The clustering algorithm to use
            
        Returns:
            Clustering results dictionary
        """
        try:
            if algorithm == "leiden":
                import leidenalg as la
                import igraph as ig
                
                # Convert NetworkX graph to igraph
                ig_graph = ig.Graph.from_networkx(self.graph)
                partition = la.find_partition(ig_graph, la.ModularityVertexPartition)
                
                clusters = defaultdict(list)
                for i, cluster_id in enumerate(partition.membership):
                    node_id = ig_graph.vs[i]["_nx_name"]
                    clusters[cluster_id].append(node_id)
                
                return {
                    "algorithm": algorithm,
                    "clusters": dict(clusters),
                    "modularity": partition.quality() / ig_graph.ecount()
                }
            else:
                return {"algorithm": algorithm, "clusters": {}}
        except ImportError:
            logger.warning(f"Clustering algorithm {algorithm} not available")
            return {"algorithm": algorithm, "clusters": {}}
    
    async def initialize(self) -> bool:
        """
        Initialize the NetworkX storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return await self.load_graph()
        except Exception as e:
            logger.error(f"Failed to initialize NetworkX storage: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """
        Clean up NetworkX storage resources.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.graph.clear()
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup NetworkX storage: {e}")
            return False
    
    def _get_graphml_file_path(self) -> str:
        """Get the path for GraphML file."""
        if self.namespace is None:
            raise ValueError("Namespace must be configured for NetworkX storage")
        return self.namespace.get_save_path(self.name)
    
    def get_networkx_info(self) -> Dict[str, Any]:
        """Get information about the NetworkX storage configuration."""
        return {
            **self.get_graph_info(),
            "filename": self.name,
            "graphml_file_path": self._get_graphml_file_path(),
            "num_nodes": self.get_node_num(),
            "num_edges": self.get_edge_num(),
            "is_directed": self.graph.is_directed(),
            "is_weighted": any("weight" in edge_data for _, _, edge_data in self.graph.edges(data=True))
        }
