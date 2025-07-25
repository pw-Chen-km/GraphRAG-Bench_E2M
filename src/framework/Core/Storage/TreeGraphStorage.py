"""
Tree-based graph storage implementation.

Provides efficient storage and retrieval of hierarchical tree structures.
This implementation is optimized for storing tree-based knowledge representations.
"""

import os
import pickle
from typing import Any, Dict, List, Optional

from pydantic import Field

from Core.Common.Logger import logger
from Core.Schema.TreeSchema import TreeSchema, TreeNode
from Core.Storage.BaseGraphStorage import BaseGraphStorage


class TreeGraphStorage(BaseGraphStorage):
    """
    Tree-based graph storage implementation.
    
    This storage backend is specifically designed for storing hierarchical tree structures
    with efficient layer-based access and tree traversal capabilities. It supports
    multi-layer tree representations commonly used in knowledge organization.
    
    Attributes:
        name: Filename for tree data storage
        _tree: Internal tree schema instance
    """
    
    name: str = Field(default="tree_data.pkl", description="Tree data filename")
    tree: TreeSchema = Field(default_factory=TreeSchema, description="Internal tree schema")
    
    def clear(self) -> None:
        """Clear all tree data."""
        self.tree = TreeSchema()
    
    async def _persist(self, force: bool = False) -> bool:
        """
        Internal method to persist tree data.
        
        Args:
            force: Force persistence even if file exists
            
        Returns:
            True if successful, False otherwise
        """
        if os.path.exists(self._get_tree_file_path()) and not force:
            return True
        
        try:
            logger.info(f"Writing tree data to {self._get_tree_file_path()}")
            self._write_tree_data(self.tree, self._get_tree_file_path())
            return True
        except Exception as e:
            logger.error(f"Failed to persist tree data: {e}")
            return False
    
    @staticmethod
    def _write_tree_data(tree: TreeSchema, tree_file_path: str) -> None:
        """
        Write tree data to pickle file.
        
        Args:
            tree: Tree schema to write
            tree_file_path: File path for tree data
        """
        with open(tree_file_path, "wb") as file:
            pickle.dump(tree, file)
    
    async def load_tree_graph(self, force: bool = False) -> bool:
        """
        Load tree data from persistent storage.
        
        Args:
            force: Force reload even if data is already loaded
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Attempting to load tree data from: {self._get_tree_file_path()}")
        
        if not os.path.exists(self._get_tree_file_path()):
            logger.info("Tree data file does not exist, need to build tree from scratch")
            return False
        
        try:
            with open(self._get_tree_file_path(), "rb") as file:
                self.tree = pickle.load(file)
            
            logger.info(f"Successfully loaded tree with {len(self.tree.all_nodes)} nodes and {self.tree.num_layers} layers")
            return True
        except Exception as e:
            logger.error(f"Failed to load tree data: {e}")
            return False
    
    async def write_tree_leaves(self) -> bool:
        """
        Write tree leaves to separate file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._write_tree_data(self.tree, self._get_tree_leaves_file_path())
            return True
        except Exception as e:
            logger.error(f"Failed to write tree leaves: {e}")
            return False
    
    async def load_tree_graph_from_leaves(self, force: bool = False) -> bool:
        """
        Load tree data from leaves file.
        
        Args:
            force: Force reload even if data is already loaded
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Attempting to load tree leaves from: {self._get_tree_leaves_file_path()}")
        
        if not os.path.exists(self._get_tree_leaves_file_path()):
            logger.info("Tree leaves file does not exist, need to build tree from scratch")
            return False
        
        try:
            with open(self._get_tree_leaves_file_path(), "rb") as file:
                self.tree = pickle.load(file)
            
            logger.info(f"Successfully loaded tree leaves with {len(self.tree.leaf_nodes)} leaves")
            return True
        except Exception as e:
            logger.error(f"Failed to load tree leaves: {e}")
            return False
    
    @property
    def tree_schema(self) -> TreeSchema:
        """Get the tree schema instance."""
        return self.tree
    
    @property
    def root_nodes(self) -> List[TreeNode]:
        """Get all root nodes in the tree."""
        return self.tree.root_nodes
    
    @property
    def leaf_nodes(self) -> List[TreeNode]:
        """Get all leaf nodes in the tree."""
        return self.tree.leaf_nodes
    
    @property
    def num_layers(self) -> int:
        """Get the number of layers in the tree."""
        return self.tree.num_layers
    
    @property
    def num_nodes(self) -> int:
        """Get the total number of nodes in the tree."""
        return self.tree.num_nodes
    
    def add_layer(self) -> None:
        """Add a new layer to the tree."""
        if self.num_layers == 0:
            self.tree.layer_to_nodes = []
            self.tree.all_nodes = []
        self.tree.layer_to_nodes.append([])
    
    def get_layer(self, layer: int) -> List[TreeNode]:
        """
        Get all nodes in a specific layer.
        
        Args:
            layer: Layer index
            
        Returns:
            List of nodes in the specified layer
        """
        return self.tree.layer_to_nodes[layer]
    
    async def upsert_node(self, node_id: int, node_data: Dict[str, Any]) -> bool:
        """
        Insert or update a node in the tree.
        
        Args:
            node_id: The node identifier
            node_data: Node data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            node = TreeNode(
                index=node_id,
                text=node_data['text'],
                children=node_data['children'],
                embedding=node_data['embedding']
            )
            layer = node_data['layer']
            
            if layer >= len(self.tree.layer_to_nodes):
                self.tree.layer_to_nodes.extend([[] for _ in range(layer - len(self.tree.layer_to_nodes) + 1)])
            
            self.tree.layer_to_nodes[layer].append(node)
            self.tree.all_nodes.append(node)
            return True
        except Exception as e:
            logger.error(f"Failed to upsert node {node_id}: {e}")
            return False
    
    async def load_graph(self, force: bool = False) -> bool:
        """
        Load graph data from storage.
        
        Args:
            force: Force reload even if data is already loaded
            
        Returns:
            True if successful, False otherwise
        """
        return await self.load_tree_graph(force)
    
    async def persist(self, force: bool = False) -> bool:
        """
        Persist tree data to storage.
        
        Args:
            force: Force persistence even if data hasn't changed
            
        Returns:
            True if successful, False otherwise
        """
        return await self._persist(force)
    
    async def get_nodes_data(self) -> List[Dict[str, Any]]:
        """
        Get data for all nodes in the tree.
        
        Returns:
            List of node data dictionaries
        """
        return [{"content": node.text, "index": node.index} for node in self.tree.all_nodes]
    
    async def get_node_metadata(self) -> List[str]:
        """
        Get available metadata fields for nodes.
        
        Returns:
            List of metadata field names
        """
        return ["index"]
    
    async def get_node(self, node_id: int) -> Optional[TreeNode]:
        """
        Get a node by its index.
        
        Args:
            node_id: The node index
            
        Returns:
            The node if found, None otherwise
        """
        if 0 <= node_id < len(self.tree.all_nodes):
            return self.tree.all_nodes[node_id]
        return None
    
    @property
    def nodes(self) -> List[TreeNode]:
        """Get all nodes in the tree."""
        return self.tree.all_nodes
    
    async def neighbors(self, node: TreeNode) -> List[TreeNode]:
        """
        Get neighboring nodes for a given node.
        
        Args:
            node: The node to find neighbors for
            
        Returns:
            List of neighboring nodes
        """
        if not node.children:
            return []
        
        return [self.tree.all_nodes[node_idx] for node_idx in node.children if 0 <= node_idx < len(self.tree.all_nodes)]
    
    async def get_community_schema(self) -> Optional[Any]:
        """
        Get community schema for the tree.
        
        Returns:
            None (trees don't support community detection)
        """
        return None
    
    async def get_subgraph_metadata(self) -> Optional[List[str]]:
        """
        Get available metadata fields for subgraphs.
        
        Returns:
            None (trees don't support subgraph metadata)
        """
        return None
    
    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node exists in the tree.
        
        Args:
            node_id: The node identifier
            
        Returns:
            True if node exists, False otherwise
        """
        try:
            node_index = int(node_id)
            return 0 <= node_index < len(self.tree.all_nodes)
        except (ValueError, TypeError):
            return False
    
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Check if an edge exists between two nodes.
        
        Args:
            source_node_id: Source node identifier
            target_node_id: Target node identifier
            
        Returns:
            True if edge exists, False otherwise
        """
        try:
            source_index = int(source_node_id)
            target_index = int(target_node_id)
            
            if not (0 <= source_index < len(self.tree.all_nodes)):
                return False
            
            source_node = self.tree.all_nodes[source_index]
            return target_index in source_node.children
        except (ValueError, TypeError):
            return False
    
    async def node_degree(self, node_id: str) -> int:
        """
        Get the degree of a node.
        
        Args:
            node_id: The node identifier
            
        Returns:
            The degree of the node
        """
        try:
            node_index = int(node_id)
            if 0 <= node_index < len(self.tree.all_nodes):
                return len(self.tree.all_nodes[node_index].children)
            return 0
        except (ValueError, TypeError):
            return 0
    
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """
        Get the weight of an edge.
        
        Args:
            src_id: Source node identifier
            tgt_id: Target node identifier
            
        Returns:
            Edge weight (1 if edge exists, 0 otherwise)
        """
        return 1 if await self.has_edge(src_id, tgt_id) else 0
    
    async def get_node_edges(self, source_node_id: str) -> Optional[List[tuple[str, str]]]:
        """
        Get all edges from a source node.
        
        Args:
            source_node_id: Source node identifier
            
        Returns:
            List of (source, target) tuples or None if node not found
        """
        try:
            source_index = int(source_node_id)
            if not (0 <= source_index < len(self.tree.all_nodes)):
                return None
            
            source_node = self.tree.all_nodes[source_index]
            return [(source_node_id, str(child_index)) for child_index in source_node.children]
        except (ValueError, TypeError):
            return None
    
    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]) -> bool:
        """
        Insert or update an edge in the tree.
        
        Args:
            source_node_id: Source node identifier
            target_node_id: Target node identifier
            edge_data: Edge data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            source_index = int(source_node_id)
            target_index = int(target_node_id)
            
            if not (0 <= source_index < len(self.tree.all_nodes)):
                return False
            
            source_node = self.tree.all_nodes[source_index]
            if target_index not in source_node.children:
                source_node.children.append(target_index)
            
            return True
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to upsert edge: {e}")
            return False
    
    async def clustering(self, algorithm: str) -> Dict[str, Any]:
        """
        Perform tree clustering using specified algorithm.
        
        Args:
            algorithm: The clustering algorithm to use
            
        Returns:
            Clustering results dictionary
        """
        # Tree-based clustering is not implemented
        return {"algorithm": algorithm, "clusters": []}
    
    async def embed_nodes(self, algorithm: str) -> tuple:
        """
        Generate node embeddings using specified algorithm.
        
        Args:
            algorithm: The embedding algorithm to use
            
        Returns:
            Tuple of (embeddings array, node identifiers)
        """
        raise NotImplementedError("Node embedding is not supported for tree storage")
    
    async def initialize(self) -> bool:
        """
        Initialize the tree storage.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return await self.load_tree_graph()
        except Exception as e:
            logger.error(f"Failed to initialize tree storage: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """
        Clean up tree storage resources.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.tree = TreeSchema()
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup tree storage: {e}")
            return False
    
    def _get_tree_file_path(self) -> str:
        """Get the path for tree data file."""
        if self.namespace is None:
            raise ValueError("Namespace must be configured for tree storage")
        return self.namespace.get_save_path(self.name)
    
    def _get_tree_leaves_file_path(self) -> str:
        """Get the path for tree leaves file."""
        tree_path = self._get_tree_file_path()
        name, extension = os.path.splitext(tree_path)
        return f"{name}_leaves{extension}"
    
    def get_tree_info(self) -> Dict[str, Any]:
        """Get information about the tree storage configuration."""
        return {
            **self.get_graph_info(),
            "filename": self.name,
            "tree_file_path": self._get_tree_file_path(),
            "leaves_file_path": self._get_tree_leaves_file_path(),
            "num_nodes": self.num_nodes,
            "num_layers": self.num_layers,
            "root_nodes": len(self.root_nodes),
            "leaf_nodes": len(self.leaf_nodes)
        }
    
