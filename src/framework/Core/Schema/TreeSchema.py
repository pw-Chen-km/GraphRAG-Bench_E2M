from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class TreeNode:
    """
    Represents a node in a hierarchical tree structure.
    
    Attributes:
        text: Text content of the node
        index: Unique identifier for the node
        children: Set of child node indices
        embedding: Vector embedding of the node
    """
    text: str
    index: int
    children: Set[int]
    embedding: Any

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node (no children)."""
        return len(self.children) == 0

    def is_root(self, parent_dict: Dict[int, Set[int]]) -> bool:
        """Check if the node is a root node (no parents)."""
        return len(parent_dict.get(self.index, set())) == 0


@dataclass
class TreeSchema:
    """
    Schema for representing hierarchical tree structures.
    
    Attributes:
        all_nodes: List of all nodes in the tree
        layer_to_nodes: List of nodes organized by layers
    """
    all_nodes: List[TreeNode] = field(default_factory=list)
    layer_to_nodes: List[List[TreeNode]] = field(default_factory=list)

    @property
    def num_layers(self) -> int:
        """Get the number of layers in the tree."""
        return len(self.layer_to_nodes)

    @property
    def num_nodes(self) -> int:
        """Get the total number of nodes in the tree."""
        return len(self.all_nodes)

    @property
    def leaf_nodes(self) -> Optional[List[TreeNode]]:
        """Get all leaf nodes (nodes in the first layer)."""
        if self.num_layers == 0:
            return None
        return self.layer_to_nodes[0]

    @property
    def root_nodes(self) -> Optional[List[TreeNode]]:
        """Get all root nodes (nodes in the last layer)."""
        if self.num_layers == 0:
            return None
        return self.layer_to_nodes[-1]

    def find_all_descendants(self, node_index: int) -> Set[int]:
        """
        Recursively find all descendant nodes of a given node.
        
        Args:
            node_index: Index of the node to find descendants for
            
        Returns:
            Set of descendant node indices
        """
        descendants = set()
        node = self._get_node_by_index(node_index)
        
        if node:
            for child_index in node.children:
                descendants.add(child_index)
                descendants.update(self.find_all_descendants(child_index))
        
        return descendants

    def _get_node_by_index(self, node_index: int) -> Optional[TreeNode]:
        """Get a node by its index."""
        for node in self.all_nodes:
            if node.index == node_index:
                return node
        return None

    def _build_parent_dict(self) -> Dict[int, Set[int]]:
        """Build a dictionary mapping node indices to their parent indices."""
        parent_dict = {node.index: set() for node in self.all_nodes}
        
        for node in self.all_nodes:
            for child_index in node.children:
                parent_dict[child_index].add(node.index)
        
        return parent_dict

    def is_subtree_isolated(self, node_index: int, parent_dict: Dict[int, Set[int]]) -> bool:
        """
        Check if a subtree rooted at the given node is isolated.
        
        Args:
            node_index: Index of the root node of the subtree
            parent_dict: Dictionary mapping nodes to their parents
            
        Returns:
            True if the subtree is isolated, False otherwise
        """
        descendants = self.find_all_descendants(node_index)
        all_nodes_in_subtree = {node_index} | descendants
        
        for node_idx in all_nodes_in_subtree:
            parents = parent_dict.get(node_idx, set())
            if parents and not parents.issubset(all_nodes_in_subtree):
                return False
        
        return True

    def get_isolated_ratio(self) -> float:
        """
        Calculate the ratio of isolated nodes and subtrees to total nodes.
        
        Returns:
            Ratio of isolated nodes to total nodes (0.0 to 1.0)
        """
        if self.num_nodes == 0:
            return 0.0

        parent_dict = self._build_parent_dict()
        isolated_nodes = set()

        # Find isolated nodes (nodes with no children and no parents)
        for node in self.all_nodes:
            if node.is_leaf() and node.is_root(parent_dict):
                isolated_nodes.add(node.index)

        # Find isolated subtrees
        for node in self.all_nodes:
            if (node.index not in isolated_nodes and 
                self.is_subtree_isolated(node.index, parent_dict)):
                descendants = self.find_all_descendants(node.index)
                isolated_nodes.add(node.index)
                isolated_nodes.update(descendants)

        return len(isolated_nodes) / self.num_nodes
