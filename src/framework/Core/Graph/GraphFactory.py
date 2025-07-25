"""
Graph factory for creating different types of graph implementations.
Provides a unified interface for instantiating various graph types based on configuration.
"""

from Core.Graph.BaseGraph import BaseGraph
from Core.Graph.ERGraph import ERGraph
from Core.Graph.PassageGraph import PassageGraph
from Core.Graph.RKGraph import RKGraph
from Core.Graph.TreeGraph import TreeGraph
from Core.Graph.TreeGraphBalanced import TreeGraphBalanced


class GraphFactory:
    """
    Factory class for creating different types of graph implementations.
    
    This factory provides a centralized way to instantiate various graph types
    based on configuration parameters, supporting entity-relationship graphs,
    tree graphs, passage graphs, and more.
    """

    def __init__(self):
        """Initialize the factory with graph type mappings."""
        self.creators = {
            "er_graph": self._create_entity_relationship_graph,
            "rkg_graph": self._create_relationship_knowledge_graph,
            "tree_graph": self._create_tree_graph,
            "tree_graph_balanced": self._create_balanced_tree_graph,
            "passage_graph": self._create_passage_graph
        }

    def get_graph(self, config, **kwargs) -> BaseGraph:
        """
        Create a graph instance based on configuration.
        
        Args:
            config: Configuration object containing graph parameters
            **kwargs: Additional keyword arguments for graph creation
            
        Returns:
            BaseGraph: Instance of the specified graph type
            
        Raises:
            KeyError: If the graph type is not supported
        """
        graph_type = config.graph.graph_type
        if graph_type not in self.creators:
            raise KeyError(f"Unsupported graph type: {graph_type}")
        
        return self.creators[graph_type](config, **kwargs)

    @staticmethod
    def _create_entity_relationship_graph(config, **kwargs):
        """Create an entity-relationship graph."""
        return ERGraph(config.graph, **kwargs)

    @staticmethod
    def _create_relationship_knowledge_graph(config, **kwargs):
        """Create a relationship knowledge graph."""
        return RKGraph(config.graph, **kwargs)

    @staticmethod
    def _create_tree_graph(config, **kwargs):
        """Create a tree graph."""
        return TreeGraph(config, **kwargs)

    @staticmethod
    def _create_balanced_tree_graph(config, **kwargs):
        """Create a balanced tree graph."""
        return TreeGraphBalanced(config, **kwargs)

    @staticmethod
    def _create_passage_graph(config, **kwargs):
        """Create a passage graph."""
        return PassageGraph(config.graph, **kwargs)


# Global factory instance for convenience
get_graph = GraphFactory().get_graph
