from abc import ABC, abstractmethod
import asyncio
from typing import List, Tuple, Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class VdbResult:
    """
    Simple result class for vector database queries.
    
    Attributes:
        content: The content/text of the result
        metadata: Additional metadata for the result
        score: Similarity score for the result
    """
    content: str
    metadata: Dict[str, Any]
    score: float = 0.0


class BaseResult(ABC):
    """Base class for all result types."""
    
    @abstractmethod
    async def get_data(self, graph: Any, score: bool = False) -> Any:
        """Get data from the result."""
        pass


class EntityResult(BaseResult):
    """Base class for entity-related results."""
    
    @abstractmethod
    async def get_node_data(self, graph: Any, score: bool = False) -> Any:
        """Get node data from the result."""
        pass


class RelationResult(BaseResult):
    """Base class for relationship-related results."""
    
    @abstractmethod
    async def get_edge_data(self, graph: Any, score: bool = False) -> Any:
        """Get edge data from the result."""
        pass


class SubgraphResult(BaseResult):
    """Base class for subgraph-related results."""
    
    @abstractmethod
    async def get_subgraph_data(self, score: bool = False) -> Any:
        """Get subgraph data from the result."""
        pass


class ColbertNodeResult(EntityResult):
    """
    Result class for ColBERT node queries.
    
    Attributes:
        node_idxs: List of node indices
        ranks: List of rank scores
        scores: List of similarity scores
    """
    
    def __init__(self, node_idxs: List[int], ranks: List[int], scores: List[float]):
        self.node_idxs = node_idxs
        self.ranks = ranks
        self.scores = scores

    async def get_node_data(self, graph: Any, score: bool = False) -> Any:
        """Get node data using graph index lookup."""
        nodes = await asyncio.gather(
            *[graph.get_node_by_index(node_idx) for node_idx in self.node_idxs]
        )
        
        if score:
            return nodes, self.scores.copy()
        return nodes

    async def get_tree_node_data(self, graph: Any, score: bool = False) -> Any:
        """Get tree node data using graph node lookup."""
        nodes = await asyncio.gather(
            *[graph.get_node(node_idx) for node_idx in self.node_idxs]
        )
        
        if score:
            return nodes, self.scores.copy()
        return nodes

    async def get_data(self, graph: Any, score: bool = False) -> Any:
        """Get data using the default node data method."""
        return await self.get_node_data(graph, score)


class VectorIndexNodeResult(EntityResult):
    """
    Result class for vector index node queries.
    
    Attributes:
        results: List of search results with metadata
    """
    
    def __init__(self, results: List[Any]):
        self.results = results

    async def get_node_data(self, graph: Any, score: bool = False) -> Any:
        """Get node data using entity name from metadata."""
        nodes = await asyncio.gather(
            *[graph.get_node(r.metadata["entity_name"]) for r in self.results]
        )
        
        # Ensure all nodes have source_id
        processed_nodes = []
        for i, node in enumerate(nodes):
            if node is not None:
                # If source_id is missing, use entity_name as fallback
                if 'source_id' not in node or not node['source_id']:
                    node['source_id'] = node.get('entity_name', f'node_{i}')
                processed_nodes.append(node)
            else:
                # Create fallback node if graph.get_node returns None
                fallback_node = {
                    'entity_name': self.results[i].metadata.get("entity_name", f'node_{i}'),
                    'source_id': self.results[i].metadata.get("entity_name", f'node_{i}'),
                    'entity_type': '',
                    'description': ''
                }
                processed_nodes.append(fallback_node)
        
        if score:
            return processed_nodes, [r.score for r in self.results]
        return processed_nodes

    async def get_tree_node_data(self, graph: Any, score: bool = False) -> Any:
        """Get tree node data using entity metadata key."""
        nodes = await asyncio.gather(
            *[graph.get_node(r.metadata[graph.entity_metakey]) for r in self.results]
        )
        
        # Ensure all nodes have source_id
        processed_nodes = []
        for i, node in enumerate(nodes):
            if node is not None:
                # If source_id is missing, use entity_name as fallback
                if 'source_id' not in node or not node['source_id']:
                    node['source_id'] = node.get('entity_name', f'node_{i}')
                processed_nodes.append(node)
            else:
                # Create fallback node if graph.get_node returns None
                fallback_node = {
                    'entity_name': self.results[i].metadata.get(graph.entity_metakey, f'node_{i}'),
                    'source_id': self.results[i].metadata.get(graph.entity_metakey, f'node_{i}'),
                    'entity_type': '',
                    'description': ''
                }
                processed_nodes.append(fallback_node)
        
        if score:
            return processed_nodes, [r.score for r in self.results]
        return processed_nodes

    async def get_data(self, graph: Any, score: bool = False) -> Any:
        """Get data using the default node data method."""
        return await self.get_node_data(graph, score)


class VectorIndexEdgeResult(RelationResult):
    """
    Result class for vector index edge queries.
    
    Attributes:
        results: List of search results with metadata
    """
    
    def __init__(self, results: List[Any]):
        self.results = results

    async def get_edge_data(self, graph: Any, score: bool = False) -> Any:
        """Get edge data using source and target IDs from metadata."""
        edges = []
        for r in self.results:
            try:
                # Try to get src_id and tgt_id from metadata
                if "src_id" in r.metadata and "tgt_id" in r.metadata:
                    edge = await graph.get_edge(r.metadata["src_id"], r.metadata["tgt_id"])
                else:
                    # If metadata doesn't contain src_id/tgt_id, try to extract from content
                    # This is a fallback for cases where the edge data is stored differently
                    edge = None
                    if hasattr(r, 'text') and r.text:
                        # Try to parse edge information from text content
                        # This is a simplified approach - in practice, you might need more sophisticated parsing
                        edge = {"description": r.text, "content": r.text}
                
                # Ensure edge has source_id if it exists
                if edge is not None and 'source_id' not in edge:
                    edge['source_id'] = r.metadata.get("source_id", "unknown")
                
                edges.append(edge)
            except Exception as e:
                # If we can't get the edge, create a fallback edge with available information
                edge = {
                    "description": r.text if hasattr(r, 'text') else "Unknown edge",
                    "content": r.text if hasattr(r, 'text') else "Unknown edge",
                    "src_id": r.metadata.get("src_id", "unknown"),
                    "tgt_id": r.metadata.get("tgt_id", "unknown"),
                    "source_id": r.metadata.get("source_id", "unknown")
                }
                edges.append(edge)
        
        if score:
            return edges, [r.score for r in self.results]
        return edges

    async def get_data(self, graph: Any, score: bool = False) -> Any:
        """Get data using the edge data method."""
        return await self.get_edge_data(graph, score)


class VectorIndexSubgraphResult(SubgraphResult):
    """
    Result class for vector index subgraph queries.
    
    Attributes:
        results: List of search results with metadata
    """
    
    def __init__(self, results: List[Any]):
        self.results = results

    async def get_subgraph_data(self, score: bool = False) -> Any:
        """Get subgraph data from results."""
        subgraphs_data = []
        for r in self.results:
            # Ensure source_id is always present
            source_id = r.metadata.get("source_id", "")
            if not source_id:
                # Use a fallback source_id if not available
                source_id = r.metadata.get("entity_name", "unknown")
            
            subgraphs_data.append({
                "source_id": source_id, 
                "subgraph_content": r.text
            })
        
        if score:
            return subgraphs_data, [r.score for r in self.results]
        return subgraphs_data

    async def get_data(self, graph: Any, score: bool = False) -> Any:
        """Get data using the subgraph data method."""
        return await self.get_subgraph_data(score)


class ColbertEdgeResult(RelationResult):
    """
    Result class for ColBERT edge queries.
    
    Attributes:
        edge_idxs: List of edge indices
        ranks: List of rank scores
        scores: List of similarity scores
    """
    
    def __init__(self, edge_idxs: List[int], ranks: List[int], scores: List[float]):
        self.edge_idxs = edge_idxs
        self.ranks = ranks
        self.scores = scores

    async def get_edge_data(self, graph: Any, score: bool = False) -> Any:
        """Get edge data using graph index lookup."""
        if score:
            return await asyncio.gather(
                *[(graph.get_edge_by_index(edge_idx), self.scores[idx]) 
                  for idx, edge_idx in enumerate(self.edge_idxs)]
            )
        else:
            return await asyncio.gather(
                *[graph.get_edge_by_index(edge_idx) for edge_idx in self.edge_idxs]
            )

    async def get_data(self, graph: Any, score: bool = False) -> Any:
        """Get data using the edge data method."""
        return await self.get_edge_data(graph, score)