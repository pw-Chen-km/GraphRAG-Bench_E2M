from dataclasses import dataclass
from typing import List

from Core.Schema.EntityRelation import Entity, Relationship


@dataclass
class ERGraphSchema:
    """
    Entity-Relationship Graph Schema for representing knowledge graphs.
    
    Attributes:
        nodes: List of entities in the graph
        edges: List of relationships between entities
    """
    nodes: List[Entity]
    edges: List[Relationship]

    def get_entity_by_name(self, entity_name: str) -> Entity:
        """Get entity by its name."""
        for node in self.nodes:
            if node.entity_name == entity_name:
                return node
        raise ValueError(f"Entity '{entity_name}' not found")

    def get_relationships_by_entity(self, entity_name: str) -> List[Relationship]:
        """Get all relationships involving the specified entity."""
        return [
            edge for edge in self.edges 
            if edge.src_id == entity_name or edge.tgt_id == entity_name
        ]
