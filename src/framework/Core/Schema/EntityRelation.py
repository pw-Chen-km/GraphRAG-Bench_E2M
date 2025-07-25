from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Entity:
    """
    Represents an entity in the knowledge graph.
    
    Attributes:
        entity_name: Primary key for the entity
        source_id: Unique identifier of the source document
        entity_type: Type classification of the entity
        description: Description or summary of the entity
    """
    entity_name: str
    source_id: str
    entity_type: str = field(default="")
    description: str = field(default="")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entity to a dictionary representation."""
        return {
            'entity_name': self.entity_name,
            'source_id': self.source_id,
            'entity_type': self.entity_type,
            'description': self.description
        }


@dataclass
class Relationship:
    """
    Represents a relationship between two entities in the knowledge graph.
    
    Attributes:
        src_id: Source entity identifier
        tgt_id: Target entity identifier
        source_id: Unique identifier of the source document
        relation_name: Name or type of the relationship
        weight: Weight score for the relationship (used in GraphRAG and LightRAG)
        description: Description of the relationship
        keywords: Keywords associated with the relationship (used in LightRAG)
        rank: Ranking score of the relationship (used in LightRAG)
    """
    src_id: str
    tgt_id: str
    source_id: str
    relation_name: str = field(default="")
    weight: float = field(default=0.0)
    description: str = field(default="")
    keywords: str = field(default="")
    rank: int = field(default=0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the relationship to a dictionary representation."""
        return {
            'src_id': self.src_id,
            'tgt_id': self.tgt_id,
            'source_id': self.source_id,
            'relation_name': self.relation_name,
            'weight': self.weight,
            'description': self.description,
            'keywords': self.keywords,
            'rank': self.rank
        }
