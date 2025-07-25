from dataclasses import dataclass, field
from typing import Set, List, Dict, Any


@dataclass
class CommunityReportsResult:
    """
    Container for community analysis results.
    
    Attributes:
        report_string: String representation of the report
        report_json: JSON representation of the report
    """
    report_string: str
    report_json: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary representation."""
        return {
            'report_string': self.report_string,
            'report_json': self.report_json
        }


@dataclass
class LeidenInfo:
    """
    Represents community information from Leiden algorithm.
    
    Attributes:
        level: Hierarchical level of the community
        title: Title or name of the community
        edges: Set of edge identifiers in the community
        nodes: Set of node identifiers in the community
        chunk_ids: Set of chunk identifiers in the community
        occurrence: Frequency or occurrence score
        sub_communities: List of sub-community identifiers
    """
    level: str = field(default="")
    title: str = field(default="")
    edges: Set[str] = field(default_factory=set)
    nodes: Set[str] = field(default_factory=set)
    chunk_ids: Set[str] = field(default_factory=set)
    occurrence: float = field(default=0.0)
    sub_communities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the community info to a dictionary representation."""
        return {
            'level': self.level,
            'title': self.title,
            'edges': list(self.edges),
            'nodes': list(self.nodes),
            'chunk_ids': list(self.chunk_ids),
            'occurrence': self.occurrence,
            'sub_communities': self.sub_communities
        }