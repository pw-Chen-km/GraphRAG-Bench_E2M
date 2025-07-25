"""
Query factory for creating different types of query processors.
Provides a unified interface for instantiating various query strategies based on configuration.
"""

from Core.Query.BaseQuery import BaseQuery
from Core.Query.BasicQuery import BasicQuery
from Core.Query.DalkQuery import DalkQuery
from Core.Query.GRQuery import GRQuery
from Core.Query.KGPQuery import KGPQuery
from Core.Query.MedQuery import MedQuery
from Core.Query.PPRQuery import PPRQuery
from Core.Query.ToGQuery import ToGQuery


class QueryFactory:
    """
    Factory class for creating different types of query processors.
    
    This factory provides a centralized way to instantiate various query strategies
    based on configuration parameters, supporting different retrieval and generation
    approaches for graph-based RAG systems.
    """

    def __init__(self):
        """Initialize the factory with query type mappings."""
        self.creators = {
            "basic": self._create_basic_query,
            "ppr": self._create_ppr_query,
            "kgp": self._create_kgp_query,
            "tog": self._create_tog_query,
            "gr": self._create_gr_query,
            "med": self._create_medical_query,
            "dalk": self._create_dalk_query,
        }

    def get_query(self, name: str, config, retriever) -> BaseQuery:
        """
        Create a query processor instance based on name and configuration.
        
        Args:
            name: Name of the query type to create
            config: Configuration object containing query parameters
            retriever: Retriever instance for document retrieval
            
        Returns:
            BaseQuery: Instance of the specified query processor
            
        Raises:
            KeyError: If the query type is not supported
        """
        if name not in self.creators:
            raise KeyError(f"Unsupported query type: {name}")
        
        return self.creators[name](config, retriever)

    @staticmethod
    def _create_basic_query(config, retriever):
        """Create a basic query processor."""
        return BasicQuery(config, retriever)

    @staticmethod
    def _create_ppr_query(config, retriever):
        """Create a PPR (Personalized PageRank) query processor."""
        return PPRQuery(config, retriever)

    @staticmethod
    def _create_kgp_query(config, retriever):
        """Create a KGP (Knowledge Graph Path) query processor."""
        return KGPQuery(config, retriever)

    @staticmethod
    def _create_tog_query(config, retriever):
        """Create a ToG (Tree of Thoughts on Graph) query processor."""
        return ToGQuery(config, retriever)

    @staticmethod
    def _create_gr_query(config, retriever):
        """Create a GR (Graph Retrieval) query processor."""
        return GRQuery(config, retriever)

    @staticmethod
    def _create_medical_query(config, retriever):
        """Create a medical domain query processor."""
        return MedQuery(config, retriever)
    
    @staticmethod
    def _create_dalk_query(config, retriever):
        """Create a DALK query processor."""
        return DalkQuery(config, retriever)


# Global factory instance for convenience
get_query = QueryFactory().get_query
