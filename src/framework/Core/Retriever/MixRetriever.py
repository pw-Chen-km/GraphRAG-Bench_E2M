"""
Mixed retriever that combines multiple retriever types.
Provides a unified interface for accessing different retriever implementations.
"""

from typing import Dict, Any, Optional

from Core.Common.Constants import Retriever
from Core.Retriever import EntityRetriever, CommunityRetriever, ChunkRetriever, RelationshipRetriever, SubgraphRetriever


class MixRetriever:
    """
    Mixed retriever that combines multiple retriever types.
    
    Provides a unified interface for accessing different retriever implementations
    including entity, community, chunk, relationship, and subgraph retrievers.
    """

    def __init__(self, retriever_context):
        """
        Initialize the mixed retriever with context.
        
        Args:
            retriever_context: Context object containing configuration and dependencies
        """
        self.context = retriever_context
        self.retrievers: Dict[str, Any] = {}
        self.register_retrievers()

    def register_retrievers(self):
        """Register all available retriever types."""
        context_dict = self.context.to_dict()
        
        # Create retriever instances
        self.retrievers[Retriever.ENTITY] = EntityRetriever(**context_dict)
        self.retrievers[Retriever.COMMUNITY] = CommunityRetriever(**context_dict)
        self.retrievers[Retriever.CHUNK] = ChunkRetriever(**context_dict)
        self.retrievers[Retriever.RELATION] = RelationshipRetriever(**context_dict)
        self.retrievers[Retriever.SUBGRAPH] = SubgraphRetriever(**context_dict)
        
        # Set entity-relationship mappings and vector databases for retrievers that need them
        for retriever in self.retrievers.values():
            if hasattr(retriever, 'entities_to_relationships'):
                retriever.entities_to_relationships = context_dict.get('entities_to_relationships')
            if hasattr(retriever, 'relationships_to_chunks'):
                retriever.relationships_to_chunks = context_dict.get('relationships_to_chunks')
            if hasattr(retriever, '_entities_to_relationships'):
                retriever._entities_to_relationships = context_dict.get('entities_to_relationships')
            if hasattr(retriever, 'entities_vdb'):
                retriever.entities_vdb = context_dict.get('entities_vdb')
            if hasattr(retriever, 'relations_vdb'):
                retriever.relations_vdb = context_dict.get('relations_vdb')
            if hasattr(retriever, 'subgraphs_vdb'):
                retriever.subgraphs_vdb = context_dict.get('subgraphs_vdb')
        


    async def retrieve_relevant_content(self, type: Retriever, mode: str, **kwargs) -> Optional[Any]:
        """
        Retrieve relevant content using the specified retriever type and mode.
        
        Args:
            type: Type of retriever to use
            mode: Retrieval mode for the retriever
            **kwargs: Additional arguments for the retrieval operation
            
        Returns:
            Retrieved content or None if retrieval fails
        """
        if type not in self.retrievers:
            raise ValueError(f"Unsupported retriever type: {type}")
            
        return await self.retrievers[type].retrieve_relevant_content(mode=mode, **kwargs)

    def get_retriever(self, type: Retriever) -> Optional[Any]:
        """
        Get a specific retriever instance.
        
        Args:
            type: Type of retriever to get
            
        Returns:
            Retriever instance or None if not found
        """
        return self.retrievers.get(type)

    @property
    def llm(self):
        """Get the LLM instance from context."""
        return self.context.llm

    @property
    def config(self):
        """Get the configuration from context."""
        return self.context.config
