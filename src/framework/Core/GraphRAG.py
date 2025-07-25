"""
Main GraphRAG system implementation.
Provides a comprehensive framework for graph-based retrieval-augmented generation.
"""

from typing import Union, Any, Optional

import tiktoken

from pydantic import BaseModel, model_validator


from Core.Chunk.DocChunk import DocumentChunker
from Core.Common.ContextMixin import ContextMixin
from Core.Common.Logger import logger
from Core.Common.TimeStatistic import TimeStatistic
from Core.Community.ClusterFactory import get_community
from Core.Graph import get_graph
from Core.Index import get_index, get_index_config
from Core.Query import get_query
from Core.Schema.RetrieverContext import RetrieverContext
from Core.Storage.NameSpace import Workspace
from Core.Storage.PickleBlobStorage import PickleBlobStorage




class GraphRAG(ContextMixin, BaseModel):
    """
    A comprehensive Graph-based Retrieval-Augmented Generation system.
    
    This class provides a unified interface for building, querying, and managing
    graph-based RAG systems with support for multiple graph types, retrieval
    strategies, and storage backends.
    
    Attributes:
        config: Configuration object containing all system parameters
        workspace: Workspace manager for storage organization
        graph: Graph instance for knowledge representation
        doc_chunk: Document chunking processor
        time_manager: Time statistics manager
        retriever_context: Context for retrieval operations
    """
    
    # Core components (will be initialized in validator)
    config: Any
    ENCODER: Optional[Any] = None
    workspace: Optional[Any] = None
    graph: Optional[Any] = None
    doc_chunk: Optional[Any] = None
    time_manager: Optional[Any] = None
    retriever_context: Optional[Any] = None

    def __init__(self, config):
        """Initialize the GraphRAG system with configuration."""
        super().__init__(config=config)



    @model_validator(mode="after")
    def initialize_system_components(cls, data):
        """Initialize all system components after model validation."""
        # Initialize tokenizer
        cls.ENCODER = tiktoken.encoding_for_model(data.config.token_model)
        
        # Initialize workspace and core components
        cls.workspace = Workspace(data.config.working_dir, data.config.index_name)
        cls.graph = get_graph(data.config, llm=data.llm, encoder=cls.ENCODER)
        cls.doc_chunk = DocumentChunker(data.config.chunk, cls.ENCODER, data.workspace.make_for("chunk_storage"))
        cls.time_manager = TimeStatistic()
        cls.retriever_context = RetrieverContext()
        
        # Initialize storage and register components
        data = cls._initialize_storage_namespace(data)
        data = cls._register_vector_databases(data)
        data = cls._register_community_system(data)
        data = cls._register_entity_relationship_mappings(data)
        data = cls._register_retriever_context(data)
        
        return data

    @classmethod
    def _initialize_storage_namespace(cls, data):
        """Initialize storage namespaces for different components."""
        data.graph.namespace = data.workspace.make_for("graph_storage")
        
        if data.config.use_entities_vdb:
            data.entities_vdb_namespace = data.workspace.make_for("entities_vdb")
        if data.config.use_relations_vdb:
            data.relations_vdb_namespace = data.workspace.make_for("relations_vdb")
        if data.config.use_subgraphs_vdb:
            data.subgraphs_vdb_namespace = data.workspace.make_for("subgraphs_vdb")
        if data.config.graph.use_community:
            data.community_namespace = data.workspace.make_for("community_storage")
        if data.config.use_entity_link_chunk:
            data.e2r_namespace = data.workspace.make_for("map_e2r")
            data.r2c_namespace = data.workspace.make_for("map_r2c")
        
        return data

    @classmethod
    def _register_vector_databases(cls, data):
        """Register vector databases for different data types."""
        if data.config.use_entities_vdb:
            data.entities_vdb = get_index(data.config, data.entities_vdb_namespace, "entities")
        if data.config.use_relations_vdb:
            data.relations_vdb = get_index(data.config, data.relations_vdb_namespace, "relations")
        if data.config.use_subgraphs_vdb:
            data.subgraphs_vdb = get_index(data.config, data.subgraphs_vdb_namespace, "subgraphs")
        
        return data

    @classmethod
    def _register_community_system(cls, data):
        """Register community detection system if enabled."""
        if data.config.graph.use_community:
            data.community = get_community(data.config, data.community_namespace)
        
        return data

    @classmethod
    def _register_entity_relationship_mappings(cls, data):
        """
        Register entity-to-relationship and relationship-to-chunk mapping matrices.
        
        These matrices facilitate the entity -> relationship -> chunk linkage,
        which is integral to the HippoRAG and FastGraphRAG models.
        
        Entity Matrix: Represents the entities in the dataset.
        Chunk Matrix: Represents the chunks associated with the entities.
        """
        if data.config.use_entity_link_chunk:
            data.e2r_matrix = PickleBlobStorage(data.e2r_namespace)
            data.r2c_matrix = PickleBlobStorage(data.r2c_namespace)
        
        return data

    @classmethod
    def _register_retriever_context(cls, data):
        """Register retriever context and related components."""
        data.retriever = get_query(data.config, data.graph, data.llm, data.ENCODER)
        
        if data.config.use_entities_vdb:
            data.retriever_context.entities_vdb = data.entities_vdb
        if data.config.use_relations_vdb:
            data.retriever_context.relations_vdb = data.relations_vdb
        if data.config.use_subgraphs_vdb:
            data.retriever_context.subgraphs_vdb = data.subgraphs_vdb
        if data.config.graph.use_community:
            data.retriever_context.community = data.community
        if data.config.use_entity_link_chunk:
            data.retriever_context.e2r_matrix = data.e2r_matrix
            data.retriever_context.r2c_matrix = data.r2c_matrix
        
        return data

    async def _build_retriever_context(self):
        """Build the retriever context with necessary data."""
        if self.config.use_entities_vdb:
            await self.retriever_context.build_entities_context(self.graph)
        if self.config.use_relations_vdb:
            await self.retriever_context.build_relations_context(self.graph)
        if self.config.use_subgraphs_vdb:
            await self.retriever_context.build_subgraphs_context(self.graph)
        if self.config.graph.use_community:
            await self.retriever_context.build_community_context(self.graph)
        if self.config.use_entity_link_chunk:
            await self.build_e2r_r2c_maps()

    async def build_e2r_r2c_maps(self, force=False):
        """Build entity-to-relationship and relationship-to-chunk mapping matrices."""
        if not self.config.use_entity_link_chunk:
            return
        
        if not force and self.e2r_matrix.exists() and self.r2c_matrix.exists():
            logger.info("Entity-relationship mappings already exist, skipping build.")
            return
        
        logger.info("Building entity-relationship mappings...")
        await self._build_entity_relationship_mappings()
        logger.info("Entity-relationship mappings built successfully.")

    def _update_cost_information(self, stage: str):
        """Update cost information for the current processing stage."""
        self.time_manager.update_stage(stage)
        if hasattr(self, 'cost_manager'):
            self.cost_manager.update_stage(stage)

    async def insert(self, docs: Union[str, list[Any]]):
        """
        Insert documents into the GraphRAG system.
        
        Args:
            docs: Document or list of documents to insert
        """
        self._update_cost_information("insert_start")
        
        # Process documents into chunks
        chunks = await self.doc_chunk.process(docs)
        logger.info(f"Processed {len(chunks)} chunks from documents")
        
        # Build graph from chunks
        await self.graph.build(chunks)
        logger.info("Graph built successfully")
        
        # Build retriever context
        await self._build_retriever_context()
        logger.info("Retriever context built successfully")
        
        self._update_cost_information("insert_end")
        logger.info("Document insertion completed")

    async def query(self, query: str):
        """
        Process a query through the GraphRAG system.
        
        Args:
            query: Query string to process
            
        Returns:
            Query result from the system
        """
        self._update_cost_information("query_start")
        
        result = await self.retriever.query(query, self.retriever_context)
        
        self._update_cost_information("query_end")
        return result
        


   

    
    
   
      
        

   



   

  
  