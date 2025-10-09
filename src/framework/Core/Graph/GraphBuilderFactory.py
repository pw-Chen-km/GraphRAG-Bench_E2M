"""
Graph Builder Factory - Refactored graph building logic
Adopts strategy pattern and factory pattern, providing flexible graph building capabilities
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import networkx as nx

from Core.Common.Logger import logger
from Core.Utils.Display import StatusDisplay, ProgressDisplay
from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.EntityRelation import Entity, Relationship


class GraphType(Enum):
    """Graph types"""
    ENTITY_RELATION = "er_graph"
    RICH_KNOWLEDGE = "rkg_graph"
    EVOLVING = "evolving_graph"
    TREE = "tree_graph"
    TREE_BALANCED = "tree_graph_balanced"
    PASSAGE = "passage_graph"


@dataclass
class GraphBuilderConfig:
    """Graph building configuration"""
    graph_type: GraphType = GraphType.ENTITY_RELATION
    enable_entity_description: bool = True
    enable_entity_type: bool = False
    enable_edge_description: bool = True
    enable_edge_name: bool = True
    enable_edge_keywords: bool = False
    extract_two_step: bool = True
    max_gleaning: int = 1
    force_rebuild: bool = False
    evolution: Dict[str, Any] = field(default_factory=dict)


class GraphBuilder(ABC):
    """Abstract base class for graph builders"""
    
    def __init__(self, config: GraphBuilderConfig, context: Any):
        self.config = config
        self.context = context
        self.graph = None
        self.llm = context.llm
        self.encoder = context.encoder
    
    @abstractmethod
    async def execute(self, chunks: List[TextChunk], force_rebuild: bool = False) -> Any:
        """Execute graph building"""
        pass
    
    @abstractmethod
    async def _build_graph(self, chunks: List[TextChunk]) -> Any:
        """Build graph"""
        pass
    
    @abstractmethod
    async def _extract_entities_relations(self, chunk: TextChunk) -> tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships"""
        pass
    
    def get_graph(self) -> Any:
        """Get the built graph"""
        return self.graph
    
    async def augment_graph(self, entity_index: Any) -> None:
        """Augment graph (optional)"""
        if hasattr(self, '_augment_graph_by_similarity'):
            await self._augment_graph_by_similarity(entity_index)
    
    async def detect_communities(self) -> None:
        """Detect communities (optional)"""
        if hasattr(self, '_detect_communities'):
            await self._detect_communities()


class EntityRelationGraphBuilder(GraphBuilder):
    """Entity-relation graph builder"""
    
    async def execute(self, chunks: List[TextChunk], force_rebuild: bool = False) -> Any:
        """Execute entity-relation graph building"""
        StatusDisplay.show_processing_status("Graph Building", details="Entity-Relation Graph")
        
        # Check if rebuild is needed
        if not force_rebuild and self.graph is not None:
            StatusDisplay.show_info("Using existing graph")
            return self.graph
        
        # Build graph
        self.graph = await self._build_graph(chunks)
        StatusDisplay.show_success(f"Entity-relation graph building completed, nodes: {self.graph.number_of_nodes()}, edges: {self.graph.number_of_edges()}")
        return self.graph
    
    async def _build_graph(self, chunks: List[TextChunk]) -> nx.Graph:
        """Build entity-relation graph"""
        graph = nx.Graph()
        
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            ProgressDisplay.show_progress(i + 1, total_chunks, "Processing text chunks")
            
            entities, relationships = await self._extract_entities_relations(chunk)
            
            # Add entity nodes
            for entity in entities:
                if not graph.has_node(entity.entity_name):
                    graph.add_node(entity.entity_name, **entity.to_dict())
                else:
                    # Merge entity information
                    existing_data = graph.nodes[entity.entity_name]
                    merged_data = self._merge_entity_data(existing_data, entity.to_dict())
                    graph.nodes[entity.entity_name].update(merged_data)
            
            # Add relationship edges
            for relation in relationships:
                edge_key = (relation.src_id, relation.tgt_id)
                if not graph.has_edge(*edge_key):
                    graph.add_edge(*edge_key, **relation.to_dict())
                else:
                    # Merge relationship information
                    existing_data = graph.edges[edge_key]
                    merged_data = self._merge_relation_data(existing_data, relation.to_dict())
                    graph.edges[edge_key].update(merged_data)
        
        return graph
    
    async def _extract_entities_relations(self, chunk: TextChunk) -> tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships"""
        # Use LLM for named entity recognition
        entities = await self._named_entity_recognition(chunk.content)
        
        # Use LLM for relationship extraction
        relationships = await self._openie_extraction(chunk.content, entities)
        
        # Convert to Entity and Relationship objects
        entity_objects = []
        for entity_name in entities:
            entity = Entity(
                entity_name=entity_name,
                entity_type="",  # Can be extracted from NER results
                description="",
                source_id=chunk.chunk_id
            )
            entity_objects.append(entity)
        
        relationship_objects = []
        for rel in relationships:
            if len(rel) >= 3:
                relationship = Relationship(
                    src_id=rel[0],
                    tgt_id=rel[2],
                    relation_name=rel[1],
                    description="",
                    weight=1.0,
                    source_id=chunk.chunk_id
                )
                relationship_objects.append(relationship)
        
        return entity_objects, relationship_objects
    
    async def _named_entity_recognition(self, text: str) -> List[str]:
        """Named entity recognition"""
        from Core.Prompt import GraphPrompt
        
        ner_messages = GraphPrompt.NER.format(user_input=text)
        response = await self.llm.aask(ner_messages, format="json")
        
        entities = []
        if 'named_entities' in response:
            entities = response['named_entities']
        
        return entities
    
    async def _openie_extraction(self, text: str, entities: List[str]) -> List[tuple]:
        """Open information extraction"""
        from Core.Prompt import GraphPrompt
        import json
        
        named_entity_json = {"named_entities": entities}
        openie_messages = GraphPrompt.OPENIE_POST_NET.format(
            passage=text,
            named_entity_json=json.dumps(named_entity_json)
        )
        
        response = await self.llm.aask(openie_messages, format="json")
        
        triples = []
        if 'triples' in response:
            triples = response['triples']
        
        return triples
    
    def _merge_entity_data(self, existing: Dict, new: Dict) -> Dict:
        """Merge entity data"""
        merged = existing.copy()
        
        # Merge descriptions
        if self.config.enable_entity_description:
            existing_desc = existing.get('description', '')
            new_desc = new.get('description', '')
            if existing_desc and new_desc:
                merged['description'] = f"{existing_desc}; {new_desc}"
            elif new_desc:
                merged['description'] = new_desc
        
        # Merge types
        if self.config.enable_entity_type:
            existing_type = existing.get('entity_type', '')
            new_type = new.get('entity_type', '')
            if existing_type and new_type and existing_type != new_type:
                merged['entity_type'] = f"{existing_type}, {new_type}"
            elif new_type:
                merged['entity_type'] = new_type
        
        # Merge source IDs
        existing_sources = existing.get('source_id', '').split(',') if existing.get('source_id') else []
        new_sources = new.get('source_id', '').split(',') if new.get('source_id') else []
        all_sources = list(set(existing_sources + new_sources))
        merged['source_id'] = ','.join(all_sources)
        
        return merged
    
    def _merge_relation_data(self, existing: Dict, new: Dict) -> Dict:
        """Merge relationship data"""
        merged = existing.copy()
        
        # Merge descriptions
        if self.config.enable_edge_description:
            existing_desc = existing.get('description', '')
            new_desc = new.get('description', '')
            if existing_desc and new_desc:
                merged['description'] = f"{existing_desc}; {new_desc}"
            elif new_desc:
                merged['description'] = new_desc
        
        # Merge relationship names
        if self.config.enable_edge_name:
            existing_name = existing.get('relation_name', '')
            new_name = new.get('relation_name', '')
            if existing_name and new_name and existing_name != new_name:
                merged['relation_name'] = f"{existing_name}, {new_name}"
            elif new_name:
                merged['relation_name'] = new_name
        
        # Merge weights
        existing_weight = existing.get('weight', 1.0)
        new_weight = new.get('weight', 1.0)
        merged['weight'] = (existing_weight + new_weight) / 2
        
        return merged


class RichKnowledgeGraphBuilder(GraphBuilder):
    """Rich knowledge graph builder"""
    
    async def execute(self, chunks: List[TextChunk], force_rebuild: bool = False) -> Any:
        """Execute rich knowledge graph building"""
        StatusDisplay.show_processing_status("Graph Building", details="Rich Knowledge Graph")
        
        if not force_rebuild and self.graph is not None:
            StatusDisplay.show_info("Using existing graph")
            return self.graph
        
        self.graph = await self._build_graph(chunks)
        StatusDisplay.show_success(f"Rich knowledge graph building completed, nodes: {self.graph.number_of_nodes()}, edges: {self.graph.number_of_edges()}")
        return self.graph
    
    async def _build_graph(self, chunks: List[TextChunk]) -> nx.Graph:
        """Build rich knowledge graph"""
        graph = nx.Graph()
        
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            ProgressDisplay.show_progress(i + 1, total_chunks, "Processing text chunks")
            
            entities, relationships = await self._extract_entities_relations(chunk)
            
            # Add entity nodes (containing richer information)
            for entity in entities:
                if not graph.has_node(entity.entity_name):
                    graph.add_node(entity.entity_name, **entity.to_dict())
                else:
                    existing_data = graph.nodes[entity.entity_name]
                    merged_data = self._merge_entity_data(existing_data, entity.to_dict())
                    graph.nodes[entity.entity_name].update(merged_data)
            
            # Add relationship edges (containing keyword information)
            for relation in relationships:
                edge_key = (relation.src_id, relation.tgt_id)
                if not graph.has_edge(*edge_key):
                    graph.add_edge(*edge_key, **relation.to_dict())
                else:
                    existing_data = graph.edges[edge_key]
                    merged_data = self._merge_relation_data(existing_data, relation.to_dict())
                    graph.edges[edge_key].update(merged_data)
        
        return graph
    
    async def _extract_entities_relations(self, chunk: TextChunk) -> tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships (containing rich information)"""
        # Use more complex prompts for entity and relationship extraction
        from Core.Prompt import GraphPrompt
        
        # Extract records
        records = await self._extract_records_from_chunk(chunk)
        
        # Build graph
        entities, relationships = await self._build_graph_from_records(records, chunk.chunk_id)
        
        return entities, relationships
    
    async def _extract_records_from_chunk(self, chunk: TextChunk) -> List[str]:
        """Extract records from text chunk"""
        from Core.Common.Utils import split_string_by_multiple_delimiters
        
        # Use LLM to extract structured records
        extraction_prompt = f"""
        Please extract entity and relationship information from the following text in structured format:
        
        {chunk.content}
        
        Please return in the following format:
        "entity" | Entity Name | Entity Type | Entity Description
        "relation" | Source Entity | Relationship Name | Target Entity | Relationship Description | Keywords
        """
        
        response = await self.llm.aask(extraction_prompt)
        
        # Parse response
        records = split_string_by_multiple_delimiters(response, ['"entity"', '"relation"'])
        return records
    
    async def _build_graph_from_records(self, records: List[str], chunk_id: str) -> tuple[List[Entity], List[Relationship]]:
        """Build graph from records"""
        entities = []
        relationships = []
        
        for record in records:
            if record.startswith('"entity"'):
                entity = await self._handle_single_entity_extraction(record.split('|'), chunk_id)
                if entity:
                    entities.append(entity)
            elif record.startswith('"relation"'):
                relationship = await self._handle_single_relation_extraction(record.split('|'), chunk_id)
                if relationship:
                    relationships.append(relationship)
        
        return entities, relationships
    
    async def _handle_single_entity_extraction(self, record_attributes: List[str], chunk_id: str) -> Optional[Entity]:
        """Handle single entity extraction"""
        from Core.Common.Utils import clean_str
        
        if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
            return None
        
        entity_name = clean_str(record_attributes[1])
        if not entity_name.strip():
            return None
        
        entity = Entity(
            entity_name=entity_name,
            entity_type=clean_str(record_attributes[2]),
            description=clean_str(record_attributes[3]),
            source_id=chunk_id
        )
        
        return entity
    
    async def _handle_single_relation_extraction(self, record_attributes: List[str], chunk_id: str) -> Optional[Relationship]:
        """Handle single relationship extraction"""
        from Core.Common.Utils import clean_str
        
        if len(record_attributes) < 6 or record_attributes[0] != '"relation"':
            return None
        
        relationship = Relationship(
            src_id=clean_str(record_attributes[1]),
            tgt_id=clean_str(record_attributes[3]),
            relation_name=clean_str(record_attributes[2]),
            description=clean_str(record_attributes[4]),
            keywords=clean_str(record_attributes[5]) if len(record_attributes) > 5 else "",
            weight=1.0,
            source_id=chunk_id
        )
        
        return relationship


class TreeGraphBuilder(GraphBuilder):
    """Tree graph builder"""
    
    async def execute(self, chunks: List[TextChunk], force_rebuild: bool = False) -> Any:
        """Execute tree graph building"""
        StatusDisplay.show_processing_status("Graph Building", details="Tree Graph")
        
        if not force_rebuild and self.graph is not None:
            StatusDisplay.show_info("Using existing graph")
            return self.graph
        
        self.graph = await self._build_graph(chunks)
        StatusDisplay.show_success(f"Tree graph building completed, nodes: {self.graph.number_of_nodes()}, edges: {self.graph.number_of_edges()}")
        return self.graph
    
    async def _build_graph(self, chunks: List[TextChunk]) -> nx.Graph:
        """Build tree graph"""
        # Here implement tree graph building logic
        # Can refer to original TreeGraph implementation
        graph = nx.Graph()
        
        # Simplified tree graph building
        for i, chunk in enumerate(chunks):
            node_id = f"chunk_{i}"
            graph.add_node(node_id, content=chunk.content, metadata=chunk.metadata)
            
            # Add hierarchical relationships
            if i > 0:
                parent_id = f"chunk_{(i-1)//2}"  # Simple parent-child relationship
                graph.add_edge(parent_id, node_id, relation_type="hierarchy")
        
        return graph
    
    async def _extract_entities_relations(self, chunk: TextChunk) -> tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships (not needed for tree graph)"""
        return [], []


class GraphBuilderFactory:
    """Graph builder factory"""
    
    _builders = {
        GraphType.ENTITY_RELATION: EntityRelationGraphBuilder,
        GraphType.RICH_KNOWLEDGE: RichKnowledgeGraphBuilder,
        GraphType.TREE: TreeGraphBuilder,
        GraphType.TREE_BALANCED: TreeGraphBuilder,  # Can create specialized balanced tree builder
        GraphType.PASSAGE: EntityRelationGraphBuilder  # Can create specialized passage graph builder
    }
    
    @classmethod
    def create_builder(cls, config: Any, context: Any) -> GraphBuilder:
        """Create graph builder"""
        # Extract graph building configuration from config
        graph_config = GraphBuilderConfig(
            graph_type=GraphType(config.graph.graph_type),
            enable_entity_description=config.graph.enable_entity_description,
            enable_entity_type=config.graph.enable_entity_type,
            enable_edge_description=config.graph.enable_edge_description,
            enable_edge_name=config.graph.enable_edge_name,
            enable_edge_keywords=config.graph.enable_edge_keywords,
            extract_two_step=config.graph.extract_two_step,
            max_gleaning=config.graph.max_gleaning,
            force_rebuild=config.graph.force,
            evolution=getattr(config.graph, "evolution", {})
        )

        builder_class = cls._builders.get(graph_config.graph_type, EntityRelationGraphBuilder)
        return builder_class(graph_config, context)

    @classmethod
    def register_builder(cls, graph_type: Union[GraphType, str], builder_class: type):
        """Register new builder"""
        if isinstance(graph_type, str):
            graph_type = GraphType(graph_type)
        cls._builders[graph_type] = builder_class
        logger.info(f"Registered new graph builder: {graph_type.value}")
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get available graph types"""
        return [graph_type.value for graph_type in cls._builders.keys()]


# Ensure evolving builder registration on import
try:  # pragma: no cover - defensive import
    from .EvolvingGraphBuilder import EvolvingGraphBuilder  # noqa: F401
except ImportError:  # pragma: no cover - optional component
    pass
