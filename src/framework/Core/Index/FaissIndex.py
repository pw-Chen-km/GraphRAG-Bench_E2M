"""
FAISS Index Module

This module provides a FAISS-based vector index implementation using LlamaIndex.
It is designed for high-performance approximate nearest neighbor search.
"""

import asyncio
from typing import Any, List, Optional, Tuple
import numpy as np
import faiss

from llama_index.core.schema import Document, TextNode
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

from Core.Index.VectorIndexBase import VectorIndexBase
from Core.Common.Utils import mdhash_id
from Core.Common.Logger import logger


class FaissIndex(VectorIndexBase):
    """
    FAISS-based vector index implementation using LlamaIndex.
    
    This class provides high-performance approximate nearest neighbor (ANN) 
    search using FAISS indexing with HNSW algorithm.
    """

    def __init__(self, config):
        """
        Initialize the FAISS index with configuration.
        
        Args:
            config: Configuration object containing index parameters
        """
        super().__init__(config)

    def _embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Text embedding as list of floats
        """
        return self.embedding_model._get_text_embedding(text)

    async def _update_index(self, datas: List[dict], meta_data: List[str]) -> None:
        """
        Update the FAISS index with new documents.
        
        Args:
            datas: List of data dictionaries
            meta_data: List of metadata keys to include
        """
        async def process_document(data: dict) -> Document:
            """Process a single data item into a Document."""
            return Document(
                doc_id=mdhash_id(data["content"]),
                text=data["content"],
                metadata={key: data[key] for key in meta_data},
                excluded_embed_metadata_keys=meta_data,
            )

        Settings.embed_model = self.config.embed_model
        documents = await asyncio.gather(*[process_document(data) for data in datas])
        texts = [doc.text for doc in documents]
        
        # Generate embeddings with batching for OpenAI models
        text_embeddings = []
        if isinstance(self.embedding_model, OpenAIEmbedding):
            batch_size = self.embedding_model.embed_batch_size
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model._get_text_embeddings(batch)
                text_embeddings.extend(batch_embeddings)
        else:
            text_embeddings = self.embedding_model._get_text_embeddings(texts)

        # Create FAISS vector store with HNSW index
        vector_store = FaissVectorStore(faiss_index=faiss.IndexHNSWFlat(1024, 32))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self._index = VectorStoreIndex(
            [], 
            storage_context=storage_context,
            embed_model=self.config.embed_model
        )

        # Insert nodes with pre-computed embeddings
        nodes = []
        for doc, embedding in zip(documents, text_embeddings):
            node = TextNode(text=doc.text, embedding=embedding, metadata=doc.metadata)
            nodes.append(node)
        self._index.insert_nodes(nodes)

        logger.info(f"Updated FAISS index with {len(documents)} documents")

    def _get_index(self) -> VectorStoreIndex:
        """
        Get a new FAISS-based VectorStoreIndex instance.
        
        Returns:
            FAISS-based VectorStoreIndex instance
        """
        Settings.embed_model = self.config.embed_model
        # TODO: Configure FAISS index parameters from config
        vector_store = FaissVectorStore(faiss_index=faiss.IndexHNSWFlat(1024, 32))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=self.config.embed_model,
        )

    def _load_index_from_storage(self, storage_context: StorageContext) -> VectorStoreIndex:
        """
        Load FAISS-based VectorStoreIndex from storage context.
        
        Args:
            storage_context: Storage context containing the index
            
        Returns:
            Loaded FAISS-based VectorStoreIndex instance
        """
        vector_store = FaissVectorStore.from_persist_dir(str(self.config.persist_path))
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, 
            persist_dir=self.config.persist_path
        )
        return load_index_from_storage(
            storage_context=storage_context, 
            embed_model=self.config.embed_model
        )

    async def retrieval_nodes_with_score_matrix(self, query_list: List[str], top_k: int, graph: Any) -> np.ndarray:
        """
        Retrieve nodes and return a score matrix for multiple queries.
        
        Args:
            query_list: List of search queries
            top_k: Number of top results to return per query
            graph: Graph instance to search in
            
        Returns:
            Score matrix where each row represents scores for one query
        """
        if isinstance(query_list, str):
            query_list = [query_list]
            
        results = await asyncio.gather(
            *[self.retrieval_nodes(query, top_k, graph, need_score=True) for query in query_list]
        )
        
        reset_prob_matrix = np.zeros((len(query_list), graph.node_num))
        entity_indices = []
        scores = []

        async def set_idx_score(idx: int, res: Tuple[List[dict], List[float]]) -> None:
            """Set index and score for a single result."""
            for entity, score in zip(res[0], res[1]):
                entity_indices.append(await graph.get_node_index(entity["entity_name"]))
                scores.append(score)

        await asyncio.gather(*[set_idx_score(idx, res) for idx, res in enumerate(results)])
        reset_prob_matrix[np.arange(len(query_list)).reshape(-1, 1), entity_indices] = scores
        all_entity_weights = reset_prob_matrix.max(axis=0)  # (1, #all_entities)

        # Normalize the scores
        all_entity_weights /= all_entity_weights.sum()
        return all_entity_weights
