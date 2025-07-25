"""
Document Chunking Manager for processing and storing document chunks.

This module provides functionality for breaking down documents into chunks,
storing them efficiently, and retrieving them for RAG operations.
"""

import asyncio
from typing import List, Union, Dict, Any, Optional
from Core.Chunk.ChunkFactory import create_chunk_method
from Core.Common.Utils import mdhash_id
from Core.Common.Logger import logger
from Core.Schema.ChunkSchema import TextChunk
from Core.Storage.ChunkKVStorage import ChunkKVStorage


class DocumentChunker:
    """
    Manages document chunking operations and storage.
    
    This class handles the process of breaking down documents into smaller
    chunks, storing them in a key-value storage system, and providing
    efficient retrieval mechanisms.
    """
    
    def __init__(self, config: Any, token_model: Any, namespace: str):
        """
        Initialize the document chunker.
        
        Args:
            config: Configuration object containing chunking parameters
            token_model: Tokenizer model for text processing
            namespace: Storage namespace for organizing chunks
        """
        self.config = config
        self.chunk_method = create_chunk_method(self.config.chunk_method)
        self._storage = ChunkKVStorage(namespace=namespace)
        self.token_model = token_model
        self._namespace = namespace
    
    @property
    def namespace(self) -> str:
        """Get the current storage namespace."""
        return self._namespace
    
    @namespace.setter
    def namespace(self, namespace: str) -> None:
        """Set the storage namespace."""
        self._namespace = namespace
    
    async def build_chunks(
        self, 
        documents: Union[str, List[str], List[Dict[str, str]]], 
        force_rebuild: bool = True
    ) -> None:
        """
        Build chunks from input documents.
        
        Args:
            documents: Input documents as string, list of strings, or list of dicts
            force_rebuild: Whether to force rebuilding existing chunks
        """
        logger.info("Starting document chunking process")
        
        # Check if chunks already exist
        chunks_exist = await self._load_existing_chunks(force_rebuild)
        
        if not chunks_exist or force_rebuild:
            # Process and chunk the documents
            processed_docs = self._normalize_documents(documents)
            await self._process_and_store_chunks(processed_docs)
            await self._storage.persist()
        
        logger.info("✅ Document chunking completed successfully")
    
    def _normalize_documents(
        self, 
        documents: Union[str, List[str], List[Dict[str, str]]]
    ) -> Dict[str, Dict[str, str]]:
        """
        Normalize input documents to a standard format.
        
        Args:
            documents: Input documents in various formats
            
        Returns:
            Normalized documents as a dictionary with document IDs as keys
        """
        if isinstance(documents, str):
            documents = [documents]
        
        if isinstance(documents, list):
            if all(isinstance(doc, dict) for doc in documents):
                # Handle list of dictionaries
                return {
                    mdhash_id(doc["content"].strip(), prefix="doc-"): {
                        "content": doc["content"].strip(),
                        "title": doc.get("title", ""),
                    }
                    for doc in documents
                }
            else:
                # Handle list of strings
                return {
                    mdhash_id(doc.strip(), prefix="doc-"): {
                        "content": doc.strip(),
                        "title": "",
                    }
                    for doc in documents
                }
        
        return documents
    
    async def _process_and_store_chunks(
        self, 
        documents: Dict[str, Dict[str, str]]
    ) -> None:
        """
        Process documents into chunks and store them.
        
        Args:
            documents: Normalized documents dictionary
        """
        # Extract document components
        doc_items = list(documents.items())
        doc_contents = [item[1]["content"] for item in doc_items]
        doc_keys = [item[0] for item in doc_items]
        doc_titles = [item[1]["title"] for item in doc_items]
        
        # Tokenize documents
        tokens = self.token_model.encode_batch(doc_contents, num_threads=16)
        
        # Generate chunks using the configured method
        chunks = await self.chunk_method(
            tokens,
            doc_keys=doc_keys,
            tiktoken_model=self.token_model,
            title_list=doc_titles,
            overlap_token_size=self.config.chunk_overlap_token_size,
            max_token_size=self.config.chunk_token_size,
        )
        
        # Store chunks with unique IDs
        for chunk in chunks:
            chunk["chunk_id"] = mdhash_id(chunk["content"], prefix="chunk-")
            await self._storage.upsert(chunk["chunk_id"], TextChunk(**chunk))
    
    async def _load_existing_chunks(self, force: bool) -> bool:
        """
        Load existing chunks from storage.
        
        Args:
            force: Whether to force reload
            
        Returns:
            True if chunks were loaded successfully, False otherwise
        """
        if force:
            return False
        return await self._storage.load_chunk()
    
    async def get_chunks(self) -> List[TextChunk]:
        """
        Retrieve all stored chunks.
        
        Returns:
            List of all stored text chunks
        """
        return await self._storage.get_chunks()
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[str]:
        """
        Get chunk content by chunk ID.
        
        Args:
            chunk_id: The unique identifier of the chunk
            
        Returns:
            The content of the chunk or None if not found
        """
        chunk = await self._storage.get_by_key(chunk_id)
        return chunk.content if chunk else None
    
    async def get_chunk_by_index(self, index: int) -> Optional[str]:
        """
        Get chunk content by index.
        
        Args:
            index: The index of the chunk
            
        Returns:
            The content of the chunk or None if not found
        """
        chunk = await self._storage.get_data_by_index(index)
        return chunk.content if chunk else None
    
    async def get_chunk_id_by_index(self, index: int) -> Optional[str]:
        """
        Get chunk ID by index.
        
        Args:
            index: The index of the chunk
            
        Returns:
            The chunk ID or None if not found
        """
        return await self._storage.get_key_by_index(index)
    
    async def get_chunks_by_indices(self, indices: List[int]) -> List[str]:
        """
        Get multiple chunks by their indices.
        
        Args:
            indices: List of chunk indices
            
        Returns:
            List of chunk contents
        """
        tasks = [self.get_chunk_by_index(index) for index in indices]
        return await asyncio.gather(*tasks)
    
    async def get_index_by_merge_key(self, chunk_id: str) -> Optional[int]:
        """
        Get chunk index by merge key.
        
        Args:
            chunk_id: The chunk ID to look up
            
        Returns:
            The index of the chunk or None if not found
        """
        return await self._storage.get_index_by_merge_key(chunk_id)
    
    async def get_index_by_key(self, key: str) -> Optional[int]:
        """
        Get chunk index by key.
        
        Args:
            key: The key to look up
            
        Returns:
            The index of the chunk or None if not found
        """
        return await self._storage.get_index_by_key(key)
    
    @property
    async def chunk_count(self) -> int:
        """
        Get the total number of stored chunks.
        
        Returns:
            The number of chunks in storage
        """
        return await self._storage.size()
