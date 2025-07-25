"""
Chunk module for document processing and text chunking operations.

This module provides various chunking strategies for breaking down documents
into smaller, manageable pieces for RAG (Retrieval-Augmented Generation) systems.
"""

from Core.Chunk.Separator import chunking_by_seperators
from Core.Chunk.Tokensize import chunking_by_token_size
from Core.Chunk.DocChunk import DocumentChunker
from Core.Chunk.ChunkFactory import ChunkingMethodRegistry, register_chunking_method, create_chunk_method

# Export main chunking functions and classes
__all__ = [
    "chunking_by_seperators", 
    "chunking_by_token_size",
    "DocumentChunker",
    "ChunkingMethodRegistry",
    "register_chunking_method",
    "create_chunk_method"
]
