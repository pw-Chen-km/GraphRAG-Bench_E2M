"""
Token size-based text chunking implementation.

This module provides functionality for splitting text into chunks based on
token count, with support for overlap between chunks.
"""

from typing import List, Optional, Any, Dict
from Core.Chunk.ChunkFactory import register_chunking_method


class TokenSizeChunker:
    """
    Chunks text based on token size constraints.
    
    This class implements a simple but effective chunking strategy that
    splits text into fixed-size chunks while maintaining overlap between
    consecutive chunks for better context preservation.
    """
    
    def __init__(self, max_token_size: int = 1024, overlap_token_size: int = 128):
        """
        Initialize the token size chunker.
        
        Args:
            max_token_size: Maximum number of tokens per chunk
            overlap_token_size: Number of tokens to overlap between chunks
        """
        self.max_token_size = max_token_size
        self.overlap_token_size = overlap_token_size
        self.step_size = max_token_size - overlap_token_size
    
    def chunk_tokens(self, tokens: List[int]) -> List[List[int]]:
        """
        Split tokens into chunks based on size constraints.
        
        Args:
            tokens: Input token sequence
            
        Returns:
            List of token chunks
        """
        if not tokens:
            return []
        
        chunks = []
        for start_position in range(0, len(tokens), self.step_size):
            end_position = start_position + self.max_token_size
            chunk = tokens[start_position:end_position]
            
            # Only add non-empty chunks
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def get_chunk_lengths(self, tokens: List[int]) -> List[int]:
        """
        Calculate the length of each chunk without creating the chunks.
        
        Args:
            tokens: Input token sequence
            
        Returns:
            List of chunk lengths
        """
        if not tokens:
            return []
        
        lengths = []
        for start_position in range(0, len(tokens), self.step_size):
            end_position = start_position + self.max_token_size
            chunk_length = min(self.max_token_size, len(tokens) - start_position)
            lengths.append(chunk_length)
        
        return lengths


@register_chunking_method("chunking_by_token_size")
async def chunking_by_token_size(
    tokens_list: List[List[int]], 
    doc_keys: List[str], 
    tiktoken_model: Any, 
    overlap_token_size: int = 128,
    max_token_size: int = 1024, 
    title_list: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Chunk documents using token size-based splitting.
    
    This function processes a list of tokenized documents and splits them
    into chunks based on token count while maintaining overlap between
    consecutive chunks for better context preservation.
    
    Args:
        tokens_list: List of tokenized documents
        doc_keys: Document identifiers
        tiktoken_model: Tokenizer model for encoding/decoding
        overlap_token_size: Number of tokens to overlap between chunks
        max_token_size: Maximum tokens per chunk
        title_list: Optional list of document titles
        
    Returns:
        List of chunk dictionaries with metadata
    """
    # Initialize chunker
    chunker = TokenSizeChunker(max_token_size, overlap_token_size)
    
    results = []
    chunk_counter = 0
    title_list = title_list or [""] * len(doc_keys)
    
    for doc_index, tokens in enumerate(tokens_list):
        # Generate chunks and their lengths
        chunk_tokens = chunker.chunk_tokens(tokens)
        chunk_lengths = chunker.get_chunk_lengths(tokens)
        
        # Decode chunks back to text
        decoded_chunks = tiktoken_model.decode_batch(chunk_tokens)
        
        # Create chunk metadata
        for chunk_index, (chunk_text, chunk_length) in enumerate(
            zip(decoded_chunks, chunk_lengths)
        ):
            results.append({
                "tokens": chunk_length,
                "content": chunk_text.strip(),
                "index": chunk_counter,
                "doc_id": doc_keys[doc_index],
                "title": title_list[doc_index],
            })
            chunk_counter += 1
    
    return results