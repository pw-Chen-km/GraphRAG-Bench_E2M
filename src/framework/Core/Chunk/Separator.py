"""
Separator-based text chunking implementation.

This module provides functionality for splitting text into chunks based on
custom separators, with support for overlap and size constraints.
"""

from typing import List, Optional, Union, Literal, Callable, Any, Dict, List, Any
from Core.Chunk.ChunkFactory import register_chunking_method
from Core.Common.Constants import DEFAULT_TEXT_SEPARATORS


class SeparatorBasedSplitter:
    """
    Splits text tokens based on specified separators.
    
    This class implements a sophisticated text splitting algorithm that
    respects natural boundaries defined by separators while maintaining
    chunk size constraints and overlap requirements.
    """
    
    def __init__(
        self,
        separators: Optional[List[List[int]]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = "end",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable = len,
    ):
        """
        Initialize the separator-based splitter.
        
        Args:
            separators: List of token sequences that act as separators
            keep_separator: Whether and where to keep separators in chunks
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of tokens to overlap between chunks
            length_function: Function to calculate chunk length
        """
        self._separators = separators or []
        self._keep_separator = keep_separator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
    
    def split_tokens(self, tokens: List[int]) -> List[List[int]]:
        """
        Split tokens into chunks based on separators.
        
        Args:
            tokens: Input token sequence
            
        Returns:
            List of token chunks
        """
        initial_splits = self._split_by_separators(tokens)
        merged_chunks = self._merge_splits_into_chunks(initial_splits)
        return self._apply_overlap_strategy(merged_chunks)
    
    def _split_by_separators(self, tokens: List[int]) -> List[List[int]]:
        """
        Split tokens at separator boundaries.
        
        Args:
            tokens: Input token sequence
            
        Returns:
            List of token splits
        """
        splits = []
        current_split = []
        position = 0
        
        while position < len(tokens):
            separator_found = self._find_separator_at_position(tokens, position)
            
            if separator_found:
                separator_tokens = separator_found
                self._handle_separator_found(current_split, splits, separator_tokens)
                position += len(separator_tokens)
            else:
                current_split.append(tokens[position])
                position += 1
        
        # Add remaining tokens
        if current_split:
            splits.append(current_split)
        
        return [split for split in splits if split]
    
    def _find_separator_at_position(
        self, 
        tokens: List[int], 
        position: int
    ) -> Optional[List[int]]:
        """
        Find if any separator exists at the given position.
        
        Args:
            tokens: Token sequence to search in
            position: Position to check for separators
            
        Returns:
            Matching separator tokens or None
        """
        for separator in self._separators:
            if (position + len(separator) <= len(tokens) and 
                tokens[position:position + len(separator)] == separator):
                return separator
        return None
    
    def _handle_separator_found(
        self, 
        current_split: List[int], 
        splits: List[List[int]], 
        separator_tokens: List[int]
    ) -> None:
        """
        Handle the case when a separator is found.
        
        Args:
            current_split: Current token split being built
            splits: List of all splits
            separator_tokens: The separator tokens found
        """
        if self._keep_separator in [True, "end"]:
            current_split.extend(separator_tokens)
        
        if current_split:
            splits.append(current_split)
            current_split.clear()
        
        if self._keep_separator == "start":
            current_split.extend(separator_tokens)
    
    def _merge_splits_into_chunks(self, splits: List[List[int]]) -> List[List[int]]:
        """
        Merge splits into chunks respecting size constraints.
        
        Args:
            splits: List of token splits
            
        Returns:
            List of merged chunks
        """
        if not splits:
            return []
        
        merged_chunks = []
        current_chunk = []
        
        for split in splits:
            if not current_chunk:
                current_chunk = split
            elif self._can_merge_chunks(current_chunk, split):
                current_chunk.extend(split)
            else:
                merged_chunks.append(current_chunk)
                current_chunk = split
        
        if current_chunk:
            merged_chunks.append(current_chunk)
        
        # Handle oversized chunks
        if len(merged_chunks) == 1 and self._is_chunk_oversized(merged_chunks[0]):
            return self._split_oversized_chunk(merged_chunks[0])
        
        return merged_chunks
    
    def _can_merge_chunks(self, current_chunk: List[int], split: List[int]) -> bool:
        """
        Check if a split can be merged into the current chunk.
        
        Args:
            current_chunk: Current chunk being built
            split: Split to potentially merge
            
        Returns:
            True if merge is possible
        """
        return (self._length_function(current_chunk) + 
                self._length_function(split) <= self._chunk_size)
    
    def _is_chunk_oversized(self, chunk: List[int]) -> bool:
        """
        Check if a chunk exceeds the maximum size.
        
        Args:
            chunk: Chunk to check
            
        Returns:
            True if chunk is oversized
        """
        return self._length_function(chunk) > self._chunk_size
    
    def _split_oversized_chunk(self, chunk: List[int]) -> List[List[int]]:
        """
        Split an oversized chunk into smaller pieces.
        
        Args:
            chunk: Oversized chunk to split
            
        Returns:
            List of smaller chunks
        """
        result = []
        step_size = self._chunk_size - self._chunk_overlap
        
        for start in range(0, len(chunk), step_size):
            end = start + self._chunk_size
            new_chunk = chunk[start:end]
            
            # Only add chunks that are larger than overlap
            if len(new_chunk) > self._chunk_overlap:
                result.append(new_chunk)
        
        return result
    
    def _apply_overlap_strategy(self, chunks: List[List[int]]) -> List[List[int]]:
        """
        Apply overlap strategy to chunks.
        
        Args:
            chunks: List of chunks to process
            
        Returns:
            List of chunks with overlap applied
        """
        if self._chunk_overlap <= 0:
            return chunks
        
        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
            else:
                overlapped_chunk = self._create_overlapped_chunk(chunks[i - 1], chunk)
                result.append(overlapped_chunk)
        
        return result
    
    def _create_overlapped_chunk(
        self, 
        previous_chunk: List[int], 
        current_chunk: List[int]
    ) -> List[int]:
        """
        Create a chunk with overlap from the previous chunk.
        
        Args:
            previous_chunk: Previous chunk to overlap from
            current_chunk: Current chunk to add overlap to
            
        Returns:
            Chunk with overlap applied
        """
        overlap_tokens = previous_chunk[-self._chunk_overlap:]
        overlapped_chunk = overlap_tokens + current_chunk
        
        # Ensure the chunk doesn't exceed maximum size
        if self._length_function(overlapped_chunk) > self._chunk_size:
            overlapped_chunk = overlapped_chunk[:self._chunk_size]
        
        return overlapped_chunk


@register_chunking_method("chunking_by_seperators")
async def chunking_by_seperators(
    tokens_list: List[List[int]], 
    doc_keys: List[str], 
    tiktoken_model: Any, 
    overlap_token_size: int = 128,
    max_token_size: int = 1024, 
    title_list: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Chunk documents using separator-based splitting.
    
    This function processes a list of tokenized documents and splits them
    into chunks based on natural separators while respecting size and
    overlap constraints.
    
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
    # Initialize splitter with default separators
    splitter = SeparatorBasedSplitter(
        separators=[tiktoken_model.encode(sep) for sep in DEFAULT_TEXT_SEPARATORS],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    
    results = []
    title_list = title_list or [""] * len(doc_keys)
    
    for doc_index, tokens in enumerate(tokens_list):
        # Split tokens into chunks
        chunk_tokens = splitter.split_tokens(tokens)
        chunk_lengths = [len(chunk) for chunk in chunk_tokens]
        
        # Decode chunks back to text
        decoded_chunks = tiktoken_model.decode_batch(chunk_tokens)
        
        # Create chunk metadata
        for chunk_index, (chunk_text, chunk_length) in enumerate(
            zip(decoded_chunks, chunk_lengths)
        ):
            results.append({
                "tokens": chunk_length,
                "content": chunk_text.strip(),
                "index": chunk_index,
                "doc_id": doc_keys[doc_index],
                "title": title_list[doc_index],
            })
    
    return results
