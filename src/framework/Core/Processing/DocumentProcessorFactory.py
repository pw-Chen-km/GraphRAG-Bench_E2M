"""
Document processor factory for managing different document processing strategies.
Provides a unified interface for various document chunking approaches using factory and strategy patterns.
"""

import asyncio
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import tiktoken

from Core.Common.Logger import logger
from Core.Schema.ChunkSchema import TextChunk
from Core.Utils.Display import StatusDisplay, ProgressDisplay


class ChunkingStrategy(Enum):
    """Supported chunking strategies."""
    TOKEN_BASED = "token_based"
    SENTENCE_BASED = "sentence_based"
    PARAGRAPH_BASED = "paragraph_based"
    SEMANTIC_BASED = "semantic_based"
    HYBRID = "hybrid"


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.TOKEN_BASED
    max_tokens: int = 1200
    overlap_tokens: int = 100
    min_tokens: int = 50
    max_chunks: int = 1000
    preserve_formatting: bool = True
    include_metadata: bool = True


class DocumentProcessor(ABC):
    """
    Abstract base class for document processors.
    
    Provides a unified interface for different document processing strategies
    with common functionality for chunking and metadata management.
    """
    
    def __init__(self, config: ChunkingConfig, context: Any):
        """Initialize document processor with configuration."""
        self.config = config
        self.context = context
        self.encoder = context.encoder
        self.processed_chunks: List[TextChunk] = []
    
    @abstractmethod
    async def process_documents(self, documents: Union[str, List[Any]]) -> List[TextChunk]:
        """Process documents and return text chunks."""
        pass
    
    @abstractmethod
    def _create_chunk(self, content: str, metadata: Dict[str, Any]) -> TextChunk:
        """Create a text chunk with content and metadata."""
        pass
    
    async def execute(self, documents: Union[str, List[Any]], *args, **kwargs) -> List[TextChunk]:
        """Execute document processing (alias for process_documents)."""
        return await self.process_documents(documents)
    
    async def get_processed_chunks(self) -> List[TextChunk]:
        """Get all processed text chunks."""
        return self.processed_chunks
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the encoder."""
        return len(self.encoder.encode(text))


class TokenBasedProcessor(DocumentProcessor):
    """Document processor based on token count chunking."""
    
    async def process_documents(self, documents: Union[str, List[Any]]) -> List[TextChunk]:
        """Process documents using token-based chunking."""
        StatusDisplay.show_processing_status("Document chunking", details="Token-based")
        
        if isinstance(documents, str):
            documents = [documents]
        
        all_chunks = []
        total_docs = len(documents)
        
        for i, doc in enumerate(documents):
            ProgressDisplay.show_progress(i + 1, total_docs, "Processing documents")
            
            if isinstance(doc, str):
                chunks = await self._chunk_text(doc, f"doc_{i}")
            else:
                chunks = await self._chunk_document_object(doc, f"doc_{i}")
            
            all_chunks.extend(chunks)
        
        self.processed_chunks = all_chunks
        StatusDisplay.show_success(f"Document chunking completed, generated {len(all_chunks)} text chunks")
        return all_chunks
    
    async def _chunk_text(self, text: str, doc_id: str) -> List[TextChunk]:
        """Chunk text based on token count."""
        chunks = []
        tokens = self.encoder.encode(text)
        
        start = 0
        while start < len(tokens):
            end = min(start + self.config.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            
            metadata = {
                "doc_id": doc_id,
                "chunk_id": len(chunks),
                "start_token": start,
                "end_token": end,
                "token_count": len(chunk_tokens)
            }
            
            chunk = self._create_chunk(chunk_text, metadata)
            chunks.append(chunk)
            
            # Calculate overlap
            overlap_start = max(0, end - self.config.overlap_tokens)
            start = overlap_start if start < len(tokens) - self.config.max_tokens else end
        
        return chunks
    
    async def _chunk_document_object(self, doc: Any, doc_id: str) -> List[TextChunk]:
        """Chunk document object based on its structure."""
        # Handle different document object structures
        if hasattr(doc, 'text'):
            return await self._chunk_text(doc.text, doc_id)
        elif hasattr(doc, 'content'):
            return await self._chunk_text(doc.content, doc_id)
        else:
            return await self._chunk_text(str(doc), doc_id)
    
    def _create_chunk(self, content: str, metadata: Dict[str, Any]) -> TextChunk:
        """Create a text chunk."""
        return TextChunk(
            tokens=metadata.get("token_count", 0),
            chunk_id=str(metadata.get("chunk_id", 0)),
            content=content,
            doc_id=metadata.get("doc_id", "unknown"),
            index=metadata.get("chunk_id", 0),
            title=metadata.get("title", None)
        )
    
    # Adapter methods for compatibility with existing retriever code
    async def get_data_by_key(self, chunk_id: str) -> Optional[str]:
        """
        Get chunk content by chunk ID (adapter method for compatibility).
        
        Args:
            chunk_id: The chunk ID to look up
            
        Returns:
            The chunk content as string or None if not found
        """
        try:
            # Try to parse chunk_id as integer index
            index = int(chunk_id)
            if 0 <= index < len(self.processed_chunks):
                return self.processed_chunks[index].content
        except (ValueError, TypeError):
            # If chunk_id is not an integer, try to find by chunk_id field
            for chunk in self.processed_chunks:
                if chunk.chunk_id == chunk_id:
                    return chunk.content
        
        return None
    
    async def get_data_by_indices(self, indices: List[int]) -> List[str]:
        """
        Get chunk contents by indices (adapter method for compatibility).
        
        Args:
            indices: List of chunk indices
            
        Returns:
            List of chunk contents as strings
        """
        results = []
        for index in indices:
            if 0 <= index < len(self.processed_chunks):
                results.append(self.processed_chunks[index].content)
            else:
                results.append("")  # Return empty string for invalid indices
        return results
    
    async def size(self) -> int:
        """
        Get the total number of chunks (adapter method for compatibility).
        
        Returns:
            The number of processed chunks
        """
        return len(self.processed_chunks)


class SentenceBasedProcessor(DocumentProcessor):
    """Document processor based on sentence boundaries."""
    
    async def process_documents(self, documents: Union[str, List[Any]]) -> List[TextChunk]:
        """Process documents using sentence-based chunking."""
        StatusDisplay.show_processing_status("Document chunking", details="Sentence-based")
        
        if isinstance(documents, str):
            documents = [documents]
        
        all_chunks = []
        total_docs = len(documents)
        
        for i, doc in enumerate(documents):
            ProgressDisplay.show_progress(i + 1, total_docs, "Processing documents")
            
            if isinstance(doc, str):
                chunks = await self._chunk_by_sentences(doc, f"doc_{i}")
            else:
                chunks = await self._chunk_document_object(doc, f"doc_{i}")
            
            all_chunks.extend(chunks)
        
        self.processed_chunks = all_chunks
        StatusDisplay.show_success(f"Document chunking completed, generated {len(all_chunks)} text chunks")
        return all_chunks
    
    async def _chunk_by_sentences(self, text: str, doc_id: str) -> List[TextChunk]:
        """Chunk text by sentence boundaries."""
        # Simple sentence splitting (can be improved as needed)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.config.max_tokens and current_chunk:
                # Create current chunk
                chunk_text = " ".join(current_chunk)
                metadata = {
                    "doc_id": doc_id,
                    "chunk_id": len(chunks),
                    "sentence_count": len(current_chunk),
                    "token_count": current_tokens
                }
                
                chunk = self._create_chunk(chunk_text, metadata)
                chunks.append(chunk)
                
                # Reset
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Process last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            metadata = {
                "doc_id": doc_id,
                "chunk_id": len(chunks),
                "sentence_count": len(current_chunk),
                "token_count": current_tokens
            }
            
            chunk = self._create_chunk(chunk_text, metadata)
            chunks.append(chunk)
        
        return chunks
    
    async def _chunk_document_object(self, doc: Any, doc_id: str) -> List[TextChunk]:
        """Chunk document object based on its structure."""
        # Handle different document object structures
        if hasattr(doc, 'text'):
            return await self._chunk_by_sentences(doc.text, doc_id)
        elif hasattr(doc, 'content'):
            return await self._chunk_by_sentences(doc.content, doc_id)
        else:
            return await self._chunk_by_sentences(str(doc), doc_id)
    
    def _create_chunk(self, content: str, metadata: Dict[str, Any]) -> TextChunk:
        """Create a text chunk."""
        return TextChunk(
            tokens=metadata.get("token_count", 0),
            chunk_id=str(metadata.get("chunk_id", 0)),
            content=content,
            doc_id=metadata.get("doc_id", "unknown"),
            index=metadata.get("chunk_id", 0),
            title=metadata.get("title", None)
        )
    
    # Adapter methods for compatibility with existing retriever code
    async def get_data_by_key(self, chunk_id: str) -> Optional[str]:
        """
        Get chunk content by chunk ID (adapter method for compatibility).
        
        Args:
            chunk_id: The chunk ID to look up
            
        Returns:
            The chunk content as string or None if not found
        """
        try:
            # Try to parse chunk_id as integer index
            index = int(chunk_id)
            if 0 <= index < len(self.processed_chunks):
                return self.processed_chunks[index].content
        except (ValueError, TypeError):
            # If chunk_id is not an integer, try to find by chunk_id field
            for chunk in self.processed_chunks:
                if chunk.chunk_id == chunk_id:
                    return chunk.content
        
        return None
    
    async def get_data_by_indices(self, indices: List[int]) -> List[str]:
        """
        Get chunk contents by indices (adapter method for compatibility).
        
        Args:
            indices: List of chunk indices
            
        Returns:
            List of chunk contents as strings
        """
        results = []
        for index in indices:
            if 0 <= index < len(self.processed_chunks):
                results.append(self.processed_chunks[index].content)
            else:
                results.append("")  # Return empty string for invalid indices
        return results
    
    async def size(self) -> int:
        """
        Get the total number of chunks (adapter method for compatibility).
        
        Returns:
            The number of processed chunks
        """
        return len(self.processed_chunks)


class SemanticProcessor(DocumentProcessor):
    """Document processor based on semantic similarity."""
    
    async def process_documents(self, documents: Union[str, List[Any]]) -> List[TextChunk]:
        """Process documents using semantic-based chunking."""
        StatusDisplay.show_processing_status("Document chunking", details="Semantic-based")
        
        if isinstance(documents, str):
            documents = [documents]
        
        all_chunks = []
        total_docs = len(documents)
        
        for i, doc in enumerate(documents):
            ProgressDisplay.show_progress(i + 1, total_docs, "Processing documents")
            
            if isinstance(doc, str):
                chunks = await self._chunk_by_semantics(doc, f"doc_{i}")
            else:
                chunks = await self._chunk_document_object(doc, f"doc_{i}")
            
            all_chunks.extend(chunks)
        
        self.processed_chunks = all_chunks
        StatusDisplay.show_success(f"Document chunking completed, generated {len(all_chunks)} text chunks")
        return all_chunks
    
    async def _chunk_by_semantics(self, text: str, doc_id: str) -> List[TextChunk]:
        """Chunk text based on semantic similarity."""
        # Implement more complex semantic chunking logic here
        # For example, using sentence embeddings and clustering
        return await self._fallback_to_token_based(text, doc_id)
    
    async def _fallback_to_token_based(self, text: str, doc_id: str) -> List[TextChunk]:
        """Fallback to token-based chunking."""
        processor = TokenBasedProcessor(self.config, self.context)
        return await processor._chunk_text(text, doc_id)
    
    async def _chunk_document_object(self, doc: Any, doc_id: str) -> List[TextChunk]:
        """Chunk document object based on its structure."""
        # Handle different document object structures
        if hasattr(doc, 'text'):
            return await self._chunk_by_semantics(doc.text, doc_id)
        elif hasattr(doc, 'content'):
            return await self._chunk_by_semantics(doc.content, doc_id)
        else:
            return await self._chunk_by_semantics(str(doc), doc_id)
    
    def _create_chunk(self, content: str, metadata: Dict[str, Any]) -> TextChunk:
        """Create a text chunk."""
        return TextChunk(
            tokens=metadata.get("token_count", 0),
            chunk_id=str(metadata.get("chunk_id", 0)),
            content=content,
            doc_id=metadata.get("doc_id", "unknown"),
            index=metadata.get("chunk_id", 0),
            title=metadata.get("title", None)
        )
    
    # Adapter methods for compatibility with existing retriever code
    async def get_data_by_key(self, chunk_id: str) -> Optional[str]:
        """
        Get chunk content by chunk ID (adapter method for compatibility).
        
        Args:
            chunk_id: The chunk ID to look up
            
        Returns:
            The chunk content as string or None if not found
        """
        try:
            # Try to parse chunk_id as integer index
            index = int(chunk_id)
            if 0 <= index < len(self.processed_chunks):
                return self.processed_chunks[index].content
        except (ValueError, TypeError):
            # If chunk_id is not an integer, try to find by chunk_id field
            for chunk in self.processed_chunks:
                if chunk.chunk_id == chunk_id:
                    return chunk.content
        
        return None
    
    async def get_data_by_indices(self, indices: List[int]) -> List[str]:
        """
        Get chunk contents by indices (adapter method for compatibility).
        
        Args:
            indices: List of chunk indices
            
        Returns:
            List of chunk contents as strings
        """
        results = []
        for index in indices:
            if 0 <= index < len(self.processed_chunks):
                results.append(self.processed_chunks[index].content)
            else:
                results.append("")  # Return empty string for invalid indices
        return results
    
    async def size(self) -> int:
        """
        Get the total number of chunks (adapter method for compatibility).
        
        Returns:
            The number of processed chunks
        """
        return len(self.processed_chunks)


class HybridProcessor(DocumentProcessor):
    """Document processor using hybrid chunking strategy."""
    
    async def process_documents(self, documents: Union[str, List[Any]]) -> List[TextChunk]:
        """Process documents using hybrid chunking strategy."""
        StatusDisplay.show_processing_status("Document chunking", details="Hybrid strategy")
        
        if isinstance(documents, str):
            documents = [documents]
        
        all_chunks = []
        total_docs = len(documents)
        
        for i, doc in enumerate(documents):
            ProgressDisplay.show_progress(i + 1, total_docs, "Processing documents")
            
            if isinstance(doc, str):
                chunks = await self._hybrid_chunking(doc, f"doc_{i}")
            else:
                chunks = await self._chunk_document_object(doc, f"doc_{i}")
            
            all_chunks.extend(chunks)
        
        self.processed_chunks = all_chunks
        StatusDisplay.show_success(f"Document chunking completed, generated {len(all_chunks)} text chunks")
        return all_chunks
    
    async def _hybrid_chunking(self, text: str, doc_id: str) -> List[TextChunk]:
        """Apply hybrid chunking strategy."""
        # First split by sentences
        sentence_processor = SentenceBasedProcessor(self.config, self.context)
        sentence_chunks = await sentence_processor._chunk_by_sentences(text, doc_id)
        
        # Then split overly long chunks at token level
        final_chunks = []
        for chunk in sentence_chunks:
            if self._count_tokens(chunk.content) > self.config.max_tokens:
                # Split overly long chunks at token level
                token_processor = TokenBasedProcessor(self.config, self.context)
                sub_chunks = await token_processor._chunk_text(chunk.content, f"{doc_id}_sub")
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    async def _chunk_document_object(self, doc: Any, doc_id: str) -> List[TextChunk]:
        """Chunk document object based on its structure."""
        # Handle different document object structures
        if hasattr(doc, 'text'):
            return await self._hybrid_chunking(doc.text, doc_id)
        elif hasattr(doc, 'content'):
            return await self._hybrid_chunking(doc.content, doc_id)
        else:
            return await self._hybrid_chunking(str(doc), doc_id)
    
    def _create_chunk(self, content: str, metadata: Dict[str, Any]) -> TextChunk:
        """Create a text chunk."""
        return TextChunk(
            tokens=metadata.get("token_count", 0),
            chunk_id=str(metadata.get("chunk_id", 0)),
            content=content,
            doc_id=metadata.get("doc_id", "unknown"),
            index=metadata.get("chunk_id", 0),
            title=metadata.get("title", None)
        )
    
    # Adapter methods for compatibility with existing retriever code
    async def get_data_by_key(self, chunk_id: str) -> Optional[str]:
        """
        Get chunk content by chunk ID (adapter method for compatibility).
        
        Args:
            chunk_id: The chunk ID to look up
            
        Returns:
            The chunk content as string or None if not found
        """
        try:
            # Try to parse chunk_id as integer index
            index = int(chunk_id)
            if 0 <= index < len(self.processed_chunks):
                return self.processed_chunks[index].content
        except (ValueError, TypeError):
            # If chunk_id is not an integer, try to find by chunk_id field
            for chunk in self.processed_chunks:
                if chunk.chunk_id == chunk_id:
                    return chunk.content
        
        return None
    
    async def get_data_by_indices(self, indices: List[int]) -> List[str]:
        """
        Get chunk contents by indices (adapter method for compatibility).
        
        Args:
            indices: List of chunk indices
            
        Returns:
            List of chunk contents as strings
        """
        results = []
        for index in indices:
            if 0 <= index < len(self.processed_chunks):
                results.append(self.processed_chunks[index].content)
            else:
                results.append("")  # Return empty string for invalid indices
        return results
    
    async def size(self) -> int:
        """
        Get the total number of chunks (adapter method for compatibility).
        
        Returns:
            The number of processed chunks
        """
        return len(self.processed_chunks)


class DocumentProcessorFactory:
    """
    Factory class for creating document processors.
    
    Provides a centralized way to instantiate different document processing
    strategies based on configuration parameters.
    """
    
    _processors = {
        ChunkingStrategy.TOKEN_BASED: TokenBasedProcessor,
        ChunkingStrategy.SENTENCE_BASED: SentenceBasedProcessor,
        ChunkingStrategy.PARAGRAPH_BASED: TokenBasedProcessor,  # Fallback to token-based for paragraph
        ChunkingStrategy.SEMANTIC_BASED: SemanticProcessor,
        ChunkingStrategy.HYBRID: HybridProcessor
    }
    
    @classmethod
    def create_processor(cls, config: Any, context: Any) -> DocumentProcessor:
        """
        Create a document processor based on configuration.
        
        Args:
            config: Configuration object containing chunking parameters
            context: Context object for the processor
            
        Returns:
            DocumentProcessor: Instance of the appropriate document processor
        """
        # Map configuration chunk method to ChunkingStrategy
        chunk_method = config.chunk.chunk_method.replace("chunking_by_", "")
        strategy_mapping = {
            "token_size": ChunkingStrategy.TOKEN_BASED,
            "sentence": ChunkingStrategy.SENTENCE_BASED,
            "paragraph": ChunkingStrategy.PARAGRAPH_BASED,
            "semantic": ChunkingStrategy.SEMANTIC_BASED,
            "hybrid": ChunkingStrategy.HYBRID
        }
        
        strategy = strategy_mapping.get(chunk_method, ChunkingStrategy.TOKEN_BASED)
        
        # Extract chunking configuration from config
        chunking_config = ChunkingConfig(
            strategy=strategy,
            max_tokens=config.chunk.chunk_token_size,
            overlap_tokens=config.chunk.chunk_overlap_token_size,
            min_tokens=50,
            max_chunks=1000,
            preserve_formatting=True,
            include_metadata=True
        )
        
        processor_class = cls._processors.get(chunking_config.strategy, TokenBasedProcessor)
        return processor_class(chunking_config, context)
    
    @classmethod
    def register_processor(cls, strategy: ChunkingStrategy, processor_class: type):
        """
        Register a new document processor.
        
        Args:
            strategy: Chunking strategy to register
            processor_class: Class implementing the document processor
        """
        cls._processors[strategy] = processor_class
        logger.info(f"Registered new document processor: {strategy.value}")
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available chunking strategies."""
        return [strategy.value for strategy in cls._processors.keys()] 