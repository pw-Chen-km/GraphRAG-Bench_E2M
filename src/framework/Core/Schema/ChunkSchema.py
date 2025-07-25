from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextChunk:
    """
    Represents a text chunk with metadata for document processing.
    
    Attributes:
        tokens: Number of tokens in the chunk
        chunk_id: Unique identifier for the chunk
        content: The actual text content
        doc_id: Document identifier this chunk belongs to
        index: Position index of the chunk in the document
        title: Optional title for the chunk
    """
    tokens: int
    chunk_id: str
    content: str
    doc_id: str
    index: int
    title: Optional[str] = field(default=None)

    def to_dict(self) -> dict:
        """Convert the chunk to a dictionary representation."""
        return {
            'tokens': self.tokens,
            'chunk_id': self.chunk_id,
            'content': self.content,
            'doc_id': self.doc_id,
            'index': self.index,
            'title': self.title
        }