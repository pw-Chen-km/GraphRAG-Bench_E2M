"""
Configuration for document chunking operations.
Defines parameters for splitting documents into manageable chunks.
"""

from Core.Utils.YamlModel import YamlModel


class ChunkConfig(YamlModel):
    """
    Configuration class for document chunking parameters.
    
    Attributes:
        chunk_token_size: Maximum number of tokens per chunk
        chunk_overlap_token_size: Number of overlapping tokens between chunks
        chunk_method: Method used for chunking documents
    """
    
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    chunk_method: str = "chunking_by_token_size"
