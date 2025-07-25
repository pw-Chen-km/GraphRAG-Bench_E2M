"""
Constants Module - System-wide constants and configuration
Defines all constants used throughout the GraphRAG system
"""

import os
from pathlib import Path
from enum import Enum
from typing import List

from loguru import logger


# Animation and UI Constants
PROCESS_TICKERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Text Processing Constants
DEFAULT_TEXT_SEPARATORS = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]

# Graph Processing Constants
GRAPH_FIELD_SEPARATOR = "<SEP>"
GRAPH_FIELD_SEP = GRAPH_FIELD_SEPARATOR  # Alias for backward compatibility
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"

# Message Routing Constants
IGNORED_MESSAGE_ID = "0"
MESSAGE_ROUTE_FROM = "sent_from"
MESSAGE_ROUTE_TO = "send_to"
MESSAGE_ROUTE_CAUSE_BY = "cause_by"
MESSAGE_META_ROLE = "role"
MESSAGE_ROUTE_TO_ALL = "<all>"
MESSAGE_ROUTE_TO_NONE = "<none>"

# Medical Graph RAG Constants
NODE_PATTERN = r"Node\(id='(.*?)', type='(.*?)'\)"
RELATIONSHIP_PATTERN = r"Relationship\(subj=Node\(id='(.*?)', type='(.*?)'\), obj=Node\(id='(.*?)', type='(.*?)'\), type='(.*?)'\)"

# External API Constants
GCUBE_TOKEN = '07e1bd33-c0f5-41b0-979b-4c9a859eec3f-843339462'

# UI and Display Constants
HEX_COLOR = "#ea6eaf"
ANSI_COLOR = f"\033[38;2;{int(HEX_COLOR[1:3], 16)};{int(HEX_COLOR[3:5], 16)};{int(HEX_COLOR[5:7], 16)}m"
TOKEN_TO_CHAR_RATIO = 4

# Timeout Constants
USE_CONFIG_TIMEOUT = 0  # Using llm.timeout configuration
LLM_API_TIMEOUT = 300


class RetrieverType(Enum):
    """Enumeration of retriever types"""
    ENTITY = "entity"
    RELATION = "relationship"
    CHUNK = "chunk"
    COMMUNITY = "community"
    SUBGRAPH = "subgraph"


class Retriever:
    """Retriever constants for different retrieval types"""
    ENTITY = "entity"
    RELATION = "relationship"
    CHUNK = "chunk"
    COMMUNITY = "community"
    SUBGRAPH = "subgraph"


class PathManager:
    """Manages system paths and directories"""
    
    @staticmethod
    def get_package_root() -> Path:
        """
        Get the current working directory as package root.
        
        Returns:
            Path object representing package root
        """
        return Path.cwd()

    @staticmethod
    def get_project_root() -> Path:
        """
        Get the project root directory.
        
        Checks environment variable METAGPT_PROJECT_ROOT first,
        falls back to package root if not set.
        
        Returns:
            Path object representing project root
        """
        project_root_env = os.getenv("METAGPT_PROJECT_ROOT")
        if project_root_env:
            project_root = Path(project_root_env)
            logger.info(f"PROJECT_ROOT set from environment variable to {str(project_root)}")
        else:
            project_root = PathManager.get_package_root()
        
        return project_root

    @staticmethod
    def get_config_root() -> Path:
        """
        Get the configuration root directory.
        
        Returns:
            Path object representing config root
        """
        return Path.home() / "Option"


# Global path constants
GRAPHRAG_ROOT = PathManager.get_project_root()
CONFIG_ROOT = PathManager.get_config_root()


class SystemConfig:
    """System configuration constants"""
    
    # Default values
    DEFAULT_MAX_BUDGET = 10.0
    DEFAULT_TIMEOUT = 300
    
    # File extensions
    SUPPORTED_TEXT_EXTENSIONS = ['.txt', '.md', '.py', '.js', '.html', '.css']
    SUPPORTED_DOCUMENT_EXTENSIONS = ['.pdf', '.docx', '.doc']
    
    # Encoding
    DEFAULT_ENCODING = 'utf-8'
    
    # Cache settings
    DEFAULT_CACHE_SIZE = 1000
    DEFAULT_CACHE_TTL = 3600  # 1 hour


class ValidationRules:
    """Validation rules and constraints"""
    
    # Text validation
    MIN_TEXT_LENGTH = 1
    MAX_TEXT_LENGTH = 10000
    
    # List validation
    MIN_LIST_ITEMS = 0
    MAX_LIST_ITEMS = 10000
    
    # Token limits
    MAX_TOKENS_PER_REQUEST = 8192
    MAX_TOKENS_PER_RESPONSE = 4096
    
    # File size limits (in bytes)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
