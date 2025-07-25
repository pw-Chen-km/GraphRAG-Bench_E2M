"""
Memory Module - Message storage and retrieval system
Provides flexible memory management for storing and retrieving messages
"""

from collections import defaultdict
from typing import DefaultDict, Iterable, Set, List, Optional, Callable

from pydantic import BaseModel, Field, SerializeAsAny

from Core.Common.Constants import IGNORED_MESSAGE_ID
from Core.Schema.Message import Message
from Core.Common.Utils import any_to_str, any_to_str_set


class Memory(BaseModel):
    """
    Basic memory system for storing and retrieving messages.
    Provides indexing and search capabilities for efficient message retrieval.
    """

    storage: List[SerializeAsAny[Message]] = Field(default_factory=list)
    ignore_id: bool = False
    
    # Index for efficient content-based search
    content_index: DefaultDict[str, Set[int]] = Field(default_factory=lambda: defaultdict(set), exclude=True)
    keyword_index: DefaultDict[str, Set[int]] = Field(default_factory=lambda: defaultdict(set), exclude=True)

    def add(self, message: Message) -> None:
        """
        Add a new message to storage and update indices.
        
        Args:
            message: Message to add
        """
        if self.ignore_id:
            message.id = IGNORED_MESSAGE_ID
            
        if message in self.storage:
            return
            
        # Add to storage
        self.storage.append(message)
        
        # Update indices
        self._update_indices(message, len(self.storage) - 1)

    def add_batch(self, messages: Iterable[Message]) -> None:
        """
        Add multiple messages to storage.
        
        Args:
            messages: Iterable of messages to add
        """
        for message in messages:
            self.add(message)

    def _update_indices(self, message: Message, index: int) -> None:
        """
        Update search indices for a message.
        
        Args:
            message: Message to index
            index: Position of message in storage
        """
        # Content index
        content_words = set(message.content.lower().split())
        for word in content_words:
            self.content_index[word].add(index)
        
        # Keyword index (using any_to_str_set for broader matching)
        keywords = any_to_str_set(message.content)
        for keyword in keywords:
            self.keyword_index[keyword].add(index)

    def get_by_content(self, content: str) -> List[Message]:
        """
        Retrieve messages containing specified content.
        
        Args:
            content: Content to search for
            
        Returns:
            List of messages containing the content
        """
        return [message for message in self.storage if content in message.content]

    def get_by_keyword(self, keyword: str) -> List[Message]:
        """
        Retrieve messages by keyword using index.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of messages containing the keyword
        """
        keyword_lower = keyword.lower()
        if keyword_lower in self.keyword_index:
            indices = self.keyword_index[keyword_lower]
            return [self.storage[i] for i in indices if i < len(self.storage)]
        return []

    def delete_newest(self) -> Optional[Message]:
        """
        Remove and return the newest message from storage.
        
        Returns:
            Removed message or None if storage is empty
        """
        if self.storage:
            newest_msg = self.storage.pop()
            self._remove_from_indices(len(self.storage))
            return newest_msg
        return None

    def delete(self, message: Message) -> None:
        """
        Remove a specific message from storage.
        
        Args:
            message: Message to remove
        """
        if self.ignore_id:
            message.id = IGNORED_MESSAGE_ID
            
        if message in self.storage:
            index = self.storage.index(message)
            self.storage.remove(message)
            self._remove_from_indices(index)

    def _remove_from_indices(self, index: int) -> None:
        """
        Remove message at index from all indices.
        
        Args:
            index: Index of message to remove
        """
        # Remove from content index
        for word_indices in self.content_index.values():
            word_indices.discard(index)
        
        # Remove from keyword index
        for keyword_indices in self.keyword_index.values():
            keyword_indices.discard(index)

    def clear(self) -> None:
        """Clear all messages and indices"""
        self.storage.clear()
        self.content_index.clear()
        self.keyword_index.clear()

    def count(self) -> int:
        """
        Get the number of messages in storage.
        
        Returns:
            Number of messages
        """
        return len(self.storage)

    def try_remember(self, keyword: str) -> List[Message]:
        """
        Try to recall messages containing a keyword.
        Uses both direct content search and indexed search.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of messages containing the keyword
        """
        # Try indexed search first
        indexed_results = self.get_by_keyword(keyword)
        if indexed_results:
            return indexed_results
        
        # Fall back to content search
        return self.get_by_content(keyword)

    def get(self, k: int = 0) -> List[Message]:
        """
        Get the most recent k messages.
        
        Args:
            k: Number of recent messages to retrieve. If 0, returns all messages
            
        Returns:
            List of recent messages
        """
        if k == 0:
            return self.storage.copy()
        return self.storage[-k:]

    def find_new_messages(self, observed: List[Message], k: int = 0) -> List[Message]:
        """
        Find messages that are not in the most recent k messages.
        
        Args:
            observed: List of observed messages
            k: Number of recent messages to check against. If 0, checks all messages
            
        Returns:
            List of new messages not in recent k messages
        """
        already_observed = self.get(k)
        new_messages = []
        
        for message in observed:
            if message not in already_observed:
                new_messages.append(message)
                
        return new_messages

    def search_by_criteria(self, criteria: Callable[[Message], bool]) -> List[Message]:
        """
        Search messages using custom criteria function.
        
        Args:
            criteria: Function that takes a message and returns True if it matches
            
        Returns:
            List of messages matching the criteria
        """
        return [message for message in self.storage if criteria(message)]

    def get_statistics(self) -> dict:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            "total_messages": len(self.storage),
            "unique_keywords": len(self.keyword_index),
            "unique_content_words": len(self.content_index),
            "storage_size": len(self.storage)
        }

    def export_messages(self, format_type: str = "list") -> any:
        """
        Export messages in specified format.
        
        Args:
            format_type: Export format ("list", "dict", "json")
            
        Returns:
            Exported messages in specified format
        """
        if format_type == "list":
            return self.storage.copy()
        elif format_type == "dict":
            return [message.model_dump() for message in self.storage]
        elif format_type == "json":
            return [message.model_dump_json() for message in self.storage]
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

