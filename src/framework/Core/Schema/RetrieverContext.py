from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class RetrieverContext:
    """
    Context container for retriever operations.
    
    Attributes:
        context: Dictionary storing context information
    """
    context: Dict[str, Any] = field(default_factory=dict)

    def register_context(self, key: str, value: Any) -> None:
        """Register a key-value pair in the context."""
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a value from context by key."""
        return self.context.get(key, default)

    def has_context(self, key: str) -> bool:
        """Check if a key exists in the context."""
        return key in self.context

    def to_dict(self) -> Dict[str, Any]:
        """Convert the context to a dictionary representation."""
        return self.context.copy()

    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Get configuration from context."""
        return self.context.get("config")

    @property
    def llm(self) -> Optional[Any]:
        """Get LLM instance from context."""
        return self.context.get("llm")