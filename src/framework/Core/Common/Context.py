"""
Context Module - Application context and configuration management
Provides centralized context management for the GraphRAG system
"""

import os
from pathlib import Path
from typing import Any, Optional, Dict

from pydantic import BaseModel, ConfigDict

from Option.merged_config import MergedConfig as Config
from Config.LLMConfig import LLMConfig, LLMType
# Import BaseLLM as type hint only to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Core.Provider.BaseLLM import BaseLLM
# Import create_llm_instance as type hint only to avoid circular import
if TYPE_CHECKING:
    from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Common.CostManager import (
    CostManager,
    FireworksCostManager,
    TokenCostManager,
)


class AttrDict(BaseModel):
    """
    Dictionary-like object that allows attribute access to keys.
    Compatible with Pydantic and provides flexible attribute management.
    """

    model_config = ConfigDict(extra="allow")

    def __init__(self, **kwargs):
        """Initialize with keyword arguments"""
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

    def __getattr__(self, key: str) -> Any:
        """Get attribute value, returns None if not found"""
        return self.__dict__.get(key, None)

    def __setattr__(self, key: str, value: Any) -> None:
        """Set attribute value"""
        self.__dict__[key] = value

    def __delattr__(self, key: str) -> None:
        """Delete attribute if it exists"""
        if key in self.__dict__:
            del self.__dict__[key]
        else:
            raise AttributeError(f"No such attribute: {key}")

    def set(self, key: str, value: Any) -> None:
        """
        Set a key-value pair.
        
        Args:
            key: Attribute name
            value: Value to set
        """
        self.__dict__[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value by key with optional default.
        
        Args:
            key: Attribute name
            default: Default value if key doesn't exist
            
        Returns:
            Value associated with key or default
        """
        return self.__dict__.get(key, default)

    def remove(self, key: str) -> None:
        """
        Remove an attribute if it exists.
        
        Args:
            key: Attribute name to remove
        """
        if key in self.__dict__:
            del self.__dict__[key]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to regular dictionary.
        
        Returns:
            Dictionary representation of attributes
        """
        return self.__dict__.copy()

    def update(self, **kwargs) -> None:
        """
        Update multiple attributes at once.
        
        Args:
            **kwargs: Key-value pairs to update
        """
        self.__dict__.update(kwargs)

    def clear(self) -> None:
        """Clear all attributes"""
        self.__dict__.clear()


class Context(BaseModel):
    """
    Application context that manages configuration, LLM instances, and cost tracking.
    Provides centralized access to system resources and configuration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core attributes
    kwargs: AttrDict = AttrDict()
    config: Config = Config.default()
    src_workspace: Optional[Path] = None
    cost_manager: CostManager = CostManager()

    # Private attributes
    _llm: Optional["BaseLLM"] = None
    _environment_cache: Optional[Dict[str, str]] = None

    def new_environment(self) -> Dict[str, str]:
        """
        Create a new environment dictionary with current context.
        
        Returns:
            Copy of current environment with context updates
        """
        if self._environment_cache is None:
            env = os.environ.copy()
            # Future: Add context-specific environment variables here
            self._environment_cache = env
        return self._environment_cache.copy()

    def _select_cost_manager(self, llm_config: LLMConfig) -> CostManager:
        """
        Select appropriate cost manager based on LLM configuration.
        
        Args:
            llm_config: LLM configuration object
            
        Returns:
            Appropriate cost manager instance
        """
        if llm_config.api_type == LLMType.FIREWORKS:
            return FireworksCostManager()
        elif llm_config.api_type == LLMType.OPEN_LLM:
            return TokenCostManager()
        else:
            return self.cost_manager

    def llm(self) -> "BaseLLM":
        """
        Get or create LLM instance with cost manager.
        
        Returns:
            LLM instance with cost manager
        """
        if self._llm is None:
            from Core.Provider.LLMProviderRegister import create_llm_instance
            self._llm = create_llm_instance(self.config.llm)
            if self._llm.cost_manager is None:
                self._llm.cost_manager = self.cost_manager
        return self._llm

    def llm_with_cost_manager_from_llm_config(self, llm_config: LLMConfig) -> "BaseLLM":
        """
        Create LLM instance with specific configuration and cost manager.
        
        Args:
            llm_config: LLM configuration
            
        Returns:
            LLM instance with cost manager
        """
        from Core.Provider.LLMProviderRegister import create_llm_instance
        llm = create_llm_instance(llm_config)
        llm.cost_manager = self.cost_manager
        return llm

    def reset_llm(self) -> None:
        """Reset LLM instance to force recreation"""
        self._llm = None

    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get current cost summary.
        
        Returns:
            Dictionary with cost information
        """
        return {
            "total_prompt_tokens": self.cost_manager.total_prompt_tokens,
            "total_completion_tokens": self.cost_manager.total_completion_tokens,
            "total_cost": self.cost_manager.total_cost,
            "max_budget": self.cost_manager.max_budget
        }

    def is_within_budget(self) -> bool:
        """
        Check if current costs are within budget.
        
        Returns:
            True if within budget, False otherwise
        """
        return self.cost_manager.total_cost <= self.cost_manager.max_budget

    def set_workspace(self, workspace_path: Path) -> None:
        """
        Set the source workspace path.
        
        Args:
            workspace_path: Path to workspace directory
        """
        self.src_workspace = workspace_path

    def get_workspace(self) -> Optional[Path]:
        """
        Get the current workspace path.
        
        Returns:
            Workspace path or None if not set
        """
        return self.src_workspace

  
