"""
Context Mixin Module - Mixin for context and configuration management
Provides context and configuration capabilities to classes that inherit from it
"""

from typing import Optional, Any, Dict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Option.merged_config import MergedConfig as Config
from Core.Common.Context import Context
# Import BaseLLM as type hint only to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Core.Provider.BaseLLM import BaseLLM


class ContextMixin(BaseModel):
    """
    Mixin class that provides context and configuration management capabilities.
    Allows classes to have their own context, config, and LLM instances while
    falling back to shared instances when not explicitly set.
    
    Note: Uses 'private_' prefix instead of '_private_' due to Pydantic inheritance issues.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Private attributes with 'private_' prefix to avoid Pydantic inheritance issues
    # See: https://github.com/pydantic/pydantic/issues/7142, #7083, #7091
    private_context: Optional[Context] = Field(default=None, exclude=True)
    private_config: Optional[Config] = Field(default=None, exclude=True)
    private_llm: Optional["BaseLLM"] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_context_mixin_extra(self) -> "ContextMixin":
        """
        Process extra fields after model validation.
        
        Returns:
            Self after processing extra fields
        """
        self._process_context_mixin_extra()
        return self

    def _process_context_mixin_extra(self) -> None:
        """
        Process extra fields from model_extra.
        Extracts context, config, and llm from extra fields.
        """
        kwargs = self.model_extra or {}
        self.set_context(kwargs.pop("context", None))
        self.set_config(kwargs.pop("config", None))
        self.set_llm(kwargs.pop("llm", None))

    def set(self, key: str, value: Any, override: bool = False) -> None:
        """
        Set attribute with optional override control.
        
        Args:
            key: Attribute name
            value: Value to set
            override: Whether to override existing value
        """
        if override or not self.__dict__.get(key):
            self.__dict__[key] = value

    def set_context(self, context: Optional[Context], override: bool = True) -> None:
        """
        Set the context instance.
        
        Args:
            context: Context instance to set
            override: Whether to override existing context
        """
        self.set("private_context", context, override)

    def set_config(self, config: Optional[Config], override: bool = False) -> None:
        """
        Set the configuration instance.
        
        Args:
            config: Configuration instance to set
            override: Whether to override existing config
        """
        self.set("private_config", config, override)
        if config is not None:
            # Initialize LLM when config is set
            _ = self.llm

    def set_llm(self, llm: Optional["BaseLLM"], override: bool = False) -> None:
        """
        Set the LLM instance.
        
        Args:
            llm: LLM instance to set
            override: Whether to override existing LLM
        """
        self.set("private_llm", llm, override)

    @property
    def config(self) -> Config:
        """
        Get configuration with priority: private config > context config.
        
        Returns:
            Configuration instance
        """
        if self.private_config:
            return self.private_config
        return self.context.config

    @config.setter
    def config(self, config: Config) -> None:
        """
        Set configuration.
        
        Args:
            config: Configuration instance
        """
        self.set_config(config)

    @property
    def context(self) -> Context:
        """
        Get context with priority: private context > default context.
        
        Returns:
            Context instance
        """
        if self.private_context:
            return self.private_context
        return Context()

    @context.setter
    def context(self, context: Context) -> None:
        """
        Set context.
        
        Args:
            context: Context instance
        """
        self.set_context(context)

    @property
    def llm(self) -> "BaseLLM":
        """
        Get LLM instance with lazy initialization.
        Creates LLM from config if not already set.
        
        Returns:
            Configured LLM instance
        """
        if not self.private_llm:
            self.private_llm = self.context.llm_with_cost_manager_from_llm_config(
                self.config.llm
            )
        return self.private_llm

    @llm.setter
    def llm(self, llm: "BaseLLM") -> None:
        """
        Set LLM instance.
        
        Args:
            llm: LLM instance
        """
        self.private_llm = llm

    def reset_llm(self) -> None:
        """Reset LLM instance to force recreation"""
        self.private_llm = None

    def get_context_info(self) -> Dict[str, Any]:
        """
        Get information about current context setup.
        
        Returns:
            Dictionary with context information
        """
        return {
            "has_private_context": self.private_context is not None,
            "has_private_config": self.private_config is not None,
            "has_private_llm": self.private_llm is not None,
            "config_source": "private" if self.private_config else "context",
            "context_source": "private" if self.private_context else "default"
        }

    def merge_context(self, other_context: "ContextMixin") -> None:
        """
        Merge context from another ContextMixin instance.
        Only sets values that are not already set.
        
        Args:
            other_context: ContextMixin instance to merge from
        """
        if not self.private_context and other_context.private_context:
            self.set_context(other_context.private_context)
        
        if not self.private_config and other_context.private_config:
            self.set_config(other_context.private_config)
        
        if not self.private_llm and other_context.private_llm:
            self.set_llm(other_context.private_llm)
