"""
LLM Module - Language Model interface utilities
Provides convenient access to LLM instances with proper configuration
"""

from typing import Optional

from Config.LLMConfig import LLMConfig
from Core.Common.Context import Context
from Core.Provider.BaseLLM import BaseLLM


def create_llm_instance(
    llm_config: Optional[LLMConfig] = None, 
    context: Optional[Context] = None
) -> BaseLLM:
    """
    Create and configure an LLM instance with optional configuration.
    
    Args:
        llm_config: Optional LLM configuration. If None, uses default config
        context: Optional context instance. If None, creates new context
        
    Returns:
        Configured LLM instance with cost manager
    """
    ctx = context or Context()
    
    if llm_config is not None:
        return ctx.llm_with_cost_manager_from_llm_config(llm_config)
    
    return ctx.llm()


def get_default_llm(context: Optional[Context] = None) -> BaseLLM:
    """
    Get the default LLM instance using default configuration.
    
    Args:
        context: Optional context instance. If None, creates new context
        
    Returns:
        Default LLM instance
    """
    return create_llm_instance(context=context)


def get_configured_llm(llm_config: LLMConfig, context: Optional[Context] = None) -> BaseLLM:
    """
    Get an LLM instance with specific configuration.
    
    Args:
        llm_config: LLM configuration to use
        context: Optional context instance. If None, creates new context
        
    Returns:
        Configured LLM instance
    """
    return create_llm_instance(llm_config=llm_config, context=context)


# Backward compatibility alias
LLM = create_llm_instance
