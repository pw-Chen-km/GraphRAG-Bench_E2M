"""
Cost Manager Module - API cost tracking and management
Provides comprehensive cost tracking for different LLM providers and models
"""

import re
from typing import NamedTuple, Dict, List, Optional
from abc import ABC, abstractmethod

from pydantic import BaseModel

from Core.Common.Logger import logger
from Core.Utils.TokenCounter import FIREWORKS_GRADE_TOKEN_COSTS, TOKEN_COSTS


class CostSummary(NamedTuple):
    """Named tuple for cost summary information"""
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float
    total_budget: float


class BaseCostManager(BaseModel, ABC):
    """
    Abstract base class for cost managers.
    Defines interface for cost tracking across different providers.
    """

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_budget: float = 10.0
    total_cost: float = 0
    stage_costs: List[CostSummary] = []

    @abstractmethod
    def update_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        """
        Update cost tracking with new token usage.
        
        Args:
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            model: Model name for cost calculation
        """
        pass

    def get_total_prompt_tokens(self) -> int:
        """
        Get total prompt tokens used.
        
        Returns:
            Total prompt tokens
        """
        return self.total_prompt_tokens

    def get_total_completion_tokens(self) -> int:
        """
        Get total completion tokens used.
        
        Returns:
            Total completion tokens
        """
        return self.total_completion_tokens

    def get_total_cost(self) -> float:
        """
        Get total cost incurred.
        
        Returns:
            Total cost in dollars
        """
        return self.total_cost

    def get_cost_summary(self) -> CostSummary:
        """
        Get complete cost summary.
        
        Returns:
            CostSummary with all cost information
        """
        return CostSummary(
            self.total_prompt_tokens,
            self.total_completion_tokens,
            self.total_cost,
            self.total_budget
        )

    def set_stage_cost(self) -> None:
        """Record current cost as a stage checkpoint"""
        self.stage_costs.append(self.get_cost_summary())

    def get_last_stage_cost(self) -> CostSummary:
        """
        Get cost difference since last stage.
        
        Returns:
            CostSummary with incremental costs since last stage
        """
        current_cost = self.get_cost_summary()
        
        if not self.stage_costs:
            last_cost = CostSummary(0, 0, 0, 0)
        else:
            last_cost = self.stage_costs[-1]
        
        incremental_cost = CostSummary(
            current_cost.total_prompt_tokens - last_cost.total_prompt_tokens,
            current_cost.total_completion_tokens - last_cost.total_completion_tokens,
            current_cost.total_cost - last_cost.total_cost,
            current_cost.total_budget - last_cost.total_budget
        )
        
        self.set_stage_cost()
        return incremental_cost

    def is_within_budget(self) -> bool:
        """
        Check if current costs are within budget.
        
        Returns:
            True if within budget, False otherwise
        """
        return self.total_cost <= self.total_budget

    def get_budget_remaining(self) -> float:
        """
        Get remaining budget.
        
        Returns:
            Remaining budget amount
        """
        return max(0, self.total_budget - self.total_cost)

    def reset_costs(self) -> None:
        """Reset all cost tracking to zero"""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.stage_costs.clear()


class CostManager(BaseCostManager):
    """
    Standard cost manager for OpenAI and similar providers.
    Calculates costs based on token usage and model-specific pricing.
    """

    token_costs: Dict[str, Dict[str, float]] = TOKEN_COSTS

    def update_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        """
        Update cost tracking for standard providers.
        
        Args:
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            model: Model name for cost calculation
        """
        if prompt_tokens + completion_tokens == 0 or not model:
            return
            
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        
        if model not in self.token_costs:
            logger.warning(f"Model {model} not found in TOKEN_COSTS")
            return

        cost = (
            prompt_tokens * self.token_costs[model]["prompt"] +
            completion_tokens * self.token_costs[model]["completion"]
        ) / 1000
        
        self.total_cost += cost
        
        logger.info(
            f"Total running cost: ${self.total_cost:.3f} | "
            f"Max budget: ${self.total_budget:.3f} | "
            f"Current cost: ${cost:.3f}, "
            f"prompt_tokens: {prompt_tokens}, "
            f"completion_tokens: {completion_tokens}"
        )


class TokenCostManager(BaseCostManager):
    """
    Cost manager for self-hosted/open LLM models.
    Tracks token usage without cost calculation since these are typically free.
    """

    def update_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        """
        Update token tracking for free models.
        
        Args:
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            model: Model name (for logging purposes)
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        
        logger.info(
            f"Token usage for {model}: "
            f"prompt_tokens: {prompt_tokens}, "
            f"completion_tokens: {completion_tokens}"
        )


class FireworksCostManager(BaseCostManager):
    """
    Cost manager for Fireworks AI models.
    Calculates costs based on model size and Fireworks pricing structure.
    """

    def _get_model_size(self, model: str) -> float:
        """
        Extract model size from model name.
        
        Args:
            model: Model name string
            
        Returns:
            Model size in billions, -1 if not found
        """
        size_match = re.findall(r".*-([0-9.]+)b", model)
        return float(size_match[0]) if size_match else -1

    def _get_model_token_costs(self, model: str) -> Dict[str, float]:
        """
        Get token costs for a specific Fireworks model.
        
        Args:
            model: Model name
            
        Returns:
            Dictionary with prompt and completion token costs
        """
        if "mixtral-8x7b" in model:
            return FIREWORKS_GRADE_TOKEN_COSTS["mixtral-8x7b"]
        
        model_size = self._get_model_size(model)
        
        if 0 < model_size <= 16:
            return FIREWORKS_GRADE_TOKEN_COSTS["16"]
        elif 16 < model_size <= 80:
            return FIREWORKS_GRADE_TOKEN_COSTS["80"]
        else:
            return FIREWORKS_GRADE_TOKEN_COSTS["-1"]

    def update_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        """
        Update cost tracking for Fireworks models.
        
        Args:
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            model: Model name for cost calculation
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        token_costs = self._get_model_token_costs(model)
        cost = (
            prompt_tokens * token_costs["prompt"] +
            completion_tokens * token_costs["completion"]
        ) / 1000000  # Fireworks uses per-million token pricing
        
        self.total_cost += cost
        
        logger.info(
            f"Total running cost: ${self.total_cost:.4f} | "
            f"Current cost: ${cost:.4f}, "
            f"prompt_tokens: {prompt_tokens}, "
            f"completion_tokens: {completion_tokens}"
        )


class CostManagerFactory:
    """Factory for creating appropriate cost managers based on provider type"""
    
    @staticmethod
    def create_cost_manager(provider_type: str) -> BaseCostManager:
        """
        Create cost manager for specified provider type.
        
        Args:
            provider_type: Type of LLM provider
            
        Returns:
            Appropriate cost manager instance
        """
        if provider_type == "fireworks":
            return FireworksCostManager()
        elif provider_type == "open_llm":
            return TokenCostManager()
        else:
            return CostManager()