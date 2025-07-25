"""
Entity and Relationship merging utilities for GraphRAG system.
Provides functionality for merging entity and relationship information in graph structures.
"""

from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from Core.Common.Constants import GRAPH_FIELD_SEP


@dataclass
class MergeConfig:
    """Configuration for merging operations."""
    merge_strategy: str = "frequency"  # frequency, union, intersection
    preserve_order: bool = True
    deduplicate: bool = True


class MergeStrategy:
    """
    Base class for merge strategies.
    
    Defines the interface for different merging strategies
    used in entity and relationship merging operations.
    """
    
    def merge_strings(self, existing: List[str], new: List[str]) -> str:
        """
        Merge two lists of strings according to the strategy.
        
        Args:
            existing: Existing string list
            new: New string list to merge
            
        Returns:
            Merged string
        """
        raise NotImplementedError("Subclasses must implement merge_strings")
    
    def merge_values(self, existing: Any, new: Any) -> Any:
        """
        Merge two values according to the strategy.
        
        Args:
            existing: Existing value
            new: New value to merge
            
        Returns:
            Merged value
        """
        raise NotImplementedError("Subclasses must implement merge_values")


class FrequencyMergeStrategy(MergeStrategy):
    """
    Merge strategy that selects the most frequent value.
    """
    
    def merge_strings(self, existing: List[str], new: List[str]) -> str:
        """
        Merge strings by selecting the most frequent one.
        
        Args:
            existing: Existing string list
            new: New string list
            
        Returns:
            Most frequent string
        """
        merged_list = existing + new
        if not merged_list:
            return ""
        
        counter = Counter(merged_list)
        return counter.most_common(1)[0][0]
    
    def merge_values(self, existing: Any, new: Any) -> Any:
        """
        Merge values by selecting the most frequent one.
        
        Args:
            existing: Existing value
            new: New value
            
        Returns:
            Most frequent value
        """
        if existing == new:
            return existing
        
        # For simple values, prefer the new one if different
        return new


class UnionMergeStrategy(MergeStrategy):
    """
    Merge strategy that combines all unique values.
    """
    
    def merge_strings(self, existing: List[str], new: List[str]) -> str:
        """
        Merge strings by combining all unique values.
        
        Args:
            existing: Existing string list
            new: New string list
            
        Returns:
            Combined unique strings
        """
        merged_set = set(existing) | set(new)
        if not merged_set:
            return ""
        
        # Sort for consistent output
        return GRAPH_FIELD_SEP.join(sorted(merged_set))
    
    def merge_values(self, existing: Any, new: Any) -> Any:
        """
        Merge values by combining them.
        
        Args:
            existing: Existing value
            new: New value
            
        Returns:
            Combined value
        """
        if isinstance(existing, (list, tuple)) and isinstance(new, (list, tuple)):
            return list(set(existing) | set(new))
        elif isinstance(existing, (int, float)) and isinstance(new, (int, float)):
            return existing + new
        else:
            return new


class IntersectionMergeStrategy(MergeStrategy):
    """
    Merge strategy that keeps only common values.
    """
    
    def merge_strings(self, existing: List[str], new: List[str]) -> str:
        """
        Merge strings by keeping only common values.
        
        Args:
            existing: Existing string list
            new: New string list
            
        Returns:
            Common strings
        """
        common_set = set(existing) & set(new)
        if not common_set:
            return ""
        
        return GRAPH_FIELD_SEP.join(sorted(common_set))
    
    def merge_values(self, existing: Any, new: Any) -> Any:
        """
        Merge values by keeping only common values.
        
        Args:
            existing: Existing value
            new: New value
            
        Returns:
            Common value
        """
        if isinstance(existing, (list, tuple)) and isinstance(new, (list, tuple)):
            return list(set(existing) & set(new))
        elif existing == new:
            return existing
        else:
            return None


class MergeStrategyFactory:
    """
    Factory for creating merge strategies.
    """
    
    _strategies = {
        "frequency": FrequencyMergeStrategy,
        "union": UnionMergeStrategy,
        "intersection": IntersectionMergeStrategy
    }
    
    @classmethod
    def get_strategy(cls, strategy_name: str) -> MergeStrategy:
        """
        Get a merge strategy by name.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Merge strategy instance
            
        Raises:
            ValueError: If strategy name is not recognized
        """
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown merge strategy: {strategy_name}")
        
        return cls._strategies[strategy_name]()


class EntityMerger:
    """
    Handles merging of entity information in graph structures.
    
    Provides methods for merging various entity attributes
    such as source IDs, entity types, and descriptions.
    """
    
    def __init__(self, config: MergeConfig = None):
        """
        Initialize the entity merger.
        
        Args:
            config: Merge configuration
        """
        self.config = config or MergeConfig()
        self.strategy = MergeStrategyFactory.get_strategy(self.config.merge_strategy)
        self.merge_functions = {
            "source_id": self._merge_source_ids,
            "entity_type": self._merge_entity_types,
            "description": self._merge_descriptions,
        }
    
    def _merge_source_ids(self, existing_source_ids: List[str], new_source_ids: List[str]) -> str:
        """
        Merge source IDs using the configured strategy.
        
        Args:
            existing_source_ids: Existing source IDs
            new_source_ids: New source IDs
            
        Returns:
            Merged source IDs string
        """
        return self.strategy.merge_strings(existing_source_ids, new_source_ids)
    
    def _merge_entity_types(self, existing_entity_types: List[str], new_entity_types: List[str]) -> str:
        """
        Merge entity types using the configured strategy.
        
        Args:
            existing_entity_types: Existing entity types
            new_entity_types: New entity types
            
        Returns:
            Merged entity type
        """
        return self.strategy.merge_strings(existing_entity_types, new_entity_types)
    
    def _merge_descriptions(self, existing_descriptions: List[str], new_descriptions: List[str]) -> str:
        """
        Merge descriptions using the configured strategy.
        
        Args:
            existing_descriptions: Existing descriptions
            new_descriptions: New descriptions
            
        Returns:
            Merged descriptions string
        """
        return self.strategy.merge_strings(existing_descriptions, new_descriptions)
    
    async def merge_entity_info(
        self, 
        merge_keys: List[str], 
        nodes_data: Dict[str, Any], 
        merge_dict: Dict[str, Any]
    ) -> List[Any]:
        """
        Merge entity information for a specific entity.
        
        Args:
            merge_keys: Keys to merge
            nodes_data: Existing node data
            merge_dict: New data to merge
            
        Returns:
            List of merged values
        """
        if not nodes_data:
            return []
        
        result = []
        
        for merge_key in merge_keys:
            if merge_key in merge_dict and merge_key in self.merge_functions:
                merged_value = self.merge_functions[merge_key](
                    nodes_data.get(merge_key, []), 
                    merge_dict[merge_key]
                )
                result.append(merged_value)
            else:
                result.append("")
        
        # Ensure result has the same length as merge_keys
        while len(result) < len(merge_keys):
            result.append("")
        
        return result


class RelationshipMerger:
    """
    Handles merging of relationship information in graph structures.
    
    Provides methods for merging various relationship attributes
    such as weights, descriptions, keywords, and relation names.
    """
    
    def __init__(self, config: MergeConfig = None):
        """
        Initialize the relationship merger.
        
        Args:
            config: Merge configuration
        """
        self.config = config or MergeConfig()
        self.strategy = MergeStrategyFactory.get_strategy(self.config.merge_strategy)
        self.merge_functions = {
            "weight": self._merge_weight,
            "description": self._merge_descriptions,
            "source_id": self._merge_source_ids,
            "keywords": self._merge_keywords,
            "relation_name": self._merge_relation_name
        }
    
    def _merge_weight(self, existing_weight: float, new_weight: float) -> float:
        """
        Merge weights by summing them.
        
        Args:
            existing_weight: Existing weight
            new_weight: New weight
            
        Returns:
            Sum of weights
        """
        return existing_weight + new_weight
    
    def _merge_descriptions(self, existing_descriptions: List[str], new_descriptions: List[str]) -> str:
        """
        Merge descriptions using the configured strategy.
        
        Args:
            existing_descriptions: Existing descriptions
            new_descriptions: New descriptions
            
        Returns:
            Merged descriptions string
        """
        return self.strategy.merge_strings(existing_descriptions, new_descriptions)
    
    def _merge_source_ids(self, existing_source_ids: List[str], new_source_ids: List[str]) -> str:
        """
        Merge source IDs using the configured strategy.
        
        Args:
            existing_source_ids: Existing source IDs
            new_source_ids: New source IDs
            
        Returns:
            Merged source IDs string
        """
        return self.strategy.merge_strings(existing_source_ids, new_source_ids)
    
    def _merge_keywords(self, existing_keywords: List[str], new_keywords: List[str]) -> str:
        """
        Merge keywords using the configured strategy.
        
        Args:
            existing_keywords: Existing keywords
            new_keywords: New keywords
            
        Returns:
            Merged keywords string
        """
        return self.strategy.merge_strings(existing_keywords, new_keywords)
    
    def _merge_relation_name(self, existing_relation_name: str, new_relation_name: str) -> str:
        """
        Merge relation names using the configured strategy.
        
        Args:
            existing_relation_name: Existing relation name
            new_relation_name: New relation name
            
        Returns:
            Merged relation name string
        """
        return self.strategy.merge_strings([existing_relation_name], [new_relation_name])
    
    async def merge_relationship_info(
        self, 
        edges_data: Dict[str, Any], 
        merge_dict: Dict[str, Any]
    ) -> List[Any]:
        """
        Merge relationship information for a specific relationship.
        
        Args:
            edges_data: Existing edge data
            merge_dict: New data to merge
            
        Returns:
            List of merged values
        """
        if not edges_data:
            return []
        
        result = []
        merge_keys = ["source_id", "weight", "description", "keywords", "relation_name"]
        
        for merge_key in merge_keys:
            if merge_key in merge_dict and merge_key in self.merge_functions:
                existing_value = edges_data.get(merge_key, [] if merge_key != "weight" else 0)
                new_value = merge_dict[merge_key]
                
                merged_value = self.merge_functions[merge_key](existing_value, new_value)
                result.append(merged_value)
            else:
                result.append("")
        
        return result


# Legacy classes for backward compatibility
class MergeEntity:
    """
    Legacy entity merger class for backward compatibility.
    """
    
    merge_function = None
    
    @staticmethod
    def merge_source_ids(existing_source_ids: List[str], new_source_ids: List[str]) -> str:
        """
        Merge source IDs by combining unique values.
        
        Args:
            existing_source_ids: Existing source IDs
            new_source_ids: New source IDs
            
        Returns:
            Merged source IDs string
        """
        merged_source_ids = list(set(new_source_ids) | set(existing_source_ids))
        return GRAPH_FIELD_SEP.join(merged_source_ids)
    
    @staticmethod
    def merge_types(existing_entity_types: List[str], new_entity_types: List[str]) -> str:
        """
        Merge entity types by selecting the most frequent one.
        
        Args:
            existing_entity_types: Existing entity types
            new_entity_types: New entity types
            
        Returns:
            Most frequent entity type
        """
        merged_entity_types = existing_entity_types + new_entity_types
        entity_type_counts = Counter(merged_entity_types)
        most_common_type = entity_type_counts.most_common(1)[0][0] if entity_type_counts else ''
        return most_common_type
    
    @staticmethod
    def merge_descriptions(entity_relationships: List[str], new_descriptions: List[str]) -> str:
        """
        Merge descriptions by combining unique values.
        
        Args:
            entity_relationships: Existing descriptions
            new_descriptions: New descriptions
            
        Returns:
            Merged descriptions string
        """
        merged_descriptions = list(set(new_descriptions) | set(entity_relationships))
        return GRAPH_FIELD_SEP.join(sorted(merged_descriptions))
    
    @classmethod
    async def merge_info(cls, merge_keys: List[str], nodes_data: Dict[str, Any], merge_dict: Dict[str, Any]) -> List[Any]:
        """
        Merge entity information for a specific entity name.
        
        Args:
            merge_keys: Keys to merge
            nodes_data: Existing node data
            merge_dict: New data to merge
            
        Returns:
            List of merged values
        """
        if not nodes_data:
            return []
        
        result = []
        
        if cls.merge_function is None:
            cls.merge_function = {
                "source_id": cls.merge_source_ids,
                "entity_type": cls.merge_types,
                "description": cls.merge_descriptions,
            }
        
        for merge_key in merge_keys:
            if merge_key in merge_dict:
                merged_value = cls.merge_function[merge_key](
                    nodes_data.get(merge_key, []), 
                    merge_dict[merge_key]
                )
                result.append(merged_value)
        
        # Ensure result has the same length as merge_keys
        while len(result) < len(merge_keys):
            result.append("")
        
        return result


class MergeRelationship:
    """
    Legacy relationship merger class for backward compatibility.
    """
    
    merge_keys = ["source_id", "weight", "description", "keywords", "relation_name"]
    merge_function = None
    
    @staticmethod
    def merge_weight(merge_weight: float, new_weight: float) -> float:
        """
        Merge weights by summing them.
        
        Args:
            merge_weight: Existing weight
            new_weight: New weight
            
        Returns:
            Sum of weights
        """
        return merge_weight + new_weight
    
    @staticmethod
    def merge_descriptions(entity_relationships: List[str], new_descriptions: List[str]) -> str:
        """
        Merge descriptions by combining unique values.
        
        Args:
            entity_relationships: Existing descriptions
            new_descriptions: New descriptions
            
        Returns:
            Merged descriptions string
        """
        return GRAPH_FIELD_SEP.join(sorted(set(new_descriptions + entity_relationships)))
    
    @staticmethod
    def merge_source_ids(existing_source_ids: List[str], new_source_ids: List[str]) -> str:
        """
        Merge source IDs by combining unique values.
        
        Args:
            existing_source_ids: Existing source IDs
            new_source_ids: New source IDs
            
        Returns:
            Merged source IDs string
        """
        return GRAPH_FIELD_SEP.join(set(new_source_ids + existing_source_ids))
    
    @staticmethod
    def merge_keywords(keywords: List[str], new_keywords: List[str]) -> str:
        """
        Merge keywords by combining unique values.
        
        Args:
            keywords: Existing keywords
            new_keywords: New keywords
            
        Returns:
            Merged keywords string
        """
        return GRAPH_FIELD_SEP.join(set(keywords + new_keywords))
    
    @staticmethod
    def merge_relation_name(relation_name: str, new_relation_name: str) -> str:
        """
        Merge relation names by combining unique values.
        
        Args:
            relation_name: Existing relation name
            new_relation_name: New relation name
            
        Returns:
            Merged relation name string
        """
        return GRAPH_FIELD_SEP.join(sorted(set([relation_name, new_relation_name])))
    
    @classmethod
    async def merge_info(cls, edges_data: Dict[str, Any], merge_dict: Dict[str, Any]) -> List[Any]:
        """
        Merge relationship information for a specific relationship.
        
        Args:
            edges_data: Existing edge data
            merge_dict: New data to merge
            
        Returns:
            List of merged values
        """
        if not edges_data:
            return []
        
        result = []
        
        if cls.merge_function is None:
            cls.merge_function = {
                "weight": cls.merge_weight,
                "description": cls.merge_descriptions,
                "source_id": cls.merge_source_ids,
                "keywords": cls.merge_keywords,
                "relation_name": cls.merge_relation_name
            }
        
        for merge_key in cls.merge_keys:
            if merge_key in merge_dict:
                existing_value = edges_data.get(merge_key, [] if merge_key != "weight" else 0)
                new_value = merge_dict[merge_key]
                
                merged_value = cls.merge_function[merge_key](existing_value, new_value)
                result.append(merged_value)
        
        return result
