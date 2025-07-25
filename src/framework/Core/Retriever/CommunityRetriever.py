"""
Community retriever for retrieving relevant communities from the knowledge graph.
Supports various retrieval methods including entity-based and level-based community retrieval.
"""

import json
from typing import List, Dict, Any, Optional
from collections import Counter

from Core.Common.Logger import logger
from Core.Common.Utils import truncate_list_by_token_size
from Core.Prompt import QueryPrompt
from Core.Retriever.BaseRetriever import BaseRetriever
from Core.Retriever.RetrieverMixin import RetrieverMixin
from Core.Retriever.RetrieverFactory import register_retriever_method


class CommunityRetriever(BaseRetriever, RetrieverMixin):
    """
    Retriever for communities with support for multiple retrieval strategies.
    
    Implements various methods to find relevant communities based on entities
    and community levels.
    """

    def __init__(self, **kwargs):
        """
        Initialize the community retriever.
        
        Args:
            **kwargs: Configuration and dependencies
        """
        config = kwargs.pop("config")
        super().__init__(config)
        self._mode_list = ["from_entity", "from_level"]
        self._type = "community"
        
        # Set additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def mode_list(self) -> List[str]:
        """List of supported retrieval modes."""
        return self._mode_list

    @property
    def type(self) -> str:
        """Type identifier for this retriever."""
        return self._type

    @register_retriever_method(retriever_type="community", method_name="from_entity")
    async def _find_relevant_community_from_entities(self, seed: List[Dict]) -> Optional[List[Dict]]:
        """
        Find relevant communities based on entity data.
        
        Args:
            seed: List of entity data dictionaries
            
        Returns:
            List of relevant communities or None if retrieval fails
        """
        if not self._validate_input(seed, "seed"):
            return None
            
        community_reports = self.community.community_reports
        related_communities = []
        
        # Extract community information from entity data
        for node_d in seed:
            if "clusters" not in node_d:
                continue
            related_communities.extend(json.loads(node_d["clusters"]))
            
        # Filter communities by level
        related_community_dup_keys = [
            str(dp["cluster"])
            for dp in related_communities
            if dp["level"] <= self.config.level
        ]

        # Count community occurrences
        related_community_keys_counts = dict(Counter(related_community_dup_keys))
        
        # Get community data
        _related_community_datas = await self._safe_async_gather(
            [community_reports.get_by_id(k) for k in related_community_keys_counts.keys()],
            "Failed to get community data"
        )
        
        related_community_datas = {
            k: v
            for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
            if v is not None
        }
        
        # Sort communities by occurrence count and rating
        related_community_keys = sorted(
            related_community_keys_counts.keys(),
            key=lambda k: (
                related_community_keys_counts[k],
                related_community_datas[k]["report_json"].get("rating", -1),
            ),
            reverse=True,
        )
        
        sorted_community_datas = [
            related_community_datas[k] for k in related_community_keys
        ]

        # Truncate communities based on token limit
        use_community_reports = truncate_list_by_token_size(
            sorted_community_datas,
            key=lambda x: x["report_string"],
            max_token_size=self.config.local_max_token_for_community_report,
        )
        
        # Return single community if configured
        if self.config.local_community_single_one:
            use_community_reports = use_community_reports[:1]
            
        return use_community_reports

    @register_retriever_method(retriever_type="community", method_name="from_level")
    async def find_relevant_community_by_level(self, seed: Optional[Any] = None) -> List[Dict]:
        """
        Find relevant communities by level.
        
        Args:
            seed: Optional seed data (not used in this method)
            
        Returns:
            List of relevant communities or failure response
        """
        community_schema = self.community.community_schema
        
        # Filter communities by level
        community_schema = {
            k: v for k, v in community_schema.items() if v.level <= self.config.level
        }
        
        if not len(community_schema):
            return QueryPrompt.FAIL_RESPONSE

        # Sort communities by occurrence
        sorted_community_schemas = sorted(
            community_schema.items(),
            key=lambda x: x[1].occurrence,
            reverse=True,
        )

        # Limit number of communities to consider
        sorted_community_schemas = sorted_community_schemas[
                                   : self.config.global_max_consider_community
                                   ]
        
        # Get community data
        community_datas = await self.community.community_reports.get_by_ids(
            [k[0] for k in sorted_community_schemas]
        )

        # Filter and sort communities
        community_datas = [c for c in community_datas if c is not None]
        community_datas = [
            c
            for c in community_datas
            if c["report_json"].get("rating", 0) >= self.config.global_min_community_rating
        ]
        
        community_datas = sorted(
            community_datas,
            key=lambda x: (x['occurrence'], x["report_json"].get("rating", 0)),
            reverse=True,
        )
        
        logger.info(f"Retrieved {len(community_datas)} communities")
        return community_datas
