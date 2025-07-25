"""
Configuration for retrieval system parameters.
Defines parameters for different retrieval strategies and methods.
"""

from Core.Utils.YamlModel import YamlModel


class RetrieverConfig(YamlModel):
    """
    Configuration for retrieval system parameters.
    
    This class manages all retrieval-related configuration including
    query types, PPR parameters, token limits, and context management
    for different retrieval strategies.
    
    Attributes:
        query_type: Type of query to use for retrieval
        enable_local: Enable local retrieval
        use_entity_similarity_for_ppr: Use entity similarity for PPR
        top_k_entity_for_ppr: Top-k entities for PPR
        node_specificity: Enable node specificity
        damping: Damping factor for PPR
        top_k: Number of top results to retrieve
        k_nei: Number of neighbors to consider
        max_token_for_local_context: Maximum tokens for local context
        max_token_for_global_context: Maximum tokens for global context
        local_max_token_for_text_unit: Maximum tokens for local text units
        use_relations_vdb: Use relations vector database
        use_subgraphs_vdb: Use subgraphs vector database
        global_max_consider_community: Maximum communities to consider
        global_min_community_rating: Minimum community rating threshold
        level: Retrieval level
    """

    # Basic Retrieval Configuration
    query_type: str = "ppr"
    enable_local: bool = False
    top_k: int = 5
    k_nei: int = 3
    level: int = 2

    # PPR Configuration
    use_entity_similarity_for_ppr: bool = True
    top_k_entity_for_ppr: int = 8
    node_specificity: bool = True
    damping: float = 0.1

    # Token Limits Configuration
    max_token_for_local_context: int = 4800
    max_token_for_global_context: int = 4000
    local_max_token_for_text_unit: int = 4000

    # Vector Database Configuration
    use_relations_vdb: bool = False
    use_subgraphs_vdb: bool = False

    # Community Configuration
    global_max_consider_community: int = 512
    global_min_community_rating: float = 0.0
