"""
Configuration for query processing and retrieval parameters.
Defines parameters for different query types and search strategies.
"""

from dataclasses import field

from Core.Utils.YamlModel import YamlModel


class QueryConfig(YamlModel):
    """
    Configuration for query processing and retrieval parameters.
    
    This class manages all query-related configuration including
    query types, search strategies, token limits, and method-specific
    parameters for various GraphRAG implementations.
    
    Attributes:
        query_type: Type of query to process
        only_need_context: Return only context without answer
        response_type: Format of response
        level: Query processing level
        top_k: Number of top results to retrieve
        nei_k: Number of neighbors to consider
        num_doc: Number of documents for HippoRAG
        
        # Naive Search Configuration
        naive_max_token_for_text_unit: Maximum tokens for naive search
        use_keywords: Enable keyword-based search
        use_communiy_info: Use community information
        
        # Local Search Configuration
        enable_local: Enable local search
        local_max_token_for_text_unit: Maximum tokens for local text units
        local_max_token_for_community_report: Maximum tokens for community reports
        local_community_single_one: Use single community for local search
        community_information: Enable community information for MS-GraphRAG
        max_token_for_text_unit: Maximum tokens for local text units
        
        # Global Search Configuration
        global_min_community_rating: Minimum community rating threshold
        global_max_consider_community: Maximum communities to consider
        global_max_token_for_community_report: Maximum tokens for global community reports
        max_token_for_global_context: Maximum tokens for global context
        global_special_community_map_llm_kwargs: LLM arguments for community mapping
        use_global_query: Enable global query for LightRAG and GraphRAG
        use_community: Enable community-based search for LGraphRAG and GGraphRAG
        enable_hybrid_query: Enable hybrid query for LightRAG
        
        # IR-COT Configuration
        max_ir_steps: Maximum IR-COT steps
        
        # HippoRAG Configuration
        augmentation_ppr: Enable PPR augmentation
        entities_max_tokens: Maximum tokens for entities
        relationships_max_tokens: Maximum tokens for relationships
        
        # RAPTOR Configuration
        tree_search: Enable tree-based search
        
        # TOG Configuration
        depth: Search depth for TOG
        width: Search width for TOG
        
        # G-Retriever Configuration
        max_txt_len: Maximum text length
        topk_e: Top-k entities
        cost_e: Entity cost parameter
        
        # Medical GraphRAG Configuration
        topk_entity: Top-k entities for medical queries
        k_hop: K-hop neighborhood size
    """

    # Basic Query Configuration
    query_type: str = "qa"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 20
    nei_k: int = 3
    num_doc: int = 5

    # Naive Search Configuration
    naive_max_token_for_text_unit: int = 12000
    use_keywords: bool = False
    use_communiy_info: bool = False

    # Local Search Configuration
    enable_local: bool = False
    local_max_token_for_text_unit: int = 4000
    local_max_token_for_community_report: int = 3200
    local_community_single_one: bool = False
    community_information: bool = False
    max_token_for_text_unit: int = 4000

    # Global Search Configuration
    global_min_community_rating: float = 0
    global_max_consider_community: float = 512
    global_max_token_for_community_report: int = 16384
    max_token_for_global_context: int = 4000
    global_special_community_map_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    use_global_query: bool = False
    use_community: bool = False
    enable_hybrid_query: bool = False

    # IR-COT Configuration
    max_ir_steps: int = 2

    # HippoRAG Configuration
    augmentation_ppr: bool = False
    entities_max_tokens: int = 2000
    relationships_max_tokens: int = 2000

    # RAPTOR Configuration
    tree_search: bool = False

    # TOG Configuration
    depth: int = 3
    width: int = 3

    # G-Retriever Configuration
    max_txt_len: int = 512
    topk_e: int = 3
    cost_e: float = 0.5

    # Medical GraphRAG Configuration
    topk_entity: int = 10
    k_hop: int = 2
