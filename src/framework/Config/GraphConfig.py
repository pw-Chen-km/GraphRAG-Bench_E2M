"""
Configuration for graph construction and processing.
Defines parameters for different graph types and clustering algorithms.
"""

from typing import Optional

from Core.Utils.YamlModel import YamlModel


class GraphConfig(YamlModel):
    """
    Configuration for graph construction and processing parameters.
    
    This class manages all graph-related configuration including
    graph type selection, entity/edge extraction, clustering, and
    tree graph construction parameters.
    
    Attributes:
        graph_type: Type of graph to construct (er_graph, kg_graph, etc.)
        extract_two_step: Enable two-step entity extraction
        max_gleaning: Maximum gleaning iterations
        force: Force rebuild graph even if exists
        
        # Entity and Edge Configuration
        enable_entity_description: Include entity descriptions
        enable_entity_type: Include entity types
        enable_edge_description: Include edge descriptions
        enable_edge_name: Include edge names
        prior_prob: Prior probability for edge extraction
        enable_edge_keywords: Include edge keywords
        
        # Clustering Configuration
        use_community: Enable community detection
        graph_cluster_algorithm: Algorithm for graph clustering
        max_graph_cluster_size: Maximum size of graph clusters
        graph_cluster_seed: Random seed for clustering
        summary_max_tokens: Maximum tokens for summaries
        llm_model_max_token_size: Maximum tokens for LLM model
        
        # Tree Graph Configuration
        build_tree_from_leaves: Build tree starting from leaves
        reduction_dimension: Dimension for reduction
        summarization_length: Length of summaries
        num_layers: Number of tree layers
        top_k: Top k elements to select
        threshold_cluster_num: Threshold for cluster number
        start_layer: Starting layer for tree construction
        graph_cluster_params: Additional clustering parameters
        selection_mode: Mode for element selection
        max_length_in_cluster: Maximum length within clusters
        threshold: Similarity threshold
        cluster_metric: Metric for clustering
        verbose: Enable verbose output
        random_seed: Random seed for reproducibility
        enforce_sub_communities: Enforce sub-community structure
        max_size_percentage: Maximum size as percentage
        tol: Tolerance for convergence
        max_iter: Maximum iterations
        size_of_clusters: Target size of clusters
        
        # Graph Augmentation
        similarity_threshold: Threshold for similarity matching
        similarity_top_k: Top k similar elements
        similarity_max: Maximum similarity value
    """

    # Basic Graph Configuration
    graph_type: str = "er_graph"
    extract_two_step: bool = True
    max_gleaning: int = 1
    force: bool = False

    # Entity and Edge Configuration
    enable_entity_description: bool = False
    enable_entity_type: bool = False
    enable_edge_description: bool = False
    enable_edge_name: bool = False
    prior_prob: float = 0.8
    enable_edge_keywords: bool = False
    
    # Clustering Configuration
    use_community: bool = False
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    summary_max_tokens: int = 500
    llm_model_max_token_size: int = 32768

    # Tree Graph Configuration
    build_tree_from_leaves: bool = False
    reduction_dimension: int = 5
    summarization_length: int = 100
    num_layers: int = 10
    top_k: int = 5
    threshold_cluster_num: int = 5000
    start_layer: int = 5
    graph_cluster_params: Optional[dict] = None
    selection_mode: str = "top_k"
    max_length_in_cluster: int = 3500
    threshold: float = 0.1
    cluster_metric: str = "cosine"
    verbose: bool = False
    random_seed: int = 224
    enforce_sub_communities: bool = False
    max_size_percentage: float = 0.2
    tol: float = 1e-4
    max_iter: int = 300
    size_of_clusters: int = 10

    # Graph Augmentation Configuration
    similarity_threshold: float = 0.8
    similarity_top_k: int = 10
    similarity_max: float = 1.0
