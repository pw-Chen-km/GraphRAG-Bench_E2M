#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from dataclasses import field
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from pydantic import BaseModel
from Config import *
from Core.Common.Constants import CONFIG_ROOT, GRAPHRAG_ROOT
from Core.Utils.YamlModel import YamlModel
from pydantic import model_validator


class WorkingParams(BaseModel):
    """Working parameters"""

    working_dir: str = ""
    exp_name: str = ""
    data_root: str = ""
    dataset_name: str = ""


class MergedConfig(WorkingParams, YamlModel):
    """Merged configuration file containing configuration parameters for all GraphRAG methods"""

    # Global configuration
    llm: LLMConfig
    exp_name: str = "default"
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunk: ChunkConfig = ChunkConfig()
    graph: GraphConfig = GraphConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    query: QueryConfig = QueryConfig()

    use_colbert: bool = True
    colbert_checkpoint_path: str = "/path/to/colbertv2.0"
    index_name: str = "nbits_2"
    similarity_max: float = 1.0
    vdb_type: str = "vector"  # vector/colbert/faiss
    
    # method
    use_entities_vdb: bool = True
    use_relations_vdb: bool = False
    use_subgraphs_vdb: bool = False
    use_entity_link_chunk: bool = True
    llm_model_max_token_size: int = 32768
    token_model: str = "gpt-3.5-turbo"
    enable_graph_augmentation: bool = True
    retrieve_top_k: int = 20

    # Method-specific configurations
    methods: Dict = {}
    
    @model_validator(mode="after")
    def ensure_config_types(self):
        # 自动转换为对应的Config对象
        if isinstance(self.graph, dict):
            self.graph = GraphConfig(**self.graph)
        if isinstance(self.chunk, dict):
            self.chunk = ChunkConfig(**self.chunk)
        if isinstance(self.retriever, dict):
            self.retriever = RetrieverConfig(**self.retriever)
        if isinstance(self.query, dict):
            self.query = QueryConfig(**self.query)
        if isinstance(self.embedding, dict):
            self.embedding = EmbeddingConfig(**self.embedding)
        if isinstance(self.llm, dict):
            self.llm = LLMConfig(**self.llm)
        return self

    def __init__(self, **data):
        super().__init__(**data)
        # 确保配置对象正确初始化
        self._ensure_config_types()
    
    def _ensure_config_types(self):
        """确保配置对象正确初始化"""
        if isinstance(self.graph, dict):
            self.graph = GraphConfig(**self.graph)
        if isinstance(self.chunk, dict):
            self.chunk = ChunkConfig(**self.chunk)
        if isinstance(self.retriever, dict):
            self.retriever = RetrieverConfig(**self.retriever)
        if isinstance(self.query, dict):
            self.query = QueryConfig(**self.query)
        if isinstance(self.embedding, dict):
            self.embedding = EmbeddingConfig(**self.embedding)
        if isinstance(self.llm, dict):
            self.llm = LLMConfig(**self.llm)

    @classmethod
    def from_yaml_config(cls, path: str, method_name: str = None):
        """Load configuration from YAML configuration file
        
        Args:
            path: Configuration file path
            method_name: Method name, if specified, loads the corresponding method's configuration
        """
        opt = parse(path)
        config = cls(**opt)
        
        if method_name and method_name in config.methods:
            # Merge method-specific configuration into global configuration
            method_config = config.methods[method_name]
            for key, value in method_config.items():
                setattr(config, key, value)
        
        return config

    @classmethod
    def parse(cls, _path, dataset_name, method_name: str = None):
        """Parse configuration file
        
        Args:
            _path: Configuration file path
            dataset_name: Dataset name
            method_name: Method name
        """
        opt = [parse(_path)]

        default_config_paths: List[Path] = [
            GRAPHRAG_ROOT / "Option/merged_config.yaml",
        ]
        opt += [MergedConfig.read_yaml(path) for path in default_config_paths]
    
        final = merge_dict(opt)
        final["dataset_name"] = dataset_name
        final["working_dir"] = os.path.join(final["working_dir"], dataset_name)
        
        config = cls(**final)
        
        if method_name and method_name in config.methods:
            # Merge method-specific configuration into global configuration
            method_config = config.methods[method_name]
            for key, value in method_config.items():
                setattr(config, key, value)
        
        # 确保配置对象正确初始化
        config._ensure_config_types()
        
        return config
    
    @classmethod
    def default(cls, method_name: str = None):
        """Load default configuration
        
        Args:
            method_name: Method name, if specified, loads the corresponding method's configuration
        """
        default_config_paths: List[Path] = [
            GRAPHRAG_ROOT / "Option/merged_config.yaml",
        ]

        dicts = [dict(os.environ)]
        dicts += [MergedConfig.read_yaml(path) for path in default_config_paths]

        final = merge_dict(dicts)
        config = cls(**final)
        
        if method_name and method_name in config.methods:
            # Merge method-specific configuration into global configuration
            method_config = config.methods[method_name]
            for key, value in method_config.items():
                setattr(config, key, value)
        
        return config

    def get_method_config(self, method_name: str) -> Dict:
        """Get configuration for specific method
        
        Args:
            method_name: Method name
            
        Returns:
            Method configuration dictionary
        """
        if method_name in self.methods:
            return self.methods[method_name]
        return {}

    def set_method_config(self, method_name: str, config: Dict):
        """Set configuration for specific method
        
        Args:
            method_name: Method name
            config: Configuration dictionary
        """
        self.methods[method_name] = config

    def get_available_methods(self) -> List[str]:
        """Get all available method names
        
        Returns:
            List of method names
        """
        return list(self.methods.keys())

    @property
    def extra(self):
        return self._extra

    @extra.setter
    def extra(self, value: dict):
        self._extra = value

    def get_openai_llm(self) -> Optional[LLMConfig]:
        """Get OpenAI LLMConfig. Throws exception if no OpenAI"""
        if self.llm.api_type == LLMType.OPENAI:
            return self.llm
        return None


def parse(opt_path):
    """Parse YAML configuration file"""
    with open(opt_path, mode='r') as f:
        opt = YamlModel.read_yaml(opt_path)
    return opt


def merge_dict(dicts: Iterable[Dict]) -> Dict:
    """Merge multiple dictionaries into one, later dictionaries override earlier ones"""
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


# Create default configuration instance
default_config = MergedConfig.default()

# Method name mapping (for compatibility with original filenames)
METHOD_NAME_MAPPING = {
    "HippoRAG": "hippo_rag",
    "GR": "gr", 
    "ToG": "tog",
    "KGP": "kgp",
    "RAPTOR": "raptor",
    "LightRAG": "light_rag",
    "LGraphRAG": "lgraph_rag",
    "GGraphRAG": "ggraph_rag",
    "Dalk": "dalk"
}


def load_method_config(method_name: str, dataset_name: str = None) -> MergedConfig:
    """Load configuration for specific method
    
    Args:
        method_name: Method name (can be original filename or mapped name)
        dataset_name: Dataset name
        
    Returns:
        Configuration object
    """
    # If dataset name is provided, use parse method
    if dataset_name:
        config_path = GRAPHRAG_ROOT / "Option/merged_config.yaml"
        return MergedConfig.parse(str(config_path), dataset_name, method_name)
    else:
        # Otherwise use default method
        return MergedConfig.default(method_name)


def get_method_config_by_original_name(original_name: str) -> Dict:
    """Get method configuration by original filename
    
    Args:
        original_name: Original filename (e.g., "HippoRAG.yaml")
        
    Returns:
        Method configuration dictionary
    """
    # Remove .yaml suffix
    if original_name.endswith('.yaml'):
        original_name = original_name[:-5]
    
    # Get mapped name
    mapped_name = METHOD_NAME_MAPPING.get(original_name, original_name.lower())
    
    # Load configuration
    config = MergedConfig.default()
    return config.get_method_config(mapped_name) 