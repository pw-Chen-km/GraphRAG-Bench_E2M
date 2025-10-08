"""
Configuration for Large Language Model (LLM) settings and API parameters.
Defines parameters for different LLM providers and model configurations.
"""

from enum import Enum
from typing import Optional

from pydantic import field_validator

from Core.Common.Constants import CONFIG_ROOT, LLM_API_TIMEOUT, GRAPHRAG_ROOT
from Core.Utils.YamlModel import YamlModel


class LLMType(Enum):
    """Supported LLM API providers and endpoints."""
    OPENAI = "openai"
    FIREWORKS = "fireworks"
    OPEN_LLM = "open_llm"
    GEMINI = "gemini"
    OLLAMA = "ollama"  # /chat endpoint
    OLLAMA_GENERATE = "ollama.generate"  # /generate endpoint
    OLLAMA_EMBEDDINGS = "ollama.embeddings"  # /embeddings endpoint
    OLLAMA_EMBED = "ollama.embed"  # /embed endpoint
    OPENROUTER = "openrouter"
    BEDROCK = "bedrock"
    ARK = "ark"  # Volcengine API

    def __missing__(self, key):
        """Default to OpenAI if provider not found."""
        return self.OPENAI


class LLMConfig(YamlModel):
    """
    Configuration for Large Language Model settings and API parameters.
    
    This class manages all LLM-related configuration including
    API credentials, model specifications, generation parameters,
    and network settings.
    
    Attributes:
        api_key: API key for authentication
        api_type: Type of LLM API provider
        base_url: Base URL for API requests
        api_version: API version to use
        model: Name of the LLM model or deployment
        pricing_plan: Cost settlement plan parameters
        
        # Cloud Service Provider Settings
        access_key: Access key for cloud providers
        secret_key: Secret key for cloud providers
        session_token: Session token for temporary access
        endpoint: Custom endpoint for self-deployed models
        
        # Spark/Xunfei Settings
        app_id: Application ID for Spark
        api_secret: API secret for Spark
        domain: Domain for Spark
        
        # Generation Parameters
        max_token: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repetition
        stop: Stop sequences
        presence_penalty: Presence penalty
        frequency_penalty: Frequency penalty
        best_of: Number of best responses
        n: Number of responses to generate
        stream: Enable streaming responses
        seed: Random seed for reproducibility
        logprobs: Enable log probabilities
        top_logprobs: Number of top log probabilities
        timeout: Request timeout in seconds
        context_length: Maximum input tokens
        
        # Amazon Bedrock Settings
        region_name: AWS region name
        
        # Network Settings
        proxy: Proxy configuration
        max_concurrent: Maximum concurrent requests
        
        # Cost Control
        calc_usage: Calculate usage costs
        
        # Message Control
        use_system_prompt: Use system prompt in messages
    """

    # Basic API Configuration
    api_key: str = "sk-"
    api_type: LLMType = LLMType.OPENAI
    base_url: str = "https://api.openai.com/v1"
    api_version: Optional[str] = None
    model: Optional[str] = None
    pricing_plan: Optional[str] = None

    # Cloud Service Provider Configuration
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    session_token: Optional[str] = None
    endpoint: Optional[str] = None

    # Spark/Xunfei Configuration
    app_id: Optional[str] = None
    api_secret: Optional[str] = None
    domain: Optional[str] = None

    # Generation Parameters
    max_token: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop: Optional[str] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: Optional[int] = None
    n: Optional[int] = None
    stream: bool = False
    seed: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    timeout: int = 600
    context_length: Optional[int] = None

    # Amazon Bedrock Configuration
    region_name: str = None

    # Network Configuration
    proxy: Optional[str] = None
    max_concurrent: int = 20

    # Cost Control
    calc_usage: bool = True

    # Message Control
    use_system_prompt: bool = True

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value):
        """Validate API key and provide helpful error messages."""
        if value in ["", None, "YOUR_API_KEY"]:
            repo_config_path = GRAPHRAG_ROOT / "Option/merged_config.yaml"
            root_config_path = CONFIG_ROOT / "merged_config.yaml"
            
            if root_config_path.exists():
                raise ValueError(
                    f"Please set your API key in {root_config_path}. If you also set your config in {repo_config_path}, "
                    f"the former will overwrite the latter. This may cause unexpected result."
                )
            elif repo_config_path.exists():
                raise ValueError(f"Please set your API key in {repo_config_path}")
            else:
                raise ValueError("Please set your API key in merged_config.yaml")
        return value

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, value):
        """Validate and set default timeout if not provided."""
        return value or LLM_API_TIMEOUT
