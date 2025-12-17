"""
Helix configuration.

Defines configuration for LLM provider settings, model routing, and behavior.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path

import yaml

from ordinis.ai.helix.models import ModelInfo, ModelType, ProviderType
from ordinis.engines.base import BaseEngineConfig

# Mistral AI models
MISTRAL_MODELS: dict[str, ModelInfo] = {
    "codestral": ModelInfo(
        model_id="mistralai/mistral-large-3-675b-instruct-2512",
        display_name="Mistral Large 3 (675B) Instruct 2512",
        model_type=ModelType.CODE,
        provider=ProviderType.NVIDIA_NIM,
        context_length=256000,
        default_temperature=0.1,
        max_output_tokens=4096,
    ),
    "mistral-large": ModelInfo(
        model_id="mistralai/mistral-large-3-675b-instruct-2512",
        display_name="Mistral Large 3 (675B) Instruct 2512",
        model_type=ModelType.CHAT,
        provider=ProviderType.NVIDIA_NIM,
        context_length=128000,
        default_temperature=0.3,
        max_output_tokens=4096,
    ),
}

# OpenAI models
OPENAI_MODELS: dict[str, ModelInfo] = {
    "gpt-4.1": ModelInfo(
        model_id="gpt-4.1",
        display_name="GPT-4.1",
        model_type=ModelType.CODE,
        provider=ProviderType.OPENAI_API,
        context_length=128000,
        supports_function_calling=True,
        default_temperature=0.2,
        max_output_tokens=4096,
    ),
    "gpt-4o": ModelInfo(
        model_id="gpt-4o",
        display_name="GPT-4o",
        model_type=ModelType.CHAT,
        provider=ProviderType.OPENAI_API,
        context_length=128000,
        supports_function_calling=True,
        default_temperature=0.3,
        max_output_tokens=4096,
    ),
    "text-embedding-3-large": ModelInfo(
        model_id="text-embedding-3-large",
        display_name="Text Embedding 3 Large",
        model_type=ModelType.EMBEDDING,
        provider=ProviderType.OPENAI_API,
        context_length=8192,
        embedding_dim=3072,
        supports_streaming=False,
    ),
}

# Azure OpenAI models (deployment names configurable)
AZURE_MODELS: dict[str, ModelInfo] = {
    "azure-gpt-4": ModelInfo(
        model_id="gpt-4",  # Azure deployment name
        display_name="Azure GPT-4",
        model_type=ModelType.CHAT,
        provider=ProviderType.AZURE_OPENAI_API,
        context_length=128000,
        supports_function_calling=True,
        default_temperature=0.2,
        max_output_tokens=4096,
    ),
    "azure-embedding": ModelInfo(
        model_id="text-embedding-3-large",
        display_name="Azure Text Embedding 3",
        model_type=ModelType.EMBEDDING,
        provider=ProviderType.AZURE_OPENAI_API,
        context_length=8192,
        embedding_dim=3072,
        supports_streaming=False,
    ),
}

# Pre-defined NVIDIA models
NVIDIA_MODELS: dict[str, ModelInfo] = {
    # Primary reasoning/strategy model (DeepSeek R1)
    "deepseek-r1": ModelInfo(
        model_id="deepseek-ai/deepseek-r1",
        display_name="DeepSeek R1",
        model_type=ModelType.CHAT,
        provider=ProviderType.NVIDIA_API,
        context_length=128000,
        supports_function_calling=True,
        default_temperature=0.6,
        max_output_tokens=8192,
    ),
    # Fallback reasoning model (Nemotron Ultra)
    "nemotron-ultra": ModelInfo(
        model_id="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        display_name="Nemotron Ultra 253B",
        model_type=ModelType.CHAT,
        provider=ProviderType.NVIDIA_API,
        context_length=128000,
        supports_function_calling=True,
        default_temperature=0.6,
        max_output_tokens=4096,
    ),
    # Fast synthesis/RAG model (Super)
    "nemotron-super": ModelInfo(
        model_id="meta/llama-3.3-70b-instruct",
        display_name="Llama 3.3 70B Instruct",
        model_type=ModelType.CHAT,
        provider=ProviderType.NVIDIA_API,
        context_length=128000,
        supports_function_calling=True,
        default_temperature=0.3,
        max_output_tokens=4096,
    ),
    # Reward/Judgment model
    "nemotron-reward": ModelInfo(
        model_id="nvidia/llama-3.1-nemotron-70b-reward",
        display_name="Nemotron Reward 70B",
        model_type=ModelType.CHAT,  # Acts as chat but used for scoring
        provider=ProviderType.NVIDIA_API,
        context_length=128000,
        supports_function_calling=False,
        default_temperature=0.1,
        max_output_tokens=1024,
    ),
    # Fallback chat model
    "nemotron-8b": ModelInfo(
        model_id="meta/llama-3.1-8b-instruct",
        display_name="Llama 3.1 8B",
        model_type=ModelType.CHAT,
        provider=ProviderType.NVIDIA_API,
        context_length=128000,
        default_temperature=0.3,
        max_output_tokens=2048,
    ),
    # Primary embedding model
    "nv-embedqa": ModelInfo(
        model_id="nvidia/llama-3.2-nemoretriever-300m-embed-v2",
        display_name="NeMo Retriever 300M Embed V2",
        model_type=ModelType.EMBEDDING,
        provider=ProviderType.NVIDIA_API,
        context_length=512,
        embedding_dim=1024,
        supports_streaming=False,
    ),
    # Retriever embedding model
    "nemoretriever": ModelInfo(
        model_id="nvidia/llama-3.2-nemoretriever-300m-embed-v2",
        display_name="NeMo Retriever 300M",
        model_type=ModelType.EMBEDDING,
        provider=ProviderType.NVIDIA_API,
        context_length=8192,
        embedding_dim=1024,
        supports_streaming=False,
    ),
    # BART-RAG equivalent (using Nemotron for orchestration)
    "bart-rag": ModelInfo(
        model_id="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        display_name="BART-RAG (Nemotron Wrapper)",
        model_type=ModelType.CHAT,
        provider=ProviderType.NVIDIA_API,
        context_length=128000,
        supports_function_calling=True,
        default_temperature=0.2,
        max_output_tokens=4096,
    ),
}


@dataclass
class RetryConfig:
    """Retry behavior configuration."""

    max_retries: int = 3
    initial_delay_ms: int = 1000
    max_delay_ms: int = 30000
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class CacheConfig:
    """Response caching configuration."""

    enabled: bool = True
    ttl_seconds: int = 3600
    max_entries: int = 1000
    cache_embeddings: bool = True
    cache_chat: bool = False  # Chat responses usually shouldn't be cached


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    enabled: bool = True
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    concurrent_requests: int = 10


@dataclass
class HelixConfig(BaseEngineConfig):
    """Main Helix configuration."""

    engine_id: str = "helix"
    engine_name: str = "Helix LLM Provider"

    # API credentials
    nvidia_api_key: str | None = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY"))
    mistral_api_key: str | None = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY"))
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    azure_openai_api_key: str | None = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY")
    )
    azure_openai_endpoint: str | None = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    # Model selection - Code generation (primary use case)
    default_chat_model: str = "default"  # Maps to nemotron-super by default
    fallback_chat_model: str = "fast"
    default_code_model: str = "code"
    fallback_code_model: str = "fast"

    # Embeddings
    default_embedding_model: str = "embedding"
    fallback_embedding_model: str = "embedding_fallback"

    # Model registry - combine all providers
    models: dict[str, ModelInfo] = field(
        default_factory=lambda: {
            **NVIDIA_MODELS,
            **MISTRAL_MODELS,
            **OPENAI_MODELS,
            **AZURE_MODELS,
        }
    )

    # Model aliases from external config
    aliases: dict[str, str] = field(
        default_factory=lambda: {
            "default": "nemotron-super",
            "fast": "nemotron-8b",
            "code": "nemotron-super",
            "reasoning": "deepseek-r1",
            "embedding": "nv-embedqa",
            "embedding_fallback": "nemoretriever",
        }
    )

    def __post_init__(self) -> None:
        """Initialize and load external config."""
        super().__post_init__()
        self._load_external_config()

    def _load_external_config(self) -> None:
        """Load configuration from external YAML file."""
        config_path = Path("configs/helix_models.yaml")
        if not config_path.exists():
            return

        try:
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if "aliases" in data and isinstance(data["aliases"], dict):
                self.aliases.update(data["aliases"])
        except Exception as e:
            print(f"Warning: Failed to load helix_models.yaml: {e}")

    # Generation defaults
    default_temperature: float = 0.2
    default_max_tokens: int = 2048

    # Provider settings
    prefer_local: bool = False  # Prefer local GPU over API
    allow_fallback: bool = True  # Fall back to secondary model on error

    # Retry configuration
    retry: RetryConfig = field(default_factory=RetryConfig)

    # Cache configuration
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Rate limiting
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Logging
    log_requests: bool = False
    log_responses: bool = False

    # Timeouts (milliseconds)
    connect_timeout_ms: int = 10000
    read_timeout_ms: int = 120000

    def get_model(self, name: str) -> ModelInfo | None:
        """Get model info by name or ID."""
        # 1. Check external aliases
        if name in self.aliases:
            resolved_name = self.aliases[name]
            # Check if alias maps to a model key
            if resolved_name in self.models:
                return self.models[resolved_name]
            # Check if alias maps to a model ID
            for model in self.models.values():
                if model.model_id == resolved_name:
                    return model

        # 2. Check internal keys
        if name in self.models:
            return self.models[name]

        # 3. Check by full model ID
        for model in self.models.values():
            if model.model_id == name:
                return model
        return None

    def list_models(self, model_type: ModelType | None = None) -> list[ModelInfo]:
        """List available models, optionally filtered by type."""
        models = list(self.models.values())
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        return models

    def register_model(self, alias: str, model: ModelInfo) -> None:
        """Register a custom model."""
        self.models[alias] = model

    def validate(self) -> list[str]:
        """Validate configuration, returning list of errors."""
        errors: list[str] = []

        # Check at least one provider has credentials
        has_provider = any(
            [
                self.nvidia_api_key,
                self.mistral_api_key,
                self.openai_api_key,
                self.azure_openai_api_key,
                self.prefer_local,
            ]
        )
        if not has_provider:
            errors.append("At least one provider API key required")

        if not self.get_model(self.default_chat_model):
            errors.append(f"default_chat_model '{self.default_chat_model}' not found")

        if not self.get_model(self.fallback_chat_model):
            errors.append(f"fallback_chat_model '{self.fallback_chat_model}' not found")

        if not self.get_model(self.default_embedding_model):
            errors.append(f"default_embedding_model '{self.default_embedding_model}' not found")

        return errors
