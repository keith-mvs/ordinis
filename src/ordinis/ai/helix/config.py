"""
Helix configuration.

Defines configuration for LLM provider settings, model routing, and behavior.
"""

from dataclasses import dataclass, field
import os

from ordinis.ai.helix.models import ModelInfo, ModelType, ProviderType

# Pre-defined NVIDIA models
NVIDIA_MODELS: dict[str, ModelInfo] = {
    # Primary chat model
    "nemotron-super": ModelInfo(
        model_id="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        display_name="Nemotron Super 49B",
        model_type=ModelType.CHAT,
        provider=ProviderType.NVIDIA_API,
        context_length=128000,
        supports_function_calling=True,
        default_temperature=0.2,
        max_output_tokens=4096,
    ),
    # Fallback chat model
    "nemotron-8b": ModelInfo(
        model_id="nvidia/llama-3.1-nemotron-8b",
        display_name="Nemotron 8B",
        model_type=ModelType.CHAT,
        provider=ProviderType.NVIDIA_API,
        context_length=32000,
        default_temperature=0.3,
        max_output_tokens=2048,
    ),
    # Primary embedding model
    "nv-embedqa": ModelInfo(
        model_id="nvidia/nv-embedqa-e5-v5",
        display_name="NV-EmbedQA E5",
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
class HelixConfig:
    """Main Helix configuration."""

    # API credentials
    nvidia_api_key: str | None = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY"))

    # Model selection
    default_chat_model: str = "nemotron-super"
    fallback_chat_model: str = "nemotron-8b"
    default_embedding_model: str = "nv-embedqa"
    fallback_embedding_model: str = "nemoretriever"

    # Model registry
    models: dict[str, ModelInfo] = field(default_factory=lambda: NVIDIA_MODELS.copy())

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
        # Check by alias first
        if name in self.models:
            return self.models[name]
        # Check by full model ID
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

        if not self.nvidia_api_key and not self.prefer_local:
            errors.append("nvidia_api_key required when prefer_local=False")

        if self.default_chat_model not in self.models:
            errors.append(f"default_chat_model '{self.default_chat_model}' not in models")

        if self.fallback_chat_model not in self.models:
            errors.append(f"fallback_chat_model '{self.fallback_chat_model}' not in models")

        if self.default_embedding_model not in self.models:
            errors.append(f"default_embedding_model '{self.default_embedding_model}' not in models")

        return errors
