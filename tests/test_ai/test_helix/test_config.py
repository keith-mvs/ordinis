"""
Tests for Helix configuration.

Tests cover:
- RetryConfig settings
- CacheConfig settings
- RateLimitConfig settings
- HelixConfig validation
- Model registry operations
- NVIDIA_MODELS dictionary
"""

import os

import pytest

from ordinis.ai.helix.config import (
    NVIDIA_MODELS,
    CacheConfig,
    HelixConfig,
    RateLimitConfig,
    RetryConfig,
)
from ordinis.ai.helix.models import ModelInfo, ModelType, ProviderType


@pytest.mark.unit
class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_retry_config_defaults(self):
        """Test RetryConfig default values."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.initial_delay_ms == 1000
        assert config.max_delay_ms == 30000
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_retry_config_custom(self):
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_retries=5,
            initial_delay_ms=500,
            max_delay_ms=10000,
            exponential_base=1.5,
            jitter=False,
        )

        assert config.max_retries == 5
        assert config.initial_delay_ms == 500
        assert config.max_delay_ms == 10000
        assert config.exponential_base == 1.5
        assert config.jitter is False


@pytest.mark.unit
class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_cache_config_defaults(self):
        """Test CacheConfig default values."""
        config = CacheConfig()

        assert config.enabled is True
        assert config.ttl_seconds == 3600
        assert config.max_entries == 1000
        assert config.cache_embeddings is True
        assert config.cache_chat is False

    def test_cache_config_custom(self):
        """Test CacheConfig with custom values."""
        config = CacheConfig(
            enabled=False,
            ttl_seconds=7200,
            max_entries=500,
            cache_embeddings=False,
            cache_chat=True,
        )

        assert config.enabled is False
        assert config.ttl_seconds == 7200
        assert config.max_entries == 500
        assert config.cache_embeddings is False
        assert config.cache_chat is True


@pytest.mark.unit
class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_rate_limit_config_defaults(self):
        """Test RateLimitConfig default values."""
        config = RateLimitConfig()

        assert config.enabled is True
        assert config.requests_per_minute == 60
        assert config.tokens_per_minute == 100000
        assert config.concurrent_requests == 10

    def test_rate_limit_config_custom(self):
        """Test RateLimitConfig with custom values."""
        config = RateLimitConfig(
            enabled=False,
            requests_per_minute=120,
            tokens_per_minute=200000,
            concurrent_requests=20,
        )

        assert config.enabled is False
        assert config.requests_per_minute == 120
        assert config.tokens_per_minute == 200000
        assert config.concurrent_requests == 20


@pytest.mark.unit
class TestNVIDIAModels:
    """Tests for NVIDIA_MODELS registry."""

    def test_nvidia_models_contains_expected_models(self):
        """Test NVIDIA_MODELS has expected models."""
        assert "nemotron-super" in NVIDIA_MODELS
        assert "nemotron-8b" in NVIDIA_MODELS
        assert "nv-embedqa" in NVIDIA_MODELS
        assert "nemoretriever" in NVIDIA_MODELS

    def test_nemotron_super_config(self):
        """Test Nemotron Super model configuration."""
        model = NVIDIA_MODELS["nemotron-super"]

        assert model.model_id == "nvidia/llama-3.3-nemotron-super-49b-v1.5"
        assert model.display_name == "Nemotron Super 49B"
        assert model.model_type == ModelType.CHAT
        assert model.provider == ProviderType.NVIDIA_API
        assert model.context_length == 128000
        assert model.supports_function_calling is True
        assert model.default_temperature == 0.3
        assert model.max_output_tokens == 4096

    def test_nemotron_8b_config(self):
        """Test Nemotron 8B model configuration."""
        model = NVIDIA_MODELS["nemotron-8b"]

        assert model.model_id == "meta/llama-3.1-8b-instruct"
        assert model.display_name == "Llama 3.1 8B"
        assert model.model_type == ModelType.CHAT
        assert model.provider == ProviderType.NVIDIA_API
        assert model.context_length == 128000
        assert model.default_temperature == 0.3
        assert model.max_output_tokens == 2048

    def test_nv_embedqa_config(self):
        """Test NV-EmbedQA embedding model configuration."""
        model = NVIDIA_MODELS["nv-embedqa"]

        assert model.model_id == "nvidia/nv-embedqa-e5-v5"
        assert model.display_name == "NV-EmbedQA E5"
        assert model.model_type == ModelType.EMBEDDING
        assert model.provider == ProviderType.NVIDIA_API
        assert model.context_length == 512
        assert model.embedding_dim == 1024
        assert model.supports_streaming is False

    def test_nemoretriever_config(self):
        """Test NeMo Retriever embedding model configuration."""
        model = NVIDIA_MODELS["nemoretriever"]

        assert model.model_id == "nvidia/llama-3.2-nemoretriever-300m-embed-v2"
        assert model.display_name == "NeMo Retriever 300M"
        assert model.model_type == ModelType.EMBEDDING
        assert model.provider == ProviderType.NVIDIA_API
        assert model.context_length == 8192
        assert model.embedding_dim == 1024
        assert model.supports_streaming is False


@pytest.mark.unit
class TestHelixConfig:
    """Tests for HelixConfig."""

    def test_helix_config_defaults(self):
        """Test HelixConfig default values."""
        config = HelixConfig()

        assert config.nvidia_api_key == os.getenv("NVIDIA_API_KEY")
        assert config.default_chat_model == "nemotron-super"
        assert config.fallback_chat_model == "nemotron-8b"
        assert config.default_embedding_model == "nv-embedqa"
        assert config.fallback_embedding_model == "nemoretriever"
        assert config.default_temperature == 0.2
        assert config.default_max_tokens == 2048
        assert config.prefer_local is False
        assert config.allow_fallback is True
        assert isinstance(config.retry, RetryConfig)
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.rate_limit, RateLimitConfig)
        assert config.log_requests is False
        assert config.log_responses is False
        assert config.connect_timeout_ms == 10000
        assert config.read_timeout_ms == 120000

    def test_helix_config_custom(self, mock_nvidia_api_key: str):
        """Test HelixConfig with custom values."""
        retry = RetryConfig(max_retries=5)
        cache = CacheConfig(enabled=False)
        rate_limit = RateLimitConfig(enabled=False)

        config = HelixConfig(
            nvidia_api_key=mock_nvidia_api_key,
            default_chat_model="nemotron-8b",
            fallback_chat_model="nemotron-super",
            default_embedding_model="nemoretriever",
            default_temperature=0.5,
            default_max_tokens=4096,
            prefer_local=True,
            allow_fallback=False,
            retry=retry,
            cache=cache,
            rate_limit=rate_limit,
            log_requests=True,
            log_responses=True,
        )

        assert config.nvidia_api_key == mock_nvidia_api_key
        assert config.default_chat_model == "nemotron-8b"
        assert config.fallback_chat_model == "nemotron-super"
        assert config.default_embedding_model == "nemoretriever"
        assert config.default_temperature == 0.5
        assert config.default_max_tokens == 4096
        assert config.prefer_local is True
        assert config.allow_fallback is False
        assert config.retry == retry
        assert config.cache == cache
        assert config.rate_limit == rate_limit
        assert config.log_requests is True
        assert config.log_responses is True

    def test_helix_config_models_registry(self):
        """Test that models registry is populated."""
        config = HelixConfig()

        assert len(config.models) > 0
        assert "nemotron-super" in config.models
        assert "nv-embedqa" in config.models

    def test_get_model_by_alias(self, mock_nvidia_api_key: str):
        """Test getting model by alias."""
        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)

        model = config.get_model("nemotron-super")

        assert model is not None
        assert model.model_id == "nvidia/llama-3.3-nemotron-super-49b-v1.5"
        assert model.display_name == "Nemotron Super 49B"

    def test_get_model_by_id(self, mock_nvidia_api_key: str):
        """Test getting model by full model ID."""
        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)

        model = config.get_model("nvidia/llama-3.1-nemotron-ultra-253b-v1")

        assert model is not None
        assert model.display_name == "Nemotron Ultra 253B"

    def test_get_model_not_found(self, mock_nvidia_api_key: str):
        """Test getting non-existent model."""
        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)

        model = config.get_model("non-existent-model")

        assert model is None

    def test_list_models_all(self, mock_nvidia_api_key: str):
        """Test listing all models."""
        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)

        models = config.list_models()

        assert len(models) >= 4
        model_ids = [m.model_id for m in models]
        assert "nvidia/llama-3.1-nemotron-ultra-253b-v1" in model_ids

    def test_list_models_by_type_chat(self, mock_nvidia_api_key: str):
        """Test listing chat models."""
        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)

        models = config.list_models(ModelType.CHAT)

        assert len(models) >= 2
        assert all(m.model_type == ModelType.CHAT for m in models)

    def test_list_models_by_type_embedding(self, mock_nvidia_api_key: str):
        """Test listing embedding models."""
        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)

        models = config.list_models(ModelType.EMBEDDING)

        assert len(models) >= 2
        assert all(m.model_type == ModelType.EMBEDDING for m in models)

    def test_register_model(self, mock_nvidia_api_key: str):
        """Test registering a custom model."""
        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)

        custom_model = ModelInfo(
            model_id="custom/model",
            display_name="Custom Model",
            model_type=ModelType.CHAT,
            provider=ProviderType.LOCAL_GPU,
            context_length=2048,
        )

        config.register_model("custom", custom_model)

        assert "custom" in config.models
        assert config.get_model("custom") == custom_model

    def test_validate_valid_config(self, mock_nvidia_api_key: str):
        """Test validating a valid configuration."""
        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)

        errors = config.validate()

        assert len(errors) == 0

    def test_validate_missing_api_key(self):
        """Test validation fails without API key when not using local."""
        config = HelixConfig(
            nvidia_api_key=None,
            prefer_local=False,
        )

        errors = config.validate()

        assert len(errors) > 0
        assert any("At least one provider API key required" in err for err in errors)

    def test_validate_api_key_not_required_for_local(self):
        """Test validation passes without API key when prefer_local=True."""
        config = HelixConfig(
            nvidia_api_key=None,
            prefer_local=True,
        )

        errors = config.validate()

        # Should not error about API key
        assert not any("nvidia_api_key" in err for err in errors)

    def test_validate_invalid_default_chat_model(self, mock_nvidia_api_key: str):
        """Test validation fails with invalid default chat model."""
        config = HelixConfig(
            nvidia_api_key=mock_nvidia_api_key,
            default_chat_model="non-existent",
        )

        errors = config.validate()

        assert len(errors) > 0
        assert any("default_chat_model" in err for err in errors)

    def test_validate_invalid_fallback_chat_model(self, mock_nvidia_api_key: str):
        """Test validation fails with invalid fallback chat model."""
        config = HelixConfig(
            nvidia_api_key=mock_nvidia_api_key,
            fallback_chat_model="non-existent",
        )

        errors = config.validate()

        assert len(errors) > 0
        assert any("fallback_chat_model" in err for err in errors)

    def test_validate_invalid_embedding_model(self, mock_nvidia_api_key: str):
        """Test validation fails with invalid embedding model."""
        config = HelixConfig(
            nvidia_api_key=mock_nvidia_api_key,
            default_embedding_model="non-existent",
        )

        errors = config.validate()

        assert len(errors) > 0
        assert any("default_embedding_model" in err for err in errors)
