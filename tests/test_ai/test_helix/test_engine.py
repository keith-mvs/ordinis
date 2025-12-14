"""
Tests for Helix engine.

Tests cover:
- Helix initialization
- generate() and generate_sync() methods
- generate_stream() for streaming
- embed() and embed_sync() methods
- Caching behavior
- Rate limiting
- Retry with backoff
- Error handling and fallback
- Model resolution
- Metrics and health checks
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ordinis.ai.helix.config import HelixConfig
from ordinis.ai.helix.engine import Helix
from ordinis.ai.helix.models import (
    ChatMessage,
    ChatResponse,
    EmbeddingResponse,
    HelixError,
    ModelNotFoundError,
    ModelType,
    ProviderError,
    ProviderType,
    RateLimitError,
    UsageInfo,
)


@pytest.mark.unit
class TestHelixInitialization:
    """Tests for Helix initialization."""

    def test_helix_init_with_config(self, test_helix_config: HelixConfig):
        """Test Helix initialization with config."""
        helix = Helix(test_helix_config)

        assert helix.config == test_helix_config
        assert len(helix._cache) == 0
        assert helix._total_requests == 0
        assert helix._total_tokens == 0
        assert helix._cache_hits == 0
        assert helix._cache_misses == 0

    def test_helix_init_without_config(self):
        """Test Helix initialization without config uses defaults."""
        # This will fail validation if no API key in env
        # Mock the provider initialization to avoid this
        with patch("ordinis.ai.helix.engine.NVIDIAProvider") as mock_provider:
            mock_provider.return_value.is_available = False
            helix = Helix()

            assert helix.config is not None
            assert isinstance(helix.config, HelixConfig)

    def test_helix_init_invalid_config(self):
        """Test Helix initialization fails with invalid config."""
        invalid_config = HelixConfig(
            nvidia_api_key=None,
            prefer_local=False,
            default_chat_model="non-existent",
        )

        with pytest.raises(ValueError, match="Invalid Helix configuration"):
            Helix(invalid_config)

    def test_helix_init_providers(self, test_helix_config: HelixConfig):
        """Test provider initialization."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider") as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider.is_available = True
            mock_provider_class.return_value = mock_provider

            helix = Helix(test_helix_config)

            assert ProviderType.NVIDIA_API in helix._providers
            assert helix._providers[ProviderType.NVIDIA_API] == mock_provider


@pytest.mark.unit
class TestModelResolution:
    """Tests for model resolution."""

    def test_get_model_by_name(self, test_helix_config: HelixConfig):
        """Test getting model by name."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            model = helix._get_model("nemotron-super", ModelType.CHAT)

            assert model is not None
            assert model.model_id == "nvidia/llama-3.3-nemotron-super-49b-v1.5"

    def test_get_model_default_chat(self, test_helix_config: HelixConfig):
        """Test getting default chat model."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            model = helix._get_model(None, ModelType.CHAT)

            assert model is not None
            assert model.model_id == "nvidia/llama-3.3-nemotron-super-49b-v1.5"

    def test_get_model_default_embedding(self, test_helix_config: HelixConfig):
        """Test getting default embedding model."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            model = helix._get_model(None, ModelType.EMBEDDING)

            assert model is not None
            assert model.model_id == "nvidia/nv-embedqa-e5-v5"

    def test_get_model_not_found(self, test_helix_config: HelixConfig):
        """Test error when model not found."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            with pytest.raises(ModelNotFoundError, match="non-existent"):
                helix._get_model("non-existent", ModelType.CHAT)

    def test_get_fallback_model_chat(self, test_helix_config: HelixConfig):
        """Test getting fallback chat model."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            model = helix._get_fallback_model(ModelType.CHAT)

            assert model is not None
            assert model.model_id == "nvidia/llama-3.1-nemotron-8b"

    def test_get_fallback_model_embedding(self, test_helix_config: HelixConfig):
        """Test getting fallback embedding model."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            model = helix._get_fallback_model(ModelType.EMBEDDING)

            assert model is not None
            assert model.model_id == "nvidia/llama-3.2-nemoretriever-300m-embed-v2"

    def test_get_fallback_model_disabled(self, test_helix_config: HelixConfig):
        """Test fallback disabled."""
        test_helix_config.allow_fallback = False

        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            model = helix._get_fallback_model(ModelType.CHAT)

            assert model is None


@pytest.mark.unit
class TestCaching:
    """Tests for caching behavior."""

    def test_cache_key_generation(
        self, test_helix_config: HelixConfig, sample_chat_messages: list[ChatMessage]
    ):
        """Test cache key generation."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            key1 = helix._cache_key(sample_chat_messages, "model1")
            key2 = helix._cache_key(sample_chat_messages, "model1")
            key3 = helix._cache_key(sample_chat_messages, "model2")

            assert key1 == key2
            assert key1 != key3
            assert len(key1) == 32

    def test_embed_cache_key_generation(self, test_helix_config: HelixConfig):
        """Test embedding cache key generation."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            key1 = helix._embed_cache_key(["text1", "text2"], "model1")
            key2 = helix._embed_cache_key(["text1", "text2"], "model1")
            key3 = helix._embed_cache_key(["text2", "text1"], "model1")

            assert key1 == key2
            assert key1 == key3  # Sorted, so order doesn't matter
            assert len(key1) == 32

    def test_cache_set_and_get(
        self, test_helix_config: HelixConfig, sample_chat_response: ChatResponse
    ):
        """Test setting and getting cached responses."""
        # Enable chat caching for this test
        test_helix_config.cache.cache_chat = True

        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            helix._set_cached("test_key", sample_chat_response)
            cached = helix._get_cached("test_key")

            assert cached is not None
            assert cached.content == sample_chat_response.content
            assert cached.cached is True
            assert helix._cache_hits == 1

    def test_cache_miss(self, test_helix_config: HelixConfig):
        """Test cache miss."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            cached = helix._get_cached("non_existent_key")

            assert cached is None
            assert helix._cache_misses == 1

    def test_cache_ttl_expiration(
        self, test_helix_config: HelixConfig, sample_chat_response: ChatResponse
    ):
        """Test cache TTL expiration."""
        test_helix_config.cache.ttl_seconds = 0  # Expire immediately

        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            helix._set_cached("test_key", sample_chat_response)
            time.sleep(0.1)
            cached = helix._get_cached("test_key")

            assert cached is None
            assert "test_key" not in helix._cache

    def test_cache_max_entries(
        self, test_helix_config: HelixConfig, sample_embedding_response: EmbeddingResponse
    ):
        """Test cache eviction when max entries reached."""
        test_helix_config.cache.max_entries = 3
        test_helix_config.cache.cache_embeddings = True

        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            # Use embedding responses since they're cached by default
            for i in range(5):
                helix._set_cached(f"key_{i}", sample_embedding_response)

            assert len(helix._cache) == 3
            assert "key_0" not in helix._cache
            assert "key_1" not in helix._cache
            assert "key_4" in helix._cache

    def test_cache_disabled(
        self, test_helix_config: HelixConfig, sample_chat_response: ChatResponse
    ):
        """Test caching when disabled."""
        test_helix_config.cache.enabled = False

        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            helix._set_cached("test_key", sample_chat_response)
            cached = helix._get_cached("test_key")

            assert cached is None
            assert len(helix._cache) == 0

    def test_cache_chat_disabled(
        self, test_helix_config: HelixConfig, sample_chat_response: ChatResponse
    ):
        """Test chat caching disabled specifically."""
        test_helix_config.cache.cache_chat = False

        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            helix._set_cached("test_key", sample_chat_response)

            assert len(helix._cache) == 0

    def test_cache_embeddings_enabled(
        self, test_helix_config: HelixConfig, sample_embedding_response: EmbeddingResponse
    ):
        """Test embedding caching enabled."""
        test_helix_config.cache.cache_embeddings = True

        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            helix._set_cached("test_key", sample_embedding_response)

            assert len(helix._cache) == 1


@pytest.mark.unit
class TestRateLimiting:
    """Tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_check_passes(self, test_helix_config: HelixConfig):
        """Test rate limit check passes when under limits."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            await helix._check_rate_limit()

    @pytest.mark.asyncio
    async def test_rate_limit_request_exceeded(self, test_helix_config: HelixConfig):
        """Test rate limit exceeded for requests."""
        test_helix_config.rate_limit.requests_per_minute = 2

        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            # Fill up the rate limit
            helix._rate_limit.request_times = [time.time(), time.time()]

            with pytest.raises(RateLimitError, match="Request rate limit exceeded"):
                await helix._check_rate_limit()

    @pytest.mark.asyncio
    async def test_rate_limit_token_exceeded(self, test_helix_config: HelixConfig):
        """Test rate limit exceeded for tokens."""
        test_helix_config.rate_limit.tokens_per_minute = 100

        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            # Fill up token limit
            now = time.time()
            helix._rate_limit.token_counts = [(now, 60), (now, 50)]

            with pytest.raises(RateLimitError, match="Token rate limit exceeded"):
                await helix._check_rate_limit()

    @pytest.mark.asyncio
    async def test_rate_limit_disabled(self, test_helix_config: HelixConfig):
        """Test rate limiting disabled."""
        test_helix_config.rate_limit.enabled = False

        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            # Should not raise even with many requests
            for _ in range(100):
                await helix._check_rate_limit()

    def test_record_usage(self, test_helix_config: HelixConfig):
        """Test usage recording."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            helix._record_usage(100)

            assert helix._total_requests == 1
            assert helix._total_tokens == 100
            assert len(helix._rate_limit.request_times) == 1
            assert len(helix._rate_limit.token_counts) == 1


@pytest.mark.unit
class TestRetryLogic:
    """Tests for retry with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_first_attempt(self, test_helix_config: HelixConfig):
        """Test retry succeeds on first attempt."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            async def success_func():
                return "success"

            result = await helix._retry_with_backoff(success_func)

            assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failures(self, test_helix_config: HelixConfig):
        """Test retry succeeds after initial failures."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            call_count = 0

            async def flaky_func():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise HelixError("Temporary error", retriable=True)
                return "success"

            result = await helix._retry_with_backoff(flaky_func)

            assert result == "success"
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_fails_non_retriable(self, test_helix_config: HelixConfig):
        """Test retry fails immediately for non-retriable errors."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            async def fail_func():
                raise HelixError("Non-retriable error", retriable=False)

            with pytest.raises(HelixError, match="Non-retriable error"):
                await helix._retry_with_backoff(fail_func)

    @pytest.mark.asyncio
    async def test_retry_exhausts_attempts(self, test_helix_config: HelixConfig):
        """Test retry exhausts max attempts."""
        test_helix_config.retry.max_retries = 2

        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            async def always_fail():
                raise HelixError("Always fails", retriable=True)

            with pytest.raises(HelixError):
                await helix._retry_with_backoff(always_fail)

    @pytest.mark.asyncio
    async def test_retry_rate_limit_error(self, test_helix_config: HelixConfig):
        """Test retry handles rate limit errors with retry_after."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            call_count = 0

            async def rate_limited_func():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RateLimitError("Rate limited", retry_after=0.01)
                return "success"

            result = await helix._retry_with_backoff(rate_limited_func)

            assert result == "success"
            assert call_count == 2


@pytest.mark.asyncio
@pytest.mark.unit
class TestChatGeneration:
    """Tests for chat generation."""

    async def test_generate_basic(
        self, test_helix_config: HelixConfig, sample_chat_messages: list[ChatMessage]
    ):
        """Test basic chat generation."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider") as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.is_available = True
            mock_provider.chat.return_value = ChatResponse(
                content="Test response",
                model="test/model",
                provider=ProviderType.NVIDIA_API,
                usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            )
            mock_provider_class.return_value = mock_provider

            helix = Helix(test_helix_config)
            helix._providers[ProviderType.NVIDIA_API] = mock_provider

            response = await helix.generate(sample_chat_messages)

            assert response.content == "Test response"
            assert mock_provider.chat.called

    async def test_generate_with_dict_messages(self, test_helix_config: HelixConfig):
        """Test generation with dict messages."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider") as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.is_available = True
            mock_provider.chat.return_value = ChatResponse(
                content="Test response",
                model="test/model",
                provider=ProviderType.NVIDIA_API,
                usage=UsageInfo.empty(),
            )
            mock_provider_class.return_value = mock_provider

            helix = Helix(test_helix_config)
            helix._providers[ProviderType.NVIDIA_API] = mock_provider

            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]

            response = await helix.generate(messages)

            assert response.content == "Test response"

    async def test_generate_with_custom_params(
        self, test_helix_config: HelixConfig, sample_chat_messages: list[ChatMessage]
    ):
        """Test generation with custom parameters."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider") as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.is_available = True
            mock_provider.chat.return_value = ChatResponse(
                content="Test response",
                model="test/model",
                provider=ProviderType.NVIDIA_API,
                usage=UsageInfo.empty(),
            )
            mock_provider_class.return_value = mock_provider

            helix = Helix(test_helix_config)
            helix._providers[ProviderType.NVIDIA_API] = mock_provider

            response = await helix.generate(
                sample_chat_messages,
                temperature=0.8,
                max_tokens=1024,
                stop=["STOP"],
            )

            assert response.content == "Test response"
            assert mock_provider.chat.called

    async def test_generate_with_caching(
        self, test_helix_config: HelixConfig, sample_chat_messages: list[ChatMessage]
    ):
        """Test generation with caching."""
        test_helix_config.cache.cache_chat = True

        with patch("ordinis.ai.helix.engine.NVIDIAProvider") as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.is_available = True
            mock_provider.chat.return_value = ChatResponse(
                content="Test response",
                model="test/model",
                provider=ProviderType.NVIDIA_API,
                usage=UsageInfo.empty(),
            )
            mock_provider_class.return_value = mock_provider

            helix = Helix(test_helix_config)
            helix._providers[ProviderType.NVIDIA_API] = mock_provider

            # First call
            response1 = await helix.generate(sample_chat_messages, use_cache=True)
            # Second call should hit cache
            response2 = await helix.generate(sample_chat_messages, use_cache=True)

            assert response1.content == response2.content
            assert response2.cached is True
            assert mock_provider.chat.call_count == 1  # Only called once

    async def test_generate_fallback_on_error(self, test_helix_config: HelixConfig):
        """Test fallback to secondary model on error."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider") as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.is_available = True

            # First call fails, second succeeds
            call_count = 0

            async def chat_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ProviderError("Primary failed")
                return ChatResponse(
                    content="Fallback response",
                    model="fallback/model",
                    provider=ProviderType.NVIDIA_API,
                    usage=UsageInfo.empty(),
                )

            mock_provider.chat.side_effect = chat_side_effect
            mock_provider_class.return_value = mock_provider

            helix = Helix(test_helix_config)
            helix._providers[ProviderType.NVIDIA_API] = mock_provider

            messages = [ChatMessage(role="user", content="Test")]
            response = await helix.generate(messages)

            assert response.content == "Fallback response"
            assert call_count == 2  # Primary + fallback

    def test_generate_sync(
        self, test_helix_config: HelixConfig, sample_chat_messages: list[ChatMessage]
    ):
        """Test synchronous generation wrapper."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider") as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.is_available = True
            mock_provider.chat.return_value = ChatResponse(
                content="Sync response",
                model="test/model",
                provider=ProviderType.NVIDIA_API,
                usage=UsageInfo.empty(),
            )
            mock_provider_class.return_value = mock_provider

            helix = Helix(test_helix_config)
            helix._providers[ProviderType.NVIDIA_API] = mock_provider

            response = helix.generate_sync(sample_chat_messages)

            assert response.content == "Sync response"


@pytest.mark.asyncio
@pytest.mark.unit
class TestStreamingGeneration:
    """Tests for streaming chat generation."""

    async def test_generate_stream_basic(
        self, test_helix_config: HelixConfig, sample_chat_messages: list[ChatMessage]
    ):
        """Test basic streaming generation."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider") as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.is_available = True

            async def mock_stream(*args, **kwargs):
                for chunk in ["Hello", " ", "world"]:
                    yield chunk

            # Mock chat_stream to return the async generator
            mock_provider.chat_stream = mock_stream
            mock_provider_class.return_value = mock_provider

            helix = Helix(test_helix_config)
            helix._providers[ProviderType.NVIDIA_API] = mock_provider

            chunks = []
            async for chunk in helix.generate_stream(sample_chat_messages):
                chunks.append(chunk)

            assert chunks == ["Hello", " ", "world"]


@pytest.mark.asyncio
@pytest.mark.unit
class TestEmbedding:
    """Tests for embedding generation."""

    async def test_embed_single_text(self, test_helix_config: HelixConfig):
        """Test embedding single text."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider") as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.is_available = True
            mock_provider.embed.return_value = EmbeddingResponse(
                embeddings=[[0.1, 0.2, 0.3]],
                model="test/embed",
                provider=ProviderType.NVIDIA_API,
                usage=UsageInfo.empty(),
                dimension=3,
            )
            mock_provider_class.return_value = mock_provider

            helix = Helix(test_helix_config)
            helix._providers[ProviderType.NVIDIA_API] = mock_provider

            response = await helix.embed("test text")

            assert response.count == 1
            assert response.dimension == 3

    async def test_embed_multiple_texts(self, test_helix_config: HelixConfig):
        """Test embedding multiple texts."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider") as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.is_available = True
            mock_provider.embed.return_value = EmbeddingResponse(
                embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                model="test/embed",
                provider=ProviderType.NVIDIA_API,
                usage=UsageInfo.empty(),
                dimension=2,
            )
            mock_provider_class.return_value = mock_provider

            helix = Helix(test_helix_config)
            helix._providers[ProviderType.NVIDIA_API] = mock_provider

            response = await helix.embed(["text1", "text2", "text3"])

            assert response.count == 3
            assert response.dimension == 2

    async def test_embed_with_caching(self, test_helix_config: HelixConfig):
        """Test embedding with caching."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider") as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.is_available = True
            mock_provider.embed.return_value = EmbeddingResponse(
                embeddings=[[0.1, 0.2]],
                model="test/embed",
                provider=ProviderType.NVIDIA_API,
                usage=UsageInfo.empty(),
                dimension=2,
            )
            mock_provider_class.return_value = mock_provider

            helix = Helix(test_helix_config)
            helix._providers[ProviderType.NVIDIA_API] = mock_provider

            # First call
            response1 = await helix.embed("test text")
            # Second call should hit cache
            response2 = await helix.embed("test text")

            assert response2.cached is True
            assert mock_provider.embed.call_count == 1

    async def test_embed_fallback_on_error(self, test_helix_config: HelixConfig):
        """Test fallback to secondary embedding model on error."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider") as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.is_available = True

            call_count = 0

            async def embed_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ProviderError("Primary failed")
                return EmbeddingResponse(
                    embeddings=[[0.1, 0.2]],
                    model="fallback/embed",
                    provider=ProviderType.NVIDIA_API,
                    usage=UsageInfo.empty(),
                    dimension=2,
                )

            mock_provider.embed.side_effect = embed_side_effect
            mock_provider_class.return_value = mock_provider

            helix = Helix(test_helix_config)
            helix._providers[ProviderType.NVIDIA_API] = mock_provider

            response = await helix.embed("test text")

            assert response.count == 1
            assert call_count == 2

    def test_embed_sync(self, test_helix_config: HelixConfig):
        """Test synchronous embedding wrapper."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider") as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.is_available = True
            mock_provider.embed.return_value = EmbeddingResponse(
                embeddings=[[0.1, 0.2]],
                model="test/embed",
                provider=ProviderType.NVIDIA_API,
                usage=UsageInfo.empty(),
                dimension=2,
            )
            mock_provider_class.return_value = mock_provider

            helix = Helix(test_helix_config)
            helix._providers[ProviderType.NVIDIA_API] = mock_provider

            response = helix.embed_sync("test text")

            assert response.count == 1


@pytest.mark.unit
class TestUtilityMethods:
    """Tests for utility methods."""

    def test_list_models(self, test_helix_config: HelixConfig):
        """Test listing all models."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            models = helix.list_models()

            assert len(models) > 0

    def test_list_models_by_type(self, test_helix_config: HelixConfig):
        """Test listing models by type."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            chat_models = helix.list_models(ModelType.CHAT)
            embed_models = helix.list_models(ModelType.EMBEDDING)

            assert all(m.model_type == ModelType.CHAT for m in chat_models)
            assert all(m.model_type == ModelType.EMBEDDING for m in embed_models)

    def test_get_metrics(self, test_helix_config: HelixConfig):
        """Test getting metrics."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider"):
            helix = Helix(test_helix_config)

            helix._total_requests = 10
            helix._total_tokens = 1000
            helix._cache_hits = 3
            helix._cache_misses = 7

            metrics = helix.get_metrics()

            assert metrics["total_requests"] == 10
            assert metrics["total_tokens"] == 1000
            assert metrics["cache_hits"] == 3
            assert metrics["cache_misses"] == 7
            assert metrics["cache_hit_rate"] == 0.3
            assert metrics["cache_size"] == 0

    @pytest.mark.asyncio
    async def test_health_check(self, test_helix_config: HelixConfig):
        """Test health check."""
        with patch("ordinis.ai.helix.engine.NVIDIAProvider") as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.is_available = True
            mock_provider.health_check.return_value = True
            mock_provider_class.return_value = mock_provider

            helix = Helix(test_helix_config)
            helix._providers[ProviderType.NVIDIA_API] = mock_provider

            health = await helix.health_check()

            assert health[ProviderType.NVIDIA_API.value] is True
