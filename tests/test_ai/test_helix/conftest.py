"""
Shared test fixtures for Helix tests.

Provides:
- Mock NVIDIA API responses
- HelixConfig with test settings
- Mock providers
- Common test data
"""

from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest

# Add src to path to allow direct imports
src_path = Path(__file__).parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Mock chromadb to avoid import errors
sys.modules["chromadb"] = MagicMock()
sys.modules["chromadb.config"] = MagicMock()

# Import directly from helix submodule
from ordinis.ai.helix.config import (  # noqa: E402
    CacheConfig,
    HelixConfig,
    RateLimitConfig,
    RetryConfig,
)
from ordinis.ai.helix.models import (  # noqa: E402
    ChatMessage,
    ChatResponse,
    EmbeddingResponse,
    ModelInfo,
    ModelType,
    ProviderType,
    UsageInfo,
)
from ordinis.ai.helix.providers.base import BaseProvider  # noqa: E402


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    def __init__(self, provider_type: ProviderType = ProviderType.MOCK):
        """Initialize mock provider."""
        self._provider_type = provider_type
        self._is_available = True
        self.chat_called = False
        self.embed_called = False
        self.stream_called = False

    @property
    def provider_type(self) -> ProviderType:
        """Return provider type."""
        return self._provider_type

    @property
    def is_available(self) -> bool:
        """Check availability."""
        return self._is_available

    async def chat(
        self,
        messages: list[ChatMessage],
        model: ModelInfo,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: object,
    ) -> ChatResponse:
        """Mock chat completion."""
        self.chat_called = True
        return ChatResponse(
            content="Mock response",
            model=model.model_id,
            provider=self.provider_type,
            usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            latency_ms=100.0,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: ModelInfo,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: object,
    ) -> AsyncIterator[str]:
        """Mock streaming chat completion."""
        self.stream_called = True
        for chunk in ["Mock ", "streamed ", "response"]:
            yield chunk

    async def embed(
        self,
        texts: list[str],
        model: ModelInfo,
        **kwargs: object,
    ) -> EmbeddingResponse:
        """Mock embedding generation."""
        self.embed_called = True
        return EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3] for _ in texts],
            model=model.model_id,
            provider=self.provider_type,
            usage=UsageInfo(prompt_tokens=5, completion_tokens=0, total_tokens=5),
            dimension=3,
            latency_ms=50.0,
        )


@pytest.fixture
def mock_nvidia_api_key() -> str:
    """Mock NVIDIA API key."""
    return "nvapi-test-key-1234567890"


@pytest.fixture
def test_retry_config() -> RetryConfig:
    """Fast retry config for tests."""
    return RetryConfig(
        max_retries=2,
        initial_delay_ms=10,
        max_delay_ms=100,
        exponential_base=2.0,
        jitter=False,
    )


@pytest.fixture
def test_cache_config() -> CacheConfig:
    """Cache config for tests."""
    return CacheConfig(
        enabled=True,
        ttl_seconds=60,
        max_entries=100,
        cache_embeddings=True,
        cache_chat=False,
    )


@pytest.fixture
def test_rate_limit_config() -> RateLimitConfig:
    """Rate limit config for tests."""
    return RateLimitConfig(
        enabled=True,
        requests_per_minute=100,
        tokens_per_minute=10000,
        concurrent_requests=5,
    )


@pytest.fixture
def test_helix_config(
    mock_nvidia_api_key: str,
    test_retry_config: RetryConfig,
    test_cache_config: CacheConfig,
    test_rate_limit_config: RateLimitConfig,
) -> HelixConfig:
    """HelixConfig with test settings."""
    return HelixConfig(
        nvidia_api_key=mock_nvidia_api_key,
        default_chat_model="nemotron-super",
        fallback_chat_model="nemotron-8b",
        default_embedding_model="nv-embedqa",
        fallback_embedding_model="nemoretriever",
        default_temperature=0.2,
        default_max_tokens=2048,
        prefer_local=False,
        allow_fallback=True,
        retry=test_retry_config,
        cache=test_cache_config,
        rate_limit=test_rate_limit_config,
        log_requests=False,
        log_responses=False,
        connect_timeout_ms=5000,
        read_timeout_ms=30000,
    )


@pytest.fixture
def test_model_info() -> ModelInfo:
    """Test model info."""
    return ModelInfo(
        model_id="test/model",
        display_name="Test Model",
        model_type=ModelType.CHAT,
        provider=ProviderType.MOCK,
        context_length=4096,
        supports_streaming=True,
        default_temperature=0.2,
        max_output_tokens=2048,
    )


@pytest.fixture
def test_embedding_model() -> ModelInfo:
    """Test embedding model info."""
    return ModelInfo(
        model_id="test/embed-model",
        display_name="Test Embed Model",
        model_type=ModelType.EMBEDDING,
        provider=ProviderType.MOCK,
        context_length=512,
        embedding_dim=768,
        supports_streaming=False,
    )


@pytest.fixture
def sample_chat_messages() -> list[ChatMessage]:
    """Sample chat messages."""
    return [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hello, how are you?"),
    ]


@pytest.fixture
def sample_chat_response() -> ChatResponse:
    """Sample chat response."""
    return ChatResponse(
        content="I'm doing well, thank you!",
        model="test/model",
        provider=ProviderType.MOCK,
        usage=UsageInfo(prompt_tokens=15, completion_tokens=8, total_tokens=23),
        latency_ms=150.0,
        finish_reason="stop",
        timestamp=datetime.now(),
    )


@pytest.fixture
def sample_embedding_response() -> EmbeddingResponse:
    """Sample embedding response."""
    return EmbeddingResponse(
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        model="test/embed-model",
        provider=ProviderType.MOCK,
        usage=UsageInfo(prompt_tokens=10, completion_tokens=0, total_tokens=10),
        dimension=3,
        latency_ms=75.0,
        timestamp=datetime.now(),
    )


@pytest.fixture
def mock_provider() -> MockProvider:
    """Mock provider instance."""
    return MockProvider()


@pytest.fixture
def mock_langchain_chat_nvidia(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock langchain ChatNVIDIA client."""
    mock_response = MagicMock()
    mock_response.content = "Mock NVIDIA response"
    mock_response.response_metadata = {
        "token_usage": {
            "prompt_tokens": 12,
            "completion_tokens": 25,
            "total_tokens": 37,
        }
    }

    mock_client = MagicMock()
    mock_client.invoke.return_value = mock_response
    mock_client.stream.return_value = [
        MagicMock(content="Mock "),
        MagicMock(content="streaming "),
        MagicMock(content="response"),
    ]

    mock_class = MagicMock(return_value=mock_client)

    # Mock the import and class
    monkeypatch.setattr(
        "ordinis.ai.helix.providers.nvidia.ChatNVIDIA",
        mock_class,
        raising=False,
    )

    return mock_client


@pytest.fixture
def mock_langchain_embeddings_nvidia(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock langchain NVIDIAEmbeddings client."""
    mock_client = MagicMock()
    mock_client.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    mock_class = MagicMock(return_value=mock_client)

    # Mock the import and class
    monkeypatch.setattr(
        "ordinis.ai.helix.providers.nvidia.NVIDIAEmbeddings",
        mock_class,
        raising=False,
    )

    return mock_client
