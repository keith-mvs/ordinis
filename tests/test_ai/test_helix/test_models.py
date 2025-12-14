"""
Tests for Helix data models.

Tests cover:
- ModelType and ProviderType enums
- ModelInfo metadata
- ChatMessage conversion
- UsageInfo handling
- ChatResponse and EmbeddingResponse
- Exception hierarchy
"""

from datetime import datetime

import pytest

from ordinis.ai.helix.models import (
    ChatMessage,
    ChatResponse,
    EmbeddingResponse,
    HelixError,
    ModelInfo,
    ModelNotFoundError,
    ModelType,
    ProviderError,
    ProviderType,
    RateLimitError,
    UsageInfo,
)


@pytest.mark.unit
class TestEnums:
    """Tests for enum types."""

    def test_model_type_values(self):
        """Test ModelType enum values."""
        assert ModelType.CHAT.value == "chat"
        assert ModelType.EMBEDDING.value == "embedding"
        assert ModelType.CODE.value == "code"
        assert ModelType.MULTIMODAL.value == "multimodal"

    def test_provider_type_values(self):
        """Test ProviderType enum values."""
        assert ProviderType.NVIDIA_API.value == "nvidia_api"
        assert ProviderType.LOCAL_GPU.value == "local_gpu"
        assert ProviderType.LOCAL_CPU.value == "local_cpu"
        assert ProviderType.MOCK.value == "mock"


@pytest.mark.unit
class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test creating ModelInfo."""
        model = ModelInfo(
            model_id="test/model",
            display_name="Test Model",
            model_type=ModelType.CHAT,
            provider=ProviderType.NVIDIA_API,
            context_length=4096,
        )

        assert model.model_id == "test/model"
        assert model.display_name == "Test Model"
        assert model.model_type == ModelType.CHAT
        assert model.provider == ProviderType.NVIDIA_API
        assert model.context_length == 4096
        assert model.embedding_dim is None
        assert model.supports_streaming is True
        assert model.supports_function_calling is False
        assert model.cost_per_1k_tokens == 0.0
        assert model.default_temperature == 0.7
        assert model.max_output_tokens == 4096

    def test_model_info_embedding_model(self):
        """Test embedding model with dimension."""
        model = ModelInfo(
            model_id="test/embed",
            display_name="Test Embedder",
            model_type=ModelType.EMBEDDING,
            provider=ProviderType.NVIDIA_API,
            context_length=512,
            embedding_dim=768,
            supports_streaming=False,
        )

        assert model.embedding_dim == 768
        assert model.supports_streaming is False

    def test_model_info_str(self):
        """Test ModelInfo string representation."""
        model = ModelInfo(
            model_id="test/model",
            display_name="Test Model",
            model_type=ModelType.CHAT,
            provider=ProviderType.NVIDIA_API,
            context_length=4096,
        )

        assert str(model) == "Test Model (test/model)"

    def test_model_info_frozen(self):
        """Test that ModelInfo is immutable (frozen)."""
        model = ModelInfo(
            model_id="test/model",
            display_name="Test Model",
            model_type=ModelType.CHAT,
            provider=ProviderType.NVIDIA_API,
            context_length=4096,
        )

        with pytest.raises(AttributeError):
            model.context_length = 8192


@pytest.mark.unit
class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_chat_message_creation(self):
        """Test creating ChatMessage."""
        msg = ChatMessage(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None
        assert msg.metadata == {}

    def test_chat_message_with_name(self):
        """Test ChatMessage with name."""
        msg = ChatMessage(role="assistant", content="Hi", name="bot")

        assert msg.name == "bot"

    def test_chat_message_with_metadata(self):
        """Test ChatMessage with metadata."""
        msg = ChatMessage(
            role="user",
            content="Test",
            metadata={"timestamp": "2025-01-01", "source": "api"},
        )

        assert msg.metadata["timestamp"] == "2025-01-01"
        assert msg.metadata["source"] == "api"

    def test_chat_message_to_dict(self):
        """Test ChatMessage.to_dict() conversion."""
        msg = ChatMessage(role="user", content="Hello")
        result = msg.to_dict()

        assert result == {"role": "user", "content": "Hello"}

    def test_chat_message_to_dict_with_name(self):
        """Test ChatMessage.to_dict() with name."""
        msg = ChatMessage(role="assistant", content="Hi", name="bot")
        result = msg.to_dict()

        assert result == {"role": "assistant", "content": "Hi", "name": "bot"}

    def test_chat_message_to_dict_excludes_metadata(self):
        """Test that to_dict() excludes metadata."""
        msg = ChatMessage(
            role="user",
            content="Test",
            metadata={"internal": "data"},
        )
        result = msg.to_dict()

        assert "metadata" not in result


@pytest.mark.unit
class TestUsageInfo:
    """Tests for UsageInfo dataclass."""

    def test_usage_info_creation(self):
        """Test creating UsageInfo."""
        usage = UsageInfo(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=0.0015,
        )

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30
        assert usage.cost_usd == 0.0015

    def test_usage_info_empty(self):
        """Test UsageInfo.empty() constructor."""
        usage = UsageInfo.empty()

        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cost_usd == 0.0


@pytest.mark.unit
class TestChatResponse:
    """Tests for ChatResponse dataclass."""

    def test_chat_response_creation(self):
        """Test creating ChatResponse."""
        usage = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = ChatResponse(
            content="Hello!",
            model="test/model",
            provider=ProviderType.NVIDIA_API,
            usage=usage,
        )

        assert response.content == "Hello!"
        assert response.model == "test/model"
        assert response.provider == ProviderType.NVIDIA_API
        assert response.usage == usage
        assert response.finish_reason == "stop"
        assert response.latency_ms == 0.0
        assert isinstance(response.timestamp, datetime)
        assert response.raw_response is None
        assert response.cached is False

    def test_chat_response_with_all_fields(self):
        """Test ChatResponse with all optional fields."""
        usage = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        raw = {"choices": [{"message": {"content": "Hello!"}}]}
        timestamp = datetime.now()

        response = ChatResponse(
            content="Hello!",
            model="test/model",
            provider=ProviderType.NVIDIA_API,
            usage=usage,
            finish_reason="length",
            latency_ms=150.5,
            timestamp=timestamp,
            raw_response=raw,
            cached=True,
        )

        assert response.finish_reason == "length"
        assert response.latency_ms == 150.5
        assert response.timestamp == timestamp
        assert response.raw_response == raw
        assert response.cached is True

    def test_chat_response_str(self):
        """Test ChatResponse string representation."""
        usage = UsageInfo.empty()
        response = ChatResponse(
            content="Test response",
            model="test/model",
            provider=ProviderType.MOCK,
            usage=usage,
        )

        assert str(response) == "Test response"


@pytest.mark.unit
class TestEmbeddingResponse:
    """Tests for EmbeddingResponse dataclass."""

    def test_embedding_response_creation(self):
        """Test creating EmbeddingResponse."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        usage = UsageInfo(prompt_tokens=5, completion_tokens=0, total_tokens=5)

        response = EmbeddingResponse(
            embeddings=embeddings,
            model="test/embed",
            provider=ProviderType.NVIDIA_API,
            usage=usage,
            dimension=3,
        )

        assert response.embeddings == embeddings
        assert response.model == "test/embed"
        assert response.provider == ProviderType.NVIDIA_API
        assert response.usage == usage
        assert response.dimension == 3
        assert response.latency_ms == 0.0
        assert isinstance(response.timestamp, datetime)
        assert response.cached is False

    def test_embedding_response_count(self):
        """Test EmbeddingResponse.count property."""
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        usage = UsageInfo.empty()

        response = EmbeddingResponse(
            embeddings=embeddings,
            model="test/embed",
            provider=ProviderType.MOCK,
            usage=usage,
            dimension=2,
        )

        assert response.count == 3

    def test_embedding_response_as_numpy(self):
        """Test EmbeddingResponse.as_numpy() conversion."""
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        usage = UsageInfo.empty()

        response = EmbeddingResponse(
            embeddings=embeddings,
            model="test/embed",
            provider=ProviderType.MOCK,
            usage=usage,
            dimension=2,
        )

        try:
            import numpy as np

            array = response.as_numpy()
            assert isinstance(array, np.ndarray)
            assert array.shape == (2, 2)
            assert array[0, 0] == 0.1
            assert array[1, 1] == 0.4
        except ImportError:
            pytest.skip("NumPy not installed")


@pytest.mark.unit
class TestExceptions:
    """Tests for exception classes."""

    def test_helix_error_basic(self):
        """Test basic HelixError."""
        error = HelixError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.provider is None
        assert error.model is None
        assert error.retriable is False

    def test_helix_error_with_context(self):
        """Test HelixError with context."""
        error = HelixError(
            "Provider failed",
            provider=ProviderType.NVIDIA_API,
            model="test/model",
            retriable=True,
        )

        assert str(error) == "Provider failed"
        assert error.provider == ProviderType.NVIDIA_API
        assert error.model == "test/model"
        assert error.retriable is True

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError(
            "Rate limit exceeded",
            retry_after=60.0,
            provider=ProviderType.NVIDIA_API,
        )

        assert str(error) == "Rate limit exceeded"
        assert error.retry_after == 60.0
        assert error.provider == ProviderType.NVIDIA_API
        assert error.retriable is True

    def test_rate_limit_error_default_message(self):
        """Test RateLimitError with default message."""
        error = RateLimitError()

        assert str(error) == "Rate limit exceeded"
        assert error.retry_after is None

    def test_model_not_found_error(self):
        """Test ModelNotFoundError."""
        error = ModelNotFoundError(
            "invalid-model",
            available=["model1", "model2", "model3"],
        )

        assert "invalid-model" in str(error)
        assert "not found" in str(error)
        assert error.model == "invalid-model"
        assert error.available == ["model1", "model2", "model3"]

    def test_model_not_found_error_with_long_list(self):
        """Test ModelNotFoundError truncates long available list."""
        available = [f"model{i}" for i in range(10)]
        error = ModelNotFoundError("invalid", available=available)

        error_str = str(error)
        assert "invalid" in error_str
        assert "Available:" in error_str
        # Should only show first 5
        assert "model0" in error_str
        assert "model4" in error_str

    def test_provider_error(self):
        """Test ProviderError."""
        error = ProviderError(
            "API request failed",
            status_code=500,
            provider=ProviderType.NVIDIA_API,
        )

        assert str(error) == "API request failed"
        assert error.status_code == 500
        assert error.provider == ProviderType.NVIDIA_API

    def test_provider_error_without_status(self):
        """Test ProviderError without status code."""
        error = ProviderError("Connection failed")

        assert error.status_code is None

    def test_exception_inheritance(self):
        """Test exception inheritance hierarchy."""
        assert issubclass(RateLimitError, HelixError)
        assert issubclass(ModelNotFoundError, HelixError)
        assert issubclass(ProviderError, HelixError)
        assert issubclass(HelixError, Exception)
