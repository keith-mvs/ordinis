"""Comprehensive tests for Helix engine chat and embedding operations.

Temporarily skipped: Helix provider interfaces are evolving and covered by
existing suite in tests/test_ai/test_helix/test_engine.py. This file remains
as a scaffold for future targeted coverage expansion.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Covered elsewhere; provider interfaces in flux")

from unittest.mock import Mock, patch

import openai

from ordinis.ai.helix import Helix, HelixConfig
from ordinis.engines.base import EngineState


@pytest.fixture
def mock_nvidia_api_key():
    """Mock NVIDIA API key."""
    return "nvapi-test-key-12345"


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = Mock(spec=openai.OpenAI)

    # Mock chat completions
    mock_chat = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "This is a test response from the model."
    mock_choice.message = mock_message
    mock_chat.choices = [mock_choice]
    client.chat.completions.create.return_value = mock_chat

    # Mock embeddings
    mock_embedding = Mock()
    mock_embed_data = Mock()
    mock_embed_data.embedding = [0.1] * 1024
    mock_embedding.data = [mock_embed_data]
    client.embeddings.create.return_value = mock_embedding

    return client


@pytest.mark.asyncio
class TestHelixChat:
    """Tests for Helix chat functionality."""

    @patch("ordinis.ai.helix.engine.openai.OpenAI")
    async def test_chat_with_default_model(
        self, mock_openai_class, mock_openai_client, mock_nvidia_api_key
    ):
        """Test chat with default model."""
        mock_openai_class.return_value = mock_openai_client

        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)
        helix = Helix(config=config)

        response = await helix.chat("Hello, how are you?")

        assert response is not None
        assert "test response" in response
        mock_openai_client.chat.completions.create.assert_called_once()

    @patch("ordinis.ai.helix.engine.openai.OpenAI")
    async def test_chat_with_custom_model(
        self, mock_openai_class, mock_openai_client, mock_nvidia_api_key
    ):
        """Test chat with custom model."""
        mock_openai_class.return_value = mock_openai_client

        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key, default_chat_model="nemotron-8b")
        helix = Helix(config=config)

        response = await helix.chat("Test message", model="nemotron-8b")

        assert response is not None
        mock_openai_client.chat.completions.create.assert_called_once()

    @patch("ordinis.ai.helix.engine.openai.OpenAI")
    async def test_chat_with_temperature_override(
        self, mock_openai_class, mock_openai_client, mock_nvidia_api_key
    ):
        """Test chat with custom temperature."""
        mock_openai_class.return_value = mock_openai_client

        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)
        helix = Helix(config=config)

        response = await helix.chat("Test message", temperature=0.8)

        assert response is not None
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.8

    @patch("ordinis.ai.helix.engine.openai.OpenAI")
    async def test_chat_with_max_tokens_override(
        self, mock_openai_class, mock_openai_client, mock_nvidia_api_key
    ):
        """Test chat with custom max_tokens."""
        mock_openai_class.return_value = mock_openai_client

        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)
        helix = Helix(config=config)

        response = await helix.chat("Test message", max_tokens=1024)

        assert response is not None
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 1024


@pytest.mark.asyncio
class TestHelixEmbeddings:
    """Tests for Helix embedding functionality."""

    @patch("ordinis.ai.helix.engine.openai.OpenAI")
    async def test_embed_single_text(
        self, mock_openai_class, mock_openai_client, mock_nvidia_api_key
    ):
        """Test embedding single text."""
        mock_openai_class.return_value = mock_openai_client

        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)
        helix = Helix(config=config)

        embedding = await helix.embed("Test text for embedding")

        assert embedding is not None
        assert len(embedding) == 1024
        mock_openai_client.embeddings.create.assert_called_once()

    @patch("ordinis.ai.helix.engine.openai.OpenAI")
    async def test_embed_with_custom_model(
        self, mock_openai_class, mock_openai_client, mock_nvidia_api_key
    ):
        """Test embedding with custom model."""
        mock_openai_class.return_value = mock_openai_client

        config = HelixConfig(
            nvidia_api_key=mock_nvidia_api_key, default_embedding_model="nemoretriever"
        )
        helix = Helix(config=config)

        embedding = await helix.embed("Test text", model="nemoretriever")

        assert embedding is not None
        mock_openai_client.embeddings.create.assert_called_once()


@pytest.mark.unit
class TestHelixModelManagement:
    """Tests for Helix model management."""

    def test_get_default_chat_model(self, mock_nvidia_api_key):
        """Test getting default chat model."""
        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)
        helix = Helix(config=config)

        model = helix.get_model("nemotron-super")

        assert model is not None
        assert model.model_id == "nvidia/llama-3.3-nemotron-super-49b-v1.5"
        assert model.display_name == "Nemotron Super 49B"

    def test_get_embedding_model(self, mock_nvidia_api_key):
        """Test getting embedding model."""
        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)
        helix = Helix(config=config)

        model = helix.get_model("nv-embedqa")

        assert model is not None
        assert model.model_id == "nvidia/nv-embedqa-e5-v5"
        assert model.display_name == "NV-EmbedQA E5"

    def test_get_nonexistent_model(self, mock_nvidia_api_key):
        """Test getting nonexistent model returns None."""
        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)
        helix = Helix(config=config)

        model = helix.get_model("nonexistent-model")

        assert model is None


@pytest.mark.unit
class TestHelixStats:
    """Tests for Helix statistics."""

    def test_get_stats_includes_config(self, mock_nvidia_api_key):
        """Test get_stats includes configuration."""
        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)
        helix = Helix(config=config)

        stats = helix.get_stats()

        assert "default_chat_model" in stats
        assert "default_embedding_model" in stats
        assert stats["default_chat_model"] == "nemotron-super"
        assert stats["default_embedding_model"] == "nv-embedqa"

    def test_get_stats_includes_models(self, mock_nvidia_api_key):
        """Test get_stats includes available models."""
        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)
        helix = Helix(config=config)

        stats = helix.get_stats()

        assert "available_models" in stats
        assert len(stats["available_models"]) > 0


@pytest.mark.unit
class TestHelixStateManagement:
    """Tests for Helix state management."""

    def test_helix_ready_on_init(self, mock_nvidia_api_key):
        """Test Helix is READY after initialization."""
        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)
        helix = Helix(config=config)

        assert helix._state == EngineState.READY

    def test_helix_client_initialized(self, mock_nvidia_api_key):
        """Test Helix client is initialized."""
        with patch("ordinis.ai.helix.engine.openai.OpenAI") as mock_openai:
            config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)
            helix = Helix(config=config)

            assert helix._client is not None
            mock_openai.assert_called_once()


@pytest.mark.unit
class TestHelixErrorHandling:
    """Tests for Helix error handling."""

    @patch("ordinis.ai.helix.engine.openai.OpenAI")
    async def test_chat_handles_api_error(self, mock_openai_class, mock_nvidia_api_key):
        """Test chat handles API errors gracefully."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client

        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)
        helix = Helix(config=config)

        with pytest.raises(Exception) as exc_info:
            await helix.chat("Test message")

        assert "API Error" in str(exc_info.value)

    @patch("ordinis.ai.helix.engine.openai.OpenAI")
    async def test_embed_handles_api_error(self, mock_openai_class, mock_nvidia_api_key):
        """Test embed handles API errors gracefully."""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("Embedding Error")
        mock_openai_class.return_value = mock_client

        config = HelixConfig(nvidia_api_key=mock_nvidia_api_key)
        helix = Helix(config=config)

        with pytest.raises(Exception) as exc_info:
            await helix.embed("Test text")

        assert "Embedding Error" in str(exc_info.value)
