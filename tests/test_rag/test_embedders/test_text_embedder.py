"""
Tests for TextEmbedder module.

Tests cover:
- Initialization with local and API modes
- Embedding functionality
- Fallback behavior
- Utility methods
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestTextEmbedderInit:
    """Tests for TextEmbedder initialization."""

    @pytest.mark.unit
    def test_init_with_local_mode_no_fallback(self):
        """Test initialization in local mode without GPU falls back."""
        with (
            patch("ordinis.rag.embedders.text_embedder.get_config") as mock_config,
            patch(
                "ordinis.rag.embedders.text_embedder.TextEmbedder._init_local_model"
            ) as mock_init,
        ):
            mock_config.return_value = MagicMock(
                use_local_embeddings=True,
                nvidia_api_key=None,
                text_embedding_model="nvidia/test-model",
                text_embedding_dimension=1024,
            )

            from ordinis.rag.embedders.text_embedder import TextEmbedder

            embedder = TextEmbedder(use_local=True)

            mock_init.assert_called_once()
            assert embedder.use_local is True

    @pytest.mark.unit
    def test_init_with_api_mode(self):
        """Test initialization in API mode."""
        with (
            patch("ordinis.rag.embedders.text_embedder.get_config") as mock_config,
            patch(
                "ordinis.rag.embedders.text_embedder.TextEmbedder._init_api_client"
            ) as mock_init,
        ):
            mock_config.return_value = MagicMock(
                use_local_embeddings=False,
                nvidia_api_key="test-key",
                text_embedding_model="nvidia/test-model",
                text_embedding_dimension=1024,
            )

            from ordinis.rag.embedders.text_embedder import TextEmbedder

            embedder = TextEmbedder(use_local=False, api_key="test-key")

            mock_init.assert_called_once()
            assert embedder.use_local is False

    @pytest.mark.unit
    def test_init_api_mode_no_key_raises(self):
        """Test API mode without key raises ValueError."""
        with (
            patch("ordinis.rag.embedders.text_embedder.get_config") as mock_config,
            patch.dict("os.environ", {}, clear=True),
        ):
            mock_config.return_value = MagicMock(
                use_local_embeddings=False,
                nvidia_api_key=None,
                text_embedding_model="nvidia/test-model",
                text_embedding_dimension=1024,
            )

            from ordinis.rag.embedders.text_embedder import TextEmbedder

            with pytest.raises(ValueError, match="API key required"):
                TextEmbedder(use_local=False, api_key=None)


class TestTextEmbedderEmbed:
    """Tests for embed method."""

    @pytest.fixture
    def embedder(self):
        """Create embedder with mocked model."""
        with (
            patch("ordinis.rag.embedders.text_embedder.get_config") as mock_config,
            patch(
                "ordinis.rag.embedders.text_embedder.TextEmbedder._init_local_model"
            ),
        ):
            mock_config.return_value = MagicMock(
                use_local_embeddings=True,
                nvidia_api_key=None,
                text_embedding_model="nvidia/test-model",
                text_embedding_dimension=1024,
            )

            from ordinis.rag.embedders.text_embedder import TextEmbedder

            embedder = TextEmbedder(use_local=True)
            embedder._model = MagicMock()
            return embedder

    @pytest.mark.unit
    def test_embed_single_text(self, embedder):
        """Test embedding single text returns 1D array."""
        embedder._model.encode.return_value = np.array([[0.1] * 1024])

        result = embedder.embed("test text")

        assert result.shape == (1024,)
        embedder._model.encode.assert_called_once()

    @pytest.mark.unit
    def test_embed_multiple_texts(self, embedder):
        """Test embedding multiple texts returns 2D array."""
        embedder._model.encode.return_value = np.array([[0.1] * 1024, [0.2] * 1024])

        result = embedder.embed(["text 1", "text 2"])

        assert result.shape == (2, 1024)

    @pytest.mark.unit
    def test_embed_empty_list(self, embedder):
        """Test embedding empty list returns empty array."""
        result = embedder.embed([])

        assert len(result) == 0

    @pytest.mark.unit
    def test_embed_truncates_to_dimension(self, embedder):
        """Test embedding truncates to configured dimension."""
        # Return embeddings larger than configured dimension
        embedder._model.encode.return_value = np.array([[0.1] * 2048])
        embedder.embedding_dim = 1024

        result = embedder.embed("test")

        assert result.shape == (1024,)


class TestTextEmbedderAPI:
    """Tests for API-mode embedding."""

    @pytest.fixture
    def embedder(self):
        """Create embedder with mocked API client."""
        with (
            patch("ordinis.rag.embedders.text_embedder.get_config") as mock_config,
            patch(
                "ordinis.rag.embedders.text_embedder.TextEmbedder._init_api_client"
            ),
        ):
            mock_config.return_value = MagicMock(
                use_local_embeddings=False,
                nvidia_api_key="test-key",
                text_embedding_model="nvidia/test-model",
                text_embedding_dimension=1024,
            )

            from ordinis.rag.embedders.text_embedder import TextEmbedder

            embedder = TextEmbedder(use_local=False, api_key="test-key")
            embedder._client = MagicMock()
            return embedder

    @pytest.mark.unit
    def test_embed_with_api_client(self, embedder):
        """Test embedding with API client."""
        embedder._client.embed_documents.return_value = [[0.1] * 1024]

        result = embedder.embed("test text")

        assert result.shape == (1024,)
        embedder._client.embed_documents.assert_called_once()


class TestTextEmbedderUtilities:
    """Tests for utility methods."""

    @pytest.fixture
    def embedder(self):
        """Create embedder with mocked model."""
        with (
            patch("ordinis.rag.embedders.text_embedder.get_config") as mock_config,
            patch(
                "ordinis.rag.embedders.text_embedder.TextEmbedder._init_local_model"
            ),
        ):
            mock_config.return_value = MagicMock(
                use_local_embeddings=True,
                nvidia_api_key=None,
                text_embedding_model="nvidia/test-model",
                text_embedding_dimension=1024,
            )

            from ordinis.rag.embedders.text_embedder import TextEmbedder

            embedder = TextEmbedder(use_local=True)
            return embedder

    @pytest.mark.unit
    def test_get_embedding_dimension(self, embedder):
        """Test get_embedding_dimension returns configured dimension."""
        assert embedder.get_embedding_dimension() == 1024

    @pytest.mark.unit
    def test_is_available_with_model(self, embedder):
        """Test is_available returns True when model is loaded."""
        embedder._model = MagicMock()

        assert embedder.is_available() is True

    @pytest.mark.unit
    def test_is_available_without_model(self, embedder):
        """Test is_available returns False when model not loaded."""
        embedder._model = None

        assert embedder.is_available() is False

    @pytest.mark.unit
    def test_unload_clears_model(self, embedder):
        """Test unload clears the model."""
        embedder._model = MagicMock()

        embedder.unload()

        assert embedder._model is None

    @pytest.mark.unit
    def test_unload_without_model(self, embedder):
        """Test unload is safe when no model."""
        embedder._model = None

        # Should not raise
        embedder.unload()

        assert embedder._model is None
