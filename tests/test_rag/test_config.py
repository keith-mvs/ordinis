"""Tests for RAG configuration module.

Tests cover:
- RAGConfig model validation
- Default values
- get_config singleton behavior
- set_config functionality
"""

from pathlib import Path

import pytest

from ordinis.rag.config import RAGConfig, get_config, set_config


class TestRAGConfig:
    """Tests for RAGConfig model."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test RAGConfig has sensible defaults."""
        config = RAGConfig()

        assert isinstance(config.chroma_persist_directory, Path)
        assert config.text_collection_name == "kb_text"
        assert config.code_collection_name == "codebase"

    @pytest.mark.unit
    def test_embedding_model_defaults(self):
        """Test default embedding models."""
        config = RAGConfig()

        assert "nvidia" in config.text_embedding_model
        assert "nvidia" in config.code_embedding_model
        assert "nvidia" in config.rerank_model

    @pytest.mark.unit
    def test_embedding_dimension(self):
        """Test default embedding dimension."""
        config = RAGConfig()

        assert config.text_embedding_dimension == 1024

    @pytest.mark.unit
    def test_use_local_embeddings_default(self):
        """Test use_local_embeddings defaults to False."""
        config = RAGConfig()

        assert config.use_local_embeddings is False

    @pytest.mark.unit
    def test_retrieval_params(self):
        """Test retrieval parameter defaults."""
        config = RAGConfig()

        assert config.top_k_retrieval == 20
        assert config.top_k_rerank == 5
        assert 0 < config.similarity_threshold < 1

    @pytest.mark.unit
    def test_chunk_params(self):
        """Test chunk parameter defaults."""
        config = RAGConfig()

        assert config.text_chunk_size > 0
        assert config.text_chunk_overlap > 0
        assert config.text_chunk_overlap < config.text_chunk_size
        assert config.code_chunk_size > 0

    @pytest.mark.unit
    def test_paths_are_path_objects(self):
        """Test path fields are Path objects."""
        config = RAGConfig()

        assert isinstance(config.kb_base_path, Path)
        assert isinstance(config.code_base_path, Path)
        assert isinstance(config.project_root, Path)
        assert isinstance(config.cache_directory, Path)

    @pytest.mark.unit
    def test_nvidia_api_key_optional(self):
        """Test nvidia_api_key can be None."""
        config = RAGConfig()

        assert config.nvidia_api_key is None

    @pytest.mark.unit
    def test_api_fallback_default(self):
        """Test use_api_fallback defaults to True."""
        config = RAGConfig()

        assert config.use_api_fallback is True

    @pytest.mark.unit
    def test_vram_settings(self):
        """Test VRAM management settings."""
        config = RAGConfig()

        assert config.max_vram_usage_gb > 0
        assert config.check_vram_before_load is True

    @pytest.mark.unit
    def test_cache_settings(self):
        """Test cache settings."""
        config = RAGConfig()

        assert config.enable_query_cache is True
        assert config.cache_max_size_mb > 0

    @pytest.mark.unit
    def test_custom_values(self):
        """Test creating config with custom values."""
        config = RAGConfig(
            text_collection_name="custom_text",
            code_collection_name="custom_code",
            top_k_retrieval=10,
            use_local_embeddings=True,
        )

        assert config.text_collection_name == "custom_text"
        assert config.code_collection_name == "custom_code"
        assert config.top_k_retrieval == 10
        assert config.use_local_embeddings is True

    @pytest.mark.unit
    def test_validate_assignment(self):
        """Test config validates on assignment."""
        config = RAGConfig()

        # Should accept valid assignment
        config.top_k_retrieval = 15
        assert config.top_k_retrieval == 15


class TestGetConfig:
    """Tests for get_config function."""

    @pytest.mark.unit
    def test_get_config_returns_rag_config(self):
        """Test get_config returns RAGConfig instance."""
        config = get_config()

        assert isinstance(config, RAGConfig)

    @pytest.mark.unit
    def test_get_config_singleton(self):
        """Test get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2


class TestSetConfig:
    """Tests for set_config function."""

    @pytest.mark.unit
    def test_set_config(self):
        """Test set_config sets global config."""
        custom_config = RAGConfig(text_collection_name="test_collection")

        set_config(custom_config)
        result = get_config()

        assert result.text_collection_name == "test_collection"

        # Clean up - reset to default
        set_config(RAGConfig())
