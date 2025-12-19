"""Tests for SynapseConfig."""

from ordinis.ai.synapse import SearchScope, SynapseConfig


class TestSynapseConfig:
    """Tests for SynapseConfig dataclass."""

    def test_default_values(self):
        """Test SynapseConfig default values."""
        config = SynapseConfig()
        assert config.default_scope == SearchScope.AUTO
        assert config.default_top_k == 5
        assert config.similarity_threshold == 0.5
        assert config.top_k_retrieval == 20
        assert config.top_k_rerank == 10
        assert config.use_helix_embeddings is False
        assert config.embedding_model == "nv-embedqa"
        assert config.chroma_persist_dir == "./data/chroma"
        assert config.text_collection == "ordinis_kb"
        assert config.code_collection == "ordinis_code"
        assert config.max_context_tokens == 2000
        assert config.include_citations is True
        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 300

    def test_custom_scope(self):
        """Test SynapseConfig with custom default_scope."""
        config = SynapseConfig(default_scope=SearchScope.CODE)
        assert config.default_scope == SearchScope.CODE

    def test_custom_top_k(self):
        """Test SynapseConfig with custom default_top_k."""
        config = SynapseConfig(default_top_k=10)
        assert config.default_top_k == 10

    def test_custom_similarity_threshold(self):
        """Test SynapseConfig with custom similarity_threshold."""
        config = SynapseConfig(similarity_threshold=0.7)
        assert config.similarity_threshold == 0.7

    def test_custom_retrieval_settings(self):
        """Test SynapseConfig with custom retrieval settings."""
        config = SynapseConfig(
            top_k_retrieval=30,
            top_k_rerank=15,
        )
        assert config.top_k_retrieval == 30
        assert config.top_k_rerank == 15

    def test_custom_embedding_settings(self):
        """Test SynapseConfig with custom embedding settings."""
        config = SynapseConfig(
            use_helix_embeddings=True,
            embedding_model="custom-model",
        )
        assert config.use_helix_embeddings is True
        assert config.embedding_model == "custom-model"

    def test_custom_chroma_settings(self):
        """Test SynapseConfig with custom ChromaDB settings."""
        config = SynapseConfig(
            chroma_persist_dir="/custom/path",
            text_collection="custom_kb",
            code_collection="custom_code",
        )
        assert config.chroma_persist_dir == "/custom/path"
        assert config.text_collection == "custom_kb"
        assert config.code_collection == "custom_code"

    def test_custom_context_settings(self):
        """Test SynapseConfig with custom context settings."""
        config = SynapseConfig(
            max_context_tokens=4000,
            include_citations=False,
        )
        assert config.max_context_tokens == 4000
        assert config.include_citations is False

    def test_custom_cache_settings(self):
        """Test SynapseConfig with custom cache settings."""
        config = SynapseConfig(
            cache_enabled=False,
            cache_ttl_seconds=600,
        )
        assert config.cache_enabled is False
        assert config.cache_ttl_seconds == 600

    def test_validate_success(self):
        """Test validate() returns empty list for valid config."""
        config = SynapseConfig()
        errors = config.validate()
        assert errors == []

    def test_validate_invalid_top_k(self):
        """Test validate() detects invalid default_top_k."""
        config = SynapseConfig(default_top_k=0)
        errors = config.validate()
        assert len(errors) == 1
        assert "default_top_k must be >= 1" in errors[0]

    def test_validate_invalid_top_k_negative(self):
        """Test validate() detects negative default_top_k."""
        config = SynapseConfig(default_top_k=-5)
        errors = config.validate()
        assert len(errors) == 1
        assert "default_top_k must be >= 1" in errors[0]

    def test_validate_invalid_similarity_threshold_low(self):
        """Test validate() detects similarity_threshold too low."""
        config = SynapseConfig(similarity_threshold=-0.1)
        errors = config.validate()
        assert len(errors) == 1
        assert "similarity_threshold must be between 0.0 and 1.0" in errors[0]

    def test_validate_invalid_similarity_threshold_high(self):
        """Test validate() detects similarity_threshold too high."""
        config = SynapseConfig(similarity_threshold=1.5)
        errors = config.validate()
        assert len(errors) == 1
        assert "similarity_threshold must be between 0.0 and 1.0" in errors[0]

    def test_validate_boundary_similarity_threshold_zero(self):
        """Test validate() allows similarity_threshold=0.0."""
        config = SynapseConfig(similarity_threshold=0.0)
        errors = config.validate()
        assert errors == []

    def test_validate_boundary_similarity_threshold_one(self):
        """Test validate() allows similarity_threshold=1.0."""
        config = SynapseConfig(similarity_threshold=1.0)
        errors = config.validate()
        assert errors == []

    def test_validate_invalid_retrieval_ratio(self):
        """Test validate() detects top_k_retrieval < top_k_rerank."""
        config = SynapseConfig(
            top_k_retrieval=5,
            top_k_rerank=10,
        )
        errors = config.validate()
        assert len(errors) == 1
        assert "top_k_retrieval must be >= top_k_rerank" in errors[0]

    def test_validate_equal_retrieval_values(self):
        """Test validate() allows top_k_retrieval == top_k_rerank."""
        config = SynapseConfig(
            top_k_retrieval=10,
            top_k_rerank=10,
        )
        errors = config.validate()
        assert errors == []

    def test_validate_multiple_errors(self):
        """Test validate() returns multiple errors."""
        config = SynapseConfig(
            default_top_k=0,
            similarity_threshold=1.5,
            top_k_retrieval=5,
            top_k_rerank=10,
        )
        errors = config.validate()
        assert len(errors) == 3
        assert any("default_top_k" in e for e in errors)
        assert any("similarity_threshold" in e for e in errors)
        assert any("top_k_retrieval" in e for e in errors)

    def test_all_search_scopes_valid(self):
        """Test that all SearchScope values work with SynapseConfig."""
        for scope in SearchScope:
            config = SynapseConfig(default_scope=scope)
            assert config.default_scope == scope
            errors = config.validate()
            assert errors == []
