"""
Tests for the RAG vector ID generator module.

Tests cover:
- Deterministic ID generation
- Content hashing
- ID parsing and validation
- VectorIdGenerator class
"""

import pytest

from ordinis.rag.vectordb.id_generator import (
    VectorIdGenerator,
    generate_code_chunk_id,
    generate_content_hash,
    generate_kb_chunk_id,
    generate_message_vector_id,
    generate_session_chunk_id,
    generate_summary_vector_id,
    generate_trade_vector_id,
    generate_vector_id,
    get_id_generator,
    is_valid_vector_id,
    parse_vector_id,
)


class TestGenerateVectorId:
    """Tests for generate_vector_id function."""

    @pytest.mark.unit
    def test_basic_id_generation(self):
        """Test basic vector ID generation."""
        result = generate_vector_id("trade", "t_123", "AAPL long entry")

        assert result.startswith("trade:t_123:")
        assert result.endswith(":0")
        parts = result.split(":")
        assert len(parts) == 4
        assert len(parts[2]) == 12  # Content hash

    @pytest.mark.unit
    def test_id_with_chunk_index(self):
        """Test ID generation with chunk index."""
        result = generate_vector_id("session", "sess_abc", "chunk content", chunk_index=5)

        assert ":5" in result
        parts = result.split(":")
        assert parts[-1] == "5"

    @pytest.mark.unit
    def test_deterministic_ids(self):
        """Test that IDs are deterministic for same input."""
        id1 = generate_vector_id("trade", "t_123", "same content")
        id2 = generate_vector_id("trade", "t_123", "same content")

        assert id1 == id2

    @pytest.mark.unit
    def test_different_content_different_hash(self):
        """Test that different content produces different hash."""
        id1 = generate_vector_id("trade", "t_123", "content A")
        id2 = generate_vector_id("trade", "t_123", "content B")

        # Same prefix but different hash
        assert id1.split(":")[2] != id2.split(":")[2]


class TestGenerateContentHash:
    """Tests for generate_content_hash function."""

    @pytest.mark.unit
    def test_hash_length(self):
        """Test content hash is 12 characters."""
        result = generate_content_hash("test content")

        assert len(result) == 12

    @pytest.mark.unit
    def test_hash_deterministic(self):
        """Test hash is deterministic."""
        hash1 = generate_content_hash("same content")
        hash2 = generate_content_hash("same content")

        assert hash1 == hash2

    @pytest.mark.unit
    def test_hash_hex_format(self):
        """Test hash is valid hexadecimal."""
        result = generate_content_hash("test content")

        # Should not raise
        int(result, 16)


class TestSpecificIdGenerators:
    """Tests for specific ID generator functions."""

    @pytest.mark.unit
    def test_generate_trade_vector_id(self):
        """Test trade vector ID generation."""
        result = generate_trade_vector_id("trade_123", "AAPL buy order")

        assert result.startswith("trade:trade_123:")
        assert result.endswith(":0")

    @pytest.mark.unit
    def test_generate_session_chunk_id(self):
        """Test session chunk ID generation."""
        result = generate_session_chunk_id("sess_abc", "message content", 3)

        assert result.startswith("session:sess_abc:")
        assert result.endswith(":3")

    @pytest.mark.unit
    def test_generate_message_vector_id(self):
        """Test message vector ID generation."""
        result = generate_message_vector_id("sess_abc", 42, "user message")

        assert result.startswith("message:sess_abc:42:")
        assert result.endswith(":0")

    @pytest.mark.unit
    def test_generate_summary_vector_id(self):
        """Test summary vector ID generation."""
        result = generate_summary_vector_id("sess_abc", "rolling", 1, 100, "summary content")

        assert result.startswith("summary:sess_abc:rolling:1-100:")
        assert result.endswith(":0")

    @pytest.mark.unit
    def test_generate_kb_chunk_id(self):
        """Test KB chunk ID generation."""
        result = generate_kb_chunk_id("docs/readme.md", "documentation", 2)

        assert result.startswith("kb:docs/readme.md:")
        assert result.endswith(":2")

    @pytest.mark.unit
    def test_generate_kb_chunk_id_normalizes_path(self):
        """Test KB chunk ID normalizes Windows paths."""
        result = generate_kb_chunk_id("docs\\subfolder\\file.md", "content", 0)

        assert "\\" not in result
        assert "docs/subfolder/file.md" in result

    @pytest.mark.unit
    def test_generate_code_chunk_id(self):
        """Test code chunk ID generation."""
        result = generate_code_chunk_id("src/main.py", "def function():", 1)

        assert result.startswith("code:src/main.py:")
        assert result.endswith(":1")

    @pytest.mark.unit
    def test_generate_code_chunk_id_normalizes_path(self):
        """Test code chunk ID normalizes Windows paths."""
        result = generate_code_chunk_id("src\\module\\file.py", "content", 0)

        assert "\\" not in result
        assert "src/module/file.py" in result


class TestParseVectorId:
    """Tests for parse_vector_id function."""

    @pytest.mark.unit
    def test_parse_simple_id(self):
        """Test parsing a simple vector ID."""
        vector_id = "trade:t_123:abcdef123456:0"
        result = parse_vector_id(vector_id)

        assert result["entity_type"] == "trade"
        assert result["source_id"] == "t_123"
        assert result["content_hash"] == "abcdef123456"
        assert result["chunk_index"] == 0

    @pytest.mark.unit
    def test_parse_id_with_colon_in_source(self):
        """Test parsing ID where source_id contains colons."""
        vector_id = "summary:sess_abc:rolling:1-100:abcdef123456:0"
        result = parse_vector_id(vector_id)

        assert result["entity_type"] == "summary"
        assert result["source_id"] == "sess_abc:rolling:1-100"
        assert result["content_hash"] == "abcdef123456"
        assert result["chunk_index"] == 0

    @pytest.mark.unit
    def test_parse_invalid_id_raises(self):
        """Test parsing invalid ID raises ValueError."""
        with pytest.raises(ValueError, match="Invalid vector ID format"):
            parse_vector_id("invalid:id")

    @pytest.mark.unit
    def test_parse_roundtrip(self):
        """Test parsing a generated ID returns correct values."""
        original_id = generate_vector_id("trade", "t_456", "test content", chunk_index=7)
        parsed = parse_vector_id(original_id)

        assert parsed["entity_type"] == "trade"
        assert parsed["source_id"] == "t_456"
        assert parsed["chunk_index"] == 7


class TestIsValidVectorId:
    """Tests for is_valid_vector_id function."""

    @pytest.mark.unit
    def test_valid_trade_id(self):
        """Test valid trade vector ID."""
        vector_id = generate_trade_vector_id("t_123", "content")

        assert is_valid_vector_id(vector_id) is True

    @pytest.mark.unit
    def test_valid_session_id(self):
        """Test valid session vector ID."""
        vector_id = generate_session_chunk_id("sess_abc", "content", 0)

        assert is_valid_vector_id(vector_id) is True

    @pytest.mark.unit
    def test_valid_message_id(self):
        """Test valid message vector ID."""
        vector_id = generate_message_vector_id("sess_abc", 1, "content")

        assert is_valid_vector_id(vector_id) is True

    @pytest.mark.unit
    def test_valid_summary_id(self):
        """Test valid summary vector ID."""
        vector_id = generate_summary_vector_id("sess_abc", "rolling", 1, 10, "content")

        assert is_valid_vector_id(vector_id) is True

    @pytest.mark.unit
    def test_valid_kb_id(self):
        """Test valid KB vector ID."""
        vector_id = generate_kb_chunk_id("file.md", "content", 0)

        assert is_valid_vector_id(vector_id) is True

    @pytest.mark.unit
    def test_valid_code_id(self):
        """Test valid code vector ID."""
        vector_id = generate_code_chunk_id("file.py", "content", 0)

        assert is_valid_vector_id(vector_id) is True

    @pytest.mark.unit
    def test_invalid_entity_type(self):
        """Test invalid entity type returns False."""
        vector_id = "invalid_type:source:abcdef123456:0"

        assert is_valid_vector_id(vector_id) is False

    @pytest.mark.unit
    def test_invalid_hash_length(self):
        """Test invalid hash length returns False."""
        vector_id = "trade:source:short:0"

        assert is_valid_vector_id(vector_id) is False

    @pytest.mark.unit
    def test_invalid_format(self):
        """Test completely invalid format returns False."""
        assert is_valid_vector_id("not:valid") is False
        assert is_valid_vector_id("") is False

    @pytest.mark.unit
    def test_negative_chunk_index_invalid(self):
        """Test negative chunk index is invalid."""
        # Manually construct an ID with negative chunk
        vector_id = "trade:source:abcdef123456:-1"

        assert is_valid_vector_id(vector_id) is False


class TestVectorIdGenerator:
    """Tests for VectorIdGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create generator with default config."""
        return VectorIdGenerator()

    @pytest.mark.unit
    def test_init_default_values(self, generator):
        """Test initialization with default values."""
        assert generator.embedding_model == "nvidia/llama-3.2-nemoretriever-300m-embed-v2"
        assert generator.embedding_dim == 1024

    @pytest.mark.unit
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        generator = VectorIdGenerator(
            embedding_model="custom/model",
            embedding_dim=512,
        )

        assert generator.embedding_model == "custom/model"
        assert generator.embedding_dim == 512

    @pytest.mark.unit
    def test_create_trade_id(self, generator):
        """Test creating trade ID with metadata."""
        vector_id, metadata = generator.create_trade_id("trade_123", "AAPL buy order")

        assert vector_id.startswith("trade:trade_123:")
        assert metadata["entity_type"] == "trade"
        assert metadata["source_id"] == "trade_123"
        assert metadata["source_table"] == "trades"
        assert "content_hash" in metadata
        assert "indexed_at" in metadata
        assert metadata["embedding_model"] == generator.embedding_model
        assert metadata["embedding_dim"] == generator.embedding_dim

    @pytest.mark.unit
    def test_create_trade_id_with_extra_metadata(self, generator):
        """Test creating trade ID with extra metadata."""
        extra = {"symbol": "AAPL", "side": "long"}
        vector_id, metadata = generator.create_trade_id(
            "trade_123", "content", extra_metadata=extra
        )

        assert metadata["symbol"] == "AAPL"
        assert metadata["side"] == "long"

    @pytest.mark.unit
    def test_create_session_chunk_id(self, generator):
        """Test creating session chunk ID with metadata."""
        vector_id, metadata = generator.create_session_chunk_id(
            "sess_abc", "chunk content", 5
        )

        assert vector_id.startswith("session:sess_abc:")
        assert ":5" in vector_id
        assert metadata["entity_type"] == "session"
        assert metadata["session_id"] == "sess_abc"
        assert metadata["chunk_index"] == 5
        assert metadata["source_table"] == "messages"

    @pytest.mark.unit
    def test_create_session_chunk_id_with_extra_metadata(self, generator):
        """Test creating session chunk ID with extra metadata."""
        extra = {"topic": "trading"}
        vector_id, metadata = generator.create_session_chunk_id(
            "sess_abc", "content", 0, extra_metadata=extra
        )

        assert metadata["topic"] == "trading"

    @pytest.mark.unit
    def test_create_summary_id(self, generator):
        """Test creating summary ID with metadata."""
        vector_id, metadata = generator.create_summary_id(
            "sess_abc", "rolling", 1, 100, "summary content"
        )

        assert vector_id.startswith("summary:")
        assert metadata["entity_type"] == "summary"
        assert metadata["session_id"] == "sess_abc"
        assert metadata["summary_type"] == "rolling"
        assert metadata["start_sequence"] == 1
        assert metadata["end_sequence"] == 100
        assert metadata["source_table"] == "session_summaries"

    @pytest.mark.unit
    def test_create_summary_id_with_extra_metadata(self, generator):
        """Test creating summary ID with extra metadata."""
        extra = {"importance": "high"}
        vector_id, metadata = generator.create_summary_id(
            "sess_abc", "final", 1, 50, "content", extra_metadata=extra
        )

        assert metadata["importance"] == "high"


class TestGetIdGenerator:
    """Tests for get_id_generator function."""

    @pytest.mark.unit
    def test_returns_generator(self):
        """Test get_id_generator returns a VectorIdGenerator."""
        generator = get_id_generator()

        assert isinstance(generator, VectorIdGenerator)

    @pytest.mark.unit
    def test_returns_singleton(self):
        """Test get_id_generator returns same instance."""
        gen1 = get_id_generator()
        gen2 = get_id_generator()

        assert gen1 is gen2
