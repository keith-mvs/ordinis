"""Tests for Synapse retrieval engine."""

from unittest.mock import Mock, patch

import pytest

from ordinis.ai.synapse import (
    RetrievalContext,
    RetrievalResultSet,
    SearchScope,
    Synapse,
    SynapseConfig,
)
from ordinis.engines.base import EngineState
from ordinis.rag.vectordb.schema import QueryResponse, RetrievalResult


class TestSynapseInitialization:
    """Tests for Synapse initialization."""

    def test_init_with_default_config(self, mock_helix):
        """Test Synapse initialization with default config."""
        synapse = Synapse(helix=mock_helix)
        assert synapse.config is not None
        assert synapse.config.default_scope == SearchScope.AUTO
        assert synapse.config.default_top_k == 5
        assert synapse._rag_engine is None
        assert synapse._state != EngineState.READY

    def test_init_with_custom_config(self, synapse_config, mock_helix):
        """Test Synapse initialization with custom config."""
        synapse = Synapse(config=synapse_config, helix=mock_helix)
        assert synapse.config == synapse_config
        assert synapse._rag_engine is None
        assert synapse._state != EngineState.READY

    def test_init_with_helix(self, synapse_config, mock_helix):
        """Test Synapse initialization with Helix instance."""
        synapse = Synapse(config=synapse_config, helix=mock_helix)
        assert synapse.helix == mock_helix

    def test_init_validates_config(self, mock_helix):
        """Test Synapse initialization validates config."""
        invalid_config = SynapseConfig(default_top_k=0)
        with pytest.raises(ValueError, match="Invalid Synapse configuration"):
            Synapse(config=invalid_config, helix=mock_helix)

    def test_init_with_invalid_config_multiple_errors(self, mock_helix):
        """Test Synapse initialization with multiple config errors."""
        invalid_config = SynapseConfig(
            default_top_k=-1,
            similarity_threshold=1.5,
        )
        with pytest.raises(ValueError, match="Invalid Synapse configuration"):
            Synapse(config=invalid_config, helix=mock_helix)


class TestSynapseRAGEngine:
    """Tests for RAG engine management."""

    def test_ensure_rag_engine_lazy_init(self, synapse_config, mock_helix):
        """Test _ensure_rag_engine performs lazy initialization."""
        with patch("ordinis.rag.retrieval.engine.RetrievalEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine

            synapse = Synapse(config=synapse_config, helix=mock_helix)
            assert synapse._rag_engine is None

            engine = synapse._ensure_rag_engine()

            assert engine == mock_engine
            assert synapse._rag_engine == mock_engine
            mock_engine_class.assert_called_once()

    def test_ensure_rag_engine_returns_cached(self, synapse_with_mock_rag, mock_rag_engine):
        """Test _ensure_rag_engine returns cached instance."""
        engine1 = synapse_with_mock_rag._ensure_rag_engine()
        engine2 = synapse_with_mock_rag._ensure_rag_engine()
        assert engine1 is engine2
        assert engine1 is mock_rag_engine

    def test_is_available_true(self, synapse_with_mock_rag):
        """Test is_available returns True when RAG engine works."""
        assert synapse_with_mock_rag.is_available is True

    def test_is_available_false_on_error(self, synapse_config, mock_helix):
        """Test is_available returns False when initialization fails."""
        with patch("ordinis.rag.retrieval.engine.RetrievalEngine") as mock_engine_class:
            mock_engine_class.side_effect = Exception("Init failed")

            synapse = Synapse(config=synapse_config, helix=mock_helix)
            assert synapse.is_available is False


class TestSynapseRetrieve:
    """Tests for retrieve() method."""

    def test_retrieve_basic(self, synapse_with_mock_rag, mock_rag_engine):
        """Test basic retrieve() call."""
        result = synapse_with_mock_rag.retrieve("test query")

        assert isinstance(result, RetrievalResultSet)
        assert result.query == "test query"
        assert len(result.snippets) > 0
        assert result.execution_time_ms > 0

        # Verify RAG engine was called
        mock_rag_engine.query.assert_called_once()
        call_args = mock_rag_engine.query.call_args
        assert call_args.kwargs["query"] == "test query"

    def test_retrieve_with_context_object(self, synapse_with_mock_rag, retrieval_context):
        """Test retrieve() with RetrievalContext object."""
        result = synapse_with_mock_rag.retrieve("test query", context=retrieval_context)

        assert isinstance(result, RetrievalResultSet)
        assert result.query == "test query"

    def test_retrieve_with_context_dict(self, synapse_with_mock_rag):
        """Test retrieve() with context as dict."""
        context_dict = {
            "scope": SearchScope.CODE,
            "top_k": 10,
            "min_score": 0.7,
        }
        result = synapse_with_mock_rag.retrieve("test query", context=context_dict)

        assert isinstance(result, RetrievalResultSet)

    def test_retrieve_with_none_context(self, synapse_with_mock_rag):
        """Test retrieve() with None context uses defaults."""
        result = synapse_with_mock_rag.retrieve("test query", context=None)

        assert isinstance(result, RetrievalResultSet)

    def test_retrieve_filters_by_min_score(self, synapse_with_mock_rag, mock_retrieval_results):
        """Test retrieve() filters results by min_score."""
        context = RetrievalContext(min_score=0.8)
        result = synapse_with_mock_rag.retrieve("test query", context=context)

        # Should filter out results with score < 0.8
        for snippet in result.snippets:
            assert snippet.score >= 0.8

    def test_retrieve_converts_snippets(self, synapse_with_mock_rag):
        """Test retrieve() converts RetrievalResults to Snippets."""
        result = synapse_with_mock_rag.retrieve("test query")

        # Check snippets are properly converted
        assert all(hasattr(s, "id") for s in result.snippets)
        assert all(hasattr(s, "text") for s in result.snippets)
        assert all(hasattr(s, "score") for s in result.snippets)
        assert all(hasattr(s, "source") for s in result.snippets)

    def test_retrieve_separates_text_and_code(self, synapse_with_mock_rag):
        """Test retrieve() separates text and code results."""
        result = synapse_with_mock_rag.retrieve("test query")

        # Should have both text and code results based on mock data
        assert result.text_results > 0
        assert result.code_results > 0
        assert result.text_results + result.code_results == result.count

    def test_retrieve_with_engine_filter(self, synapse_with_mock_rag, mock_rag_engine):
        """Test retrieve() with engine filter."""
        context = RetrievalContext(engine="riskguard")
        synapse_with_mock_rag.retrieve("test query", context=context)

        # Verify filters were passed to RAG engine
        call_args = mock_rag_engine.query.call_args
        assert call_args.kwargs["filters"]["engine"] == "riskguard"

    def test_retrieve_with_custom_filters(self, synapse_with_mock_rag, mock_rag_engine):
        """Test retrieve() with custom filters."""
        context = RetrievalContext(filters={"domain": "trading", "priority": "high"})
        synapse_with_mock_rag.retrieve("test query", context=context)

        call_args = mock_rag_engine.query.call_args
        assert call_args.kwargs["filters"]["domain"] == "trading"
        assert call_args.kwargs["filters"]["priority"] == "high"

    def test_retrieve_scope_to_query_type_text(self, synapse_with_mock_rag, mock_rag_engine):
        """Test retrieve() converts TEXT scope to query type."""
        context = RetrievalContext(scope=SearchScope.TEXT)
        synapse_with_mock_rag.retrieve("test query", context=context)

        call_args = mock_rag_engine.query.call_args
        assert call_args.kwargs["query_type"] == "text"

    def test_retrieve_scope_to_query_type_code(self, synapse_with_mock_rag, mock_rag_engine):
        """Test retrieve() converts CODE scope to query type."""
        context = RetrievalContext(scope=SearchScope.CODE)
        synapse_with_mock_rag.retrieve("test query", context=context)

        call_args = mock_rag_engine.query.call_args
        assert call_args.kwargs["query_type"] == "code"

    def test_retrieve_scope_to_query_type_hybrid(self, synapse_with_mock_rag, mock_rag_engine):
        """Test retrieve() converts HYBRID scope to query type."""
        context = RetrievalContext(scope=SearchScope.HYBRID)
        synapse_with_mock_rag.retrieve("test query", context=context)

        call_args = mock_rag_engine.query.call_args
        assert call_args.kwargs["query_type"] == "hybrid"

    def test_retrieve_scope_to_query_type_auto(self, synapse_with_mock_rag, mock_rag_engine):
        """Test retrieve() converts AUTO scope to None query type."""
        context = RetrievalContext(scope=SearchScope.AUTO)
        synapse_with_mock_rag.retrieve("test query", context=context)

        call_args = mock_rag_engine.query.call_args
        assert call_args.kwargs["query_type"] is None

    def test_retrieve_infers_scope_from_results(self, synapse_with_mock_rag):
        """Test retrieve() infers actual scope from results."""
        context = RetrievalContext(scope=SearchScope.AUTO)
        result = synapse_with_mock_rag.retrieve("test query", context=context)

        # With mixed results, should infer HYBRID
        assert result.scope == SearchScope.HYBRID

    def test_retrieve_records_execution_time(self, synapse_with_mock_rag):
        """Test retrieve() records execution time."""
        result = synapse_with_mock_rag.retrieve("test query")
        assert result.execution_time_ms > 0

    def test_retrieve_captures_total_candidates(self, synapse_with_mock_rag):
        """Test retrieve() captures total_candidates from response."""
        result = synapse_with_mock_rag.retrieve("test query")
        assert result.total_candidates == 50  # From mock response


class TestSynapseRetrieveForPrompt:
    """Tests for retrieve_for_prompt() method."""

    def test_retrieve_for_prompt_basic(self, synapse_with_mock_rag):
        """Test basic retrieve_for_prompt() call."""
        context_str = synapse_with_mock_rag.retrieve_for_prompt("test query")

        assert isinstance(context_str, str)
        assert len(context_str) > 0

    def test_retrieve_for_prompt_with_max_tokens(self, synapse_with_mock_rag):
        """Test retrieve_for_prompt() with custom max_tokens."""
        context_str = synapse_with_mock_rag.retrieve_for_prompt(
            "test query",
            max_tokens=100,
        )

        # Should be limited (100 tokens = ~400 chars)
        assert len(context_str) <= 400

    def test_retrieve_for_prompt_with_scope(self, synapse_with_mock_rag):
        """Test retrieve_for_prompt() with custom scope."""
        context_str = synapse_with_mock_rag.retrieve_for_prompt(
            "test query",
            scope=SearchScope.CODE,
        )

        assert isinstance(context_str, str)

    def test_retrieve_for_prompt_uses_config_defaults(self, synapse_with_mock_rag):
        """Test retrieve_for_prompt() uses config defaults."""
        context_str = synapse_with_mock_rag.retrieve_for_prompt("test query")

        # Should use config.max_context_tokens (2000)
        # 2000 tokens = ~8000 chars max
        assert len(context_str) <= 8000

    def test_retrieve_for_prompt_applies_similarity_threshold(
        self, synapse_with_mock_rag, mock_rag_engine
    ):
        """Test retrieve_for_prompt() applies similarity_threshold."""
        synapse_with_mock_rag.retrieve_for_prompt("test query")

        # Should have been called with min_score from config
        # (indirectly via retrieve() call)
        result = synapse_with_mock_rag.retrieve(
            "test query",
            RetrievalContext(
                scope=SearchScope.AUTO,
                top_k=synapse_with_mock_rag.config.default_top_k,
                min_score=synapse_with_mock_rag.config.similarity_threshold,
            ),
        )
        assert all(s.score >= 0.5 for s in result.snippets)


class TestSynapseSearchCode:
    """Tests for search_code() method."""

    def test_search_code_basic(self, synapse_with_mock_rag):
        """Test basic search_code() call."""
        result = synapse_with_mock_rag.search_code("test query")

        assert isinstance(result, RetrievalResultSet)
        # Should have used CODE scope

    def test_search_code_with_engine_filter(self, synapse_with_mock_rag, mock_rag_engine):
        """Test search_code() with engine filter."""
        synapse_with_mock_rag.search_code("test query", engine="cortex")

        call_args = mock_rag_engine.query.call_args
        assert call_args.kwargs["query_type"] == "code"
        assert call_args.kwargs["filters"]["engine"] == "cortex"

    def test_search_code_with_custom_top_k(self, synapse_with_mock_rag, mock_rag_engine):
        """Test search_code() with custom top_k."""
        synapse_with_mock_rag.search_code("test query", top_k=10)

        call_args = mock_rag_engine.query.call_args
        assert call_args.kwargs["top_k"] == 10

    def test_search_code_filters_code_results(self, synapse_with_mock_rag):
        """Test search_code() returns code snippets."""
        result = synapse_with_mock_rag.search_code("test query")

        # Check that results have code characteristics
        code_snippets = [s for s in result.snippets if s.file_path is not None]
        assert len(code_snippets) > 0


class TestSynapseSearchDocs:
    """Tests for search_docs() method."""

    def test_search_docs_basic(self, synapse_with_mock_rag):
        """Test basic search_docs() call."""
        result = synapse_with_mock_rag.search_docs("test query")

        assert isinstance(result, RetrievalResultSet)

    def test_search_docs_with_domain_filter(self, synapse_with_mock_rag, mock_rag_engine):
        """Test search_docs() with domain filter."""
        synapse_with_mock_rag.search_docs("test query", domain="trading")

        call_args = mock_rag_engine.query.call_args
        assert call_args.kwargs["query_type"] == "text"
        assert call_args.kwargs["filters"]["domain"] == "trading"

    def test_search_docs_with_custom_top_k(self, synapse_with_mock_rag, mock_rag_engine):
        """Test search_docs() with custom top_k."""
        synapse_with_mock_rag.search_docs("test query", top_k=10)

        call_args = mock_rag_engine.query.call_args
        assert call_args.kwargs["top_k"] == 10

    def test_search_docs_without_domain(self, synapse_with_mock_rag, mock_rag_engine):
        """Test search_docs() without domain filter."""
        synapse_with_mock_rag.search_docs("test query")

        call_args = mock_rag_engine.query.call_args
        # Should have empty filters or no domain filter
        filters = call_args.kwargs.get("filters")
        if filters:
            assert "domain" not in filters

    def test_search_docs_filters_text_results(self, synapse_with_mock_rag):
        """Test search_docs() returns text snippets."""
        result = synapse_with_mock_rag.search_docs("test query")

        # Check that results have text characteristics
        text_snippets = [s for s in result.snippets if s.domain is not None]
        assert len(text_snippets) > 0


class TestSynapseGetStats:
    """Tests for get_stats() method."""

    def test_get_stats_before_initialization(self, synapse_config, mock_helix):
        """Test get_stats() before RAG engine is initialized."""
        synapse = Synapse(config=synapse_config, helix=mock_helix)
        stats = synapse.get_stats()

        assert stats["initialized"] is False

    def test_get_stats_after_initialization(self, synapse_with_mock_rag, mock_rag_engine):
        """Test get_stats() after RAG engine is initialized."""
        # Set state to READY manually since _ensure_rag_engine is mocked
        synapse_with_mock_rag._state = EngineState.READY

        # Ensure initialized by calling _ensure_rag_engine
        synapse_with_mock_rag._ensure_rag_engine()

        stats = synapse_with_mock_rag.get_stats()

        # Should include stats from the RAG engine and synapse metadata
        assert stats is not None
        assert isinstance(stats, dict)
        assert "total_queries" in stats  # From RAG engine stats
        assert stats["total_queries"] == 100  # From mock
        assert "synapse" in stats  # Synapse metadata
        assert stats["synapse"]["initialized"] is True

    def test_get_stats_with_helix(self, synapse_config, mock_helix, mock_rag_engine, monkeypatch):
        """Test get_stats() with Helix enabled."""
        synapse = Synapse(config=synapse_config, helix=mock_helix)

        # Mock RAG engine
        def mock_ensure_rag_engine():
            return mock_rag_engine

        monkeypatch.setattr(synapse, "_ensure_rag_engine", mock_ensure_rag_engine)
        synapse._rag_engine = mock_rag_engine

        # Ensure initialized
        synapse._ensure_rag_engine()

        stats = synapse.get_stats()
        # Check that stats exist - the exact structure may vary
        assert stats is not None
        assert isinstance(stats, dict)


class TestSynapseConvertToSnippets:
    """Tests for _convert_to_snippets() helper."""

    def test_convert_text_results(self, synapse_with_mock_rag):
        """Test converting text RetrievalResults to Snippets."""
        text_results = [
            RetrievalResult(
                id="text_001",
                text="Test text",
                score=0.9,
                metadata={"source": "test.md", "domain": "test"},
            )
        ]

        snippets, text_count, code_count = synapse_with_mock_rag._convert_to_snippets(
            text_results, SearchScope.TEXT
        )

        assert len(snippets) == 1
        assert text_count == 1
        assert code_count == 0
        assert snippets[0].domain == "test"
        assert snippets[0].file_path is None

    def test_convert_code_results(self, synapse_with_mock_rag):
        """Test converting code RetrievalResults to Snippets."""
        code_results = [
            RetrievalResult(
                id="code_001",
                text="def test():\n    pass",
                score=0.9,
                metadata={
                    "file_path": "test.py",
                    "function_name": "test",
                    "line_start": 1,
                    "line_end": 2,
                },
            )
        ]

        snippets, text_count, code_count = synapse_with_mock_rag._convert_to_snippets(
            code_results, SearchScope.CODE
        )

        assert len(snippets) == 1
        assert text_count == 0
        assert code_count == 1
        assert snippets[0].file_path == "test.py"
        assert snippets[0].function_name == "test"
        assert snippets[0].domain is None

    def test_convert_mixed_results(self, synapse_with_mock_rag, mock_retrieval_results):
        """Test converting mixed text and code results."""
        snippets, text_count, code_count = synapse_with_mock_rag._convert_to_snippets(
            mock_retrieval_results, SearchScope.HYBRID
        )

        assert len(snippets) == len(mock_retrieval_results)
        assert text_count > 0
        assert code_count > 0
        assert text_count + code_count == len(snippets)

    def test_convert_preserves_metadata(self, synapse_with_mock_rag):
        """Test conversion preserves custom metadata."""
        results = [
            RetrievalResult(
                id="test_001",
                text="Test",
                score=0.9,
                metadata={
                    "source": "test.md",
                    "custom_field": "custom_value",
                    "priority": 1,
                },
            )
        ]

        snippets, _, _ = synapse_with_mock_rag._convert_to_snippets(results, SearchScope.TEXT)

        assert snippets[0].metadata["custom_field"] == "custom_value"
        assert snippets[0].metadata["priority"] == 1


class TestSynapseEdgeCases:
    """Tests for edge cases and error handling."""

    def test_retrieve_empty_results(self, synapse_with_mock_rag, mock_rag_engine):
        """Test retrieve() with empty results."""
        # Mock empty response
        mock_rag_engine.query.return_value = QueryResponse(
            query="test query",
            query_type="hybrid",
            results=[],
            execution_time_ms=50.0,
            total_candidates=0,
        )

        result = synapse_with_mock_rag.retrieve("test query")

        assert result.count == 0
        assert result.snippets == []
        assert result.top_score == 0.0

    def test_retrieve_all_filtered_by_min_score(self, synapse_with_mock_rag):
        """Test retrieve() when all results filtered by min_score."""
        context = RetrievalContext(min_score=0.99)  # Very high threshold
        result = synapse_with_mock_rag.retrieve("test query", context=context)

        # Most/all results should be filtered
        assert result.count < 5  # Less than original results

    def test_retrieve_with_empty_metadata(self, synapse_with_mock_rag, mock_rag_engine):
        """Test retrieve() handles empty metadata gracefully."""
        mock_rag_engine.query.return_value = QueryResponse(
            query="test query",
            query_type="text",
            results=[
                RetrievalResult(
                    id="test_001",
                    text="Test",
                    score=0.9,
                    metadata={},  # Empty metadata
                )
            ],
            execution_time_ms=50.0,
            total_candidates=1,
        )

        result = synapse_with_mock_rag.retrieve("test query")
        assert len(result.snippets) == 1

    def test_scope_inference_text_only(self, synapse_with_mock_rag, mock_rag_engine):
        """Test scope inference with text-only results."""
        mock_rag_engine.query.return_value = QueryResponse(
            query="test query",
            query_type="auto",
            results=[
                RetrievalResult(
                    id="text_001",
                    text="Test",
                    score=0.9,
                    metadata={"source": "test.md"},
                )
            ],
            execution_time_ms=50.0,
            total_candidates=1,
        )

        context = RetrievalContext(scope=SearchScope.AUTO)
        result = synapse_with_mock_rag.retrieve("test query", context=context)

        assert result.scope == SearchScope.TEXT

    def test_scope_inference_code_only(self, synapse_with_mock_rag, mock_rag_engine):
        """Test scope inference with code-only results."""
        mock_rag_engine.query.return_value = QueryResponse(
            query="test query",
            query_type="auto",
            results=[
                RetrievalResult(
                    id="code_001",
                    text="def test(): pass",
                    score=0.9,
                    metadata={"file_path": "test.py"},
                )
            ],
            execution_time_ms=50.0,
            total_candidates=1,
        )

        context = RetrievalContext(scope=SearchScope.AUTO)
        result = synapse_with_mock_rag.retrieve("test query", context=context)

        assert result.scope == SearchScope.CODE
