"""Tests for Synapse data models."""

from datetime import datetime

from ordinis.ai.synapse import (
    RetrievalContext,
    RetrievalResultSet,
    SearchScope,
    Snippet,
)


class TestSearchScope:
    """Tests for SearchScope enum."""

    def test_search_scope_values(self):
        """Test SearchScope enum values."""
        assert SearchScope.TEXT.value == "text"
        assert SearchScope.CODE.value == "code"
        assert SearchScope.HYBRID.value == "hybrid"
        assert SearchScope.AUTO.value == "auto"

    def test_search_scope_membership(self):
        """Test SearchScope enum membership."""
        assert SearchScope.TEXT in SearchScope
        assert SearchScope.CODE in SearchScope
        assert SearchScope.HYBRID in SearchScope
        assert SearchScope.AUTO in SearchScope

    def test_search_scope_iteration(self):
        """Test SearchScope enum iteration."""
        scopes = list(SearchScope)
        assert len(scopes) == 4
        assert SearchScope.TEXT in scopes
        assert SearchScope.CODE in scopes
        assert SearchScope.HYBRID in scopes
        assert SearchScope.AUTO in scopes


class TestRetrievalContext:
    """Tests for RetrievalContext dataclass."""

    def test_default_values(self):
        """Test RetrievalContext default values."""
        ctx = RetrievalContext()
        assert ctx.scope == SearchScope.AUTO
        assert ctx.domain is None
        assert ctx.engine is None
        assert ctx.filters == {}
        assert ctx.min_score == 0.0
        assert ctx.top_k == 5
        assert ctx.include_metadata is True
        assert ctx.source_file is None
        assert ctx.source_function is None

    def test_custom_scope(self):
        """Test RetrievalContext with custom scope."""
        ctx = RetrievalContext(scope=SearchScope.CODE)
        assert ctx.scope == SearchScope.CODE

    def test_custom_filters(self):
        """Test RetrievalContext with custom filters."""
        filters = {"engine": "riskguard", "domain": "risk"}
        ctx = RetrievalContext(filters=filters)
        assert ctx.filters == filters

    def test_custom_top_k(self):
        """Test RetrievalContext with custom top_k."""
        ctx = RetrievalContext(top_k=10)
        assert ctx.top_k == 10

    def test_custom_min_score(self):
        """Test RetrievalContext with custom min_score."""
        ctx = RetrievalContext(min_score=0.7)
        assert ctx.min_score == 0.7

    def test_engine_filter(self):
        """Test RetrievalContext with engine filter."""
        ctx = RetrievalContext(engine="cortex")
        assert ctx.engine == "cortex"

    def test_domain_filter(self):
        """Test RetrievalContext with domain filter."""
        ctx = RetrievalContext(domain="portfolio")
        assert ctx.domain == "portfolio"

    def test_source_context(self):
        """Test RetrievalContext with source context."""
        ctx = RetrievalContext(
            source_file="src/ordinis/engines/cortex/core/engine.py",
            source_function="execute_analysis",
        )
        assert ctx.source_file == "src/ordinis/engines/cortex/core/engine.py"
        assert ctx.source_function == "execute_analysis"


class TestSnippet:
    """Tests for Snippet dataclass."""

    def test_text_snippet_creation(self):
        """Test creating a text snippet."""
        snippet = Snippet(
            id="text_001",
            text="Portfolio rebalancing maintains target allocations.",
            score=0.85,
            source="docs/portfolio.md",
            domain="portfolio",
            section="Overview",
        )
        assert snippet.id == "text_001"
        assert "Portfolio rebalancing" in snippet.text
        assert snippet.score == 0.85
        assert snippet.source == "docs/portfolio.md"
        assert snippet.domain == "portfolio"
        assert snippet.section == "Overview"
        assert snippet.file_path is None
        assert snippet.function_name is None

    def test_code_snippet_creation(self):
        """Test creating a code snippet."""
        snippet = Snippet(
            id="code_001",
            text="def evaluate_position(self, position):\n    pass",
            score=0.92,
            source="src/ordinis/engines/riskguard/core/engine.py",
            file_path="src/ordinis/engines/riskguard/core/engine.py",
            function_name="evaluate_position",
            class_name="RiskGuard",
            line_start=42,
            line_end=45,
        )
        assert snippet.id == "code_001"
        assert "def evaluate_position" in snippet.text
        assert snippet.score == 0.92
        assert snippet.file_path == "src/ordinis/engines/riskguard/core/engine.py"
        assert snippet.function_name == "evaluate_position"
        assert snippet.class_name == "RiskGuard"
        assert snippet.line_start == 42
        assert snippet.line_end == 45
        assert snippet.domain is None

    def test_snippet_with_metadata(self):
        """Test snippet with custom metadata."""
        metadata = {"custom_field": "value", "priority": 1}
        snippet = Snippet(
            id="snippet_001",
            text="Test snippet",
            score=0.8,
            source="test.md",
            metadata=metadata,
        )
        assert snippet.metadata == metadata
        assert snippet.metadata["custom_field"] == "value"
        assert snippet.metadata["priority"] == 1

    def test_snippet_str_with_file_path(self):
        """Test Snippet __str__ with file_path."""
        snippet = Snippet(
            id="code_001",
            text="This is a long text snippet that should be truncated when displayed in the string representation",
            score=0.85,
            source="src/test.py",
            file_path="src/test.py",
        )
        result = str(snippet)
        assert "[0.85]" in result
        assert "src/test.py" in result
        assert "..." in result  # Should be truncated

    def test_snippet_str_with_domain(self):
        """Test Snippet __str__ with domain."""
        snippet = Snippet(
            id="text_001",
            text="Short text",
            score=0.90,
            source="docs/test.md",
            domain="trading",
        )
        result = str(snippet)
        assert "[0.90]" in result
        assert "trading" in result

    def test_snippet_str_unknown_location(self):
        """Test Snippet __str__ with no location info."""
        snippet = Snippet(
            id="snippet_001",
            text="Test snippet",
            score=0.75,
            source="unknown",
        )
        result = str(snippet)
        assert "[0.75]" in result
        assert "unknown" in result


class TestRetrievalResultSet:
    """Tests for RetrievalResultSet dataclass."""

    def test_result_set_creation(self, code_snippets, text_snippets):
        """Test creating a RetrievalResultSet."""
        all_snippets = code_snippets + text_snippets
        result_set = RetrievalResultSet(
            query="test query",
            snippets=all_snippets,
            scope=SearchScope.HYBRID,
            execution_time_ms=125.5,
            total_candidates=50,
            text_results=2,
            code_results=2,
        )
        assert result_set.query == "test query"
        assert len(result_set.snippets) == 4
        assert result_set.scope == SearchScope.HYBRID
        assert result_set.execution_time_ms == 125.5
        assert result_set.total_candidates == 50
        assert result_set.text_results == 2
        assert result_set.code_results == 2

    def test_result_set_count_property(self, code_snippets):
        """Test RetrievalResultSet count property."""
        result_set = RetrievalResultSet(
            query="test query",
            snippets=code_snippets,
            scope=SearchScope.CODE,
            execution_time_ms=100.0,
            total_candidates=20,
        )
        assert result_set.count == len(code_snippets)
        assert result_set.count == 2

    def test_result_set_top_score_property(self, code_snippets):
        """Test RetrievalResultSet top_score property."""
        result_set = RetrievalResultSet(
            query="test query",
            snippets=code_snippets,
            scope=SearchScope.CODE,
            execution_time_ms=100.0,
            total_candidates=20,
        )
        assert result_set.top_score == 0.92  # Highest score from code_snippets

    def test_result_set_top_score_empty(self):
        """Test RetrievalResultSet top_score with no snippets."""
        result_set = RetrievalResultSet(
            query="test query",
            snippets=[],
            scope=SearchScope.CODE,
            execution_time_ms=50.0,
            total_candidates=0,
        )
        assert result_set.top_score == 0.0

    def test_get_context_string_basic(self, code_snippets):
        """Test get_context_string with basic snippets."""
        result_set = RetrievalResultSet(
            query="test query",
            snippets=code_snippets,
            scope=SearchScope.CODE,
            execution_time_ms=100.0,
            total_candidates=20,
        )
        context = result_set.get_context_string()
        assert "[1]" in context
        assert "[2]" in context
        assert code_snippets[0].source in context
        assert code_snippets[1].source in context

    def test_get_context_string_with_max_tokens(self):
        """Test get_context_string respects max_tokens."""
        # Create snippet with known text length
        snippet = Snippet(
            id="test_001",
            text="A" * 100,  # 100 characters
            score=0.9,
            source="test.py",
        )
        result_set = RetrievalResultSet(
            query="test query",
            snippets=[snippet],
            scope=SearchScope.CODE,
            execution_time_ms=50.0,
            total_candidates=1,
        )

        # Max 10 tokens = 40 chars, should not include full snippet
        context = result_set.get_context_string(max_tokens=10)
        assert len(context) <= 40

    def test_get_context_string_truncates_long_results(self, code_snippets, text_snippets):
        """Test get_context_string truncates when exceeding max_tokens."""
        # Create many snippets
        many_snippets = code_snippets + text_snippets
        for i in range(10):
            many_snippets.append(
                Snippet(
                    id=f"extra_{i}",
                    text="Extra snippet " * 50,  # Long text
                    score=0.7,
                    source=f"extra_{i}.py",
                )
            )

        result_set = RetrievalResultSet(
            query="test query",
            snippets=many_snippets,
            scope=SearchScope.HYBRID,
            execution_time_ms=150.0,
            total_candidates=50,
        )

        context = result_set.get_context_string(max_tokens=100)  # Very small limit
        # Should have truncated, not all snippets included
        assert context.count("[") < len(many_snippets)

    def test_get_context_string_empty_snippets(self):
        """Test get_context_string with no snippets."""
        result_set = RetrievalResultSet(
            query="test query",
            snippets=[],
            scope=SearchScope.AUTO,
            execution_time_ms=50.0,
            total_candidates=0,
        )
        context = result_set.get_context_string()
        assert context == ""

    def test_to_citations_basic(self, code_snippets):
        """Test to_citations conversion."""
        result_set = RetrievalResultSet(
            query="test query",
            snippets=code_snippets,
            scope=SearchScope.CODE,
            execution_time_ms=100.0,
            total_candidates=20,
        )
        citations = result_set.to_citations()
        assert len(citations) == 2
        assert citations[0]["id"] == code_snippets[0].id
        assert citations[0]["source"] == code_snippets[0].source
        assert citations[0]["score"] == code_snippets[0].score
        assert citations[0]["file"] == code_snippets[0].file_path
        assert citations[0]["lines"] == "45-47"

    def test_to_citations_with_text_snippets(self, text_snippets):
        """Test to_citations with text snippets (no line numbers)."""
        result_set = RetrievalResultSet(
            query="test query",
            snippets=text_snippets,
            scope=SearchScope.TEXT,
            execution_time_ms=100.0,
            total_candidates=10,
        )
        citations = result_set.to_citations()
        assert len(citations) == 2
        assert citations[0]["id"] == text_snippets[0].id
        assert citations[0]["lines"] is None  # No line numbers for text
        assert citations[0]["file"] is None  # No file_path for text

    def test_to_citations_empty(self):
        """Test to_citations with no snippets."""
        result_set = RetrievalResultSet(
            query="test query",
            snippets=[],
            scope=SearchScope.AUTO,
            execution_time_ms=50.0,
            total_candidates=0,
        )
        citations = result_set.to_citations()
        assert citations == []

    def test_timestamp_field(self):
        """Test that timestamp is auto-generated."""
        before = datetime.now()
        result_set = RetrievalResultSet(
            query="test query",
            snippets=[],
            scope=SearchScope.AUTO,
            execution_time_ms=50.0,
            total_candidates=0,
        )
        after = datetime.now()
        assert before <= result_set.timestamp <= after

    def test_custom_timestamp(self):
        """Test RetrievalResultSet with custom timestamp."""
        custom_time = datetime(2025, 1, 1, 12, 0, 0)
        result_set = RetrievalResultSet(
            query="test query",
            snippets=[],
            scope=SearchScope.AUTO,
            execution_time_ms=50.0,
            total_candidates=0,
            timestamp=custom_time,
        )
        assert result_set.timestamp == custom_time
