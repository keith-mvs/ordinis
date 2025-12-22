"""
Tests for query classifier module.

Tests cover:
- QueryType enum
- classify_query function
- Keyword matching
"""

import pytest

from ordinis.rag.retrieval.query_classifier import (
    CODE_KEYWORDS,
    HYBRID_KEYWORDS,
    QueryType,
    classify_query,
)


class TestQueryType:
    """Tests for QueryType enum."""

    @pytest.mark.unit
    def test_query_type_values(self):
        """Test QueryType has expected values."""
        assert QueryType.TEXT == "text"
        assert QueryType.CODE == "code"
        assert QueryType.HYBRID == "hybrid"

    @pytest.mark.unit
    def test_query_type_is_str_enum(self):
        """Test QueryType is a string enum."""
        assert isinstance(QueryType.TEXT.value, str)
        assert QueryType.TEXT.value == "text"


class TestClassifyQuery:
    """Tests for classify_query function."""

    @pytest.mark.unit
    def test_classify_text_query(self):
        """Test classification of text query."""
        result = classify_query("What is the trading strategy?")

        assert result == QueryType.TEXT

    @pytest.mark.unit
    def test_classify_code_query_implement(self):
        """Test classification of code query with 'implement'."""
        result = classify_query("implement a new trading function")

        assert result == QueryType.CODE

    @pytest.mark.unit
    def test_classify_code_query_function(self):
        """Test classification of code query with 'function'."""
        result = classify_query("show me the function for risk calculation")

        assert result == QueryType.CODE

    @pytest.mark.unit
    def test_classify_code_query_class(self):
        """Test classification of code query with 'class'."""
        result = classify_query("what does this class do?")

        assert result == QueryType.CODE

    @pytest.mark.unit
    def test_classify_code_query_show_me(self):
        """Test classification of code query with 'show me'."""
        result = classify_query("show me the code")

        assert result == QueryType.CODE

    @pytest.mark.unit
    def test_classify_hybrid_query_architecture(self):
        """Test classification of hybrid query with 'architecture'."""
        result = classify_query("explain the architecture of the system")

        assert result == QueryType.HYBRID

    @pytest.mark.unit
    def test_classify_hybrid_query_design(self):
        """Test classification of hybrid query with 'design'."""
        result = classify_query("design a new feature")

        assert result == QueryType.HYBRID

    @pytest.mark.unit
    def test_classify_hybrid_query_how_to(self):
        """Test classification of hybrid query with 'how to'."""
        result = classify_query("how to build a trading bot")

        assert result == QueryType.HYBRID

    @pytest.mark.unit
    def test_classify_hybrid_query_best_practice(self):
        """Test classification of hybrid query with 'best practice'."""
        result = classify_query("best practice for error handling")

        assert result == QueryType.HYBRID

    @pytest.mark.unit
    def test_hybrid_takes_precedence_over_code(self):
        """Test hybrid keywords take precedence over code keywords."""
        # Contains both 'implement' (code) and 'architecture' (hybrid)
        result = classify_query("implement the architecture pattern")

        assert result == QueryType.HYBRID

    @pytest.mark.unit
    def test_classify_case_insensitive(self):
        """Test classification is case insensitive."""
        result = classify_query("IMPLEMENT A FUNCTION")

        assert result == QueryType.CODE

    @pytest.mark.unit
    def test_classify_code_with_import(self):
        """Test classification with 'import' keyword."""
        result = classify_query("how do I import this module")

        assert result == QueryType.CODE

    @pytest.mark.unit
    def test_classify_code_with_api(self):
        """Test classification with 'api' keyword."""
        result = classify_query("what api does this use")

        assert result == QueryType.CODE


class TestKeywords:
    """Tests for keyword sets."""

    @pytest.mark.unit
    def test_code_keywords_exist(self):
        """Test CODE_KEYWORDS is not empty."""
        assert len(CODE_KEYWORDS) > 0

    @pytest.mark.unit
    def test_hybrid_keywords_exist(self):
        """Test HYBRID_KEYWORDS is not empty."""
        assert len(HYBRID_KEYWORDS) > 0

    @pytest.mark.unit
    def test_code_keywords_lowercase(self):
        """Test all code keywords are lowercase for matching."""
        for keyword in CODE_KEYWORDS:
            assert keyword == keyword.lower() or " " in keyword

    @pytest.mark.unit
    def test_hybrid_keywords_lowercase(self):
        """Test all hybrid keywords are lowercase for matching."""
        for keyword in HYBRID_KEYWORDS:
            assert keyword == keyword.lower() or " " in keyword
