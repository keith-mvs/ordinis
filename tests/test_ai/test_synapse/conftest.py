"""Test fixtures for Synapse RAG wrapper tests."""

import sys
from unittest.mock import MagicMock, Mock

import pytest

# Mock chromadb before imports
sys.modules["chromadb"] = MagicMock()
sys.modules["chromadb.config"] = MagicMock()

from ordinis.ai.synapse import (  # noqa: E402
    RetrievalContext,
    SearchScope,
    Snippet,
    Synapse,
    SynapseConfig,
)
from ordinis.rag.vectordb.schema import QueryResponse, RetrievalResult  # noqa: E402


@pytest.fixture
def synapse_config() -> SynapseConfig:
    """Default test configuration for Synapse."""
    return SynapseConfig(
        default_scope=SearchScope.AUTO,
        default_top_k=5,
        similarity_threshold=0.5,
        top_k_retrieval=20,
        top_k_rerank=10,
        max_context_tokens=2000,
        cache_enabled=False,  # Disable cache for deterministic tests
    )


@pytest.fixture
def mock_retrieval_results() -> list[RetrievalResult]:
    """Mock RAG retrieval results with mixed text and code."""
    return [
        # Code result 1
        RetrievalResult(
            id="code_001",
            text="class RiskGuard:\n    def evaluate_position(self, position):\n        pass",
            score=0.95,
            metadata={
                "file_path": "src/ordinis/engines/riskguard/core/engine.py",
                "function_name": "evaluate_position",
                "class_name": "RiskGuard",
                "engine": "riskguard",
                "line_start": 42,
                "line_end": 45,
            },
        ),
        # Text result 1
        RetrievalResult(
            id="text_001",
            text="RiskGuard validates positions against risk rules before execution.",
            score=0.88,
            metadata={
                "source": "docs/engines/riskguard.md",
                "domain": "risk",
                "section": "Overview",
            },
        ),
        # Code result 2
        RetrievalResult(
            id="code_002",
            text="def check_position_size(position, max_size):\n    return position.size <= max_size",
            score=0.82,
            metadata={
                "file_path": "src/ordinis/engines/riskguard/rules/size.py",
                "function_name": "check_position_size",
                "engine": "riskguard",
                "line_start": 10,
                "line_end": 12,
            },
        ),
        # Text result 2
        RetrievalResult(
            id="text_002",
            text="Position sizing rules prevent excessive exposure to single positions.",
            score=0.75,
            metadata={
                "source": "docs/strategies/risk-management.md",
                "domain": "risk",
                "section": "Position Sizing",
            },
        ),
        # Low score result (below threshold)
        RetrievalResult(
            id="text_003",
            text="Market data is provided by various plugins.",
            score=0.35,
            metadata={
                "source": "docs/plugins/market-data.md",
                "domain": "data",
                "section": "Providers",
            },
        ),
    ]


@pytest.fixture
def mock_query_response(mock_retrieval_results: list[RetrievalResult]) -> QueryResponse:
    """Mock RAG QueryResponse."""
    return QueryResponse(
        query="How does RiskGuard evaluate positions?",
        query_type="hybrid",
        results=mock_retrieval_results,
        execution_time_ms=125.5,
        total_candidates=50,
    )


@pytest.fixture
def mock_rag_engine(mock_query_response: QueryResponse) -> Mock:
    """Mock RAG RetrievalEngine."""
    engine = Mock()
    engine.query.return_value = mock_query_response
    engine.get_stats.return_value = {
        "total_queries": 100,
        "avg_latency_ms": 120.0,
        "text_collection_size": 1000,
        "code_collection_size": 500,
    }
    return engine


@pytest.fixture
def synapse_with_mock_rag(
    synapse_config: SynapseConfig,
    mock_rag_engine: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> Synapse:
    """Synapse instance with mocked RAG engine."""
    synapse = Synapse(config=synapse_config)

    # Mock the _ensure_rag_engine method to return our mock
    def mock_ensure_rag_engine() -> Mock:
        synapse._initialized = True
        return mock_rag_engine

    monkeypatch.setattr(synapse, "_ensure_rag_engine", mock_ensure_rag_engine)
    synapse._rag_engine = mock_rag_engine

    return synapse


@pytest.fixture
def retrieval_context() -> RetrievalContext:
    """Default retrieval context for tests."""
    return RetrievalContext(
        scope=SearchScope.HYBRID,
        top_k=5,
        min_score=0.5,
    )


@pytest.fixture
def code_snippets() -> list[Snippet]:
    """Sample code snippets."""
    return [
        Snippet(
            id="code_001",
            text="def calculate_returns(prices):\n    return prices.pct_change()",
            score=0.92,
            source="src/ordinis/core/metrics.py",
            file_path="src/ordinis/core/metrics.py",
            line_start=45,
            line_end=47,
            function_name="calculate_returns",
        ),
        Snippet(
            id="code_002",
            text="class Portfolio:\n    def rebalance(self):\n        pass",
            score=0.85,
            source="src/ordinis/engines/portfolio/core.py",
            file_path="src/ordinis/engines/portfolio/core.py",
            line_start=100,
            line_end=103,
            function_name="rebalance",
            class_name="Portfolio",
        ),
    ]


@pytest.fixture
def text_snippets() -> list[Snippet]:
    """Sample text snippets."""
    return [
        Snippet(
            id="text_001",
            text="Portfolio rebalancing maintains target asset allocations.",
            score=0.88,
            source="docs/strategies/portfolio.md",
            domain="portfolio",
            section="Overview",
        ),
        Snippet(
            id="text_002",
            text="Rebalancing can be threshold-based or calendar-based.",
            score=0.80,
            source="docs/strategies/rebalancing.md",
            domain="portfolio",
            section="Types",
        ),
    ]


@pytest.fixture
def mock_helix() -> Mock:
    """Mock Helix instance for embedding tests."""
    helix = Mock()
    helix.embed.return_value = [0.1] * 384  # Mock embedding vector
    return helix
