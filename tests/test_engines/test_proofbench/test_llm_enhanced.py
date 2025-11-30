"""
Tests for LLM-enhanced ProofBench analytics.

Tests cover:
- Performance narration
- Result comparison
- Optimization suggestions
- Metric explanations
- Trade pattern analysis
"""

from datetime import datetime

import pandas as pd
import pytest

from src.engines.proofbench import (
    LLMPerformanceNarrator,
    PerformanceMetrics,
    SimulationConfig,
    SimulationResults,
)
from src.engines.proofbench.core.portfolio import Portfolio


@pytest.fixture
def mock_metrics():
    """Create mock performance metrics."""
    return PerformanceMetrics(
        total_return=0.25,
        annualized_return=0.18,
        volatility=0.15,
        downside_deviation=0.10,
        sharpe_ratio=1.2,
        sortino_ratio=1.8,
        calmar_ratio=1.5,
        max_drawdown=-0.12,
        avg_drawdown=-0.05,
        max_drawdown_duration=30.0,
        num_trades=50,
        win_rate=0.55,
        profit_factor=1.8,
        avg_win=500.0,
        avg_loss=-300.0,
        largest_win=2000.0,
        largest_loss=-800.0,
        avg_trade_duration=5.0,
        expectancy=110.0,
        recovery_factor=2.1,
        equity_final=125000.0,
    )


@pytest.fixture
def mock_results(mock_metrics):
    """Create mock simulation results."""
    config = SimulationConfig(initial_capital=100000.0)
    portfolio = Portfolio(100000.0)

    # Create equity curve
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    equity_curve = pd.DataFrame(
        {
            "timestamp": dates,
            "equity": [100000.0 + i * 100 for i in range(len(dates))],
        }
    )

    # Create trades DataFrame
    trades = pd.DataFrame(
        {
            "entry_time": dates[:50],
            "exit_time": dates[1:51],
            "symbol": ["AAPL"] * 50,
            "pnl": [100.0] * 50,
        }
    )

    return SimulationResults(
        config=config,
        metrics=mock_metrics,
        portfolio=portfolio,
        equity_curve=equity_curve,
        trades=trades,
        orders=[],
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 12, 31),
    )


@pytest.mark.unit
def test_narrator_creation():
    """Test creating performance narrator."""
    narrator = LLMPerformanceNarrator(nvidia_api_key=None)

    assert narrator.nvidia_api_key is None
    assert narrator._llm_client is None


@pytest.mark.unit
def test_narrator_creation_with_api_key():
    """Test creating narrator with API key."""
    narrator = LLMPerformanceNarrator(nvidia_api_key="test-key")

    assert narrator.nvidia_api_key == "test-key"
    assert narrator._llm_client is None


@pytest.mark.unit
def test_narrate_results_no_api(mock_results):
    """Test narrating results without API key (fallback)."""
    narrator = LLMPerformanceNarrator(nvidia_api_key=None)

    narration = narrator.narrate_results(mock_results)

    # Should return basic narration
    assert "narration" in narration
    assert "llm_model" in narration
    assert narration["llm_model"] == "rule-based"
    assert "metrics_summary" in narration
    assert narration["metrics_summary"]["total_return"] == 0.25


@pytest.mark.unit
def test_narrate_results_with_api(mock_results):
    """Test narrating results with API key (will use fallback)."""
    narrator = LLMPerformanceNarrator(nvidia_api_key="test-key")

    narration = narrator.narrate_results(mock_results)

    # Should return narration (fallback since no real API)
    assert "narration" in narration
    assert isinstance(narration["narration"], str)
    assert "metrics_summary" in narration


@pytest.mark.unit
def test_compare_results_no_api(mock_results):
    """Test comparing results without API key."""
    narrator = LLMPerformanceNarrator(nvidia_api_key=None)

    results_list = [
        ("Strategy A", mock_results),
        ("Strategy B", mock_results),
    ]

    comparison = narrator.compare_results(results_list)

    # Should return basic comparison
    assert "comparison" in comparison
    assert "llm_model" in comparison
    assert comparison["llm_model"] == "rule-based"
    assert comparison["strategies_compared"] == 2


@pytest.mark.unit
def test_compare_results_with_api(mock_results):
    """Test comparing results with API key (will use fallback)."""
    narrator = LLMPerformanceNarrator(nvidia_api_key="test-key")

    results_list = [
        ("Strategy A", mock_results),
        ("Strategy B", mock_results),
        ("Strategy C", mock_results),
    ]

    comparison = narrator.compare_results(results_list)

    # Should return comparison (fallback since no real API)
    assert "comparison" in comparison
    assert comparison["strategies_compared"] == 3
    assert isinstance(comparison["comparison"], str)


@pytest.mark.unit
def test_suggest_optimizations_no_api(mock_results):
    """Test optimization suggestions without API key."""
    narrator = LLMPerformanceNarrator(nvidia_api_key=None)

    suggestions = narrator.suggest_optimizations(mock_results, focus="returns")

    # Should return basic suggestions
    assert len(suggestions) > 0
    assert isinstance(suggestions, list)
    assert all(isinstance(s, str) for s in suggestions)


@pytest.mark.unit
def test_suggest_optimizations_with_api(mock_results):
    """Test optimization suggestions with API key (will use fallback)."""
    narrator = LLMPerformanceNarrator(nvidia_api_key="test-key")

    suggestions = narrator.suggest_optimizations(mock_results, focus="risk")

    # Should return suggestions (fallback since no real API)
    assert len(suggestions) > 0
    assert isinstance(suggestions, list)


@pytest.mark.unit
def test_suggest_optimizations_different_focus(mock_results):
    """Test suggestions with different focus areas."""
    narrator = LLMPerformanceNarrator(nvidia_api_key=None)

    for focus in ["returns", "risk", "consistency", "general"]:
        suggestions = narrator.suggest_optimizations(mock_results, focus=focus)
        assert len(suggestions) > 0
        assert isinstance(suggestions, list)


@pytest.mark.unit
def test_explain_metric_no_api():
    """Test metric explanation without API key."""
    narrator = LLMPerformanceNarrator(nvidia_api_key=None)

    explanation = narrator.explain_metric("Sharpe Ratio", 1.5)

    assert "Sharpe Ratio" in explanation
    assert isinstance(explanation, str)


@pytest.mark.unit
def test_explain_metric_with_api():
    """Test metric explanation with API key (will use fallback)."""
    narrator = LLMPerformanceNarrator(nvidia_api_key="test-key")

    explanation = narrator.explain_metric("Sortino Ratio", 2.0)

    assert isinstance(explanation, str)
    assert len(explanation) > 0


@pytest.mark.unit
def test_analyze_trade_patterns_no_api(mock_results):
    """Test trade pattern analysis without API key."""
    narrator = LLMPerformanceNarrator(nvidia_api_key=None)

    analysis = narrator.analyze_trade_patterns(mock_results)

    # Should return basic analysis
    assert "analysis" in analysis
    assert "llm_model" in analysis
    assert analysis["llm_model"] == "rule-based"


@pytest.mark.unit
def test_analyze_trade_patterns_with_api(mock_results):
    """Test trade pattern analysis with API key (will use fallback)."""
    narrator = LLMPerformanceNarrator(nvidia_api_key="test-key")

    analysis = narrator.analyze_trade_patterns(mock_results)

    # Should return analysis (fallback since no real API)
    assert "analysis" in analysis
    assert isinstance(analysis["analysis"], str)


@pytest.mark.unit
def test_narration_includes_key_metrics(mock_results):
    """Test that narration includes key metrics."""
    narrator = LLMPerformanceNarrator(nvidia_api_key=None)

    narration = narrator.narrate_results(mock_results)

    # Check metrics summary includes all key metrics
    summary = narration["metrics_summary"]
    assert "total_return" in summary
    assert "sharpe_ratio" in summary
    assert "max_drawdown" in summary
    assert "win_rate" in summary


@pytest.mark.unit
def test_comparison_handles_single_strategy(mock_results):
    """Test comparison with single strategy."""
    narrator = LLMPerformanceNarrator(nvidia_api_key=None)

    results_list = [("Strategy A", mock_results)]

    comparison = narrator.compare_results(results_list)

    assert comparison["strategies_compared"] == 1
    assert isinstance(comparison["comparison"], str)


@pytest.mark.unit
def test_narrator_timestamp_included(mock_results):
    """Test that all outputs include timestamp."""
    narrator = LLMPerformanceNarrator(nvidia_api_key=None)

    narration = narrator.narrate_results(mock_results)
    comparison = narrator.compare_results([("A", mock_results)])
    pattern_analysis = narrator.analyze_trade_patterns(mock_results)

    assert "timestamp" in narration
    assert "timestamp" in comparison
    assert "timestamp" in pattern_analysis

    # Timestamps should be valid ISO format
    datetime.fromisoformat(narration["timestamp"])
    datetime.fromisoformat(comparison["timestamp"])
    datetime.fromisoformat(pattern_analysis["timestamp"])


@pytest.mark.unit
def test_basic_suggestions_coverage():
    """Test that basic suggestions cover all focus areas."""
    narrator = LLMPerformanceNarrator(nvidia_api_key=None)

    # Test all focus areas return non-empty suggestions
    for focus in ["returns", "risk", "consistency", "general"]:
        suggestions = narrator._get_basic_suggestions(focus)
        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)


@pytest.mark.unit
def test_narration_with_different_metrics():
    """Test narration with various metric values."""
    narrator = LLMPerformanceNarrator(nvidia_api_key=None)

    # Create results with different performance characteristics
    metrics_scenarios = [
        # Good performance
        PerformanceMetrics(
            total_return=0.50,
            annualized_return=0.35,
            volatility=0.12,
            downside_deviation=0.08,
            sharpe_ratio=2.5,
            sortino_ratio=3.0,
            calmar_ratio=3.5,
            max_drawdown=-0.08,
            avg_drawdown=-0.03,
            max_drawdown_duration=15.0,
            num_trades=100,
            win_rate=0.65,
            profit_factor=2.5,
            avg_win=800.0,
            avg_loss=-300.0,
            largest_win=3000.0,
            largest_loss=-500.0,
            avg_trade_duration=3.0,
            expectancy=250.0,
            recovery_factor=6.0,
            equity_final=150000.0,
        ),
        # Poor performance
        PerformanceMetrics(
            total_return=-0.15,
            annualized_return=-0.08,
            volatility=0.25,
            downside_deviation=0.20,
            sharpe_ratio=-0.5,
            sortino_ratio=-0.3,
            calmar_ratio=-0.8,
            max_drawdown=-0.25,
            avg_drawdown=-0.12,
            max_drawdown_duration=90.0,
            num_trades=30,
            win_rate=0.35,
            profit_factor=0.7,
            avg_win=400.0,
            avg_loss=-600.0,
            largest_win=1000.0,
            largest_loss=-2000.0,
            avg_trade_duration=12.0,
            expectancy=-100.0,
            recovery_factor=0.6,
            equity_final=85000.0,
        ),
    ]

    for metrics in metrics_scenarios:
        config = SimulationConfig(initial_capital=100000.0)
        portfolio = Portfolio(100000.0)
        equity_curve = pd.DataFrame(
            {"timestamp": [datetime.now()], "equity": [metrics.equity_final]}
        )
        trades = pd.DataFrame({"pnl": [0.0]})

        results = SimulationResults(
            config=config,
            metrics=metrics,
            portfolio=portfolio,
            equity_curve=equity_curve,
            trades=trades,
            orders=[],
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        narration = narrator.narrate_results(results)

        assert "narration" in narration
        assert "metrics_summary" in narration
        assert narration["metrics_summary"]["total_return"] == metrics.total_return
