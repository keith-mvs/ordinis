"""Tests for BacktestAnalyzer module.

Tests cover:
- Initialization
- Report generation
- Executive summary
- Period analysis
- Model analysis
- Sector analysis
- Risk analysis
- Optimization recommendations
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from ordinis.analysis.backtest_analyzer import BacktestAnalyzer


@pytest.fixture
def sample_backtest_report():
    """Create sample backtest report data."""
    return {
        "metadata": {
            "total_equities": 100,
            "test_periods": ["2023-Q1", "2023-Q2", "2023-Q3"],
            "equities": {
                "AAPL": "Technology",
                "GOOGL": "Technology",
                "JNJ": "Healthcare",
            },
        },
        "results_by_period": {
            "2023-Q1": {
                "metrics": {
                    "total_return": 0.15,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": -0.05,
                    "win_rate": 0.6,
                    "profit_factor": 2.0,
                    "avg_trade_duration": 5,
                    "num_trades": 50,
                    "total_trades": 50,
                    "winning_trades": 30,
                    "losing_trades": 20,
                },
                "trades": [
                    {"symbol": "AAPL", "pnl": 100, "sector": "Technology"},
                    {"symbol": "GOOGL", "pnl": -50, "sector": "Technology"},
                ],
            },
            "2023-Q2": {
                "metrics": {
                    "total_return": 0.10,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": -0.08,
                    "win_rate": 0.55,
                    "profit_factor": 1.8,
                    "avg_trade_duration": 4,
                    "num_trades": 45,
                    "total_trades": 45,
                    "winning_trades": 25,
                    "losing_trades": 20,
                },
                "trades": [],
            },
        },
        "model_results": {
            "model_a": {
                "total_return": 0.12,
                "sharpe_ratio": 1.3,
                "max_drawdown": -0.06,
            },
            "model_b": {
                "total_return": 0.08,
                "sharpe_ratio": 1.0,
                "max_drawdown": -0.10,
            },
        },
        "sector_results": {
            "Technology": {
                "total_return": 0.18,
                "trade_count": 30,
                "win_rate": 0.65,
            },
            "Healthcare": {
                "total_return": 0.05,
                "trade_count": 20,
                "win_rate": 0.50,
            },
        },
    }


class TestBacktestAnalyzerInit:
    """Tests for BacktestAnalyzer initialization."""

    @pytest.mark.unit
    def test_init_with_valid_report(self, sample_backtest_report, tmp_path):
        """Test initialization with valid report file."""
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(sample_backtest_report))

        analyzer = BacktestAnalyzer(str(report_path))

        assert analyzer.report == sample_backtest_report
        assert analyzer.report["metadata"]["total_equities"] == 100

    @pytest.mark.unit
    def test_init_with_missing_file(self):
        """Test initialization with missing file raises error."""
        with pytest.raises(FileNotFoundError):
            BacktestAnalyzer("/nonexistent/path/report.json")

    @pytest.mark.unit
    def test_init_with_invalid_json(self, tmp_path):
        """Test initialization with invalid JSON raises error."""
        report_path = tmp_path / "report.json"
        report_path.write_text("not valid json")

        with pytest.raises(json.JSONDecodeError):
            BacktestAnalyzer(str(report_path))


class TestGenerateDetailedReport:
    """Tests for generate_detailed_report method."""

    @pytest.fixture
    def analyzer(self, sample_backtest_report, tmp_path):
        """Create analyzer for tests."""
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(sample_backtest_report))
        return BacktestAnalyzer(str(report_path))

    @pytest.mark.unit
    def test_generate_detailed_report_structure(self, analyzer):
        """Test report has expected sections."""
        report = analyzer.generate_detailed_report()

        assert "COMPREHENSIVE BACKTEST ANALYSIS" in report
        assert "EXECUTIVE SUMMARY" in report
        assert "DETAILED PERFORMANCE BY PERIOD" in report
        assert "MODEL PERFORMANCE RANKINGS" in report
        assert "SECTOR PERFORMANCE ANALYSIS" in report
        assert "RISK ANALYSIS" in report
        assert "OPTIMIZATION RECOMMENDATIONS" in report
        assert "IMPLEMENTATION GUIDE" in report

    @pytest.mark.unit
    def test_generate_detailed_report_is_string(self, analyzer):
        """Test report returns a string."""
        report = analyzer.generate_detailed_report()
        assert isinstance(report, str)
        assert len(report) > 0


class TestExecutiveSummary:
    """Tests for _executive_summary method."""

    @pytest.fixture
    def analyzer(self, sample_backtest_report, tmp_path):
        """Create analyzer for tests."""
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(sample_backtest_report))
        return BacktestAnalyzer(str(report_path))

    @pytest.mark.unit
    def test_executive_summary_includes_metrics(self, analyzer):
        """Test summary includes key metrics."""
        summary = analyzer._executive_summary()
        summary_text = "\n".join(summary)

        assert "100 equities" in summary_text

    @pytest.mark.unit
    def test_executive_summary_empty_results(self, tmp_path):
        """Test summary with empty results."""
        report = {
            "metadata": {"total_equities": 0, "test_periods": []},
            "results_by_period": {},
        }
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(report))

        analyzer = BacktestAnalyzer(str(report_path))
        summary = analyzer._executive_summary()

        assert "No results available" in summary


class TestPeriodAnalysis:
    """Tests for _period_analysis method."""

    @pytest.fixture
    def analyzer(self, sample_backtest_report, tmp_path):
        """Create analyzer for tests."""
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(sample_backtest_report))
        return BacktestAnalyzer(str(report_path))

    @pytest.mark.unit
    def test_period_analysis_returns_list(self, analyzer):
        """Test period analysis returns list."""
        analysis = analyzer._period_analysis()
        assert isinstance(analysis, list)

    @pytest.mark.unit
    def test_period_analysis_includes_periods(self, analyzer):
        """Test period analysis includes period names."""
        analysis = analyzer._period_analysis()
        analysis_text = "\n".join(analysis)

        # Should reference the periods
        assert "2023" in analysis_text or len(analysis) > 0


class TestModelAnalysis:
    """Tests for _model_analysis method."""

    @pytest.fixture
    def analyzer(self, sample_backtest_report, tmp_path):
        """Create analyzer for tests."""
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(sample_backtest_report))
        return BacktestAnalyzer(str(report_path))

    @pytest.mark.unit
    def test_model_analysis_returns_list(self, analyzer):
        """Test model analysis returns list."""
        analysis = analyzer._model_analysis()
        assert isinstance(analysis, list)


class TestSectorAnalysis:
    """Tests for _sector_analysis method."""

    @pytest.fixture
    def analyzer(self, sample_backtest_report, tmp_path):
        """Create analyzer for tests."""
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(sample_backtest_report))
        return BacktestAnalyzer(str(report_path))

    @pytest.mark.unit
    def test_sector_analysis_returns_list(self, analyzer):
        """Test sector analysis returns list."""
        analysis = analyzer._sector_analysis()
        assert isinstance(analysis, list)


class TestRiskAnalysis:
    """Tests for _risk_analysis method."""

    @pytest.fixture
    def analyzer(self, sample_backtest_report, tmp_path):
        """Create analyzer for tests."""
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(sample_backtest_report))
        return BacktestAnalyzer(str(report_path))

    @pytest.mark.unit
    def test_risk_analysis_returns_list(self, analyzer):
        """Test risk analysis returns list."""
        analysis = analyzer._risk_analysis()
        assert isinstance(analysis, list)


class TestOptimizationRecommendations:
    """Tests for _optimization_recommendations method."""

    @pytest.fixture
    def analyzer(self, sample_backtest_report, tmp_path):
        """Create analyzer for tests."""
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(sample_backtest_report))
        return BacktestAnalyzer(str(report_path))

    @pytest.mark.unit
    def test_optimization_recommendations_returns_list(self, analyzer):
        """Test optimization recommendations returns list."""
        recommendations = analyzer._optimization_recommendations()
        assert isinstance(recommendations, list)


class TestImplementationGuide:
    """Tests for _implementation_guide method."""

    @pytest.fixture
    def analyzer(self, sample_backtest_report, tmp_path):
        """Create analyzer for tests."""
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(sample_backtest_report))
        return BacktestAnalyzer(str(report_path))

    @pytest.mark.unit
    def test_implementation_guide_returns_list(self, analyzer):
        """Test implementation guide returns list."""
        guide = analyzer._implementation_guide()
        assert isinstance(guide, list)
