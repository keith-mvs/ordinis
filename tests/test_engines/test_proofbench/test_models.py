"""Tests for proofbench core models."""

from datetime import datetime

import pytest

from ordinis.engines.proofbench.core.models import (
    Metric,
    PerformanceReport,
    TradeResult,
)


class TestMetric:
    """Tests for Metric dataclass."""

    def test_metric_creation_minimal(self):
        """Test creating a metric with required fields."""
        metric = Metric(name="sharpe_ratio", value=1.5)
        assert metric.name == "sharpe_ratio"
        assert metric.value == 1.5
        assert isinstance(metric.timestamp, datetime)
        assert metric.tags == {}

    def test_metric_creation_with_tags(self):
        """Test creating a metric with tags."""
        tags = {"strategy": "momentum", "symbol": "AAPL"}
        metric = Metric(name="win_rate", value=0.65, tags=tags)
        assert metric.tags == tags
        assert metric.tags["strategy"] == "momentum"

    def test_metric_with_custom_timestamp(self):
        """Test creating a metric with custom timestamp."""
        ts = datetime(2024, 1, 15, 10, 30)
        metric = Metric(name="max_drawdown", value=-0.12, timestamp=ts)
        assert metric.timestamp == ts

    def test_metric_negative_value(self):
        """Test metric with negative value."""
        metric = Metric(name="pnl", value=-500.0)
        assert metric.value == -500.0

    def test_metric_zero_value(self):
        """Test metric with zero value."""
        metric = Metric(name="fees", value=0.0)
        assert metric.value == 0.0


class TestTradeResult:
    """Tests for TradeResult dataclass."""

    def test_trade_result_buy(self):
        """Test creating a buy trade result."""
        ts = datetime.now()
        result = TradeResult(
            symbol="AAPL",
            side="buy",
            quantity=100.0,
            price=150.50,
            timestamp=ts,
            order_id="ORD-001",
        )
        assert result.symbol == "AAPL"
        assert result.side == "buy"
        assert result.quantity == 100.0
        assert result.price == 150.50
        assert result.fees == 0.0
        assert result.pnl is None

    def test_trade_result_sell_with_pnl(self):
        """Test creating a sell trade result with PnL."""
        ts = datetime.now()
        result = TradeResult(
            symbol="MSFT",
            side="sell",
            quantity=50.0,
            price=380.00,
            timestamp=ts,
            order_id="ORD-002",
            fees=1.50,
            pnl=250.00,
        )
        assert result.side == "sell"
        assert result.fees == 1.50
        assert result.pnl == 250.00

    def test_trade_result_with_negative_pnl(self):
        """Test trade result with losing trade."""
        ts = datetime.now()
        result = TradeResult(
            symbol="TSLA",
            side="sell",
            quantity=10.0,
            price=200.00,
            timestamp=ts,
            order_id="ORD-003",
            pnl=-150.00,
        )
        assert result.pnl == -150.00

    def test_trade_result_fractional_quantity(self):
        """Test trade result with fractional quantity (crypto)."""
        ts = datetime.now()
        result = TradeResult(
            symbol="BTC/USD",
            side="buy",
            quantity=0.5,
            price=45000.00,
            timestamp=ts,
            order_id="ORD-004",
        )
        assert result.quantity == 0.5


class TestPerformanceReport:
    """Tests for PerformanceReport dataclass."""

    def test_performance_report_minimal(self):
        """Test creating a minimal performance report."""
        ts = datetime.now()
        report = PerformanceReport(
            timestamp=ts,
            metrics={"sharpe": 1.2, "win_rate": 0.55},
        )
        assert report.timestamp == ts
        assert report.metrics["sharpe"] == 1.2
        assert report.narrative is None
        assert report.trade_count == 0

    def test_performance_report_full(self):
        """Test creating a full performance report."""
        ts = datetime.now()
        metrics = {
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.0,
            "max_drawdown": -0.10,
            "win_rate": 0.60,
            "profit_factor": 1.8,
        }
        report = PerformanceReport(
            timestamp=ts,
            metrics=metrics,
            narrative="Strategy performed well in trending market conditions.",
            trade_count=150,
        )
        assert report.narrative is not None
        assert report.trade_count == 150
        assert len(report.metrics) == 5

    def test_performance_report_empty_metrics(self):
        """Test performance report with empty metrics."""
        ts = datetime.now()
        report = PerformanceReport(timestamp=ts, metrics={})
        assert report.metrics == {}
        assert report.trade_count == 0
