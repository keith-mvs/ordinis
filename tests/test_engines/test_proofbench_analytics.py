"""Tests for ProofBenchEngine analytics capabilities."""

from datetime import datetime

import pytest

from ordinis.engines.proofbench.core.config import ProofBenchEngineConfig
from ordinis.engines.proofbench.core.engine import ProofBenchEngine


class TestProofBenchAnalytics:
    """Test ProofBench engine analytics functionality."""

    @pytest.fixture
    def analytics_engine(self):
        """Provide ProofBench engine instance configured for analytics."""
        config = ProofBenchEngineConfig(enabled=True)
        return ProofBenchEngine(config)

    @pytest.fixture
    def sample_trade_results(self):
        """Provide sample trade results."""
        return [
            {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "price": 150.0,
                "timestamp": datetime.now(),
                "order_id": "ord_001",
                "pnl": 250.0,
            },
            {
                "symbol": "MSFT",
                "side": "sell",
                "quantity": 50,
                "price": 300.0,
                "timestamp": datetime.now(),
                "order_id": "ord_002",
                "pnl": -100.0,
            },
            {
                "symbol": "GOOGL",
                "side": "buy",
                "quantity": 25,
                "price": 2800.0,
                "timestamp": datetime.now(),
                "order_id": "ord_003",
                "pnl": 500.0,
            },
        ]

    @pytest.mark.asyncio
    async def test_record_trade_results(self, analytics_engine, sample_trade_results):
        """Test recording trade results."""
        await analytics_engine.record(sample_trade_results)

        assert len(analytics_engine.trade_history) == 3
        assert analytics_engine.trade_history[0].symbol == "AAPL"
        assert analytics_engine.trade_history[0].pnl == 250.0

    @pytest.mark.asyncio
    async def test_calculate_metrics(self, analytics_engine, sample_trade_results):
        """Test metric calculation."""
        await analytics_engine.record(sample_trade_results)

        metrics = analytics_engine.calculate_metrics()

        assert metrics["trade_count"] == 3
        assert metrics["total_pnl"] == 650.0  # 250 - 100 + 500
        assert metrics["win_rate"] == 2 / 3  # 2 winning trades out of 3
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics

    @pytest.mark.asyncio
    async def test_get_performance_summary(self, analytics_engine, sample_trade_results):
        """Test performance summary."""
        await analytics_engine.record(sample_trade_results)

        summary = analytics_engine.get_performance_summary()

        assert "total_pnl" in summary
        assert "win_rate" in summary
        assert "sharpe_ratio" in summary
        assert "max_drawdown" in summary
        assert "timestamp" in summary
        assert summary["total_pnl"] == 650.0

    @pytest.mark.asyncio
    async def test_empty_metrics(self, analytics_engine):
        """Test metrics with no trades."""
        metrics = analytics_engine.calculate_metrics()

        assert metrics["total_pnl"] == 0.0
        assert metrics["win_rate"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["max_drawdown"] == 0.0
        assert metrics["trade_count"] == 0

    @pytest.mark.asyncio
    async def test_disabled_engine(self):
        """Test that disabled engine doesn't record."""
        config = ProofBenchEngineConfig(enabled=False)
        engine = ProofBenchEngine(config)

        await engine.record([{"symbol": "AAPL", "price": 150.0, "quantity": 100, "pnl": 100.0}])

        assert len(engine.trade_history) == 0
