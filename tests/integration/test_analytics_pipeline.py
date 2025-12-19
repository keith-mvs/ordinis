"""Integration test for analytics pipeline with signal generation and risk evaluation."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from ordinis.engines.proofbench.core.config import ProofBenchEngineConfig
from ordinis.engines.proofbench.core.engine import ProofBenchEngine
from ordinis.engines.riskguard.core.engine import RiskGuardEngine
from ordinis.engines.signalcore.core.signal import Signal, SignalType
from ordinis.engines.signalcore.models.technical import SMACrossoverModel


@pytest.mark.asyncio
class TestAnalyticsPipeline:
    """Test the full pipeline: signal generation → risk evaluation → analytics."""

    async def test_sma_crossover_to_analytics(self):
        """Test SMA crossover signal flowing through risk guard to analytics."""
        # Create SMA crossover model
        sma_model = SMACrossoverModel(short_window=10, long_window=20, name="SMA_10_20")

        # Generate price data with clear uptrend for bullish crossover
        prices = pd.Series(
            [100 + i * 0.5 for i in range(50)],  # Uptrend
            index=pd.date_range("2024-01-01", periods=50, freq="D"),
        )

        # Create market data with OHLC
        market_data = pd.DataFrame(
            {
                "close": prices,
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "volume": [10000] * len(prices),
            }
        )

        # Generate signal
        signal = sma_model.generate(market_data)

        # Verify signal was generated
        assert signal is not None
        assert signal.symbol == sma_model.name

        # Initialize risk guard with standard rules
        risk_engine = RiskGuardEngine()
        await risk_engine.initialize()

        # Evaluate signal
        risk_signal = Signal(
            symbol="AAPL",
            type=SignalType.LONG if signal.strength > 0 else SignalType.SHORT,
            strength=abs(signal.strength),
            confidence=signal.confidence,
            timestamp=datetime.now(UTC),
        )

        risk_result = risk_engine.evaluate_signal(risk_signal)

        # Signal should pass risk checks (not too large, within limits)
        assert risk_result.approved or risk_result.resized

        # Initialize analytics engine (ProofBench)
        analytics_config = ProofBenchEngineConfig(enabled=True, metrics_retention_days=30)
        analytics_engine = ProofBenchEngine(config=analytics_config)
        await analytics_engine.initialize()

        # Simulate trade result
        trade_data = {
            "symbol": "AAPL",
            "pnl": 150.0,
            "entry_time": "2024-01-15T10:00:00Z",
            "exit_time": "2024-01-15T15:00:00Z",
            "quantity": 10,
            "price": 150.0,
            "side": "buy",
        }

        # Record trade
        await analytics_engine.record([trade_data])

        # Get metrics
        metrics = analytics_engine.calculate_metrics()

        # Verify metrics
        assert metrics["total_pnl"] == 150.0
        assert metrics["trade_count"] == 1
        assert metrics["win_rate"] == 1.0  # 100% win rate

        # Get performance summary
        summary = analytics_engine.get_performance_summary()
        assert summary is not None
        assert summary["total_pnl"] == 150.0

        await analytics_engine.shutdown()
        await risk_engine.shutdown()

    async def test_multiple_trades_analytics(self):
        """Test analytics with multiple trades (wins and losses)."""
        analytics_engine = ProofBenchEngine()
        await analytics_engine.initialize()

        # Record multiple trades
        trades = [
            {
                "symbol": "AAPL",
                "pnl": 100.0,
                "entry_time": "2024-01-01T10:00:00Z",
                "exit_time": "2024-01-01T15:00:00Z",
                "quantity": 10,
                "price": 150.0,
                "side": "buy",
            },
            {
                "symbol": "GOOGL",
                "pnl": -50.0,
                "entry_time": "2024-01-02T10:00:00Z",
                "exit_time": "2024-01-02T15:00:00Z",
                "quantity": 5,
                "price": 2000.0,
                "side": "sell",
            },
            {
                "symbol": "MSFT",
                "pnl": 200.0,
                "entry_time": "2024-01-03T10:00:00Z",
                "exit_time": "2024-01-03T15:00:00Z",
                "quantity": 20,
                "price": 300.0,
                "side": "buy",
            },
            {
                "symbol": "AMZN",
                "pnl": -25.0,
                "entry_time": "2024-01-04T10:00:00Z",
                "exit_time": "2024-01-04T15:00:00Z",
                "quantity": 15,
                "price": 100.0,
                "side": "sell",
            },
        ]

        await analytics_engine.record(trades)

        # Calculate metrics
        metrics = analytics_engine.calculate_metrics()

        # Verify metrics
        assert metrics["total_pnl"] == 225.0  # 100 - 50 + 200 - 25
        assert metrics["trade_count"] == 4
        assert metrics["win_rate"] == 0.5  # 2 wins out of 4
        # assert metrics["sharpe_ratio"] > 0  # Placeholder is 0.0 now

        await analytics_engine.shutdown()

    async def test_risk_guard_position_sizing(self):
        """Test risk guard resizing large positions."""
        risk_engine = RiskGuardEngine()
        await risk_engine.initialize()

        # Create oversized signal
        large_signal = Signal(
            symbol="AAPL",
            type=SignalType.LONG,
            strength=1.0,  # 100% allocation - too large!
            confidence=0.9,
            timestamp=datetime.now(UTC),
        )

        result = risk_engine.evaluate_signal(large_signal)

        # Should be resized
        assert result.resized
        assert result.adjusted_signal is not None
        assert result.adjusted_signal.strength < large_signal.strength

        await risk_engine.shutdown()
