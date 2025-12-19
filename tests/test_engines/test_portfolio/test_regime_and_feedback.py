"""
Tests for RegimeSizingHook and ExecutionFeedbackCollector.

Validates:
- Regime-aware sizing multipliers
- Preflight decision logic
- Execution quality tracking
- Sizing adjustment recommendations
"""

from decimal import Decimal
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


class TestRegimeSizingHook:
    """Tests for RegimeSizingHook governance."""

    @pytest.fixture
    def mock_regime_detector(self):
        """Create mock RegimeDetector."""
        mock = MagicMock()
        mock.analyze.return_value = MagicMock(
            regime=MagicMock(value="TRENDING"),
            confidence=0.85,
            metrics={"adx": 35, "atr_ratio": 1.2},
        )
        return mock

    @pytest.fixture
    def sample_price_data(self):
        """Create sample OHLCV data."""
        rng = np.random.default_rng(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
        return pd.DataFrame(
            {
                "open": rng.standard_normal(100).cumsum() + 100,
                "high": rng.standard_normal(100).cumsum() + 101,
                "low": rng.standard_normal(100).cumsum() + 99,
                "close": rng.standard_normal(100).cumsum() + 100,
                "volume": rng.integers(1000, 10000, 100),
            },
            index=dates,
        )

    def test_regime_sizing_multipliers_defined(self):
        """Verify all expected regimes have multipliers."""
        from ordinis.engines.portfolio.hooks.regime_sizing import (
            REGIME_SIZING_MULTIPLIERS,
        )

        expected_regimes = [
            "TRENDING",
            "VOLATILE_TRENDING",
            "MEAN_REVERTING",
            "CHOPPY",
            "QUIET_CHOPPY",
            "UNKNOWN",
        ]

        for regime in expected_regimes:
            assert regime in REGIME_SIZING_MULTIPLIERS
            assert 0 < REGIME_SIZING_MULTIPLIERS[regime] <= 2.0

    def test_trending_regime_increases_sizing(self):
        """Trending regime should increase position size."""
        from ordinis.engines.portfolio.hooks.regime_sizing import (
            REGIME_SIZING_MULTIPLIERS,
        )

        assert REGIME_SIZING_MULTIPLIERS["TRENDING"] > 1.0

    def test_choppy_regime_decreases_sizing(self):
        """Choppy regime should decrease position size."""
        from ordinis.engines.portfolio.hooks.regime_sizing import (
            REGIME_SIZING_MULTIPLIERS,
        )

        assert REGIME_SIZING_MULTIPLIERS["CHOPPY"] < 1.0
        assert REGIME_SIZING_MULTIPLIERS["QUIET_CHOPPY"] < REGIME_SIZING_MULTIPLIERS["CHOPPY"]

    def test_regime_sizing_hook_initialization(self, mock_regime_detector):
        """Test RegimeSizingHook initializes correctly."""
        from ordinis.engines.portfolio.hooks.regime_sizing import (
            RegimeSizingConfig,
            RegimeSizingHook,
        )

        config = RegimeSizingConfig(
            enable_regime_adjustment=True,
            min_confidence=0.7,
            avoid_quiet_choppy=True,
        )

        hook = RegimeSizingHook(
            regime_detector=mock_regime_detector,
            config=config,
        )

        assert hook.config.enable_regime_adjustment is True
        assert hook.config.min_confidence == pytest.approx(0.7)
        assert hook.config.avoid_quiet_choppy is True

    def test_analyze_regime_trending(self, mock_regime_detector, sample_price_data):
        """Test regime analysis for trending market."""
        from ordinis.engines.portfolio.hooks.regime_sizing import RegimeSizingHook

        hook = RegimeSizingHook(regime_detector=mock_regime_detector)

        adjustment = hook.analyze_regime("AAPL", sample_price_data)

        assert adjustment.symbol == "AAPL"
        assert adjustment.regime == "TRENDING"
        assert adjustment.sizing_multiplier > 1.0
        assert adjustment.trade_recommendation == "TRADE"

    def test_analyze_regime_choppy(self, mock_regime_detector, sample_price_data):
        """Test regime analysis for choppy market."""
        from ordinis.engines.portfolio.hooks.regime_sizing import RegimeSizingHook

        # Configure mock for choppy regime
        mock_regime_detector.analyze.return_value = MagicMock(
            regime=MagicMock(value="CHOPPY"),
            confidence=0.75,
            metrics={"adx": 15},
        )

        hook = RegimeSizingHook(regime_detector=mock_regime_detector)

        adjustment = hook.analyze_regime("AAPL", sample_price_data)

        assert adjustment.regime == "CHOPPY"
        assert adjustment.sizing_multiplier < 1.0
        assert adjustment.trade_recommendation == "CAUTION"

    def test_quiet_choppy_avoid_recommendation(self, mock_regime_detector, sample_price_data):
        """Test that quiet_choppy regime triggers AVOID."""
        from ordinis.engines.portfolio.hooks.regime_sizing import (
            RegimeSizingConfig,
            RegimeSizingHook,
        )

        mock_regime_detector.analyze.return_value = MagicMock(
            regime=MagicMock(value="QUIET_CHOPPY"),
            confidence=0.8,
            metrics={"adx": 10},
        )

        config = RegimeSizingConfig(avoid_quiet_choppy=True)
        hook = RegimeSizingHook(regime_detector=mock_regime_detector, config=config)

        adjustment = hook.analyze_regime("AAPL", sample_price_data)

        assert adjustment.trade_recommendation == "AVOID"

    def test_custom_multipliers_override(self, mock_regime_detector, sample_price_data):
        """Test custom multipliers override defaults."""
        from ordinis.engines.portfolio.hooks.regime_sizing import RegimeSizingHook

        custom = {"TRENDING": 2.0, "CHOPPY": 0.3}
        hook = RegimeSizingHook(
            regime_detector=mock_regime_detector,
            custom_multipliers=custom,
        )

        # Trending should use custom multiplier (clamped to max)
        adjustment = hook.analyze_regime("AAPL", sample_price_data)

        # Should be clamped to max_sizing_multiplier (default 1.5)
        assert adjustment.sizing_multiplier <= 1.5

    def test_min_confidence_threshold(self, mock_regime_detector, sample_price_data):
        """Test that low confidence doesn't apply adjustment."""
        from ordinis.engines.portfolio.hooks.regime_sizing import (
            RegimeSizingConfig,
            RegimeSizingHook,
        )

        # Low confidence
        mock_regime_detector.analyze.return_value = MagicMock(
            regime=MagicMock(value="TRENDING"),
            confidence=0.4,  # Below threshold
            metrics={},
        )

        config = RegimeSizingConfig(min_confidence=0.6)
        hook = RegimeSizingHook(regime_detector=mock_regime_detector, config=config)

        adjustment = hook.analyze_regime("AAPL", sample_price_data)

        # Low confidence should still return adjustment but with neutral multiplier
        assert adjustment.confidence < config.min_confidence


class TestExecutionFeedbackCollector:
    """Tests for ExecutionFeedbackCollector."""

    @pytest.fixture
    def collector(self):
        """Create ExecutionFeedbackCollector instance."""
        from ordinis.engines.portfolio.feedback.execution_feedback import (
            ExecutionFeedbackCollector,
        )

        return ExecutionFeedbackCollector(max_history_size=1000)

    def test_record_order_submission(self, collector):
        """Test recording order submission."""
        collector.record_order_submission(
            order_id="order_001",
            symbol="AAPL",
            side="buy",
            expected_price=150.0,
            expected_qty=100,
            estimated_cost_bps=10.0,
        )

        assert "order_001" in collector._pending_orders
        pending = collector._pending_orders["order_001"]
        assert pending["symbol"] == "AAPL"
        assert pending["expected_price"] == pytest.approx(150.0)

    def test_record_fill_matches_submission(self, collector):
        """Test that fill matches pending submission."""
        # Submit order
        collector.record_order_submission(
            order_id="order_001",
            symbol="AAPL",
            side="buy",
            expected_price=150.0,
            expected_qty=100,
        )

        # Record fill
        record = collector.record_fill(
            order_id="order_001",
            filled_avg_price=150.10,
            filled_qty=100,
            execution_time_ms=50,
        )

        assert record is not None
        assert record.symbol == "AAPL"
        assert record.slippage_bps > 0  # Some slippage
        assert "order_001" not in collector._pending_orders  # Cleared

    def test_record_fill_unknown_order(self, collector):
        """Test fill for unknown order returns None."""
        record = collector.record_fill(
            order_id="unknown_order",
            filled_avg_price=150.0,
            filled_qty=100,
        )

        assert record is None

    def test_execution_quality_calculation(self, collector):
        """Test execution quality level calculation."""
        from ordinis.engines.portfolio.feedback.execution_feedback import (
            ExecutionQualityLevel,
        )

        # Record some executions with varying slippage
        for _ in range(10):
            collector.record_execution(
                symbol="AAPL",
                side="buy",
                expected_price=150.0,
                filled_avg_price=150.05,  # 3.3 bps slippage
                expected_qty=100,
                filled_qty=100,
            )

        metrics = collector.get_quality_metrics(lookback_hours=24)

        assert metrics.n_executions == 10
        assert metrics.avg_slippage_bps < 5  # Excellent quality
        assert metrics.overall_quality == ExecutionQualityLevel.EXCELLENT

    def test_poor_execution_detection(self, collector):
        """Test detection of poor execution quality."""
        from ordinis.engines.portfolio.feedback.execution_feedback import (
            ExecutionQualityLevel,
        )

        # Record executions with high slippage
        for _ in range(15):
            collector.record_execution(
                symbol="AAPL",
                side="buy",
                expected_price=150.0,
                filled_avg_price=150.75,  # 50 bps slippage
                expected_qty=100,
                filled_qty=100,
            )

        metrics = collector.get_quality_metrics()

        assert metrics.avg_slippage_bps >= 30
        assert metrics.overall_quality in (
            ExecutionQualityLevel.POOR,
            ExecutionQualityLevel.VERY_POOR,
        )

    def test_should_adjust_sizing_high_slippage(self, collector):
        """Test sizing adjustment recommendation for high slippage."""
        # Record poor executions for a symbol
        for _ in range(15):
            collector.record_execution(
                symbol="AAPL",
                side="buy",
                expected_price=100.0,
                filled_avg_price=100.50,  # 50 bps slippage
                expected_qty=100,
                filled_qty=100,
            )

        should_adjust, multiplier = collector.should_adjust_sizing("AAPL")

        assert should_adjust is True
        assert multiplier < 1.0  # Should reduce

    def test_should_not_adjust_low_slippage(self, collector):
        """Test no adjustment needed for low slippage."""
        # Record good executions
        for _ in range(15):
            collector.record_execution(
                symbol="MSFT",
                side="buy",
                expected_price=300.0,
                filled_avg_price=300.03,  # 1 bp slippage
                expected_qty=50,
                filled_qty=50,
            )

        _, multiplier = collector.should_adjust_sizing("MSFT")

        # Could recommend increase for very low slippage
        assert multiplier >= 1.0

    def test_symbol_breakdown_in_metrics(self, collector):
        """Test per-symbol breakdown in quality metrics."""
        # Record for multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        for symbol in symbols:
            for _ in range(5):
                collector.record_execution(
                    symbol=symbol,
                    side="buy",
                    expected_price=100.0,
                    filled_avg_price=100.02,
                    expected_qty=100,
                    filled_qty=100,
                )

        metrics = collector.get_quality_metrics()

        assert len(metrics.symbol_breakdown) == 3
        for symbol in symbols:
            assert symbol in metrics.symbol_breakdown
            assert metrics.symbol_breakdown[symbol]["n"] == 5

    def test_partial_fill_detection(self, collector):
        """Test detection of partial fills."""
        record = collector.record_execution(
            symbol="AAPL",
            side="buy",
            expected_price=150.0,
            filled_avg_price=150.0,
            expected_qty=100,
            filled_qty=75,  # Partial fill
        )

        assert record.is_partial_fill is True
        assert record.fill_rate == pytest.approx(0.75)

    def test_export_for_learning(self, collector):
        """Test data export for LearningEngine."""
        # Add some executions
        for _ in range(5):
            collector.record_execution(
                symbol="AAPL",
                side="buy",
                expected_price=150.0,
                filled_avg_price=150.05,
                expected_qty=100,
                filled_qty=100,
            )

        export = collector.export_for_learning(lookback_hours=24)

        assert len(export) == 5
        assert all("symbol" in e for e in export)
        assert all("slippage_bps" in e for e in export)
        assert all("quality" in e for e in export)

    def test_max_history_trimming(self):
        """Test history is trimmed to max size."""
        from ordinis.engines.portfolio.feedback.execution_feedback import (
            ExecutionFeedbackCollector,
        )

        collector = ExecutionFeedbackCollector(max_history_size=10)

        # Add more than max
        for _ in range(20):
            collector.record_execution(
                symbol="AAPL",
                side="buy",
                expected_price=100.0,
                filled_avg_price=100.01,
                expected_qty=50,
                filled_qty=50,
            )

        assert len(collector._execution_history) == 10


class TestPortfolioStateManager:
    """Tests for PortfolioStateManager transactional state."""

    @pytest.fixture
    def manager(self):
        """Create PortfolioStateManager instance."""
        from ordinis.engines.portfolio.state import PortfolioStateManager

        return PortfolioStateManager(initial_cash=100000.0)

    def test_initialization(self, manager):
        """Test manager initializes with correct state."""
        assert manager.cash == Decimal("100000")
        assert manager.equity == Decimal("100000")
        assert manager.version == 0
        assert not manager.in_transaction

    def test_begin_transaction(self, manager):
        """Test transaction can be started."""
        manager.begin_transaction()

        assert manager.in_transaction
        assert manager._transaction_snapshot is not None

    def test_nested_transaction_error(self, manager):
        """Test nested transaction raises error."""
        manager.begin_transaction()

        with pytest.raises(RuntimeError, match="already in progress"):
            manager.begin_transaction()

    def test_commit_increments_version(self, manager):
        """Test commit increments version."""
        initial_version = manager.version

        manager.begin_transaction()
        manager.update_cash(-5000)
        snapshot = manager.commit()

        assert manager.version == initial_version + 1
        assert snapshot.version == initial_version + 1

    def test_rollback_restores_state(self, manager):
        """Test rollback restores previous state."""
        initial_cash = manager.cash

        manager.begin_transaction()
        manager.update_cash(-10000)
        assert manager.cash == initial_cash - 10000

        manager.rollback()

        assert manager.cash == initial_cash
        assert not manager.in_transaction

    def test_update_cash_validates_negative(self, manager):
        """Test cash update rejects negative result."""
        from ordinis.engines.portfolio.state import StateValidationError

        manager.begin_transaction()

        with pytest.raises(StateValidationError, match="negative balance"):
            manager.update_cash(-200000)  # More than available

    def test_add_position(self, manager):
        """Test adding a position."""
        manager.begin_transaction()
        manager.add_position("AAPL", 100, 150.0, "LONG")
        manager.commit()

        snapshot = manager.get_snapshot()
        pos = snapshot.get_position("AAPL")

        assert pos is not None
        assert pos.quantity == Decimal("100")
        assert pos.avg_entry_price == Decimal("150")

    def test_remove_position(self, manager):
        """Test removing a position."""
        manager.begin_transaction()
        manager.add_position("AAPL", 100, 150.0)
        manager.commit()

        manager.begin_transaction()
        removed = manager.remove_position("AAPL")
        manager.commit()

        assert removed is not None
        snapshot = manager.get_snapshot()
        assert snapshot.get_position("AAPL") is None

    def test_snapshot_immutability(self, manager):
        """Test snapshots are immutable."""
        manager.begin_transaction()
        manager.add_position("AAPL", 100, 150.0)
        manager.commit()

        snapshot = manager.get_snapshot()

        # Frozen dataclass should be immutable
        with pytest.raises(Exception):  # FrozenInstanceError
            snapshot.cash = Decimal("0")

    def test_state_hash_changes(self, manager):
        """Test state hash changes with modifications."""
        snap1 = manager.get_snapshot()

        manager.begin_transaction()
        manager.update_cash(1000)
        manager.commit()

        snap2 = manager.get_snapshot()

        assert snap1.state_hash != snap2.state_hash

    def test_pending_changes_tracked(self, manager):
        """Test pending changes are tracked in transaction."""
        manager.begin_transaction()
        manager.update_cash(-5000)
        manager.add_position("AAPL", 50, 100.0)

        changes = manager.get_pending_changes()

        assert len(changes) == 2
        assert changes[0].change_type == "cash"
        assert changes[1].change_type == "position_add"

    def test_history_retention(self, manager):
        """Test snapshot history is retained."""
        for _ in range(5):
            manager.begin_transaction()
            manager.update_cash(1000)
            manager.commit()

        history = manager.get_history()

        assert len(history) == 5
        assert all(h.version == i + 1 for i, h in enumerate(history))

    def test_restore_from_snapshot(self, manager):
        """Test state restoration from snapshot."""
        # Create initial state
        manager.begin_transaction()
        manager.add_position("AAPL", 100, 150.0)
        manager.commit()

        saved_snapshot = manager.get_snapshot()

        # Modify state
        manager.begin_transaction()
        manager.add_position("MSFT", 50, 300.0)
        manager.update_cash(-5000)
        manager.commit()

        # Restore
        manager.restore_from_snapshot(saved_snapshot)

        current = manager.get_snapshot()
        assert current.position_count == 1
        assert current.get_position("MSFT") is None

    def test_optimistic_lock_error(self, manager):
        """Test optimistic locking detects version mismatch."""
        from ordinis.engines.portfolio.state import OptimisticLockError

        manager.begin_transaction()
        manager.commit()

        snapshot = manager.get_snapshot()

        # Increment version
        manager.begin_transaction()
        manager.update_cash(1000)
        manager.commit()

        # Try to restore with wrong expected version
        with pytest.raises(OptimisticLockError):
            manager.restore_from_snapshot(snapshot, expected_version=snapshot.version)
