"""
Tests for Unified Portfolio Rebalancing Engine.
"""

from datetime import UTC, datetime

import pytest

from ordinis.engines.portfolio.engine import (
    ExecutionResult,
    RebalancingEngine,
    RebalancingHistory,
    StrategyType,
)
from ordinis.engines.portfolio.target_allocation import (
    RebalanceDecision,
    TargetAllocation,
    TargetAllocationRebalancer,
)


class TestRebalancingHistory:
    """Tests for RebalancingHistory dataclass."""

    def test_create_history(self):
        """Test creating rebalancing history entry."""
        timestamp = datetime.now(tz=UTC)
        history = RebalancingHistory(
            timestamp=timestamp,
            strategy_type=StrategyType.TARGET_ALLOCATION,
            decisions_count=3,
            total_adjustment_value=5000.0,
            execution_status="executed",
            metadata={"cash": 10000.0},
        )
        assert history.timestamp == timestamp
        assert history.strategy_type == StrategyType.TARGET_ALLOCATION
        assert history.decisions_count == 3
        assert history.total_adjustment_value == 5000.0
        assert history.execution_status == "executed"
        assert history.metadata == {"cash": 10000.0}


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_create_result_success(self):
        """Test creating successful execution result."""
        timestamp = datetime.now(tz=UTC)
        result = ExecutionResult(
            timestamp=timestamp,
            decisions_executed=3,
            decisions_failed=0,
            total_value_traded=5000.0,
            success=True,
            errors=[],
        )
        assert result.timestamp == timestamp
        assert result.decisions_executed == 3
        assert result.decisions_failed == 0
        assert result.success is True
        assert len(result.errors) == 0

    def test_create_result_with_failures(self):
        """Test creating execution result with failures."""
        result = ExecutionResult(
            timestamp=datetime.now(tz=UTC),
            decisions_executed=2,
            decisions_failed=1,
            total_value_traded=3000.0,
            success=False,
            errors=["AAPL: Insufficient funds"],
        )
        assert result.decisions_executed == 2
        assert result.decisions_failed == 1
        assert result.success is False
        assert len(result.errors) == 1


class TestRebalancingEngine:
    """Tests for RebalancingEngine class."""

    @pytest.fixture
    def target_strategy(self):
        """Fixture: Simple target allocation strategy."""
        targets = [
            TargetAllocation("AAPL", 0.40),
            TargetAllocation("MSFT", 0.30),
            TargetAllocation("GOOGL", 0.30),
        ]
        return TargetAllocationRebalancer(targets, drift_threshold=0.05)

    @pytest.fixture
    def sample_positions(self):
        """Fixture: Sample portfolio positions."""
        return {"AAPL": 10, "MSFT": 5, "GOOGL": 5}

    @pytest.fixture
    def sample_prices(self):
        """Fixture: Sample market prices."""
        return {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 500.0}

    def test_initialization(self):
        """Test engine initialization."""
        engine = RebalancingEngine(
            default_strategy=StrategyType.TARGET_ALLOCATION,
            track_history=True,
        )
        assert engine.default_strategy == StrategyType.TARGET_ALLOCATION
        assert engine.track_history is True
        assert len(engine.strategies) == 0
        assert len(engine.history) == 0

    def test_initialization_no_tracking(self):
        """Test engine initialization without history tracking."""
        engine = RebalancingEngine(track_history=False)
        assert engine.track_history is False

    def test_register_strategy(self, target_strategy):
        """Test registering a strategy."""
        engine = RebalancingEngine()
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        assert StrategyType.TARGET_ALLOCATION in engine.strategies
        assert engine.strategies[StrategyType.TARGET_ALLOCATION] == target_strategy

    def test_register_invalid_strategy(self):
        """Test registering strategy without required methods fails."""
        engine = RebalancingEngine()

        # Object without generate_rebalance_orders method
        invalid_strategy = {"foo": "bar"}

        with pytest.raises(ValueError, match="must have 'generate_rebalance_orders' method"):
            engine.register_strategy(StrategyType.TARGET_ALLOCATION, invalid_strategy)

    def test_get_strategy_default(self, target_strategy):
        """Test getting default strategy."""
        engine = RebalancingEngine(default_strategy=StrategyType.TARGET_ALLOCATION)
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        strategy = engine.get_strategy()
        assert strategy == target_strategy

    def test_get_strategy_explicit(self, target_strategy):
        """Test getting strategy by explicit type."""
        engine = RebalancingEngine()
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        strategy = engine.get_strategy(StrategyType.TARGET_ALLOCATION)
        assert strategy == target_strategy

    def test_get_strategy_not_registered(self):
        """Test getting unregistered strategy fails."""
        engine = RebalancingEngine()

        with pytest.raises(ValueError, match="not registered"):
            engine.get_strategy(StrategyType.RISK_PARITY)

    def test_generate_rebalancing_decisions(self, target_strategy, sample_positions, sample_prices):
        """Test generating rebalancing decisions."""
        engine = RebalancingEngine(default_strategy=StrategyType.TARGET_ALLOCATION)
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        decisions = engine.generate_rebalancing_decisions(sample_positions, sample_prices)

        assert len(decisions) == 3
        assert all(isinstance(d, RebalanceDecision) for d in decisions)

    def test_generate_rebalancing_decisions_with_cash(
        self, target_strategy, sample_positions, sample_prices
    ):
        """Test generating decisions with cash parameter."""
        engine = RebalancingEngine()
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        decisions = engine.generate_rebalancing_decisions(
            sample_positions, sample_prices, cash=5000.0
        )

        assert len(decisions) == 3

    def test_generate_rebalancing_decisions_no_tracking(
        self, target_strategy, sample_positions, sample_prices
    ):
        """Test decision generation without history tracking."""
        engine = RebalancingEngine(track_history=False)
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        decisions = engine.generate_rebalancing_decisions(sample_positions, sample_prices)

        assert len(decisions) == 3
        assert len(engine.history) == 0

    def test_generate_rebalancing_decisions_tracks_history(
        self, target_strategy, sample_positions, sample_prices
    ):
        """Test that decision generation tracks history."""
        engine = RebalancingEngine(track_history=True)
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        # Generate decisions (should trigger history tracking)
        # Positions far from targets to ensure decisions are generated
        imbalanced_positions = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}

        decisions = engine.generate_rebalancing_decisions(imbalanced_positions, sample_prices)

        # Should have history entry
        assert len(engine.history) == 1
        assert engine.history[0].strategy_type == StrategyType.TARGET_ALLOCATION
        assert engine.history[0].decisions_count == 3

    def test_should_rebalance_true(self, target_strategy, sample_prices):
        """Test should_rebalance returns True when needed."""
        engine = RebalancingEngine()
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        # Portfolio far from targets
        imbalanced_positions = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}

        should_rebalance = engine.should_rebalance(imbalanced_positions, sample_prices)

        assert should_rebalance is True

    def test_should_rebalance_false(self, target_strategy, sample_prices):
        """Test should_rebalance returns False when not needed."""
        engine = RebalancingEngine()
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        # Portfolio balanced to targets (0.40, 0.30, 0.30)
        # Total = 20*200 + 10*300 + 6*500 = 4000 + 3000 + 3000 = 10000
        # Weights: AAPL=0.40, MSFT=0.30, GOOGL=0.30 (exact targets)
        balanced_positions = {"AAPL": 20, "MSFT": 10, "GOOGL": 6}

        should_rebalance = engine.should_rebalance(balanced_positions, sample_prices)

        assert should_rebalance is False

    def test_execute_rebalancing_no_callback(
        self, target_strategy, sample_positions, sample_prices
    ):
        """Test executing rebalancing without callback (simulation)."""
        engine = RebalancingEngine()
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        decisions = engine.generate_rebalancing_decisions(
            sample_positions, sample_prices, cash=5000.0
        )

        result = engine.execute_rebalancing(decisions)

        assert result.success is True
        assert result.decisions_executed == len(decisions)
        assert result.decisions_failed == 0
        assert result.total_value_traded > 0

    def test_execute_rebalancing_with_successful_callback(
        self, target_strategy, sample_positions, sample_prices
    ):
        """Test executing rebalancing with successful callback."""
        engine = RebalancingEngine()
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        decisions = engine.generate_rebalancing_decisions(
            sample_positions, sample_prices, cash=5000.0
        )

        # Callback that always succeeds
        def success_callback(decision):
            return (True, None)

        result = engine.execute_rebalancing(decisions, execution_callback=success_callback)

        assert result.success is True
        assert result.decisions_executed == len(decisions)
        assert result.decisions_failed == 0

    def test_execute_rebalancing_with_failing_callback(
        self, target_strategy, sample_positions, sample_prices
    ):
        """Test executing rebalancing with failing callback."""
        engine = RebalancingEngine()
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        decisions = engine.generate_rebalancing_decisions(
            sample_positions, sample_prices, cash=5000.0
        )

        # Callback that fails for AAPL
        def failing_callback(decision):
            if decision.symbol == "AAPL":
                return (False, "Insufficient funds")
            return (True, None)

        result = engine.execute_rebalancing(decisions, execution_callback=failing_callback)

        assert result.success is False
        assert result.decisions_executed == 2
        assert result.decisions_failed == 1
        assert len(result.errors) == 1
        assert "AAPL" in result.errors[0]

    def test_execute_rebalancing_with_exception_callback(
        self, target_strategy, sample_positions, sample_prices
    ):
        """Test executing rebalancing when callback raises exception."""
        engine = RebalancingEngine()
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        decisions = engine.generate_rebalancing_decisions(
            sample_positions, sample_prices, cash=5000.0
        )

        # Callback that raises exception
        def exception_callback(decision):
            raise RuntimeError("Network error")

        result = engine.execute_rebalancing(decisions, execution_callback=exception_callback)

        assert result.success is False
        assert result.decisions_executed == 0
        assert result.decisions_failed == len(decisions)
        assert all("Network error" in e for e in result.errors)

    def test_execute_rebalancing_empty_decisions(self):
        """Test executing rebalancing with no decisions."""
        engine = RebalancingEngine()

        result = engine.execute_rebalancing([])

        assert result.success is True
        assert result.decisions_executed == 0
        assert result.decisions_failed == 0
        assert result.total_value_traded == 0.0

    def test_execute_rebalancing_updates_history(
        self, target_strategy, sample_positions, sample_prices
    ):
        """Test that execution updates history status."""
        engine = RebalancingEngine(track_history=True)
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        # Generate decisions far from targets
        imbalanced_positions = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}
        decisions = engine.generate_rebalancing_decisions(imbalanced_positions, sample_prices)

        # Execute
        engine.execute_rebalancing(decisions)

        # Check history updated
        assert engine.history[0].execution_status == "executed"
        assert engine.last_rebalance_date is not None

    def test_get_history_summary_empty(self):
        """Test getting history summary when empty."""
        engine = RebalancingEngine()

        df = engine.get_history_summary()

        assert len(df) == 0
        assert "timestamp" in df.columns

    def test_get_history_summary_with_data(self, target_strategy, sample_positions, sample_prices):
        """Test getting history summary with data."""
        engine = RebalancingEngine(track_history=True)
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        # Generate decisions multiple times
        for _ in range(3):
            imbalanced = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}
            engine.generate_rebalancing_decisions(imbalanced, sample_prices)

        df = engine.get_history_summary()

        assert len(df) == 3
        assert list(df.columns) == [
            "timestamp",
            "strategy_type",
            "decisions_count",
            "total_adjustment_value",
            "execution_status",
        ]

    def test_get_history_summary_with_limit(self, target_strategy, sample_positions, sample_prices):
        """Test getting limited history summary."""
        engine = RebalancingEngine(track_history=True)
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        # Generate 5 rebalancing events
        for _ in range(5):
            imbalanced = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}
            engine.generate_rebalancing_decisions(imbalanced, sample_prices)

        df = engine.get_history_summary(limit=2)

        assert len(df) == 2  # Should only return last 2

    def test_clear_history(self, target_strategy, sample_positions, sample_prices):
        """Test clearing rebalancing history."""
        engine = RebalancingEngine(track_history=True)
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        # Generate some history
        imbalanced = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}
        decisions = engine.generate_rebalancing_decisions(imbalanced, sample_prices)
        engine.execute_rebalancing(decisions)

        assert len(engine.history) > 0
        assert engine.last_rebalance_date is not None

        # Clear history
        engine.clear_history()

        assert len(engine.history) == 0
        assert engine.last_rebalance_date is None

    def test_get_registered_strategies(self, target_strategy):
        """Test getting list of registered strategies."""
        engine = RebalancingEngine()

        # Initially empty
        assert len(engine.get_registered_strategies()) == 0

        # Register strategies
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        strategies = engine.get_registered_strategies()
        assert len(strategies) == 1
        assert StrategyType.TARGET_ALLOCATION in strategies

    def test_multiple_strategy_registration(self, target_strategy):
        """Test registering multiple strategies."""
        engine = RebalancingEngine()

        # Register same strategy under different types (just for testing)
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)
        engine.register_strategy(StrategyType.THRESHOLD_BASED, target_strategy)

        strategies = engine.get_registered_strategies()
        assert len(strategies) == 2
        assert StrategyType.TARGET_ALLOCATION in strategies
        assert StrategyType.THRESHOLD_BASED in strategies

    def test_strategy_switching(self, target_strategy, sample_positions, sample_prices):
        """Test switching between registered strategies."""
        engine = RebalancingEngine(default_strategy=StrategyType.TARGET_ALLOCATION)

        # Register two strategies
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)
        engine.register_strategy(StrategyType.THRESHOLD_BASED, target_strategy)

        # Generate decisions with default strategy
        decisions1 = engine.generate_rebalancing_decisions(
            sample_positions, sample_prices, cash=5000.0
        )

        # Generate decisions with explicit strategy
        decisions2 = engine.generate_rebalancing_decisions(
            sample_positions, sample_prices, strategy_type=StrategyType.THRESHOLD_BASED, cash=5000.0
        )

        # Both should generate decisions (using same underlying strategy in this test)
        assert len(decisions1) > 0
        assert len(decisions2) > 0
