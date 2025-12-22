"""
Tests for portfolio rebalancing integration:
- Event hooks
- SignalCore adapter
- ProofBench adapter
- FlowRoute adapter
"""

from datetime import UTC, datetime

import pytest

from ordinis.engines.portfolio import (
    EventHooks,
    FlowRouteAdapter,
    FlowRouteOrderRequest,
    PortfolioEngineConfig,
    ProofBenchAdapter,
    RebalanceEvent,
    RebalanceEventType,
    RebalancingEngine,
    SignalCoreAdapter,
    StrategyType,
    TargetAllocation,
    TargetAllocationRebalancer,
    ThresholdDecision,
)


class TestEventHooks:
    """Tests for EventHooks class."""

    def test_register_handler(self):
        """Test registering event handler."""
        hooks = EventHooks()
        events_received = []

        def handler(event):
            events_received.append(event)

        hooks.register(RebalanceEventType.REBALANCE_STARTED, handler)

        event = RebalanceEvent(
            timestamp=datetime.now(tz=UTC),
            event_type=RebalanceEventType.REBALANCE_STARTED,
        )
        hooks.emit(event)

        assert len(events_received) == 1
        assert events_received[0] == event

    def test_register_multiple_handlers(self):
        """Test registering multiple handlers for same event."""
        hooks = EventHooks()
        events_1 = []
        events_2 = []

        hooks.register(RebalanceEventType.REBALANCE_STARTED, lambda e: events_1.append(e))
        hooks.register(RebalanceEventType.REBALANCE_STARTED, lambda e: events_2.append(e))

        event = RebalanceEvent(
            timestamp=datetime.now(tz=UTC),
            event_type=RebalanceEventType.REBALANCE_STARTED,
        )
        hooks.emit(event)

        assert len(events_1) == 1
        assert len(events_2) == 1

    def test_register_global_handler(self):
        """Test registering global handler for all events."""
        hooks = EventHooks()
        events_received = []

        hooks.register_global(lambda e: events_received.append(e))

        # Emit different event types
        hooks.emit(
            RebalanceEvent(
                timestamp=datetime.now(tz=UTC),
                event_type=RebalanceEventType.REBALANCE_STARTED,
            )
        )
        hooks.emit(
            RebalanceEvent(
                timestamp=datetime.now(tz=UTC),
                event_type=RebalanceEventType.DECISIONS_GENERATED,
            )
        )

        assert len(events_received) == 2

    def test_unregister_handler(self):
        """Test unregistering event handler."""
        hooks = EventHooks()
        events_received = []

        def handler(event):
            events_received.append(event)

        hooks.register(RebalanceEventType.REBALANCE_STARTED, handler)
        hooks.unregister(RebalanceEventType.REBALANCE_STARTED, handler)

        event = RebalanceEvent(
            timestamp=datetime.now(tz=UTC),
            event_type=RebalanceEventType.REBALANCE_STARTED,
        )
        hooks.emit(event)

        assert len(events_received) == 0

    def test_has_handlers(self):
        """Test checking if handlers are registered."""
        hooks = EventHooks()

        assert not hooks.has_handlers(RebalanceEventType.REBALANCE_STARTED)

        hooks.register(RebalanceEventType.REBALANCE_STARTED, lambda e: None)

        assert hooks.has_handlers(RebalanceEventType.REBALANCE_STARTED)

    def test_get_handler_count(self):
        """Test getting handler count."""
        hooks = EventHooks()

        assert hooks.get_handler_count(RebalanceEventType.REBALANCE_STARTED) == 0

        hooks.register(RebalanceEventType.REBALANCE_STARTED, lambda e: None)
        hooks.register(RebalanceEventType.REBALANCE_STARTED, lambda e: None)

        assert hooks.get_handler_count(RebalanceEventType.REBALANCE_STARTED) == 2
        assert hooks.get_handler_count() == 2

    def test_clear_handlers(self):
        """Test clearing handlers."""
        hooks = EventHooks()

        hooks.register(RebalanceEventType.REBALANCE_STARTED, lambda e: None)
        hooks.register(RebalanceEventType.DECISIONS_GENERATED, lambda e: None)

        hooks.clear(RebalanceEventType.REBALANCE_STARTED)

        assert hooks.get_handler_count(RebalanceEventType.REBALANCE_STARTED) == 0
        assert hooks.get_handler_count(RebalanceEventType.DECISIONS_GENERATED) == 1

    def test_clear_all_handlers(self):
        """Test clearing all handlers."""
        hooks = EventHooks()

        hooks.register(RebalanceEventType.REBALANCE_STARTED, lambda e: None)
        hooks.register(RebalanceEventType.DECISIONS_GENERATED, lambda e: None)
        hooks.register_global(lambda e: None)

        hooks.clear()

        assert hooks.get_handler_count() == 0

    def test_handler_error_handling(self):
        """Test that handler errors don't stop other handlers."""
        hooks = EventHooks()
        events_received = []

        def failing_handler(event):
            raise RuntimeError("Handler error")

        def success_handler(event):
            events_received.append(event)

        hooks.register(RebalanceEventType.REBALANCE_STARTED, failing_handler)
        hooks.register(RebalanceEventType.REBALANCE_STARTED, success_handler)

        event = RebalanceEvent(
            timestamp=datetime.now(tz=UTC),
            event_type=RebalanceEventType.REBALANCE_STARTED,
        )
        hooks.emit(event)

        # Success handler should still run despite failing handler
        assert len(events_received) == 1

    def test_register_invalid_handler(self):
        """Test registering non-callable handler fails."""
        hooks = EventHooks()

        with pytest.raises(ValueError, match="must be callable"):
            hooks.register(RebalanceEventType.REBALANCE_STARTED, "not_callable")


class TestRebalanceEvent:
    """Tests for RebalanceEvent dataclass."""

    def test_create_event(self):
        """Test creating rebalance event."""
        timestamp = datetime.now(tz=UTC)
        event = RebalanceEvent(
            timestamp=timestamp,
            event_type=RebalanceEventType.REBALANCE_STARTED,
            strategy_type=StrategyType.TARGET_ALLOCATION,
            data={"positions": {"AAPL": 10}},
            metadata={"test": "value"},
        )

        assert event.timestamp == timestamp
        assert event.event_type == RebalanceEventType.REBALANCE_STARTED
        assert event.strategy_type == StrategyType.TARGET_ALLOCATION
        assert event.data == {"positions": {"AAPL": 10}}
        assert event.metadata == {"test": "value"}

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = RebalanceEvent(
            timestamp=datetime(2023, 12, 13, tzinfo=UTC),
            event_type=RebalanceEventType.REBALANCE_STARTED,
            strategy_type=StrategyType.TARGET_ALLOCATION,
            data={"test": "data"},
        )

        event_dict = event.to_dict()

        assert event_dict["timestamp"] == "2023-12-13T00:00:00+00:00"
        assert event_dict["event_type"] == "rebalance_started"
        assert event_dict["strategy_type"] == "target_allocation"
        assert event_dict["data"] == {"test": "data"}


class TestRebalancingEngineWithEvents:
    """Tests for RebalancingEngine with event hooks."""

    @pytest.fixture
    def target_strategy(self):
        """Fixture: Target allocation strategy."""
        targets = [
            TargetAllocation("AAPL", 0.40),
            TargetAllocation("MSFT", 0.30),
            TargetAllocation("GOOGL", 0.30),
        ]
        return TargetAllocationRebalancer(targets, drift_threshold=0.05)

    @pytest.fixture
    def sample_positions(self):
        """Fixture: Sample positions."""
        return {"AAPL": 30, "MSFT": 5, "GOOGL": 5}

    @pytest.fixture
    def sample_prices(self):
        """Fixture: Sample prices."""
        return {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 500.0}

    @pytest.mark.asyncio
    async def test_engine_emits_rebalance_started(self, target_strategy, sample_positions, sample_prices):
        """Test engine emits REBALANCE_STARTED event."""
        events_received = []

        config = PortfolioEngineConfig(enable_governance=False)
        engine = RebalancingEngine(config)
        engine.event_hooks.register(RebalanceEventType.REBALANCE_STARTED, lambda e: events_received.append(e))
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        await engine.generate_rebalancing_decisions(sample_positions, sample_prices)

        assert len(events_received) == 1
        assert events_received[0].event_type == RebalanceEventType.REBALANCE_STARTED

    @pytest.mark.asyncio
    async def test_engine_emits_decisions_generated(
        self, target_strategy, sample_positions, sample_prices
    ):
        """Test engine emits DECISIONS_GENERATED event."""
        events_received = []

        config = PortfolioEngineConfig(enable_governance=False)
        engine = RebalancingEngine(config)
        engine.event_hooks.register(RebalanceEventType.DECISIONS_GENERATED, lambda e: events_received.append(e))
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        await engine.generate_rebalancing_decisions(sample_positions, sample_prices)

        assert len(events_received) == 1
        assert events_received[0].event_type == RebalanceEventType.DECISIONS_GENERATED

    @pytest.mark.asyncio
    async def test_engine_emits_execution_events(self, target_strategy, sample_positions, sample_prices):
        """Test engine emits execution events."""
        events_received = []

        config = PortfolioEngineConfig(enable_governance=False)
        engine = RebalancingEngine(config)
        engine.event_hooks.register_global(lambda e: events_received.append(e))
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        decisions = await engine.generate_rebalancing_decisions(sample_positions, sample_prices)

        # Execute decisions
        def success_callback(decision):
            return (True, None)

        await engine.execute_rebalancing(decisions, execution_callback=success_callback)

        # Should have: REBALANCE_STARTED, DECISIONS_GENERATED, EXECUTION_STARTED,
        # ORDER_EXECUTED (x3), REBALANCE_COMPLETED
        assert (
            len(
                [e for e in events_received if e.event_type == RebalanceEventType.EXECUTION_STARTED]
            )
            == 1
        )
        assert (
            len([e for e in events_received if e.event_type == RebalanceEventType.ORDER_EXECUTED])
            == 3
        )
        assert (
            len(
                [
                    e
                    for e in events_received
                    if e.event_type == RebalanceEventType.REBALANCE_COMPLETED
                ]
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_engine_emits_order_failed(self, target_strategy, sample_positions, sample_prices):
        """Test engine emits ORDER_FAILED events."""
        failed_events = []

        config = PortfolioEngineConfig(enable_governance=False)
        engine = RebalancingEngine(config)
        engine.event_hooks.register(RebalanceEventType.ORDER_FAILED, lambda e: failed_events.append(e))
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

        decisions = await engine.generate_rebalancing_decisions(sample_positions, sample_prices)

        # Failing callback
        def failing_callback(decision):
            return (False, "Order rejected")

        await engine.execute_rebalancing(decisions, execution_callback=failing_callback)

        assert len(failed_events) == 3  # All 3 orders failed


class TestSignalCoreAdapter:
    """Tests for SignalCore integration adapter."""

    @pytest.mark.skip(reason="SignalCore availability check removed - adapter always available")
    def test_adapter_imports_fail_without_signalcore(self, monkeypatch):
        """Test adapter raises ImportError without SignalCore."""
        # Legacy test - adapter no longer has availability checking
        pass

    def test_convert_signal(self):
        """Test converting SignalCore signal to SignalInput."""
        from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType

        adapter = SignalCoreAdapter()

        signal = Signal(
            symbol="AAPL",
            timestamp=datetime.now(tz=UTC),
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            probability=0.75,
            expected_return=0.05,
            confidence_interval=(0.02, 0.08),
            score=0.6,
            model_id="test_model",
            model_version="1.0",
        )

        signal_input = adapter.convert_signal(signal)

        assert signal_input.symbol == "AAPL"
        assert -1.0 <= signal_input.signal <= 1.0  # Should be in range
        assert signal_input.confidence == 0.75  # From probability
        assert signal_input.source == "test_model"  # From model_id

    def test_convert_signal_custom_scaling(self):
        """Test signal conversion with custom scaling."""
        from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType

        adapter = SignalCoreAdapter(probability_weight=0.7, scale_to_range=(0.0, 1.0))

        signal = Signal(
            symbol="MSFT",
            timestamp=datetime.now(tz=UTC),
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            probability=0.8,
            expected_return=0.03,
            confidence_interval=(0.01, 0.05),
            score=0.4,
            model_id="value_model",
            model_version="1.0",
        )

        signal_input = adapter.convert_signal(signal)

        assert signal_input.symbol == "MSFT"
        assert 0.0 <= signal_input.signal <= 1.0  # Custom range
        assert signal_input.confidence == 0.8

    def test_convert_batch(self):
        """Test converting SignalBatch to list of SignalInput."""
        from ordinis.engines.signalcore.core.signal import (
            Direction,
            Signal,
            SignalBatch,
            SignalType,
        )

        adapter = SignalCoreAdapter()

        signals = [
            Signal(
                symbol=sym,
                timestamp=datetime.now(tz=UTC),
                signal_type=SignalType.ENTRY,
                direction=Direction.LONG,
                probability=0.7,
                expected_return=0.04,
                confidence_interval=(0.01, 0.07),
                score=0.5,
                model_id="test",
                model_version="1.0",
            )
            for sym in ["AAPL", "MSFT", "GOOGL"]
        ]

        batch = SignalBatch(
            timestamp=datetime.now(tz=UTC), signals=signals, universe=["AAPL", "MSFT", "GOOGL"]
        )

        signal_inputs = adapter.convert_batch(batch)

        assert len(signal_inputs) == 3
        assert all(s.symbol in ["AAPL", "MSFT", "GOOGL"] for s in signal_inputs)

    def test_filter_actionable(self):
        """Test filtering actionable signals."""
        from ordinis.engines.signalcore.core.signal import (
            Direction,
            Signal,
            SignalBatch,
            SignalType,
        )

        adapter = SignalCoreAdapter()

        signals = [
            Signal(
                symbol="AAPL",
                timestamp=datetime.now(tz=UTC),
                signal_type=SignalType.ENTRY,
                direction=Direction.LONG,
                probability=0.8,  # Actionable
                expected_return=0.05,
                confidence_interval=(0.02, 0.08),
                score=0.6,
                model_id="test",
                model_version="1.0",
            ),
            Signal(
                symbol="MSFT",
                timestamp=datetime.now(tz=UTC),
                signal_type=SignalType.ENTRY,
                direction=Direction.LONG,
                probability=0.4,  # Not actionable
                expected_return=0.02,
                confidence_interval=(0.0, 0.04),
                score=0.2,
                model_id="test",
                model_version="1.0",
            ),
        ]

        batch = SignalBatch(
            timestamp=datetime.now(tz=UTC), signals=signals, universe=["AAPL", "MSFT"]
        )

        actionable = adapter.filter_actionable(batch, min_probability=0.6, min_score=0.3)

        assert len(actionable) == 1
        assert actionable[0].symbol == "AAPL"


class TestProofBenchAdapter:
    """Tests for ProofBench integration adapter."""

    @pytest.mark.skip(reason="ProofBench availability check removed - adapter always available")
    def test_adapter_imports_fail_without_proofbench(self, monkeypatch):
        """Test adapter raises ImportError without ProofBench."""
        # Legacy test - adapter no longer has availability checking
        pass

    def test_emit_rebalance_started(self):
        """Test emitting rebalance started event."""
        hooks = EventHooks()
        events_received = []
        hooks.register(RebalanceEventType.REBALANCE_STARTED, lambda e: events_received.append(e))

        adapter = ProofBenchAdapter(hooks)

        positions = {"AAPL": 10}
        prices = {"AAPL": 200.0}

        adapter.emit_rebalance_started(
            datetime.now(tz=UTC), StrategyType.TARGET_ALLOCATION, positions, prices
        )

        assert len(events_received) == 1
        assert events_received[0].event_type == RebalanceEventType.REBALANCE_STARTED
        assert events_received[0].data["positions"] == positions

    def test_emit_decisions_generated(self):
        """Test emitting decisions generated event."""
        hooks = EventHooks()
        events_received = []
        hooks.register(RebalanceEventType.DECISIONS_GENERATED, lambda e: events_received.append(e))

        adapter = ProofBenchAdapter(hooks)

        decisions = [
            ThresholdDecision("AAPL", 0.5, 0.4, 0.1, 5.0, 1000.0, "test", datetime.now(tz=UTC))
        ]

        adapter.emit_decisions_generated(
            datetime.now(tz=UTC), StrategyType.THRESHOLD_BASED, decisions
        )

        assert len(events_received) == 1
        assert events_received[0].data["decisions_count"] == 1

    def test_emit_rebalance_completed(self):
        """Test emitting rebalance completed event."""
        hooks = EventHooks()
        events_received = []
        hooks.register(RebalanceEventType.REBALANCE_COMPLETED, lambda e: events_received.append(e))

        adapter = ProofBenchAdapter(hooks)

        adapter.emit_rebalance_completed(
            datetime.now(tz=UTC),
            StrategyType.TARGET_ALLOCATION,
            executed=3,
            failed=0,
            total_traded=5000.0,
        )

        assert len(events_received) == 1
        assert events_received[0].data["executed"] == 3
        assert events_received[0].data["success"] is True


class TestFlowRouteAdapter:
    """Tests for FlowRoute integration adapter."""

    def test_create_order(self):
        """Test creating order request from decision."""
        adapter = FlowRouteAdapter(account_id="TEST_ACCOUNT")

        decision = ThresholdDecision(
            symbol="AAPL",
            current_weight=0.5,
            target_weight=0.4,
            drift=0.1,
            adjustment_shares=-5.0,
            adjustment_value=-1000.0,
            trigger_reason="Above upper band",
            timestamp=datetime.now(tz=UTC),
        )

        order = adapter.create_order(decision)

        assert order.symbol == "AAPL"
        assert order.shares == -5.0
        assert order.order_type == "market"
        assert order.metadata["account_id"] == "TEST_ACCOUNT"
        assert order.metadata["trigger"] == "Above upper band"

    def test_create_orders_batch(self):
        """Test creating multiple order requests."""
        adapter = FlowRouteAdapter()

        decisions = [
            ThresholdDecision("AAPL", 0.5, 0.4, 0.1, -5.0, -1000.0, "test", datetime.now(tz=UTC)),
            ThresholdDecision("MSFT", 0.2, 0.3, -0.1, 3.0, 900.0, "test", datetime.now(tz=UTC)),
        ]

        orders = adapter.create_orders(decisions)

        assert len(orders) == 2
        assert orders[0].symbol == "AAPL"
        assert orders[1].symbol == "MSFT"

    def test_filter_min_value(self):
        """Test filtering orders by minimum value."""
        adapter = FlowRouteAdapter()

        orders = [
            FlowRouteOrderRequest(symbol="AAPL", shares=5.0),  # 5 * 200 = 1000
            FlowRouteOrderRequest(symbol="MSFT", shares=1.0),  # 1 * 300 = 300
            FlowRouteOrderRequest(symbol="GOOGL", shares=0.1),  # 0.1 * 500 = 50
        ]

        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 500.0}

        filtered = adapter.filter_min_value(orders, min_value=500.0, prices=prices)

        assert len(filtered) == 1
        assert filtered[0].symbol == "AAPL"

    def test_order_request_metadata(self):
        """Test order request with metadata."""
        adapter = FlowRouteAdapter(include_metadata=True)

        decision = ThresholdDecision(
            symbol="AAPL",
            current_weight=0.5,
            target_weight=0.4,
            drift=0.1,
            adjustment_shares=-5.0,
            adjustment_value=-1000.0,
            trigger_reason="Time threshold",
            timestamp=datetime.now(tz=UTC),
        )

        order = adapter.create_order(decision)

        assert order.metadata["source"] == "portfolio_rebalancing"
        assert order.metadata["trigger"] == "Time threshold"
        assert order.metadata["drift"] == 0.1

    def test_order_request_no_metadata(self):
        """Test order request without metadata."""
        adapter = FlowRouteAdapter(include_metadata=False)

        decision = ThresholdDecision(
            symbol="AAPL",
            current_weight=0.5,
            target_weight=0.4,
            drift=0.1,
            adjustment_shares=-5.0,
            adjustment_value=-1000.0,
            trigger_reason="Test",
            timestamp=datetime.now(tz=UTC),
        )

        order = adapter.create_order(decision)

        assert "source" not in order.metadata
        assert "trigger" not in order.metadata
