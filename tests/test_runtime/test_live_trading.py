"""Tests for Live Trading Runtime."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType
from ordinis.runtime.live_trading import LiveTradingRuntime


class TestLiveTradingRuntimeInit:
    """Tests for LiveTradingRuntime initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        mock_broker = MagicMock()
        mock_loader = MagicMock()

        runtime = LiveTradingRuntime(
            broker=mock_broker,
            strategy_loader=mock_loader,
        )

        assert runtime.broker == mock_broker
        assert runtime.loader == mock_loader
        assert runtime.mode == "paper"
        assert runtime.poll_interval == 60
        assert runtime.running is False
        assert runtime.daily_pnl == 0.0
        assert runtime.trade_count == 0

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        mock_broker = MagicMock()
        mock_loader = MagicMock()
        mock_data = MagicMock()

        runtime = LiveTradingRuntime(
            broker=mock_broker,
            strategy_loader=mock_loader,
            data_adapter=mock_data,
            mode="simulated",
            poll_interval=120,
        )

        assert runtime.mode == "simulated"
        assert runtime.poll_interval == 120
        assert runtime._data_adapter == mock_data

    def test_risk_limits(self):
        """Test risk limit defaults."""
        mock_broker = MagicMock()
        mock_loader = MagicMock()

        runtime = LiveTradingRuntime(
            broker=mock_broker,
            strategy_loader=mock_loader,
        )

        assert runtime.max_daily_loss_pct == 2.0
        assert runtime.max_position_pct == 5.0
        assert runtime.max_concurrent == 5


class TestLiveTradingRuntimeConnect:
    """Tests for connect method."""

    @pytest.fixture
    def runtime(self):
        """Create runtime with mocked dependencies."""
        mock_broker = MagicMock()
        mock_broker.connect = AsyncMock(return_value=True)
        mock_broker.get_account = AsyncMock(
            return_value=MagicMock(equity=100000.0)
        )
        mock_loader = MagicMock()

        return LiveTradingRuntime(
            broker=mock_broker,
            strategy_loader=mock_loader,
            mode="simulated",
        )

    @pytest.mark.asyncio
    async def test_connect_success(self, runtime):
        """Test successful connection."""
        result = await runtime.connect()

        assert result is True
        assert runtime.start_equity == 100000.0
        runtime.broker.connect.assert_called_once()
        runtime.broker.get_account.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure(self, runtime):
        """Test failed connection."""
        runtime.broker.connect.return_value = False

        result = await runtime.connect()

        assert result is False


class TestLiveTradingRuntimeDisconnect:
    """Tests for disconnect method."""

    @pytest.fixture
    def runtime(self):
        """Create runtime with mocked dependencies."""
        mock_broker = MagicMock()
        mock_broker.disconnect = AsyncMock()
        mock_loader = MagicMock()

        return LiveTradingRuntime(
            broker=mock_broker,
            strategy_loader=mock_loader,
        )

    @pytest.mark.asyncio
    async def test_disconnect(self, runtime):
        """Test disconnection."""
        await runtime.disconnect()

        runtime.broker.disconnect.assert_called_once()


class TestLiveTradingRuntimeRiskChecks:
    """Tests for risk check methods."""

    @pytest.fixture
    def runtime(self):
        """Create runtime with mocked dependencies."""
        mock_broker = MagicMock()
        mock_loader = MagicMock()

        runtime = LiveTradingRuntime(
            broker=mock_broker,
            strategy_loader=mock_loader,
        )
        runtime.start_equity = 100000.0
        return runtime

    def test_daily_pnl_tracking(self, runtime):
        """Test daily P&L tracking."""
        runtime.daily_pnl = -1500.0

        # Check if under limit
        loss_pct = abs(runtime.daily_pnl) / runtime.start_equity * 100
        assert loss_pct < runtime.max_daily_loss_pct

    def test_daily_loss_limit_exceeded(self, runtime):
        """Test when daily loss limit is exceeded."""
        runtime.daily_pnl = -2500.0

        loss_pct = abs(runtime.daily_pnl) / runtime.start_equity * 100
        assert loss_pct > runtime.max_daily_loss_pct


class TestLiveTradingRuntimeTrading:
    """Tests for trading operations."""

    @pytest.fixture
    def runtime(self):
        """Create runtime with mocked dependencies."""
        mock_broker = MagicMock()
        mock_broker.get_positions = AsyncMock(return_value=[])
        mock_broker.get_account = AsyncMock(
            return_value=MagicMock(equity=100000.0, buying_power=50000.0)
        )
        mock_loader = MagicMock()

        runtime = LiveTradingRuntime(
            broker=mock_broker,
            strategy_loader=mock_loader,
        )
        runtime.start_equity = 100000.0
        return runtime

    @pytest.mark.asyncio
    async def test_get_positions(self, runtime):
        """Test getting current positions."""
        positions = await runtime.broker.get_positions()

        assert positions == []

    @pytest.mark.asyncio
    async def test_get_account(self, runtime):
        """Test getting account info."""
        account = await runtime.broker.get_account()

        assert account.equity == 100000.0
        assert account.buying_power == 50000.0


class TestLiveTradingRuntimeState:
    """Tests for runtime state management."""

    @pytest.fixture
    def runtime(self):
        """Create runtime with mocked dependencies."""
        mock_broker = MagicMock()
        mock_loader = MagicMock()

        return LiveTradingRuntime(
            broker=mock_broker,
            strategy_loader=mock_loader,
        )

    def test_running_state(self, runtime):
        """Test running state."""
        assert runtime.running is False

        runtime.running = True
        assert runtime.running is True

    def test_trade_count(self, runtime):
        """Test trade count tracking."""
        assert runtime.trade_count == 0

        runtime.trade_count += 1
        assert runtime.trade_count == 1

    def test_daily_pnl_update(self, runtime):
        """Test daily P&L updates."""
        assert runtime.daily_pnl == 0.0

        runtime.daily_pnl += 500.0
        assert runtime.daily_pnl == 500.0

        runtime.daily_pnl -= 200.0
        assert runtime.daily_pnl == 300.0


class TestLiveTradingRuntimeGetMarketData:
    """Tests for get_market_data method."""

    @pytest.fixture
    def runtime(self):
        """Create runtime with mocked dependencies."""
        mock_broker = MagicMock()
        mock_loader = MagicMock()

        return LiveTradingRuntime(
            broker=mock_broker,
            strategy_loader=mock_loader,
        )

    @pytest.mark.asyncio
    async def test_get_market_data_no_adapter(self, runtime):
        """Test get_market_data returns None when no adapter."""
        result = await runtime.get_market_data("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_market_data_success(self, runtime):
        """Test successful market data fetch."""
        mock_data = MagicMock()
        mock_df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        mock_data.get_historical_bars.return_value = mock_df
        runtime._data_adapter = mock_data

        result = await runtime.get_market_data("AAPL", bars=50)

        assert result is not None
        assert len(result) == 3
        mock_data.get_historical_bars.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_market_data_empty_result(self, runtime):
        """Test get_market_data returns None when empty."""
        mock_data = MagicMock()
        mock_data.get_historical_bars.return_value = pd.DataFrame()
        runtime._data_adapter = mock_data

        result = await runtime.get_market_data("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_market_data_exception(self, runtime):
        """Test get_market_data handles exceptions."""
        mock_data = MagicMock()
        mock_data.get_historical_bars.side_effect = Exception("API error")
        runtime._data_adapter = mock_data

        result = await runtime.get_market_data("AAPL")

        assert result is None


class TestLiveTradingRuntimeProcessSymbol:
    """Tests for process_symbol method."""

    @pytest.fixture
    def runtime(self):
        """Create runtime with mocked dependencies."""
        mock_broker = MagicMock()
        mock_loader = MagicMock()

        runtime = LiveTradingRuntime(
            broker=mock_broker,
            strategy_loader=mock_loader,
        )
        return runtime

    @pytest.mark.asyncio
    async def test_process_symbol_no_model(self, runtime):
        """Test process_symbol returns None when no model."""
        runtime.loader.get_model.return_value = None

        result = await runtime.process_symbol("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_process_symbol_insufficient_data(self, runtime):
        """Test process_symbol returns None with insufficient data."""
        runtime.loader.get_model.return_value = MagicMock()

        # Mock get_market_data to return small DataFrame
        with patch.object(runtime, "get_market_data", new_callable=AsyncMock) as mock:
            mock.return_value = pd.DataFrame({"close": [100] * 10})

            result = await runtime.process_symbol("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_process_symbol_should_not_trade(self, runtime):
        """Test process_symbol skips when should not trade."""
        runtime.loader.get_model.return_value = MagicMock()
        runtime.loader.should_trade.return_value = (False, "Market closed")

        with patch.object(runtime, "get_market_data", new_callable=AsyncMock) as mock:
            mock.return_value = pd.DataFrame({"close": [100] * 60})

            result = await runtime.process_symbol("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_process_symbol_generates_signal(self, runtime):
        """Test process_symbol generates signal."""
        mock_model = MagicMock()
        mock_signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            confidence=0.85,
            timestamp=datetime.now(),
        )
        mock_model.generate = AsyncMock(return_value=mock_signal)
        runtime.loader.get_model.return_value = mock_model
        runtime.loader.should_trade.return_value = (True, "OK")

        with patch.object(runtime, "get_market_data", new_callable=AsyncMock) as mock:
            mock.return_value = pd.DataFrame({"close": [100] * 60})

            result = await runtime.process_symbol("AAPL")

        assert result is not None
        assert result.signal_type == SignalType.ENTRY

    @pytest.mark.asyncio
    async def test_process_symbol_hold_signal(self, runtime):
        """Test process_symbol returns None for HOLD signal."""
        mock_model = MagicMock()
        mock_signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.HOLD,
            direction=Direction.NEUTRAL,
            confidence=0.5,
            timestamp=datetime.now(),
        )
        mock_model.generate = AsyncMock(return_value=mock_signal)
        runtime.loader.get_model.return_value = mock_model
        runtime.loader.should_trade.return_value = (True, "OK")

        with patch.object(runtime, "get_market_data", new_callable=AsyncMock) as mock:
            mock.return_value = pd.DataFrame({"close": [100] * 60})

            result = await runtime.process_symbol("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_process_symbol_exception(self, runtime):
        """Test process_symbol handles exception in model.generate."""
        mock_model = MagicMock()
        mock_model.generate = AsyncMock(side_effect=Exception("Model error"))
        runtime.loader.get_model.return_value = mock_model
        runtime.loader.should_trade.return_value = (True, "OK")

        with patch.object(runtime, "get_market_data", new_callable=AsyncMock) as mock:
            mock.return_value = pd.DataFrame({"close": [100] * 60})

            result = await runtime.process_symbol("AAPL")

        assert result is None


class TestLiveTradingRuntimeCheckRiskLimits:
    """Tests for check_risk_limits method."""

    @pytest.fixture
    def runtime(self):
        """Create runtime with mocked dependencies."""
        mock_broker = MagicMock()
        mock_broker.get_account = AsyncMock(
            return_value=MagicMock(equity=100000.0)
        )
        mock_broker.get_positions = AsyncMock(return_value=[])
        mock_loader = MagicMock()

        runtime = LiveTradingRuntime(
            broker=mock_broker,
            strategy_loader=mock_loader,
        )
        runtime.start_equity = 100000.0
        return runtime

    @pytest.mark.asyncio
    async def test_check_risk_limits_pass(self, runtime):
        """Test risk limits pass with no positions."""
        signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            confidence=0.85,
            timestamp=datetime.now(),
        )

        result = await runtime.check_risk_limits("AAPL", signal)

        assert result is True

    @pytest.mark.asyncio
    async def test_check_risk_limits_max_positions(self, runtime):
        """Test risk limits fail when max positions reached."""
        runtime.broker.get_positions.return_value = [
            MagicMock() for _ in range(5)
        ]
        signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            confidence=0.85,
            timestamp=datetime.now(),
        )

        result = await runtime.check_risk_limits("AAPL", signal)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_risk_limits_daily_loss(self, runtime):
        """Test risk limits fail when daily loss exceeded."""
        runtime.broker.get_account.return_value = MagicMock(equity=97000.0)
        signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            confidence=0.85,
            timestamp=datetime.now(),
        )

        result = await runtime.check_risk_limits("AAPL", signal)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_risk_limits_existing_position(self, runtime):
        """Test risk limits fail when position exists in symbol."""
        runtime.broker.get_positions.return_value = {"AAPL": MagicMock()}
        signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            confidence=0.85,
            timestamp=datetime.now(),
        )

        result = await runtime.check_risk_limits("AAPL", signal)

        assert result is False


class TestLiveTradingRuntimeExecuteSignal:
    """Tests for execute_signal method."""

    @pytest.fixture
    def runtime(self):
        """Create runtime with mocked dependencies."""
        mock_broker = MagicMock()
        mock_broker.get_account = AsyncMock(
            return_value=MagicMock(equity=100000.0)
        )
        mock_broker.submit_order = AsyncMock(
            return_value=MagicMock(order_id="order_123")
        )
        mock_loader = MagicMock()
        mock_loader.get_risk_params.return_value = {"max_position_pct": 5.0}

        runtime = LiveTradingRuntime(
            broker=mock_broker,
            strategy_loader=mock_loader,
        )
        runtime.start_equity = 100000.0
        return runtime

    @pytest.mark.asyncio
    async def test_execute_signal_risk_check_fail(self, runtime):
        """Test execute_signal fails when risk check fails."""
        signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            confidence=0.85,
            timestamp=datetime.now(),
        )

        with patch.object(
            runtime, "check_risk_limits", new_callable=AsyncMock
        ) as mock_risk:
            mock_risk.return_value = False

            result = await runtime.execute_signal("AAPL", signal)

        assert result is False

    @pytest.mark.asyncio
    async def test_execute_signal_no_price(self, runtime):
        """Test execute_signal fails when no price available."""
        signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            confidence=0.85,
            timestamp=datetime.now(),
            metadata={},
        )

        with patch.object(
            runtime, "check_risk_limits", new_callable=AsyncMock
        ) as mock_risk:
            mock_risk.return_value = True

            result = await runtime.execute_signal("AAPL", signal)

        assert result is False

    @pytest.mark.asyncio
    async def test_execute_signal_long_entry(self, runtime):
        """Test execute_signal places buy order for long entry."""
        signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            confidence=0.85,
            timestamp=datetime.now(),
            metadata={"entry_price": 150.0},
        )

        with patch.object(
            runtime, "check_risk_limits", new_callable=AsyncMock
        ) as mock_risk:
            mock_risk.return_value = True

            result = await runtime.execute_signal("AAPL", signal)

        assert result is True
        assert runtime.trade_count == 1
        runtime.broker.submit_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_signal_short_entry(self, runtime):
        """Test execute_signal places sell order for short entry."""
        signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            direction=Direction.SHORT,
            confidence=0.85,
            timestamp=datetime.now(),
            metadata={"entry_price": 150.0},
        )

        with patch.object(
            runtime, "check_risk_limits", new_callable=AsyncMock
        ) as mock_risk:
            mock_risk.return_value = True

            result = await runtime.execute_signal("AAPL", signal)

        assert result is True
        assert runtime.trade_count == 1

    @pytest.mark.asyncio
    async def test_execute_signal_exit_returns_false(self, runtime):
        """Test execute_signal returns False for EXIT signals."""
        signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.EXIT,
            direction=Direction.NEUTRAL,
            confidence=0.85,
            timestamp=datetime.now(),
            metadata={"entry_price": 150.0},
        )

        with patch.object(
            runtime, "check_risk_limits", new_callable=AsyncMock
        ) as mock_risk:
            mock_risk.return_value = True

            result = await runtime.execute_signal("AAPL", signal)

        assert result is False

    @pytest.mark.asyncio
    async def test_execute_signal_neutral_direction(self, runtime):
        """Test execute_signal returns False for neutral direction."""
        signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            direction=Direction.NEUTRAL,
            confidence=0.85,
            timestamp=datetime.now(),
            metadata={"entry_price": 150.0},
        )

        with patch.object(
            runtime, "check_risk_limits", new_callable=AsyncMock
        ) as mock_risk:
            mock_risk.return_value = True

            result = await runtime.execute_signal("AAPL", signal)

        assert result is False

    @pytest.mark.asyncio
    async def test_execute_signal_order_exception(self, runtime):
        """Test execute_signal handles order exception."""
        signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            confidence=0.85,
            timestamp=datetime.now(),
            metadata={"entry_price": 150.0},
        )
        runtime.broker.submit_order.side_effect = Exception("Order failed")

        with patch.object(
            runtime, "check_risk_limits", new_callable=AsyncMock
        ) as mock_risk:
            mock_risk.return_value = True

            result = await runtime.execute_signal("AAPL", signal)

        assert result is False


class TestLiveTradingRuntimeManagePositions:
    """Tests for manage_positions method."""

    @pytest.fixture
    def runtime(self):
        """Create runtime with mocked dependencies."""
        mock_broker = MagicMock()
        mock_broker.get_positions = AsyncMock(return_value=[])
        mock_broker.submit_order = AsyncMock(
            return_value=MagicMock(order_id="order_123")
        )
        mock_loader = MagicMock()

        runtime = LiveTradingRuntime(
            broker=mock_broker,
            strategy_loader=mock_loader,
        )
        return runtime

    @pytest.mark.asyncio
    async def test_manage_positions_no_positions(self, runtime):
        """Test manage_positions with no positions."""
        await runtime.manage_positions()

        runtime.broker.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_manage_positions_no_model(self, runtime):
        """Test manage_positions skips symbol with no model."""
        runtime.broker.get_positions.return_value = [
            MagicMock(symbol="AAPL", quantity=100)
        ]
        runtime.loader.get_model.return_value = None

        await runtime.manage_positions()

        runtime.broker.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_manage_positions_exit_signal(self, runtime):
        """Test manage_positions exits position on EXIT signal."""
        position = MagicMock(symbol="AAPL", quantity=100)
        runtime.broker.get_positions.return_value = [position]

        mock_model = MagicMock()
        exit_signal = Signal(
            symbol="AAPL",
            signal_type=SignalType.EXIT,
            direction=Direction.NEUTRAL,
            confidence=0.85,
            timestamp=datetime.now(),
        )
        mock_model.generate = AsyncMock(return_value=exit_signal)
        runtime.loader.get_model.return_value = mock_model

        with patch.object(runtime, "get_market_data", new_callable=AsyncMock) as mock:
            mock.return_value = pd.DataFrame({"close": [100] * 60})

            await runtime.manage_positions()

        runtime.broker.submit_order.assert_called_once()
        assert runtime.trade_count == 1


class TestLiveTradingRuntimeStop:
    """Tests for stop method."""

    def test_stop(self):
        """Test stop method sets running to False."""
        mock_broker = MagicMock()
        mock_loader = MagicMock()

        runtime = LiveTradingRuntime(
            broker=mock_broker,
            strategy_loader=mock_loader,
        )
        runtime.running = True

        runtime.stop()

        assert runtime.running is False
