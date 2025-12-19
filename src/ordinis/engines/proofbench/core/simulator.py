"""Simulation engine for backtesting strategies.

Coordinates event-driven simulation with portfolio management,
execution simulation, and performance analytics.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from ..analytics.performance import PerformanceAnalyzer, PerformanceMetrics
from .events import Event, EventQueue, EventType
from .execution import Bar, ExecutionConfig, ExecutionSimulator, Order, OrderStatus
from .portfolio import Portfolio


@dataclass
class SimulationConfig:
    """Configuration for simulation engine.

    Attributes:
        initial_capital: Starting capital
        execution_config: Execution simulator configuration
        bar_frequency: Bar frequency (e.g., '1min', '1h', '1d')
        record_equity_frequency: How often to record equity (bars)
        enable_logging: Enable detailed logging
        risk_free_rate: Annual risk-free rate for metrics
    """

    initial_capital: float = 100000.0
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    bar_frequency: str = "1d"
    record_equity_frequency: int = 1
    enable_logging: bool = False
    risk_free_rate: float = 0.02

    def __post_init__(self):
        """Validate configuration."""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.record_equity_frequency <= 0:
            raise ValueError("Record frequency must be positive")


@dataclass
class SimulationResults:
    """Results from a backtest simulation.

    Attributes:
        config: Simulation configuration used
        metrics: Performance metrics
        portfolio: Final portfolio state
        equity_curve: Equity curve data
        trades: List of all trades
        orders: List of all orders
        start_time: Simulation start time
        end_time: Simulation end time
    """

    config: SimulationConfig
    metrics: PerformanceMetrics
    portfolio: Portfolio
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    orders: list[Order]
    start_time: datetime
    end_time: datetime


class SimulationEngine:
    """Event-driven simulation engine for backtesting.

    Runs strategies through historical data with realistic execution modeling.
    Coordinates events, portfolio updates, and performance tracking.
    """

    def __init__(self, config: SimulationConfig | None = None):
        """Initialize simulation engine.

        Args:
            config: Simulation configuration (uses defaults if None)
        """
        self.config = config or SimulationConfig()
        self.event_queue = EventQueue()
        self.portfolio = Portfolio(self.config.initial_capital)
        self.executor = ExecutionSimulator(self.config.execution_config)
        self.analyzer = PerformanceAnalyzer(self.config.risk_free_rate)

        self.data: dict[str, pd.DataFrame] = {}
        self.current_time: datetime | None = None
        self.bar_count = 0
        self.pending_orders: list[Order] = []
        self.all_orders: list[Order] = []

        # Strategy callback
        self.on_bar: Callable | None = None

    def load_data(self, symbol: str, data: pd.DataFrame):
        """Load historical data for a symbol.

        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV columns and datetime index
        """
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must have columns: {required_cols}")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex")

        self.data[symbol] = data.sort_index()

    def set_strategy(self, strategy_callback: Callable):
        """Set strategy callback function.

        The callback is called on each bar with (engine, symbol, bar).
        It should submit orders using engine.submit_order().

        Args:
            strategy_callback: Function to call on each bar
        """
        self.on_bar = strategy_callback

    def submit_order(self, order: Order):
        """Submit an order for execution.

        Args:
            order: Order to submit
        """
        if order.timestamp is None:
            order.timestamp = self.current_time

        order.status = OrderStatus.SUBMITTED
        self.pending_orders.append(order)
        self.all_orders.append(order)

        # Create order submit event
        if self.current_time is None:
            raise RuntimeError("Cannot submit order before simulation starts")

        event = Event(
            timestamp=self.current_time,
            event_type=EventType.ORDER_SUBMIT,
            data={"order": order},
            priority=5,
        )
        self.event_queue.push(event)

    def run(self, start: datetime | None = None, end: datetime | None = None) -> SimulationResults:
        """Run the backtest simulation.

        Args:
            start: Start date (uses data start if None)
            end: End date (uses data end if None)

        Returns:
            SimulationResults object
        """
        if not self.data:
            raise ValueError("No data loaded")

        if self.on_bar is None:
            raise ValueError("No strategy set")

        # Determine time range
        all_dates = []
        for df in self.data.values():
            all_dates.extend(df.index.tolist())

        if not all_dates:
            raise ValueError("No data available")

        all_dates = sorted(set(all_dates))
        start_time = start or all_dates[0]
        end_time = end or all_dates[-1]

        # Reset state
        self._reset()

        # Generate market events
        self._generate_market_events(start_time, end_time)

        # Run event loop
        while not self.event_queue.is_empty():
            event = self.event_queue.pop()

            if event is None:
                break

            if event.timestamp > end_time:
                break

            self.current_time = event.timestamp
            self._process_event(event)

        # Finalize and return results
        return self._finalize_results(start_time, end_time)

    def _reset(self):
        """Reset simulation state."""
        self.event_queue.clear()
        self.portfolio = Portfolio(self.config.initial_capital)
        self.pending_orders = []
        self.all_orders = []
        self.bar_count = 0
        self.current_time = None

    def _generate_market_events(self, start: datetime, end: datetime):
        """Generate bar update events for all symbols.

        Args:
            start: Start date
            end: End date
        """
        for symbol, df in self.data.items():
            # Filter to date range
            mask = (df.index >= start) & (df.index <= end)
            symbol_data = df[mask]

            for timestamp in symbol_data.index:
                bar = Bar(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=symbol_data.loc[timestamp, "open"],
                    high=symbol_data.loc[timestamp, "high"],
                    low=symbol_data.loc[timestamp, "low"],
                    close=symbol_data.loc[timestamp, "close"],
                    volume=int(symbol_data.loc[timestamp, "volume"]),
                )

                event = Event(
                    timestamp=timestamp,
                    event_type=EventType.BAR_UPDATE,
                    data={"symbol": symbol, "bar": bar},
                    priority=10,
                )
                self.event_queue.push(event)

    def _process_event(self, event: Event):
        """Process a simulation event.

        Args:
            event: Event to process
        """
        if event.event_type == EventType.BAR_UPDATE:
            self._process_bar_update(event)
        elif event.event_type == EventType.ORDER_SUBMIT:
            self._process_order_submit(event)
        elif event.event_type == EventType.ORDER_FILL:
            self._process_order_fill(event)

        # Record equity periodically
        self.bar_count += 1
        if self.bar_count % self.config.record_equity_frequency == 0:
            if self.current_time is None:
                raise RuntimeError("Current time not set during simulation")
            self.portfolio.record_equity(self.current_time)

    def _process_bar_update(self, event: Event):
        """Process bar update event.

        Args:
            event: Bar update event
        """
        symbol = event.data["symbol"]
        bar = event.data["bar"]

        # Update position prices (mark-to-market)
        if self.current_time is None:
            raise RuntimeError("Current time not set during bar update")
        self.portfolio.update_prices({symbol: bar.close}, self.current_time)

        # Try to fill pending orders
        self._try_fill_orders(bar)

        # Call strategy
        if self.on_bar:
            self.on_bar(self, symbol, bar)

    def _process_order_submit(self, event: Event):
        """Process order submit event.

        Args:
            event: Order submit event
        """
        # Order is already in pending_orders list

    def _process_order_fill(self, event: Event):
        """Process order fill event.

        Args:
            event: Order fill event
        """
        fill = event.data["fill"]
        order = event.data["order"]

        # Update portfolio
        if self.current_time is None:
            raise RuntimeError("Current time not set during order fill")
        self.portfolio.update_position(fill, self.current_time)

        # Update order status
        order.filled_quantity += fill.quantity
        if fill.quantity > 0:
            # Update average fill price
            total_value = order.avg_fill_price * (order.filled_quantity - fill.quantity)
            total_value += fill.price * fill.quantity
            order.avg_fill_price = total_value / order.filled_quantity

        if order.is_filled:
            order.status = OrderStatus.FILLED
        elif order.is_partial:
            order.status = OrderStatus.PARTIAL

    def _try_fill_orders(self, bar: Bar):
        """Try to fill pending orders.

        Args:
            bar: Current bar data
        """
        still_pending = []

        for order in self.pending_orders:
            if order.symbol != bar.symbol:
                still_pending.append(order)
                continue

            # Try to fill
            if self.current_time is None:
                raise RuntimeError("Current time not set during order fill attempt")

            fill = self.executor.simulate_fill(order, bar, self.current_time)

            if fill:
                # Create fill event
                event = Event(
                    timestamp=self.current_time,
                    event_type=EventType.ORDER_FILL,
                    data={"fill": fill, "order": order},
                    priority=3,
                )
                self.event_queue.push(event)

                # Keep order if not fully filled
                if not order.is_filled:
                    still_pending.append(order)
            else:
                still_pending.append(order)

        self.pending_orders = still_pending

    def _finalize_results(self, start_time: datetime, end_time: datetime) -> SimulationResults:
        """Finalize and package simulation results.

        Args:
            start_time: Simulation start time
            end_time: Simulation end time

        Returns:
            SimulationResults object
        """
        # Analyze performance
        metrics = self.analyzer.analyze(
            self.portfolio.equity_curve,
            self.portfolio.trades,
            self.config.initial_capital,
        )

        # Convert equity curve to DataFrame
        if self.portfolio.equity_curve:
            equity_df = pd.DataFrame(self.portfolio.equity_curve, columns=["timestamp", "equity"])
            equity_df.set_index("timestamp", inplace=True)
            # Remove duplicates (keep last value for each timestamp)
            equity_df = equity_df[~equity_df.index.duplicated(keep="last")]
        else:
            equity_df = pd.DataFrame(columns=["equity"])

        # Convert trades to DataFrame
        if self.portfolio.trades:
            trades_data = []
            for trade in self.portfolio.trades:
                trades_data.append(
                    {
                        "trade_id": trade.trade_id,
                        "symbol": trade.symbol,
                        "side": trade.side.value,
                        "entry_time": trade.entry_time,
                        "exit_time": trade.exit_time,
                        "entry_price": trade.entry_price,
                        "exit_price": trade.exit_price,
                        "quantity": trade.quantity,
                        "pnl": trade.pnl,
                        "pnl_pct": trade.pnl_pct,
                        "commission": trade.commission,
                        "duration": trade.duration,
                    }
                )
            trades_df = pd.DataFrame(trades_data)
        else:
            trades_df = pd.DataFrame()

        return SimulationResults(
            config=self.config,
            metrics=metrics,
            portfolio=self.portfolio,
            equity_curve=equity_df,
            trades=trades_df,
            orders=self.all_orders,
            start_time=start_time,
            end_time=end_time,
        )

    def get_position(self, symbol: str):
        """Get current position for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position object or None
        """
        return self.portfolio.get_position(symbol)

    def get_cash(self) -> float:
        """Get current cash balance.

        Returns:
            Cash balance
        """
        return self.portfolio.cash

    def get_equity(self) -> float:
        """Get current total equity.

        Returns:
            Total equity
        """
        return self.portfolio.equity
