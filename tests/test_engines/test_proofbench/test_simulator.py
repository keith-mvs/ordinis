"""Tests for simulation engine."""

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.proofbench import (
    Order,
    OrderSide,
    OrderType,
    SimulationConfig,
    SimulationEngine,
)


class TestSimulationEngine:
    """Tests for SimulationEngine."""

    def test_engine_initialization(self):
        """Test creating a simulation engine."""
        config = SimulationConfig(initial_capital=100000.0)
        engine = SimulationEngine(config)

        assert engine.config.initial_capital == 100000.0
        assert engine.portfolio.initial_capital == 100000.0
        assert engine.event_queue.is_empty()

    def test_load_data(self):
        """Test loading market data."""
        engine = SimulationEngine()

        # Create sample data
        dates = pd.date_range("2024-01-01", periods=10, freq="1d")
        data = pd.DataFrame(
            {
                "open": np.linspace(100, 110, 10),
                "high": np.linspace(102, 112, 10),
                "low": np.linspace(98, 108, 10),
                "close": np.linspace(101, 111, 10),
                "volume": [1000000] * 10,
            },
            index=dates,
        )

        engine.load_data("AAPL", data)

        assert "AAPL" in engine.data
        assert len(engine.data["AAPL"]) == 10

    def test_load_data_missing_columns(self):
        """Test error when data has missing columns."""
        engine = SimulationEngine()

        # Create data with missing columns
        dates = pd.date_range("2024-01-01", periods=10, freq="1d")
        data = pd.DataFrame({"close": [100] * 10}, index=dates)

        with pytest.raises(ValueError, match="must have columns"):
            engine.load_data("AAPL", data)

    def test_simple_buy_and_hold_strategy(self):
        """Test a simple buy-and-hold strategy."""
        # Create engine
        config = SimulationConfig(initial_capital=10000.0)
        engine = SimulationEngine(config)

        # Create sample data with upward trend
        dates = pd.date_range("2024-01-01", periods=20, freq="1d")
        data = pd.DataFrame(
            {
                "open": np.linspace(100, 120, 20),
                "high": np.linspace(102, 122, 20),
                "low": np.linspace(98, 118, 20),
                "close": np.linspace(101, 121, 20),
                "volume": [1000000] * 20,
            },
            index=dates,
        )

        engine.load_data("AAPL", data)

        # Define buy-and-hold strategy
        bought = False

        def buy_and_hold(engine, symbol, bar):
            nonlocal bought
            if not bought:
                # Buy on first bar
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=50,
                    order_type=OrderType.MARKET,
                )
                engine.submit_order(order)
                bought = True

        engine.set_strategy(buy_and_hold)

        # Run backtest
        results = engine.run()

        # Verify results
        assert results.metrics.num_trades >= 0
        assert results.metrics.equity_final > 0
        assert len(results.equity_curve) > 0
        assert results.portfolio.num_positions >= 0

    def test_simple_mean_reversion_strategy(self):
        """Test a simple mean reversion strategy."""
        config = SimulationConfig(initial_capital=10000.0)
        engine = SimulationEngine(config)

        # Create data with oscillations
        dates = pd.date_range("2024-01-01", periods=100, freq="1d")
        prices = 100 + 10 * np.sin(np.linspace(0, 4 * np.pi, 100))

        data = pd.DataFrame(
            {
                "open": prices,
                "high": prices + 1,
                "low": prices - 1,
                "close": prices,
                "volume": [1000000] * 100,
            },
            index=dates,
        )

        engine.load_data("AAPL", data)

        # Simple mean reversion: buy when below 100, sell when above 100
        def mean_reversion(engine, symbol, bar):
            pos = engine.get_position(symbol)

            if bar.close < 95 and (pos is None or pos.quantity == 0):
                # Buy when price is low
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=10,
                    order_type=OrderType.MARKET,
                )
                engine.submit_order(order)
            elif bar.close > 105 and pos and pos.quantity > 0:
                # Sell when price is high
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=10,
                    order_type=OrderType.MARKET,
                )
                engine.submit_order(order)

        engine.set_strategy(mean_reversion)

        # Run backtest
        results = engine.run()

        # Verify results structure
        assert isinstance(results.equity_curve, pd.DataFrame)
        assert isinstance(results.trades, pd.DataFrame)
        assert results.start_time < results.end_time

    def test_no_data_error(self):
        """Test error when running without data."""
        engine = SimulationEngine()

        def dummy_strategy(engine, symbol, bar):
            pass

        engine.set_strategy(dummy_strategy)

        with pytest.raises(ValueError, match="No data loaded"):
            engine.run()

    def test_no_strategy_error(self):
        """Test error when running without strategy."""
        engine = SimulationEngine()

        # Load data
        dates = pd.date_range("2024-01-01", periods=10, freq="1d")
        data = pd.DataFrame(
            {
                "open": [100] * 10,
                "high": [102] * 10,
                "low": [98] * 10,
                "close": [101] * 10,
                "volume": [1000000] * 10,
            },
            index=dates,
        )

        engine.load_data("AAPL", data)

        with pytest.raises(ValueError, match="No strategy set"):
            engine.run()

    def test_get_position(self):
        """Test getting position information."""
        config = SimulationConfig(initial_capital=10000.0)
        engine = SimulationEngine(config)

        # Initially no position
        pos = engine.get_position("AAPL")
        assert pos is None

    def test_get_cash_and_equity(self):
        """Test getting cash and equity."""
        config = SimulationConfig(initial_capital=10000.0)
        engine = SimulationEngine(config)

        assert engine.get_cash() == 10000.0
        assert engine.get_equity() == 10000.0


class TestSimulationConfig:
    """Tests for SimulationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = SimulationConfig()

        assert config.initial_capital == 100000.0
        assert config.bar_frequency == "1d"
        assert config.record_equity_frequency == 1
        assert config.risk_free_rate == 0.02

    def test_custom_config(self):
        """Test custom configuration."""
        config = SimulationConfig(
            initial_capital=50000.0,
            bar_frequency="1h",
            record_equity_frequency=10,
            risk_free_rate=0.03,
        )

        assert config.initial_capital == 50000.0
        assert config.bar_frequency == "1h"
        assert config.record_equity_frequency == 10
        assert config.risk_free_rate == 0.03

    def test_invalid_capital(self):
        """Test error with invalid initial capital."""
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            SimulationConfig(initial_capital=-1000.0)

    def test_invalid_frequency(self):
        """Test error with invalid record frequency."""
        with pytest.raises(ValueError, match="Record frequency must be positive"):
            SimulationConfig(record_equity_frequency=0)
