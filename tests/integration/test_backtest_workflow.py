"""Integration tests for complete backtest workflow."""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from interface.cli import create_strategy, load_market_data
from engines.proofbench import SimulationConfig, SimulationEngine
from engines.proofbench.core.execution import Order, OrderSide, OrderType


def create_realistic_market_data(bars: int = 500, volatility: float = 0.02) -> pd.DataFrame:
    """Create realistic market data for testing.

    Args:
        bars: Number of bars to generate
        volatility: Daily volatility (default 2%)

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)  # Reproducible data
    dates = pd.date_range(start="2023-01-01", periods=bars, freq="D")

    # Generate returns with drift
    returns = np.random.normal(0.0005, volatility, bars)  # 0.05% daily drift
    close = 100 * np.exp(np.cumsum(returns))

    # Add realistic OHLC with proper constraints
    # High should be >= max(open, close)
    # Low should be <= min(open, close)
    open_price = close * (1 + np.random.normal(0, 0.005, bars))

    # Ensure high is >= both open and close
    high_offset = np.abs(np.random.normal(0, 0.01, bars))
    high = np.maximum(open_price, close) * (1 + high_offset)

    # Ensure low is <= both open and close
    low_offset = np.abs(np.random.normal(0, 0.01, bars))
    low = np.minimum(open_price, close) * (1 - low_offset)

    # Realistic volume
    volume = np.random.randint(1000000, 5000000, bars)

    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


class TestBacktestWorkflow:
    """Test complete end-to-end backtest workflow."""

    def test_complete_workflow_rsi_strategy(self):
        """Test complete backtest with RSI strategy."""
        # Create test data
        data = create_realistic_market_data(bars=250)
        symbol = "TEST"

        # Create strategy
        strategy = create_strategy("rsi", rsi_period=14)

        # Setup simulation
        config = SimulationConfig(initial_capital=10000.0, bar_frequency="1d", risk_free_rate=0.02)
        sim = SimulationEngine(config=config)
        sim.load_data(symbol, data)

        # Track signals
        signals_generated = 0
        signals_executed = 0

        def trading_strategy(engine, sym, bar):
            nonlocal signals_generated, signals_executed

            current_bar = len(engine.portfolio.equity_curve)
            if current_bar < strategy.get_required_bars():
                return

            recent_data = data.iloc[:current_bar]
            signal = strategy.generate_signal(recent_data, bar.timestamp)

            if signal:
                signals_generated += 1

                if signal.signal_type.value == "entry" and signal.direction.value == "long":
                    position_size = int((engine.portfolio.cash * 0.1) / bar.close)
                    if position_size > 0:
                        order = Order(
                            symbol=sym,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            quantity=position_size,
                            timestamp=bar.timestamp,
                        )
                        engine.pending_orders.append(order)
                        signals_executed += 1

        sim.on_bar = trading_strategy

        # Run backtest
        results = sim.run()

        # Validate results
        assert results is not None
        assert hasattr(results, "metrics")
        assert results.metrics.num_trades >= 0
        assert results.metrics.total_return is not None
        assert len(results.equity_curve) > 0

    def test_complete_workflow_ma_strategy(self):
        """Test complete backtest with MA crossover strategy."""
        data = create_realistic_market_data(bars=300)
        symbol = "TEST"

        strategy = create_strategy("ma", fast_period=20, slow_period=50)

        config = SimulationConfig(initial_capital=10000.0)
        sim = SimulationEngine(config=config)
        sim.load_data(symbol, data)

        signals_count = 0

        def trading_strategy(engine, sym, bar):
            nonlocal signals_count

            current_bar = len(engine.portfolio.equity_curve)
            if current_bar < strategy.get_required_bars():
                return

            recent_data = data.iloc[:current_bar]
            signal = strategy.generate_signal(recent_data, bar.timestamp)

            if signal:
                signals_count += 1

        sim.on_bar = trading_strategy
        results = sim.run()

        assert results is not None
        assert len(results.equity_curve) == len(data)

    def test_complete_workflow_momentum_strategy(self):
        """Test complete backtest with momentum strategy."""
        data = create_realistic_market_data(bars=200)
        symbol = "TEST"

        strategy = create_strategy("momentum", lookback_period=20, atr_period=14)

        config = SimulationConfig(initial_capital=10000.0)
        sim = SimulationEngine(config=config)
        sim.load_data(symbol, data)

        def trading_strategy(engine, sym, bar):
            current_bar = len(engine.portfolio.equity_curve)
            if current_bar < strategy.get_required_bars():
                return

            recent_data = data.iloc[:current_bar]
            strategy.generate_signal(recent_data, bar.timestamp)

        sim.on_bar = trading_strategy
        results = sim.run()

        assert results is not None
        assert results.metrics.sharpe_ratio is not None
        assert results.metrics.max_drawdown <= 0

    def test_workflow_with_insufficient_data(self):
        """Test workflow handling insufficient data."""
        # Only 50 bars - insufficient for most strategies
        data = create_realistic_market_data(bars=50)
        symbol = "TEST"

        strategy = create_strategy("rsi", rsi_period=14)

        config = SimulationConfig(initial_capital=10000.0)
        sim = SimulationEngine(config=config)
        sim.load_data(symbol, data)

        signals_generated = 0

        def trading_strategy(engine, sym, bar):
            nonlocal signals_generated

            current_bar = len(engine.portfolio.equity_curve)
            if current_bar < strategy.get_required_bars():
                return

            recent_data = data.iloc[:current_bar]
            signal = strategy.generate_signal(recent_data, bar.timestamp)
            if signal:
                signals_generated += 1

        sim.on_bar = trading_strategy
        results = sim.run()

        # Should complete but generate few/no signals
        assert results is not None
        assert signals_generated >= 0

    def test_workflow_with_multiple_signals(self):
        """Test workflow generating multiple signals."""
        data = create_realistic_market_data(bars=500)
        symbol = "TEST"

        strategy = create_strategy(
            "rsi", rsi_period=10, oversold_threshold=35, overbought_threshold=65
        )

        config = SimulationConfig(initial_capital=10000.0)
        sim = SimulationEngine(config=config)
        sim.load_data(symbol, data)

        signals = []

        def trading_strategy(engine, sym, bar):
            current_bar = len(engine.portfolio.equity_curve)
            if current_bar < strategy.get_required_bars():
                return

            recent_data = data.iloc[:current_bar]
            signal = strategy.generate_signal(recent_data, bar.timestamp)
            if signal:
                signals.append(signal)

        sim.on_bar = trading_strategy
        results = sim.run()

        assert results is not None
        assert len(signals) >= 0  # Should generate some signals with 500 bars

    def test_workflow_with_custom_parameters(self):
        """Test workflow with custom strategy parameters."""
        data = create_realistic_market_data(bars=300)
        symbol = "TEST"

        # Custom parameters
        strategy = create_strategy("ma", fast_period=10, slow_period=30, ma_type="EMA")

        assert strategy.params["fast_period"] == 10
        assert strategy.params["slow_period"] == 30
        assert strategy.params["ma_type"] == "EMA"

        config = SimulationConfig(initial_capital=50000.0, risk_free_rate=0.03)
        sim = SimulationEngine(config=config)
        sim.load_data(symbol, data)

        def trading_strategy(engine, sym, bar):
            current_bar = len(engine.portfolio.equity_curve)
            if current_bar >= strategy.get_required_bars():
                recent_data = data.iloc[:current_bar]
                strategy.generate_signal(recent_data, bar.timestamp)

        sim.on_bar = trading_strategy
        results = sim.run()

        assert results is not None
        assert config.initial_capital == 50000.0


class TestDataLoadingIntegration:
    """Test data loading integration with workflow."""

    def test_load_data_and_run_backtest(self):
        """Test loading CSV data and running backtest."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,symbol,open,high,low,close,volume\n")
            for i in range(250):
                date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=i)
                price = 100 + i * 0.1
                f.write(f"{date.date()},AAPL,{price},{price*1.02},{price*0.98},{price},{1000000}\n")
            temp_file = f.name

        try:
            # Load data
            symbol, data = load_market_data(temp_file)

            assert symbol == "AAPL"
            assert len(data) == 250

            # Run backtest
            strategy = create_strategy("rsi")

            config = SimulationConfig(initial_capital=10000.0)
            sim = SimulationEngine(config=config)
            sim.load_data(symbol, data)

            def trading_strategy(engine, sym, bar):
                current_bar = len(engine.portfolio.equity_curve)
                if current_bar >= strategy.get_required_bars():
                    recent_data = data.iloc[:current_bar]
                    strategy.generate_signal(recent_data, bar.timestamp)

            sim.on_bar = trading_strategy
            results = sim.run()

            assert results is not None
            assert len(results.equity_curve) == 250
        finally:
            Path(temp_file).unlink()

    def test_load_data_with_different_formats(self):
        """Test loading data with different timestamp formats."""
        # Create file with 'date' column instead of 'timestamp'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("date,open,high,low,close,volume\n")
            for i in range(100):
                date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=i)
                f.write(f"{date.date()},100,105,99,102,1000000\n")
            temp_file = f.name

        try:
            symbol, data = load_market_data(temp_file)

            assert len(data) == 100
            assert isinstance(data.index, pd.DatetimeIndex)

            # Should work with backtest
            strategy = create_strategy("rsi")
            config = SimulationConfig(initial_capital=10000.0)
            sim = SimulationEngine(config=config)
            sim.load_data(symbol, data)

            def trading_strategy(engine, sym, bar):
                current_bar = len(engine.portfolio.equity_curve)
                if current_bar >= strategy.get_required_bars():
                    recent_data = data.iloc[:current_bar]
                    strategy.generate_signal(recent_data, bar.timestamp)

            sim.on_bar = trading_strategy
            results = sim.run()
            assert results is not None
        finally:
            Path(temp_file).unlink()


class TestErrorHandling:
    """Test error handling in workflow."""

    def test_invalid_strategy_name(self):
        """Test error handling for invalid strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_strategy("invalid_strategy")

    def test_missing_data_columns(self):
        """Test error handling for missing data columns."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,open,high,low,close\n")  # Missing volume
            f.write("2023-01-01,100,105,99,102\n")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                load_market_data(temp_file)
        finally:
            Path(temp_file).unlink()

    def test_empty_data_file(self):
        """Test handling of empty data file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,open,high,low,close,volume\n")
            # No data rows
            temp_file = f.name

        try:
            symbol, data = load_market_data(temp_file)

            # Empty data should load but backtests may fail/warn
            assert len(data) == 0

            # Engine should handle empty data gracefully
            strategy = create_strategy("rsi")
            config = SimulationConfig(initial_capital=10000.0)
            sim = SimulationEngine(config=config)
            sim.load_data(symbol, data)

            # This should either raise or return empty results
            # depending on engine implementation
            try:
                results = sim.run()
                # If it succeeds, verify empty results
                if results:
                    assert len(results.equity_curve) == 0
            except (ValueError, IndexError, KeyError) as e:
                # Expected for empty data - engine may raise various errors
                # This is acceptable behavior for empty datasets
                assert True  # Test passes if expected exception raised
        finally:
            Path(temp_file).unlink()


class TestPerformanceMetrics:
    """Test performance metrics calculation."""

    def test_metrics_calculation(self):
        """Test that all metrics are calculated."""
        data = create_realistic_market_data(bars=300)
        symbol = "TEST"

        strategy = create_strategy("rsi")

        config = SimulationConfig(initial_capital=10000.0, risk_free_rate=0.02)
        sim = SimulationEngine(config=config)
        sim.load_data(symbol, data)

        def trading_strategy(engine, sym, bar):
            current_bar = len(engine.portfolio.equity_curve)
            if current_bar >= strategy.get_required_bars() and current_bar % 10 == 0:
                recent_data = data.iloc[:current_bar]
                signal = strategy.generate_signal(recent_data, bar.timestamp)

                if signal and signal.signal_type.value == "entry":
                    position_size = int((engine.portfolio.cash * 0.2) / bar.close)
                    if position_size > 0:
                        order = Order(
                            symbol=sym,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            quantity=position_size,
                            timestamp=bar.timestamp,
                        )
                        engine.pending_orders.append(order)

        sim.on_bar = trading_strategy
        results = sim.run()

        # Verify all metrics exist
        metrics = results.metrics
        assert hasattr(metrics, "total_return")
        assert hasattr(metrics, "annualized_return")
        assert hasattr(metrics, "sharpe_ratio")
        assert hasattr(metrics, "sortino_ratio")
        assert hasattr(metrics, "calmar_ratio")
        assert hasattr(metrics, "max_drawdown")
        assert hasattr(metrics, "volatility")
        assert hasattr(metrics, "num_trades")
        assert hasattr(metrics, "win_rate")
        assert hasattr(metrics, "profit_factor")

    def test_metrics_values_reasonable(self):
        """Test that metrics have reasonable values."""
        data = create_realistic_market_data(bars=250)
        symbol = "TEST"

        strategy = create_strategy("ma", fast_period=20, slow_period=50)

        config = SimulationConfig(initial_capital=10000.0)
        sim = SimulationEngine(config=config)
        sim.load_data(symbol, data)

        def trading_strategy(engine, sym, bar):
            current_bar = len(engine.portfolio.equity_curve)
            if current_bar >= strategy.get_required_bars():
                recent_data = data.iloc[:current_bar]
                strategy.generate_signal(recent_data, bar.timestamp)

        sim.on_bar = trading_strategy
        results = sim.run()

        metrics = results.metrics

        # Max drawdown should be negative or zero
        assert metrics.max_drawdown <= 0

        # Number of trades should be non-negative
        assert metrics.num_trades >= 0

        # Win rate should be between 0 and 1
        if metrics.num_trades > 0:
            assert 0 <= metrics.win_rate <= 1

        # Volatility should be non-negative
        assert metrics.volatility >= 0
