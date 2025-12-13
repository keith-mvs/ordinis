"""Tests for CLI interface."""

from pathlib import Path
import tempfile

import pandas as pd
import pytest

from interface.cli import create_strategy, load_market_data


class TestLoadMarketData:
    """Tests for load_market_data function."""

    def test_load_data_with_timestamp_column(self):
        """Test loading data with timestamp column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,open,high,low,close,volume\n")
            f.write("2024-01-01,100,105,99,102,1000000\n")
            f.write("2024-01-02,102,106,101,104,1500000\n")
            f.write("2024-01-03,104,108,103,106,1200000\n")
            temp_file = f.name

        try:
            symbol, data = load_market_data(temp_file)

            assert symbol == Path(temp_file).stem.upper()
            assert len(data) == 3
            assert list(data.columns) == ["open", "high", "low", "close", "volume"]
            assert isinstance(data.index, pd.DatetimeIndex)
        finally:
            Path(temp_file).unlink()

    def test_load_data_with_date_column(self):
        """Test loading data with date column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("date,open,high,low,close,volume\n")
            f.write("2024-01-01,100,105,99,102,1000000\n")
            f.write("2024-01-02,102,106,101,104,1500000\n")
            temp_file = f.name

        try:
            symbol, data = load_market_data(temp_file)

            assert len(data) == 2
            assert isinstance(data.index, pd.DatetimeIndex)
        finally:
            Path(temp_file).unlink()

    def test_load_data_with_symbol_column(self):
        """Test loading data with symbol column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,symbol,open,high,low,close,volume\n")
            f.write("2024-01-01,AAPL,100,105,99,102,1000000\n")
            f.write("2024-01-02,AAPL,102,106,101,104,1500000\n")
            temp_file = f.name

        try:
            symbol, data = load_market_data(temp_file)

            assert symbol == "AAPL"
            assert len(data) == 2
        finally:
            Path(temp_file).unlink()

    def test_load_data_uses_filename_as_symbol(self):
        """Test symbol extraction from filename."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix="TSLA_", delete=False
        ) as f:
            f.write("timestamp,open,high,low,close,volume\n")
            f.write("2024-01-01,100,105,99,102,1000000\n")
            temp_file = f.name

        try:
            symbol, data = load_market_data(temp_file)

            assert symbol == Path(temp_file).stem.upper()
            assert len(data) == 1
        finally:
            Path(temp_file).unlink()

    def test_load_data_missing_timestamp_column(self):
        """Test error when timestamp/date column missing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("open,high,low,close,volume\n")
            f.write("100,105,99,102,1000000\n")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="timestamp.*date"):
                load_market_data(temp_file)
        finally:
            Path(temp_file).unlink()

    def test_load_data_missing_required_columns(self):
        """Test error when required columns missing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,open,high,low,close\n")  # Missing volume
            f.write("2024-01-01,100,105,99,102\n")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                load_market_data(temp_file)
        finally:
            Path(temp_file).unlink()

    def test_load_data_multiple_required_columns_missing(self):
        """Test error message lists all missing columns."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,open,high\n")  # Missing low, close, volume
            f.write("2024-01-01,100,105\n")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Missing required columns") as exc_info:
                load_market_data(temp_file)

            error_msg = str(exc_info.value)
            assert "low" in error_msg
            assert "close" in error_msg
            assert "volume" in error_msg
        finally:
            Path(temp_file).unlink()

    def test_load_data_sets_datetime_index(self):
        """Test that datetime index is properly set."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,open,high,low,close,volume\n")
            f.write("2024-01-01,100,105,99,102,1000000\n")
            f.write("2024-01-02,102,106,101,104,1500000\n")
            temp_file = f.name

        try:
            symbol, data = load_market_data(temp_file)

            assert isinstance(data.index, pd.DatetimeIndex)
            assert data.index[0] == pd.Timestamp("2024-01-01")
            assert data.index[1] == pd.Timestamp("2024-01-02")
        finally:
            Path(temp_file).unlink()


class TestCreateStrategy:
    """Tests for create_strategy function."""

    def test_create_rsi_strategy(self):
        """Test creating RSI strategy."""
        strategy = create_strategy("rsi")

        assert strategy is not None
        assert strategy.name == "rsi-backtest"
        assert "rsi" in strategy.name.lower()

    def test_create_ma_strategy(self):
        """Test creating MA crossover strategy."""
        strategy = create_strategy("ma")

        assert strategy is not None
        assert strategy.name == "ma-backtest"

    def test_create_momentum_strategy(self):
        """Test creating momentum strategy."""
        strategy = create_strategy("momentum")

        assert strategy is not None
        assert strategy.name == "momentum-backtest"

    def test_create_strategy_case_insensitive(self):
        """Test strategy creation is case insensitive."""
        strategy1 = create_strategy("RSI")
        strategy2 = create_strategy("Ma")
        strategy3 = create_strategy("MOMENTUM")

        assert strategy1 is not None
        assert strategy2 is not None
        assert strategy3 is not None

    def test_create_strategy_with_parameters(self):
        """Test creating strategy with custom parameters."""
        strategy = create_strategy("rsi", rsi_period=21, oversold_threshold=25)

        assert strategy.params["rsi_period"] == 21
        assert strategy.params["oversold_threshold"] == 25

    def test_create_ma_strategy_with_parameters(self):
        """Test creating MA strategy with parameters."""
        strategy = create_strategy("ma", fast_period=20, slow_period=50, ma_type="EMA")

        assert strategy.params["fast_period"] == 20
        assert strategy.params["slow_period"] == 50
        assert strategy.params["ma_type"] == "EMA"

    def test_create_unknown_strategy(self):
        """Test error for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_strategy("invalid_strategy")

    def test_create_strategy_error_lists_available(self):
        """Test error message lists available strategies."""
        with pytest.raises(ValueError, match="Unknown strategy") as exc_info:
            create_strategy("unknown")

        error_msg = str(exc_info.value)
        assert "rsi" in error_msg.lower()
        assert "ma" in error_msg.lower()
        assert "momentum" in error_msg.lower()

    def test_strategy_has_required_methods(self):
        """Test created strategy has required methods."""
        strategy = create_strategy("rsi")

        assert hasattr(strategy, "generate_signal")
        assert hasattr(strategy, "configure")
        assert hasattr(strategy, "get_description")
        assert callable(strategy.generate_signal)


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_load_and_validate_data_flow(self):
        """Test loading data and using with strategy."""
        # Create test data file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,symbol,open,high,low,close,volume\n")
            for i in range(100):
                date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
                f.write(f"{date.date()},TEST,100,105,99,102,1000000\n")
            temp_file = f.name

        try:
            # Load data
            symbol, data = load_market_data(temp_file)

            # Create strategy
            strategy = create_strategy("rsi")

            # Validate data is usable
            is_valid, msg = strategy.validate_data(data)

            assert symbol == "TEST"
            assert len(data) == 100
            assert is_valid
        finally:
            Path(temp_file).unlink()

    def test_all_strategies_with_loaded_data(self):
        """Test all strategies work with loaded data."""
        # Create test data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,open,high,low,close,volume\n")
            for i in range(250):
                price = 100 + i * 0.1
                f.write(
                    f"2024-01-{(i%30)+1:02d},{price},{price*1.01},{price*0.99},{price},{1000000}\n"
                )
            temp_file = f.name

        try:
            symbol, data = load_market_data(temp_file)

            # Test each strategy
            for strategy_name in ["rsi", "ma", "momentum"]:
                strategy = create_strategy(strategy_name)
                is_valid, msg = strategy.validate_data(data)

                assert is_valid, f"{strategy_name} failed validation: {msg}"
        finally:
            Path(temp_file).unlink()

    def test_data_with_extra_columns(self):
        """Test loading data with extra columns."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,open,high,low,close,volume,adj_close,extra\n")
            f.write("2024-01-01,100,105,99,102,1000000,102.5,xyz\n")
            f.write("2024-01-02,102,106,101,104,1500000,104.5,abc\n")
            temp_file = f.name

        try:
            symbol, data = load_market_data(temp_file)

            # Should load successfully and keep required columns
            assert len(data) == 2
            assert "open" in data.columns
            assert "close" in data.columns
            # Extra columns should also be present
            assert "adj_close" in data.columns
            assert "extra" in data.columns
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

            # Should load but have no data
            assert len(data) == 0
        finally:
            Path(temp_file).unlink()
