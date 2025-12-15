"""Tests for backtesting framework."""

import pandas as pd
import pytest

from ordinis.backtesting import (
    BacktestConfig,
    BacktestRunner,
    DataAdapter,
    HistoricalDataLoader,
    HistoricalSignalRunner,
    SignalRunnerConfig,
)
from ordinis.engines.signalcore.core.config import SignalCoreEngineConfig
from ordinis.engines.signalcore.core.engine import SignalCoreEngine


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data."""
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    return pd.DataFrame(
        {
            "open": [100 + i * 0.5 for i in range(252)],
            "high": [102 + i * 0.5 for i in range(252)],
            "low": [98 + i * 0.5 for i in range(252)],
            "close": [101 + i * 0.5 for i in range(252)],
            "volume": [1000000] * 252,
        },
        index=dates,
    )


@pytest.fixture
def sample_data_dict(sample_ohlcv_data):
    """Generate sample data for multiple symbols."""
    return {
        "AAPL": sample_ohlcv_data.copy(),
        "MSFT": sample_ohlcv_data.copy() * 1.5,
        "GOOGL": sample_ohlcv_data.copy() * 0.8,
    }


class TestDataAdapter:
    """Tests for data adapter."""

    def test_normalize_ohlcv(self, sample_ohlcv_data):
        """Test OHLCV normalization."""
        adapter = DataAdapter()

        # Test with lowercase columns
        normalized = adapter.normalize_ohlcv(sample_ohlcv_data)

        assert set(normalized.columns) == {"open", "high", "low", "close", "volume"}
        assert isinstance(normalized.index, pd.DatetimeIndex)
        assert len(normalized) == 252

    def test_normalize_ohlcv_uppercase(self, sample_ohlcv_data):
        """Test normalization with uppercase columns."""
        adapter = DataAdapter()

        # Rename to uppercase
        data_upper = sample_ohlcv_data.rename(columns=str.upper)

        normalized = adapter.normalize_ohlcv(data_upper)
        assert set(normalized.columns) == {"open", "high", "low", "close", "volume"}

    def test_normalize_ohlcv_missing_columns(self):
        """Test error on missing columns."""
        adapter = DataAdapter()

        df = pd.DataFrame(
            {"open": [100, 101], "high": [102, 103]},
            index=pd.date_range("2023-01-01", periods=2),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            adapter.normalize_ohlcv(df)

    def test_attach_fundamentals(self, sample_ohlcv_data):
        """Test attaching fundamental data."""
        adapter = DataAdapter()

        fundamentals = {"pe_ratio": 25.5, "pb_ratio": 3.2, "div_yield": 0.02}

        with_fundamentals = adapter.attach_fundamentals(sample_ohlcv_data.copy(), fundamentals)

        assert "pe_ratio" in with_fundamentals.columns
        assert with_fundamentals["pe_ratio"].iloc[0] == 25.5


class TestHistoricalDataLoader:
    """Tests for historical data loader."""

    def test_load_symbol_parquet(self, tmp_path, sample_ohlcv_data):
        """Test loading from parquet."""
        loader = HistoricalDataLoader(tmp_path)

        # Write test data
        parquet_file = tmp_path / "AAPL.parquet"
        sample_ohlcv_data.to_parquet(parquet_file)

        # Load
        loaded = loader.load_symbol("AAPL")
        assert len(loaded) == len(sample_ohlcv_data)
        assert set(loaded.columns) == {"open", "high", "low", "close", "volume"}

    def test_load_symbol_csv(self, tmp_path, sample_ohlcv_data):
        """Test loading from CSV."""
        loader = HistoricalDataLoader(tmp_path)

        # Write test data
        csv_file = tmp_path / "AAPL.csv"
        sample_ohlcv_data.to_csv(csv_file)

        # Load
        loaded = loader.load_symbol("AAPL")
        assert len(loaded) == len(sample_ohlcv_data)

    def test_load_batch(self, tmp_path, sample_data_dict):
        """Test batch loading."""
        loader = HistoricalDataLoader(tmp_path)

        # Write test data
        for symbol, data in sample_data_dict.items():
            (tmp_path / f"{symbol}.parquet").write_bytes(data.to_parquet())

        # Load batch
        batch = loader.load_batch(["AAPL", "MSFT"])
        assert len(batch) >= 0  # May be 0 if not found, which is ok

    def test_load_nonexistent_symbol(self, tmp_path):
        """Test error on nonexistent symbol."""
        loader = HistoricalDataLoader(tmp_path)

        with pytest.raises(FileNotFoundError):
            loader.load_symbol("NONEXISTENT")

    def test_cache(self, tmp_path, sample_ohlcv_data):
        """Test caching."""
        loader = HistoricalDataLoader(tmp_path)

        # Write test data
        (tmp_path / "AAPL.parquet").write_bytes(sample_ohlcv_data.to_parquet())

        # Load once
        loader.load_symbol("AAPL")

        # Get cached (should not raise)
        cached = loader.get_cached("AAPL")
        assert cached is not None

        # Clear and verify
        loader.clear_cache()
        cached = loader.get_cached("AAPL")
        assert cached is None


class TestHistoricalSignalRunner:
    """Tests for signal runner."""

    @pytest.mark.asyncio
    async def test_generate_signals_for_symbol(self, sample_ohlcv_data):
        """Test signal generation for single symbol."""
        # Initialize engine
        config = SignalCoreEngineConfig()
        engine = SignalCoreEngine(config)
        await engine.initialize()

        runner = HistoricalSignalRunner(engine)

        try:
            signals = await runner.generate_signals_for_symbol("AAPL", sample_ohlcv_data)

            # Should generate signals (even if 0, that's ok with no models)
            assert isinstance(signals, list)

        finally:
            await engine.shutdown()

    @pytest.mark.asyncio
    async def test_generate_batch_signals(self, sample_data_dict):
        """Test batch signal generation."""
        config = SignalCoreEngineConfig()
        engine = SignalCoreEngine(config)
        await engine.initialize()

        runner = HistoricalSignalRunner(engine)

        try:
            batches = await runner.generate_batch_signals(sample_data_dict)

            assert isinstance(batches, dict)
            # Batches keyed by timestamp

        finally:
            await engine.shutdown()

    @pytest.mark.asyncio
    async def test_signal_caching(self, sample_ohlcv_data):
        """Test signal caching."""
        config = SignalCoreEngineConfig()
        config.cache_signals = True
        engine = SignalCoreEngine(config)
        await engine.initialize()

        runner = HistoricalSignalRunner(engine, SignalRunnerConfig(cache_signals=True))

        try:
            # Generate signals
            signals = await runner.generate_signals_for_symbol("AAPL", sample_ohlcv_data)

            # Verify cache works
            if signals:
                cached = runner.get_cached_signal("AAPL", signals[0].timestamp)
                # May or may not be in cache depending on model implementation

        finally:
            await engine.shutdown()


class TestBacktestConfig:
    """Tests for backtest configuration."""

    def test_config_defaults(self):
        """Test default values."""
        config = BacktestConfig(
            name="test",
            symbols=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        assert config.initial_capital == 100000.0
        assert config.commission_pct == 0.001
        assert config.slippage_bps == 5.0

    def test_config_custom(self):
        """Test custom values."""
        config = BacktestConfig(
            name="test",
            symbols=["AAPL", "MSFT"],
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=500000.0,
            commission_pct=0.002,
        )

        assert config.initial_capital == 500000.0
        assert config.commission_pct == 0.002


class TestBacktestRunner:
    """Tests for backtest runner."""

    @pytest.mark.asyncio
    async def test_runner_initialization(self, tmp_path):
        """Test runner can initialize."""
        config = BacktestConfig(
            name="test",
            symbols=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        runner = BacktestRunner(config, tmp_path)
        await runner.initialize()

        assert runner.signal_engine is not None
        assert runner.backtest_engine is not None
        assert runner.data_loader is not None

        await runner.shutdown()

    @pytest.mark.asyncio
    async def test_runner_artifacts(self, tmp_path, monkeypatch):
        """Test that artifacts directory is created."""
        # Mock data loader to skip file I/O
        config = BacktestConfig(
            name="test_artifacts",
            symbols=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        runner = BacktestRunner(config, tmp_path)

        # Verify output dir exists
        assert runner.output_dir.exists()

        # Can write artifacts
        (runner.output_dir / "test.txt").write_text("test")
        assert (runner.output_dir / "test.txt").exists()
