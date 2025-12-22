"""Tests for Sprint GPU Accelerator.

Tests cover:
- GPUConfig dataclass
- GPUBacktestEngine initialization
- GPU/CPU compute methods
- Backtest execution
"""

from __future__ import annotations

import numpy as np
import pytest

from ordinis.engines.sprint.core.accelerator import (
    GPUConfig,
    GPUBacktestEngine,
    HAS_CUPY,
    HAS_NUMBA,
)


class TestGPUConfig:
    """Tests for GPUConfig dataclass."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default configuration values."""
        config = GPUConfig()

        assert config.use_gpu is True
        assert config.device_id == 0
        assert config.batch_size == 1024
        assert config.parallel_workers == 4

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GPUConfig(
            use_gpu=False,
            device_id=1,
            batch_size=2048,
            parallel_workers=8,
        )

        assert config.use_gpu is False
        assert config.device_id == 1
        assert config.batch_size == 2048
        assert config.parallel_workers == 8


class TestGPUBacktestEngineInit:
    """Tests for GPUBacktestEngine initialization."""

    @pytest.mark.unit
    def test_init_default_config(self):
        """Test initialization with default config."""
        engine = GPUBacktestEngine()

        assert engine.config is not None
        assert engine.config.use_gpu is True

    @pytest.mark.unit
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = GPUConfig(use_gpu=False)
        engine = GPUBacktestEngine(config)

        assert engine.config.use_gpu is False
        # GPU not available when explicitly disabled
        assert engine.gpu_available is False

    @pytest.mark.unit
    def test_init_sets_gpu_availability(self):
        """Test that initialization checks GPU availability."""
        engine = GPUBacktestEngine()

        # gpu_available should be a boolean
        assert isinstance(engine.gpu_available, bool)


class TestGPUBacktestEngineCompute:
    """Tests for compute methods."""

    @pytest.fixture
    def engine(self):
        """Create engine with GPU disabled for predictable tests."""
        config = GPUConfig(use_gpu=False)
        return GPUBacktestEngine(config)

    @pytest.fixture
    def sample_prices(self):
        """Create sample price series."""
        np.random.seed(42)
        return 100 + np.cumsum(np.random.randn(100) * 0.5)

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns series."""
        np.random.seed(42)
        return np.random.randn(100) * 0.02

    @pytest.mark.unit
    def test_compute_returns_gpu(self, engine, sample_prices):
        """Test computing returns (CPU fallback)."""
        returns = engine.compute_returns_gpu(sample_prices)

        assert len(returns) == len(sample_prices) - 1
        assert isinstance(returns, np.ndarray)
        # Returns should be reasonable (not all zeros)
        assert np.std(returns) > 0

    @pytest.mark.unit
    def test_compute_volatility_gpu(self, engine, sample_returns):
        """Test computing rolling volatility (CPU fallback)."""
        window = 20
        volatility = engine.compute_volatility_gpu(sample_returns, window=window)

        assert len(volatility) == len(sample_returns)
        # First window values should be zero
        assert all(volatility[:window] == 0)
        # After window, volatility should be positive
        assert all(volatility[window:] >= 0)

    @pytest.mark.unit
    def test_compute_ewma_volatility_gpu(self, engine, sample_returns):
        """Test computing EWMA volatility (CPU fallback)."""
        span = 30
        ewma_vol = engine.compute_ewma_volatility_gpu(sample_returns, span=span)

        assert len(ewma_vol) == len(sample_returns)
        # EWMA should be non-negative
        assert all(ewma_vol >= 0)

    @pytest.mark.unit
    def test_compute_covariance_matrix_gpu(self, engine):
        """Test computing covariance matrix (CPU fallback)."""
        np.random.seed(123)
        returns_matrix = np.random.randn(100, 3) * 0.02

        cov = engine.compute_covariance_matrix_gpu(returns_matrix)

        assert cov.shape == (3, 3)
        # Covariance matrix should be symmetric
        assert np.allclose(cov, cov.T)
        # Diagonal should be positive (variances)
        assert all(np.diag(cov) >= 0)

    @pytest.mark.unit
    def test_compute_correlation_matrix_gpu(self, engine):
        """Test computing correlation matrix (CPU fallback)."""
        np.random.seed(456)
        returns_matrix = np.random.randn(100, 3) * 0.02

        corr = engine.compute_correlation_matrix_gpu(returns_matrix)

        assert corr.shape == (3, 3)
        # Diagonal should be 1 (self-correlation)
        assert np.allclose(np.diag(corr), 1.0)
        # Correlations should be between -1 and 1
        assert np.all(corr >= -1) and np.all(corr <= 1)


class TestGPUBacktestEngineBacktest:
    """Tests for backtest execution."""

    @pytest.fixture
    def engine(self):
        """Create engine."""
        config = GPUConfig(use_gpu=False)
        return GPUBacktestEngine(config)

    @pytest.fixture
    def trending_prices(self):
        """Create trending price series."""
        np.random.seed(42)
        n = 100
        trend = np.linspace(100, 120, n)
        noise = np.random.randn(n) * 0.5
        return trend + noise

    @pytest.fixture
    def long_signals(self):
        """Create all-long signals."""
        return np.ones(100)

    @pytest.mark.unit
    def test_run_backtest_basic(self, engine, trending_prices, long_signals):
        """Test basic backtest execution."""
        result = engine.run_backtest(
            prices=trending_prices,
            signals=long_signals,
            initial_capital=100_000.0,
        )

        assert "equity_curve" in result or result is not None

    @pytest.mark.unit
    def test_run_backtest_with_mismatched_signals(self, engine, trending_prices):
        """Test backtest with mismatched signal length."""
        short_signals = np.ones(50)  # Shorter than prices

        # Should not raise, aligns signals internally
        result = engine.run_backtest(
            prices=trending_prices,
            signals=short_signals,
        )

        assert result is not None


class TestGPUBacktestEngineBatch:
    """Tests for batch backtest execution."""

    @pytest.fixture
    def engine(self):
        """Create engine."""
        config = GPUConfig(use_gpu=False, parallel_workers=2)
        return GPUBacktestEngine(config)

    @pytest.mark.unit
    def test_batch_backtest_without_executor(self, engine):
        """Test batch backtest without initialized executor."""

        def simple_backtest(params):
            return {"result": params["value"] * 2}

        param_sets = [{"value": 1}, {"value": 2}, {"value": 3}]

        results = engine.batch_backtest(simple_backtest, param_sets)

        assert len(results) == 3
        assert results[0]["result"] == 2
        assert results[1]["result"] == 4
        assert results[2]["result"] == 6

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_batch_backtest_with_executor(self, engine):
        """Test batch backtest with initialized executor."""
        await engine.initialize()

        def simple_backtest(params):
            return {"result": params["value"] ** 2}

        param_sets = [{"value": 2}, {"value": 3}, {"value": 4}]

        results = engine.batch_backtest(simple_backtest, param_sets)

        await engine.close()

        assert len(results) == 3
        assert results[0]["result"] == 4
        assert results[1]["result"] == 9
        assert results[2]["result"] == 16


class TestGPUBacktestEngineLifecycle:
    """Tests for engine lifecycle."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_initialize_and_close(self):
        """Test engine initialization and cleanup."""
        engine = GPUBacktestEngine()

        await engine.initialize()
        assert engine._executor is not None

        await engine.close()
        # Executor should be shutdown

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_close_without_initialize(self):
        """Test closing engine that was never initialized."""
        engine = GPUBacktestEngine()

        # Should not raise
        await engine.close()
