"""Tests for PortfolioOptimizer.

Tests cover:
- Initialization with valid/invalid methods
- Mean-variance optimization
- SciPy fallback behavior
- Constraint handling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.portfolioopt.optimizer import PortfolioOptimizer


class TestPortfolioOptimizerInit:
    """Tests for PortfolioOptimizer initialization."""

    @pytest.mark.unit
    def test_init_mean_variance(self):
        """Test initialization with mean_variance method."""
        optimizer = PortfolioOptimizer(method="mean_variance")
        assert optimizer.method == "mean_variance"

    @pytest.mark.unit
    def test_init_mean_cvar(self):
        """Test initialization with mean_cvar method."""
        optimizer = PortfolioOptimizer(method="mean_cvar")
        assert optimizer.method == "mean_cvar"

    @pytest.mark.unit
    def test_init_default_method(self):
        """Test initialization with default method."""
        optimizer = PortfolioOptimizer()
        assert optimizer.method == "mean_variance"

    @pytest.mark.unit
    def test_init_invalid_method(self):
        """Test initialization with invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported method"):
            PortfolioOptimizer(method="invalid_method")


class TestPortfolioOptimizerOptimize:
    """Tests for PortfolioOptimizer.optimize method."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns DataFrame for testing."""
        np.random.seed(42)
        n_assets = 3
        n_periods = 100
        returns = np.random.randn(n_periods, n_assets) * 0.02 + 0.001
        return pd.DataFrame(
            returns,
            columns=["AAPL", "GOOGL", "MSFT"],
        )

    @pytest.mark.unit
    def test_optimize_basic(self, sample_returns):
        """Test basic optimization returns valid result."""
        optimizer = PortfolioOptimizer(method="mean_variance")
        result = optimizer.optimize(sample_returns, constraints={})

        assert "weights" in result
        assert "expected_return" in result
        assert "variance" in result
        assert "method" in result
        assert "solver" in result

    @pytest.mark.unit
    def test_optimize_weights_sum_to_one(self, sample_returns):
        """Test that optimized weights sum to 1."""
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(sample_returns, constraints={})

        weights = np.array(result["weights"])
        assert np.isclose(weights.sum(), 1.0, atol=1e-4)

    @pytest.mark.unit
    def test_optimize_weights_non_negative(self, sample_returns):
        """Test that weights are non-negative (default bounds)."""
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(sample_returns, constraints={})

        weights = np.array(result["weights"])
        assert all(w >= -1e-6 for w in weights)

    @pytest.mark.unit
    def test_optimize_uses_scipy_fallback(self, sample_returns):
        """Test that SciPy fallback is used when cuOpt not available."""
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(sample_returns, constraints={})

        # cuOpt typically not installed, so should use SciPy
        assert "SciPy" in result["solver"] or "cuOpt" in result["solver"]

    @pytest.mark.unit
    def test_optimize_with_method_in_result(self, sample_returns):
        """Test that method is included in result."""
        optimizer = PortfolioOptimizer(method="mean_variance")
        result = optimizer.optimize(sample_returns, constraints={})

        assert result["method"] == "mean_variance"


class TestSciPyFallback:
    """Tests for SciPy fallback behavior."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return PortfolioOptimizer()

    @pytest.fixture
    def simple_returns(self):
        """Create simple returns array for testing."""
        np.random.seed(123)
        return np.random.randn(50, 2) * 0.01 + 0.0005

    @pytest.mark.unit
    def test_fallback_scipy_basic(self, optimizer, simple_returns):
        """Test SciPy fallback with basic inputs."""
        result = optimizer._fallback_scipy(simple_returns, constraints={})

        assert "weights" in result
        assert len(result["weights"]) == 2
        assert result["solver"] == "SciPy (CPU fallback)"

    @pytest.mark.unit
    def test_fallback_scipy_weights_valid(self, optimizer, simple_returns):
        """Test SciPy fallback returns valid weights."""
        result = optimizer._fallback_scipy(simple_returns, constraints={})

        weights = np.array(result["weights"])
        assert np.isclose(weights.sum(), 1.0, atol=1e-4)
        assert all(w >= -1e-6 for w in weights)
        assert all(w <= 1.0 + 1e-6 for w in weights)

    @pytest.mark.unit
    def test_fallback_scipy_metrics_computed(self, optimizer, simple_returns):
        """Test SciPy fallback computes return and variance."""
        result = optimizer._fallback_scipy(simple_returns, constraints={})

        assert "expected_return" in result
        assert "variance" in result
        assert isinstance(result["expected_return"], float)
        assert isinstance(result["variance"], float)
        assert result["variance"] >= 0


class TestCuOptAttempt:
    """Tests for cuOpt attempt behavior."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return PortfolioOptimizer()

    @pytest.fixture
    def simple_returns(self):
        """Create simple returns array."""
        np.random.seed(456)
        return np.random.randn(50, 2) * 0.01

    @pytest.mark.unit
    def test_try_cuopt_returns_none_when_unavailable(self, optimizer, simple_returns):
        """Test that _try_cuopt returns None when cuOpt not installed."""
        result = optimizer._try_cuopt(simple_returns, constraints={})
        # cuOpt is typically not installed in test environment
        assert result is None
