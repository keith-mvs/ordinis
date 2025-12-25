"""Tests for ML Profit Optimizer."""

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.sprint.core.ml_profit_optimizer import (
    BayesianOptimizer,
    ContextualBanditOptimizer,
    EvolutionaryOptimizer,
    MLProfitOptimizer,
    OptimizationConfig,
    OptimizationMethod,
    ParameterSpec,
    ProfitMetric,
    TrialResult,
    create_fibonacci_adx_optimizer,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_param_specs() -> list[ParameterSpec]:
    """Sample parameter specifications for testing."""
    return [
        ParameterSpec(
            name="period",
            min_value=5,
            max_value=50,
            default=14,
            integer=True,
            description="Lookback period",
        ),
        ParameterSpec(
            name="threshold",
            min_value=0.1,
            max_value=0.9,
            default=0.5,
            step=0.05,
            description="Signal threshold",
        ),
        ParameterSpec(
            name="multiplier",
            min_value=0.1,
            max_value=10.0,
            default=2.0,
            log_scale=True,
            description="ATR multiplier",
        ),
    ]


@pytest.fixture
def sample_config() -> OptimizationConfig:
    """Sample optimization configuration."""
    return OptimizationConfig(
        profit_metric=ProfitMetric.TOTAL_RETURN,
        method=OptimizationMethod.BAYESIAN,
        max_iterations=10,
        min_iterations=3,
        batch_size=2,
        patience=5,
        use_walk_forward=False,
        min_trades=5,
        seed=42,
        verbose=False,
        output_dir="",  # Don't save during tests
    )


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Generate sample OHLCV data."""
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100 * np.cumprod(1 + np.random.randn(n) * 0.02)
    
    return pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.005),
        "high": close * (1 + np.abs(np.random.randn(n)) * 0.01),
        "low": close * (1 - np.abs(np.random.randn(n)) * 0.01),
        "close": close,
        "volume": np.random.randint(1000000, 10000000, n),
    }, index=dates)


def mock_backtest(params: dict, data: pd.DataFrame) -> dict:
    """Mock backtest function for testing."""
    # Simulate performance based on parameters
    np.random.seed(int(params.get("period", 14)))
    
    n_trades = max(10, int(len(data) / params.get("period", 14)))
    pnls = np.random.randn(n_trades) * 100 + params.get("threshold", 0.5) * 50
    
    return {
        "total_return": np.sum(pnls) / 10000,
        "trades": [{"pnl": p} for p in pnls],
        "max_drawdown": abs(np.min(np.cumsum(pnls))) / 10000,
        "equity_curve": 10000 + np.cumsum(pnls),
    }


# =============================================================================
# ParameterSpec Tests
# =============================================================================

class TestParameterSpec:
    """Tests for ParameterSpec class."""
    
    def test_sample_within_bounds(self, sample_param_specs):
        """Test that sampled values are within bounds."""
        rng = np.random.default_rng(42)
        
        for spec in sample_param_specs:
            for _ in range(100):
                value = spec.sample(rng)
                assert spec.min_value <= value <= spec.max_value
    
    def test_integer_sampling(self):
        """Test integer parameter sampling."""
        spec = ParameterSpec(
            name="int_param",
            min_value=1,
            max_value=10,
            default=5,
            integer=True,
        )
        
        rng = np.random.default_rng(42)
        for _ in range(50):
            value = spec.sample(rng)
            assert value == int(value)
    
    def test_log_scale_sampling(self):
        """Test log-scale parameter sampling."""
        spec = ParameterSpec(
            name="log_param",
            min_value=0.001,
            max_value=1000,
            default=1.0,
            log_scale=True,
        )
        
        rng = np.random.default_rng(42)
        values = [spec.sample(rng) for _ in range(1000)]
        
        # Should have more samples in lower range due to log scale
        below_one = sum(1 for v in values if v < 1)
        above_one = sum(1 for v in values if v >= 1)
        
        # Log scale should give roughly equal samples in each decade
        assert below_one > 100  # Some samples below 1
        assert above_one > 100  # Some samples above 1
    
    def test_normalize_denormalize(self, sample_param_specs):
        """Test normalization and denormalization."""
        for spec in sample_param_specs:
            # Test at bounds
            assert abs(spec.normalize(spec.min_value) - 0.0) < 0.01
            assert abs(spec.normalize(spec.max_value) - 1.0) < 0.01
            
            # Test round-trip
            for value in [spec.min_value, spec.default, spec.max_value]:
                normalized = spec.normalize(value)
                recovered = spec.denormalize(normalized)
                # Allow some tolerance for discretization
                assert abs(recovered - value) < (spec.max_value - spec.min_value) * 0.1


# =============================================================================
# Optimizer Tests
# =============================================================================

class TestBayesianOptimizer:
    """Tests for Bayesian Optimizer."""
    
    def test_initial_suggestions_are_random(self, sample_param_specs, sample_config):
        """Test that initial suggestions are random exploration."""
        optimizer = BayesianOptimizer(sample_param_specs, sample_config)
        
        suggestions = optimizer.suggest(5)
        
        assert len(suggestions) == 5
        for s in suggestions:
            assert all(name in s for name in ["period", "threshold", "multiplier"])
    
    def test_update_improves_model(self, sample_param_specs, sample_config):
        """Test that updates improve the GP model."""
        optimizer = BayesianOptimizer(sample_param_specs, sample_config)
        
        # Add some results
        results = []
        for i in range(10):
            params = optimizer._random_params()
            result = TrialResult(
                trial_id=i,
                params=params,
                profit_value=np.random.randn(),
                metrics={},
                duration_seconds=1.0,
            )
            results.append(result)
        
        optimizer.update(results)
        
        assert len(optimizer.X) == 10
        assert len(optimizer.y) == 10
        assert optimizer.best_y > float("-inf")
    
    def test_expected_improvement(self, sample_param_specs, sample_config):
        """Test EI acquisition function."""
        optimizer = BayesianOptimizer(sample_param_specs, sample_config)
        
        # Add some data
        for i in range(10):
            optimizer.X.append(np.random.rand(3))
            optimizer.y.append(np.random.randn())
        optimizer.best_y = max(optimizer.y)
        
        # Calculate EI for new points
        X_new = np.random.rand(100, 3)
        ei = optimizer._expected_improvement(X_new)
        
        assert len(ei) == 100
        # EI is non-negative (allow tiny numerical errors)
        assert np.all(ei >= -1e-10)


class TestEvolutionaryOptimizer:
    """Tests for Evolutionary Optimizer."""
    
    def test_population_generation(self, sample_param_specs, sample_config):
        """Test population generation."""
        optimizer = EvolutionaryOptimizer(sample_param_specs, sample_config, population_size=10)
        
        suggestions = optimizer.suggest(10)
        
        assert len(suggestions) == 10
        for s in suggestions:
            for name, spec in optimizer.param_specs.items():
                assert spec.min_value <= s[name] <= spec.max_value
    
    def test_evolution_updates(self, sample_param_specs, sample_config):
        """Test that evolution updates improve the distribution."""
        optimizer = EvolutionaryOptimizer(sample_param_specs, sample_config, population_size=10)
        
        initial_mean = optimizer.mean.copy()
        
        # Run a few generations
        for gen in range(5):
            suggestions = optimizer.suggest(10)
            results = [
                TrialResult(
                    trial_id=gen * 10 + i,
                    params=s,
                    profit_value=np.random.randn() + gen * 0.1,  # Increasing fitness
                    metrics={},
                    duration_seconds=1.0,
                )
                for i, s in enumerate(suggestions)
            ]
            optimizer.update(results)
        
        # Mean should have shifted
        assert not np.allclose(optimizer.mean, initial_mean)
        assert optimizer.generation == 5


class TestContextualBanditOptimizer:
    """Tests for Contextual Bandit Optimizer."""
    
    def test_arm_discretization(self, sample_param_specs, sample_config):
        """Test parameter discretization into arms."""
        optimizer = ContextualBanditOptimizer(sample_param_specs, sample_config, n_arms_per_param=5)
        
        assert len(optimizer.arm_values) == 3
        for name, values in optimizer.arm_values.items():
            assert len(values) >= 2
    
    def test_thompson_sampling(self, sample_param_specs, sample_config):
        """Test Thompson sampling suggestions."""
        optimizer = ContextualBanditOptimizer(sample_param_specs, sample_config)
        
        suggestions = optimizer.suggest(5)
        
        assert len(suggestions) == 5
        for s in suggestions:
            for name, spec in optimizer.param_specs.items():
                assert spec.min_value <= s[name] <= spec.max_value
    
    def test_arm_updates(self, sample_param_specs, sample_config):
        """Test arm statistics updates."""
        optimizer = ContextualBanditOptimizer(sample_param_specs, sample_config)
        
        initial_alpha = {k: v.copy() for k, v in optimizer.alpha.items()}
        
        # Add results
        results = [
            TrialResult(
                trial_id=i,
                params=optimizer.suggest(1)[0],
                profit_value=1.0 if i % 2 == 0 else -1.0,
                metrics={},
                duration_seconds=1.0,
            )
            for i in range(10)
        ]
        optimizer.update(results)
        
        # Alpha/beta should have changed
        for name in optimizer.alpha:
            assert not np.allclose(optimizer.alpha[name], initial_alpha[name])


# =============================================================================
# MLProfitOptimizer Tests
# =============================================================================

class TestMLProfitOptimizer:
    """Tests for main MLProfitOptimizer class."""
    
    def test_initialization(self, sample_param_specs, sample_config):
        """Test optimizer initialization."""
        optimizer = MLProfitOptimizer(sample_param_specs, sample_config)
        
        assert optimizer.config == sample_config
        assert len(optimizer.param_specs) == 3
        assert optimizer.iteration == 0
    
    def test_profit_metric_computation(self, sample_param_specs, sample_config):
        """Test profit metric computation."""
        optimizer = MLProfitOptimizer(sample_param_specs, sample_config)
        
        backtest_result = {
            "total_return": 0.15,
            "trades": [{"pnl": 100}, {"pnl": -50}, {"pnl": 75}, {"pnl": -25}],
            "equity_curve": [10000, 10100, 10050, 10125, 10100],
        }
        
        profit_value, metrics = optimizer._compute_profit_metric(backtest_result)
        
        assert "total_return" in metrics
        assert "profit_factor" in metrics
        assert "win_rate" in metrics
        assert "max_drawdown" in metrics
        assert "sharpe_ratio" in metrics
        assert metrics["win_rate"] == 0.5  # 2 wins out of 4
        assert metrics["profit_factor"] > 1.0  # Gross profit > gross loss
    
    def test_constraint_checking(self, sample_param_specs, sample_config):
        """Test constraint validation."""
        sample_config.min_profit_factor = 1.0
        sample_config.max_drawdown = 0.5
        sample_config.min_trades = 5
        sample_config.min_win_rate = 0.3
        
        optimizer = MLProfitOptimizer(sample_param_specs, sample_config)
        
        # Valid metrics
        valid_metrics = {
            "profit_factor": 1.5,
            "max_drawdown": 0.2,
            "n_trades": 20,
            "win_rate": 0.5,
        }
        assert optimizer._check_constraints(valid_metrics)
        
        # Invalid: low profit factor
        invalid_metrics = valid_metrics.copy()
        invalid_metrics["profit_factor"] = 0.8
        assert not optimizer._check_constraints(invalid_metrics)
        
        # Invalid: high drawdown
        invalid_metrics = valid_metrics.copy()
        invalid_metrics["max_drawdown"] = 0.6
        assert not optimizer._check_constraints(invalid_metrics)
    
    @pytest.mark.asyncio
    async def test_optimization_loop(self, sample_param_specs, sample_config, sample_data):
        """Test full optimization loop."""
        sample_config.max_iterations = 5
        sample_config.batch_size = 2
        
        optimizer = MLProfitOptimizer(sample_param_specs, sample_config)
        
        result = await optimizer.optimize(mock_backtest, sample_data)
        
        assert result.iterations_completed > 0
        assert len(result.all_trials) > 0
        assert result.best_params is not None
        assert result.total_time_seconds > 0
    
    @pytest.mark.asyncio
    async def test_optimization_with_initial_params(self, sample_param_specs, sample_config, sample_data):
        """Test optimization with initial parameters."""
        sample_config.max_iterations = 3
        
        optimizer = MLProfitOptimizer(sample_param_specs, sample_config)
        
        initial_params = {"period": 14, "threshold": 0.5, "multiplier": 2.0}
        result = await optimizer.optimize(mock_backtest, sample_data, initial_params)
        
        assert len(result.all_trials) >= 1
        # First trial should be initial params
        assert result.all_trials[0].params == initial_params
    
    @pytest.mark.asyncio
    async def test_early_stopping(self, sample_param_specs, sample_config, sample_data):
        """Test early stopping when no improvement."""
        sample_config.max_iterations = 100
        sample_config.patience = 3
        sample_config.min_iterations = 2
        
        optimizer = MLProfitOptimizer(sample_param_specs, sample_config)
        
        # Use constant backtest to trigger early stopping
        def constant_backtest(params, data):
            return {
                "total_return": 0.1,
                "trades": [{"pnl": 10} for _ in range(20)],
                "max_drawdown": 0.05,
            }
        
        result = await optimizer.optimize(constant_backtest, sample_data)
        
        # Should stop early due to no improvement
        assert result.iterations_completed < sample_config.max_iterations


class TestMethodSelection:
    """Tests for different optimization methods."""
    
    @pytest.mark.asyncio
    async def test_bayesian_method(self, sample_param_specs, sample_data):
        """Test Bayesian optimization method."""
        config = OptimizationConfig(
            method=OptimizationMethod.BAYESIAN,
            max_iterations=3,
            batch_size=2,
            verbose=False,
            output_dir="",
        )
        
        optimizer = MLProfitOptimizer(sample_param_specs, config)
        result = await optimizer.optimize(mock_backtest, sample_data)
        
        assert result.method_used == OptimizationMethod.BAYESIAN
        assert len(result.all_trials) > 0
    
    @pytest.mark.asyncio
    async def test_evolutionary_method(self, sample_param_specs, sample_data):
        """Test evolutionary optimization method."""
        config = OptimizationConfig(
            method=OptimizationMethod.EVOLUTIONARY,
            max_iterations=3,
            batch_size=2,
            verbose=False,
            output_dir="",
        )
        
        optimizer = MLProfitOptimizer(sample_param_specs, config)
        result = await optimizer.optimize(mock_backtest, sample_data)
        
        assert result.method_used == OptimizationMethod.EVOLUTIONARY
        assert len(result.all_trials) > 0
    
    @pytest.mark.asyncio
    async def test_bandit_method(self, sample_param_specs, sample_data):
        """Test contextual bandit optimization method."""
        config = OptimizationConfig(
            method=OptimizationMethod.BANDIT,
            max_iterations=3,
            batch_size=2,
            verbose=False,
            output_dir="",
        )
        
        optimizer = MLProfitOptimizer(sample_param_specs, config)
        result = await optimizer.optimize(mock_backtest, sample_data)
        
        assert result.method_used == OptimizationMethod.BANDIT
        assert len(result.all_trials) > 0


class TestFibonacciADXOptimizer:
    """Tests for Fibonacci ADX strategy optimizer."""
    
    def test_create_fibonacci_adx_optimizer(self):
        """Test factory function for Fibonacci ADX optimizer."""
        optimizer = create_fibonacci_adx_optimizer()
        
        assert isinstance(optimizer, MLProfitOptimizer)
        param_names = [spec.name for spec in optimizer.param_specs]
        assert "adx_period" in param_names
        assert "adx_threshold" in param_names
        assert "swing_lookback" in param_names
        assert "tolerance" in param_names
        
        # Check config
        assert optimizer.config.profit_metric == ProfitMetric.RISK_ADJUSTED_RETURN
        assert optimizer.config.use_walk_forward is True


# =============================================================================
# TrialResult Tests
# =============================================================================

class TestTrialResult:
    """Tests for TrialResult class."""
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = TrialResult(
            trial_id=1,
            params={"period": 14, "threshold": 0.5},
            profit_value=0.15,
            metrics={"sharpe": 1.5, "win_rate": 0.55},
            duration_seconds=2.5,
            train_profit=0.18,
            test_profit=0.12,
            overfit_score=1.5,
        )
        
        d = result.to_dict()
        
        assert d["trial_id"] == 1
        assert d["params"] == {"period": 14, "threshold": 0.5}
        assert d["profit_value"] == 0.15
        assert d["train_profit"] == 0.18
        assert d["test_profit"] == 0.12


class TestOptimizationResult:
    """Tests for OptimizationResult class."""
    
    def test_to_dict(self, sample_param_specs, sample_config):
        """Test result serialization."""
        trials = [
            TrialResult(
                trial_id=i,
                params={"period": 14 + i},
                profit_value=0.1 + i * 0.01,
                metrics={"sharpe": 1.0},
                duration_seconds=1.0,
            )
            for i in range(3)
        ]
        
        from ordinis.engines.sprint.core.ml_profit_optimizer import OptimizationResult
        
        result = OptimizationResult(
            best_params={"period": 16},
            best_profit=0.12,
            best_metrics={"sharpe": 1.2},
            all_trials=trials,
            valid_trials=trials,
            iterations_completed=3,
            total_time_seconds=10.0,
            convergence_achieved=False,
            early_stopped=False,
        )
        
        d = result.to_dict()
        
        assert d["best_params"] == {"period": 16}
        assert d["best_profit"] == 0.12
        assert d["n_trials"] == 3
        assert len(d["trials"]) == 3
