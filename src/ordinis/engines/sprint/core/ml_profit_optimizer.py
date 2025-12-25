"""
ML Profit Optimizer - Iterative Parameter Tuning for Maximum Profit.

Implements a machine learning algorithm that iteratively searches for and tunes
model parameters to maximize trading profit using multiple optimization strategies:
- Bayesian Optimization (Gaussian Process surrogate)
- Evolutionary Algorithms (CMA-ES, Differential Evolution)
- Reinforcement Learning (Contextual Bandits)

Key Features:
- Clear profit objective with multiple measurement methods
- Iterative parameter updates based on observed performance
- Multiple optimization strategies with automatic selection
- Early stopping and convergence detection
- Walk-forward validation to prevent overfitting
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Type for parameter dictionary
ParamDict = dict[str, Any]
T = TypeVar("T")


class ProfitMetric(str, Enum):
    """Profit measurement methods."""
    
    TOTAL_RETURN = "total_return"           # Simple return percentage
    RISK_ADJUSTED_RETURN = "risk_adjusted"  # Sharpe-weighted return
    PROFIT_FACTOR = "profit_factor"         # Gross profit / gross loss
    EXPECTANCY = "expectancy"               # Average profit per trade
    CALMAR_RATIO = "calmar"                 # Return / max drawdown
    SORTINO_RATIO = "sortino"               # Return / downside deviation
    CAGR = "cagr"                           # Compound annual growth rate
    NET_PROFIT = "net_profit"               # Absolute dollar profit


class OptimizationMethod(str, Enum):
    """Optimization strategies."""
    
    BAYESIAN = "bayesian"           # Gaussian Process-based
    EVOLUTIONARY = "evolutionary"   # CMA-ES or Differential Evolution
    BANDIT = "bandit"               # Contextual multi-armed bandit
    GRID = "grid"                   # Exhaustive grid search
    RANDOM = "random"               # Random search baseline
    HYBRID = "hybrid"               # Combination of methods


@dataclass
class ParameterSpec:
    """Specification for a tunable parameter."""
    
    name: str
    min_value: float
    max_value: float
    default: float
    step: float | None = None           # Discretization step
    log_scale: bool = False              # Search in log space
    integer: bool = False                # Round to integer
    description: str = ""
    
    def sample(self, rng: np.random.Generator | None = None) -> float:
        """Sample a random value within bounds."""
        rng = rng or np.random.default_rng()
        
        if self.log_scale:
            log_min = np.log(max(self.min_value, 1e-10))
            log_max = np.log(self.max_value)
            value = np.exp(rng.uniform(log_min, log_max))
        else:
            value = rng.uniform(self.min_value, self.max_value)
        
        if self.step:
            value = round(value / self.step) * self.step
        if self.integer:
            value = int(round(value))
            
        return np.clip(value, self.min_value, self.max_value)
    
    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1] range."""
        if self.log_scale:
            log_min = np.log(max(self.min_value, 1e-10))
            log_max = np.log(self.max_value)
            return (np.log(max(value, 1e-10)) - log_min) / (log_max - log_min)
        return (value - self.min_value) / (self.max_value - self.min_value)
    
    def denormalize(self, normalized: float) -> float:
        """Convert normalized [0, 1] value back to original scale."""
        if self.log_scale:
            log_min = np.log(max(self.min_value, 1e-10))
            log_max = np.log(self.max_value)
            value = np.exp(normalized * (log_max - log_min) + log_min)
        else:
            value = normalized * (self.max_value - self.min_value) + self.min_value
        
        if self.step:
            value = round(value / self.step) * self.step
        if self.integer:
            value = int(round(value))
            
        return np.clip(value, self.min_value, self.max_value)


@dataclass
class TrialResult:
    """Result from a single parameter evaluation."""
    
    trial_id: int
    params: ParamDict
    profit_value: float                   # Primary objective value
    metrics: dict[str, float]             # All computed metrics
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_valid: bool = True
    error_message: str | None = None
    
    # Walk-forward validation results
    train_profit: float | None = None
    test_profit: float | None = None
    overfit_score: float | None = None    # train/test ratio (lower is better)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "params": self.params,
            "profit_value": self.profit_value,
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat(),
            "is_valid": self.is_valid,
            "error_message": self.error_message,
            "train_profit": self.train_profit,
            "test_profit": self.test_profit,
            "overfit_score": self.overfit_score,
        }


@dataclass
class OptimizationConfig:
    """Configuration for the profit optimizer."""
    
    # Objective settings
    profit_metric: ProfitMetric = ProfitMetric.RISK_ADJUSTED_RETURN
    method: OptimizationMethod = OptimizationMethod.BAYESIAN
    
    # Iteration settings
    max_iterations: int = 100             # Maximum optimization iterations
    min_iterations: int = 20              # Minimum before early stopping
    batch_size: int = 4                   # Parallel evaluations per iteration
    
    # Convergence settings
    convergence_threshold: float = 0.001  # Minimum improvement to continue
    patience: int = 10                    # Iterations without improvement
    
    # Regularization
    use_walk_forward: bool = True         # Validate with walk-forward
    train_ratio: float = 0.7              # Training data ratio
    max_overfit_ratio: float = 1.5        # Max train/test performance gap
    
    # Risk constraints
    min_profit_factor: float = 1.2        # Minimum gross profit / gross loss
    max_drawdown: float = 0.25            # Maximum allowed drawdown
    min_trades: int = 30                  # Minimum trades for validity
    min_win_rate: float = 0.40            # Minimum win rate
    
    # Exploration
    exploration_ratio: float = 0.2        # Fraction of random exploration
    
    # Output
    output_dir: str = "artifacts/optimization"
    save_all_trials: bool = True
    verbose: bool = True
    
    # Random seed
    seed: int | None = 42


@dataclass
class OptimizationResult:
    """Complete optimization result."""
    
    best_params: ParamDict
    best_profit: float
    best_metrics: dict[str, float]
    
    all_trials: list[TrialResult]
    valid_trials: list[TrialResult]
    
    iterations_completed: int
    total_time_seconds: float
    convergence_achieved: bool
    early_stopped: bool
    
    # Walk-forward validation
    best_train_profit: float | None = None
    best_test_profit: float | None = None
    overfit_score: float | None = None
    
    method_used: OptimizationMethod = OptimizationMethod.BAYESIAN
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_profit": self.best_profit,
            "best_metrics": self.best_metrics,
            "iterations_completed": self.iterations_completed,
            "total_time_seconds": self.total_time_seconds,
            "convergence_achieved": self.convergence_achieved,
            "early_stopped": self.early_stopped,
            "best_train_profit": self.best_train_profit,
            "best_test_profit": self.best_test_profit,
            "overfit_score": self.overfit_score,
            "method_used": self.method_used.value,
            "n_trials": len(self.all_trials),
            "n_valid_trials": len(self.valid_trials),
            "trials": [t.to_dict() for t in self.all_trials],
        }
    
    def save(self, path: str | Path) -> Path:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path


class BaseOptimizer(ABC):
    """Abstract base class for optimization strategies."""
    
    def __init__(
        self,
        param_specs: list[ParameterSpec],
        config: OptimizationConfig,
        rng: np.random.Generator | None = None,
    ):
        self.param_specs = {p.name: p for p in param_specs}
        self.config = config
        self.rng = rng or np.random.default_rng(config.seed)
        self.history: list[TrialResult] = []
    
    @abstractmethod
    def suggest(self, n_suggestions: int = 1) -> list[ParamDict]:
        """Suggest next parameter combinations to evaluate."""
        pass
    
    @abstractmethod
    def update(self, results: list[TrialResult]) -> None:
        """Update optimizer state with new results."""
        pass
    
    def _random_params(self) -> ParamDict:
        """Generate random parameter combination."""
        return {name: spec.sample(self.rng) for name, spec in self.param_specs.items()}


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian Optimization using Gaussian Process surrogate.
    
    Uses Expected Improvement (EI) acquisition function to balance
    exploration and exploitation.
    """
    
    def __init__(
        self,
        param_specs: list[ParameterSpec],
        config: OptimizationConfig,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(param_specs, config, rng)
        self.X: list[np.ndarray] = []  # Normalized parameter vectors
        self.y: list[float] = []        # Observed profits
        
        # GP hyperparameters
        self.length_scale = 0.5
        self.noise_var = 0.01
        self.amplitude = 1.0
        
        # Best observed value for EI calculation
        self.best_y = float("-inf")
    
    def _to_vector(self, params: ParamDict) -> np.ndarray:
        """Convert params dict to normalized vector."""
        return np.array([
            self.param_specs[name].normalize(params.get(name, self.param_specs[name].default))
            for name in sorted(self.param_specs.keys())
        ])
    
    def _from_vector(self, vector: np.ndarray) -> ParamDict:
        """Convert normalized vector to params dict."""
        names = sorted(self.param_specs.keys())
        return {
            name: self.param_specs[name].denormalize(vector[i])
            for i, name in enumerate(names)
        }
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (squared exponential) kernel."""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        return self.amplitude * np.exp(-0.5 * sqdist / self.length_scale**2)
    
    def _gp_predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance using Gaussian Process."""
        if len(self.X) == 0:
            n = len(X_new)
            return np.zeros(n), np.ones(n) * self.amplitude
        
        X_train = np.array(self.X)
        y_train = np.array(self.y)
        
        # Kernel matrices
        K = self._rbf_kernel(X_train, X_train) + self.noise_var * np.eye(len(X_train))
        K_s = self._rbf_kernel(X_train, X_new)
        K_ss = self._rbf_kernel(X_new, X_new)
        
        # Cholesky decomposition for numerical stability
        try:
            L = np.linalg.cholesky(K + 1e-6 * np.eye(len(K)))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            mean = K_s.T @ alpha
            
            v = np.linalg.solve(L, K_s)
            var = np.diag(K_ss) - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)  # Ensure positive variance
        except np.linalg.LinAlgError:
            # Fallback if Cholesky fails
            mean = np.mean(y_train) * np.ones(len(X_new))
            var = np.ones(len(X_new)) * self.amplitude
        
        return mean, var
    
    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Calculate Expected Improvement acquisition function."""
        mean, var = self._gp_predict(X)
        std = np.sqrt(var)
        
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (mean - self.best_y - xi) / std
            ei = (mean - self.best_y - xi) * self._norm_cdf(z) + std * self._norm_pdf(z)
            ei[std <= 0] = 0.0
        
        return ei
    
    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        """Standard normal CDF."""
        from scipy.special import erf
        return 0.5 * (1 + erf(x / np.sqrt(2)))
    
    @staticmethod
    def _norm_pdf(x: np.ndarray) -> np.ndarray:
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def suggest(self, n_suggestions: int = 1) -> list[ParamDict]:
        """Suggest parameters using Expected Improvement."""
        suggestions = []
        
        # Initial random exploration
        if len(self.X) < 5:
            for _ in range(n_suggestions):
                suggestions.append(self._random_params())
            return suggestions
        
        # Generate candidates
        n_candidates = 1000
        dim = len(self.param_specs)
        candidates = self.rng.uniform(0, 1, (n_candidates, dim))
        
        # Calculate EI for all candidates
        ei = self._expected_improvement(candidates)
        
        # Select top candidates
        for _ in range(n_suggestions):
            # Exploration vs exploitation
            if self.rng.random() < self.config.exploration_ratio:
                idx = self.rng.integers(len(candidates))
            else:
                idx = np.argmax(ei)
            
            suggestions.append(self._from_vector(candidates[idx]))
            ei[idx] = -np.inf  # Don't select same candidate twice
        
        return suggestions
    
    def update(self, results: list[TrialResult]) -> None:
        """Update GP model with new observations."""
        for result in results:
            if result.is_valid:
                x = self._to_vector(result.params)
                self.X.append(x)
                self.y.append(result.profit_value)
                
                if result.profit_value > self.best_y:
                    self.best_y = result.profit_value
        
        self.history.extend(results)
        
        # Adapt length scale based on observations
        if len(self.y) > 10:
            y_std = np.std(self.y)
            if y_std > 0:
                self.amplitude = y_std**2


class EvolutionaryOptimizer(BaseOptimizer):
    """
    Evolutionary optimization using CMA-ES-inspired strategy.
    
    Maintains a population that evolves toward better solutions
    through selection, recombination, and mutation.
    """
    
    def __init__(
        self,
        param_specs: list[ParameterSpec],
        config: OptimizationConfig,
        rng: np.random.Generator | None = None,
        population_size: int = 20,
    ):
        super().__init__(param_specs, config, rng)
        
        self.dim = len(param_specs)
        self.population_size = population_size
        
        # CMA-ES-like parameters
        self.mean = np.full(self.dim, 0.5)  # Start at center
        self.sigma = 0.3                     # Initial step size
        self.C = np.eye(self.dim)            # Covariance matrix
        
        # Evolution path
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
        
        # Learning rates
        self.mu = population_size // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        
        self.mueff = 1.0 / (self.weights**2).sum()
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff))
        
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        
        self.generation = 0
        self.best_solution = None
        self.best_fitness = float("-inf")
    
    def _to_vector(self, params: ParamDict) -> np.ndarray:
        """Convert params dict to normalized vector."""
        names = sorted(self.param_specs.keys())
        return np.array([
            self.param_specs[name].normalize(params.get(name, self.param_specs[name].default))
            for name in names
        ])
    
    def _from_vector(self, vector: np.ndarray) -> ParamDict:
        """Convert normalized vector to params dict."""
        # Clip to valid range
        vector = np.clip(vector, 0, 1)
        names = sorted(self.param_specs.keys())
        return {
            name: self.param_specs[name].denormalize(vector[i])
            for i, name in enumerate(names)
        }
    
    def suggest(self, n_suggestions: int = 1) -> list[ParamDict]:
        """Generate offspring population."""
        suggestions = []
        
        # Sample from multivariate normal
        try:
            L = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            L = np.eye(self.dim)
        
        for _ in range(n_suggestions):
            z = self.rng.standard_normal(self.dim)
            x = self.mean + self.sigma * (L @ z)
            x = np.clip(x, 0, 1)  # Bound to valid range
            suggestions.append(self._from_vector(x))
        
        return suggestions
    
    def update(self, results: list[TrialResult]) -> None:
        """Update distribution based on fitness."""
        self.history.extend(results)
        
        # Sort by fitness (descending)
        valid_results = [r for r in results if r.is_valid]
        if not valid_results:
            return
        
        sorted_results = sorted(valid_results, key=lambda r: r.profit_value, reverse=True)
        
        # Update best solution
        if sorted_results[0].profit_value > self.best_fitness:
            self.best_fitness = sorted_results[0].profit_value
            self.best_solution = self._to_vector(sorted_results[0].params)
        
        # Select top mu individuals
        selected = sorted_results[:self.mu]
        if len(selected) < 2:
            return
        
        # Weighted recombination
        vectors = np.array([self._to_vector(r.params) for r in selected])
        weights = self.weights[:len(selected)]
        weights /= weights.sum()
        
        old_mean = self.mean.copy()
        self.mean = np.average(vectors, axis=0, weights=weights)
        
        # Update evolution paths
        y = (self.mean - old_mean) / self.sigma
        
        self.p_sigma = (1 - self.cs) * self.p_sigma + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * y
        hsig = np.linalg.norm(self.p_sigma) / np.sqrt(1 - (1 - self.cs)**(2 * (self.generation + 1))) < (1.4 + 2 / (self.dim + 1)) * np.sqrt(self.dim)
        
        self.p_c = (1 - self.cc) * self.p_c
        if hsig:
            self.p_c += np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        
        # Update covariance matrix
        artmp = (vectors[:len(weights)] - old_mean) / self.sigma
        self.C = (1 - self.c1 - self.cmu) * self.C
        self.C += self.c1 * np.outer(self.p_c, self.p_c)
        self.C += self.cmu * (weights[:, None] * artmp).T @ artmp
        
        # Ensure symmetry
        self.C = (self.C + self.C.T) / 2
        
        # Update step size
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.p_sigma) / np.sqrt(self.dim) - 1))
        self.sigma = np.clip(self.sigma, 0.01, 1.0)
        
        self.generation += 1


class ContextualBanditOptimizer(BaseOptimizer):
    """
    Contextual Bandit optimizer using Thompson Sampling.
    
    Discretizes parameter space into arms and uses Bayesian
    updating to balance exploration and exploitation.
    """
    
    def __init__(
        self,
        param_specs: list[ParameterSpec],
        config: OptimizationConfig,
        rng: np.random.Generator | None = None,
        n_arms_per_param: int = 10,
    ):
        super().__init__(param_specs, config, rng)
        
        self.n_arms_per_param = n_arms_per_param
        
        # Discretize each parameter
        self.arm_values: dict[str, np.ndarray] = {}
        for name, spec in self.param_specs.items():
            if spec.integer:
                values = np.arange(spec.min_value, spec.max_value + 1, max(1, (spec.max_value - spec.min_value) // n_arms_per_param))
            elif spec.log_scale:
                values = np.logspace(np.log10(spec.min_value), np.log10(spec.max_value), n_arms_per_param)
            else:
                values = np.linspace(spec.min_value, spec.max_value, n_arms_per_param)
            self.arm_values[name] = values
        
        # Beta distribution parameters for each arm
        # Using (alpha, beta) for success/failure counts
        self.alpha: dict[str, np.ndarray] = {name: np.ones(len(vals)) for name, vals in self.arm_values.items()}
        self.beta: dict[str, np.ndarray] = {name: np.ones(len(vals)) for name, vals in self.arm_values.items()}
        
        self.best_profit = float("-inf")
    
    def _discretize(self, params: ParamDict) -> dict[str, int]:
        """Find nearest arm indices for each parameter."""
        indices = {}
        for name, value in params.items():
            arms = self.arm_values[name]
            idx = np.argmin(np.abs(arms - value))
            indices[name] = idx
        return indices
    
    def suggest(self, n_suggestions: int = 1) -> list[ParamDict]:
        """Suggest parameters using Thompson Sampling."""
        suggestions = []
        
        for _ in range(n_suggestions):
            params = {}
            for name in self.param_specs:
                # Thompson sampling: sample from Beta distribution
                samples = self.rng.beta(self.alpha[name], self.beta[name])
                best_arm = np.argmax(samples)
                params[name] = float(self.arm_values[name][best_arm])
            
            suggestions.append(params)
        
        return suggestions
    
    def update(self, results: list[TrialResult]) -> None:
        """Update arm statistics with results."""
        for result in results:
            if not result.is_valid:
                continue
            
            indices = self._discretize(result.params)
            
            # Update beta distributions
            # Treat profit > median as success
            if len(self.history) > 0:
                median_profit = np.median([r.profit_value for r in self.history if r.is_valid])
            else:
                median_profit = 0
            
            is_success = result.profit_value > median_profit
            
            for name, idx in indices.items():
                if is_success:
                    self.alpha[name][idx] += 1
                else:
                    self.beta[name][idx] += 1
            
            if result.profit_value > self.best_profit:
                self.best_profit = result.profit_value
        
        self.history.extend(results)


class MLProfitOptimizer:
    """
    Main optimizer class that orchestrates parameter tuning for profit maximization.
    
    Usage:
        optimizer = MLProfitOptimizer(param_specs, config)
        result = await optimizer.optimize(backtest_fn, data)
    """
    
    def __init__(
        self,
        param_specs: list[ParameterSpec],
        config: OptimizationConfig | None = None,
    ):
        self.param_specs = param_specs
        self.config = config or OptimizationConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        # Initialize optimizer based on method
        self._optimizer = self._create_optimizer()
        
        # State
        self.trials: list[TrialResult] = []
        self.best_trial: TrialResult | None = None
        self.iteration = 0
        self.no_improvement_count = 0
        
    def _create_optimizer(self) -> BaseOptimizer:
        """Create the appropriate optimizer based on config."""
        if self.config.method == OptimizationMethod.BAYESIAN:
            return BayesianOptimizer(self.param_specs, self.config, self.rng)
        elif self.config.method == OptimizationMethod.EVOLUTIONARY:
            return EvolutionaryOptimizer(self.param_specs, self.config, self.rng)
        elif self.config.method == OptimizationMethod.BANDIT:
            return ContextualBanditOptimizer(self.param_specs, self.config, self.rng)
        else:
            # Default to Bayesian
            return BayesianOptimizer(self.param_specs, self.config, self.rng)
    
    def _compute_profit_metric(
        self,
        backtest_result: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        """
        Compute profit value based on configured metric.
        
        Returns:
            (primary_value, all_metrics_dict)
        """
        trades = backtest_result.get("trades", [])
        equity_curve = backtest_result.get("equity_curve", [])
        
        metrics = {}
        
        # Total return
        total_return = backtest_result.get("total_return", 0.0)
        metrics["total_return"] = total_return
        
        # Net profit
        net_profit = backtest_result.get("net_profit", 0.0)
        metrics["net_profit"] = net_profit
        
        # Profit factor
        gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        metrics["profit_factor"] = profit_factor
        
        # Win rate
        winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
        win_rate = winning_trades / len(trades) if trades else 0.0
        metrics["win_rate"] = win_rate
        
        # Expectancy (average profit per trade)
        pnls = [t.get("pnl", 0) for t in trades]
        expectancy = np.mean(pnls) if pnls else 0.0
        metrics["expectancy"] = expectancy
        
        # Max drawdown
        if len(equity_curve) > 0:
            if isinstance(equity_curve, pd.Series):
                equity = equity_curve.values
            else:
                equity = np.array(equity_curve)
            
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        else:
            max_drawdown = backtest_result.get("max_drawdown", 0.0)
        metrics["max_drawdown"] = max_drawdown
        
        # Sharpe ratio
        if pnls and len(pnls) > 1:
            sharpe = np.mean(pnls) / np.std(pnls, ddof=1) if np.std(pnls) > 0 else 0.0
            sharpe *= np.sqrt(252)  # Annualize
        else:
            sharpe = backtest_result.get("sharpe_ratio", 0.0)
        metrics["sharpe_ratio"] = sharpe
        
        # Sortino ratio
        downside_returns = [p for p in pnls if p < 0]
        if len(downside_returns) > 1:
            downside_std = np.std(downside_returns, ddof=1)
            sortino = np.mean(pnls) / downside_std if downside_std > 0 else 0.0
            sortino *= np.sqrt(252)
        else:
            sortino = backtest_result.get("sortino_ratio", 0.0)
        metrics["sortino_ratio"] = sortino
        
        # Calmar ratio
        calmar = total_return / max_drawdown if max_drawdown > 0 else 0.0
        metrics["calmar_ratio"] = calmar
        
        # CAGR (if we have time info)
        years = backtest_result.get("years", 1.0)
        if years > 0 and total_return > -1:
            cagr = (1 + total_return) ** (1 / years) - 1
        else:
            cagr = total_return
        metrics["cagr"] = cagr
        
        metrics["n_trades"] = len(trades)
        
        # Select primary metric
        metric_map = {
            ProfitMetric.TOTAL_RETURN: total_return,
            ProfitMetric.NET_PROFIT: net_profit,
            ProfitMetric.PROFIT_FACTOR: profit_factor,
            ProfitMetric.EXPECTANCY: expectancy,
            ProfitMetric.RISK_ADJUSTED_RETURN: sharpe * (1 - max_drawdown),  # Penalize drawdown
            ProfitMetric.CALMAR_RATIO: calmar,
            ProfitMetric.SORTINO_RATIO: sortino,
            ProfitMetric.CAGR: cagr,
        }
        
        primary_value = metric_map.get(self.config.profit_metric, total_return)
        
        return primary_value, metrics
    
    def _check_constraints(self, metrics: dict[str, float]) -> bool:
        """Check if result satisfies risk constraints."""
        if metrics.get("profit_factor", 0) < self.config.min_profit_factor:
            return False
        if metrics.get("max_drawdown", 1) > self.config.max_drawdown:
            return False
        if metrics.get("n_trades", 0) < self.config.min_trades:
            return False
        if metrics.get("win_rate", 0) < self.config.min_win_rate:
            return False
        return True
    
    def _check_convergence(self) -> tuple[bool, str]:
        """Check if optimization has converged."""
        if self.iteration < self.config.min_iterations:
            return False, "Below minimum iterations"
        
        if self.no_improvement_count >= self.config.patience:
            return True, f"No improvement for {self.config.patience} iterations"
        
        # Check if recent improvements are below threshold
        if len(self.trials) >= 10:
            recent_bests = []
            current_best = float("-inf")
            for trial in self.trials[-20:]:
                if trial.is_valid and trial.profit_value > current_best:
                    current_best = trial.profit_value
                    recent_bests.append(current_best)
            
            if len(recent_bests) >= 3:
                improvements = np.diff(recent_bests)
                if np.all(np.abs(improvements) < self.config.convergence_threshold):
                    return True, "Improvements below threshold"
        
        return False, ""
    
    async def _evaluate_params(
        self,
        params: ParamDict,
        backtest_fn: Callable[[ParamDict, pd.DataFrame], dict[str, Any]],
        data: pd.DataFrame,
        trial_id: int,
    ) -> TrialResult:
        """Evaluate a single parameter combination."""
        start_time = time.time()
        
        try:
            # Run backtest
            if self.config.use_walk_forward:
                # Split data for walk-forward validation
                split_idx = int(len(data) * self.config.train_ratio)
                train_data = data.iloc[:split_idx]
                test_data = data.iloc[split_idx:]
                
                train_result = backtest_fn(params, train_data)
                test_result = backtest_fn(params, test_data)
                
                train_profit, train_metrics = self._compute_profit_metric(train_result)
                test_profit, test_metrics = self._compute_profit_metric(test_result)
                
                # Use test performance as primary (prevents overfitting)
                profit_value = test_profit
                metrics = test_metrics
                
                # Calculate overfit score
                if test_profit != 0:
                    overfit_score = train_profit / test_profit if test_profit > 0 else float("inf")
                else:
                    overfit_score = float("inf") if train_profit > 0 else 1.0
                
                # Penalize overfitting
                if overfit_score > self.config.max_overfit_ratio:
                    profit_value *= 0.5  # Heavy penalty
                
                is_valid = self._check_constraints(metrics) and overfit_score < self.config.max_overfit_ratio * 2
                
                return TrialResult(
                    trial_id=trial_id,
                    params=params,
                    profit_value=profit_value,
                    metrics=metrics,
                    duration_seconds=time.time() - start_time,
                    is_valid=is_valid,
                    train_profit=train_profit,
                    test_profit=test_profit,
                    overfit_score=overfit_score,
                )
            else:
                result = backtest_fn(params, data)
                profit_value, metrics = self._compute_profit_metric(result)
                is_valid = self._check_constraints(metrics)
                
                return TrialResult(
                    trial_id=trial_id,
                    params=params,
                    profit_value=profit_value,
                    metrics=metrics,
                    duration_seconds=time.time() - start_time,
                    is_valid=is_valid,
                )
                
        except Exception as e:
            logger.warning(f"Trial {trial_id} failed: {e}")
            return TrialResult(
                trial_id=trial_id,
                params=params,
                profit_value=float("-inf"),
                metrics={},
                duration_seconds=time.time() - start_time,
                is_valid=False,
                error_message=str(e),
            )
    
    async def optimize(
        self,
        backtest_fn: Callable[[ParamDict, pd.DataFrame], dict[str, Any]],
        data: pd.DataFrame,
        initial_params: ParamDict | None = None,
    ) -> OptimizationResult:
        """
        Run the optimization loop.
        
        Args:
            backtest_fn: Function that takes (params, data) and returns backtest result dict
            data: DataFrame with OHLCV data
            initial_params: Optional starting parameters
            
        Returns:
            OptimizationResult with best parameters and all trial history
        """
        start_time = time.time()
        trial_id = 0
        
        logger.info(f"Starting {self.config.method.value} optimization with {self.config.max_iterations} max iterations")
        
        # Evaluate initial params if provided
        if initial_params:
            trial = await self._evaluate_params(initial_params, backtest_fn, data, trial_id)
            self.trials.append(trial)
            self._optimizer.update([trial])
            trial_id += 1
            
            if trial.is_valid:
                self.best_trial = trial
                logger.info(f"Initial params: profit={trial.profit_value:.4f}")
        
        # Main optimization loop
        while self.iteration < self.config.max_iterations:
            self.iteration += 1
            
            # Get parameter suggestions
            suggestions = self._optimizer.suggest(self.config.batch_size)
            
            # Evaluate suggestions
            batch_results = []
            for params in suggestions:
                trial = await self._evaluate_params(params, backtest_fn, data, trial_id)
                batch_results.append(trial)
                self.trials.append(trial)
                trial_id += 1
            
            # Update optimizer
            self._optimizer.update(batch_results)
            
            # Track best result
            improvement = False
            for trial in batch_results:
                if trial.is_valid:
                    if self.best_trial is None or trial.profit_value > self.best_trial.profit_value:
                        self.best_trial = trial
                        improvement = True
                        self.no_improvement_count = 0
                        
                        if self.config.verbose:
                            logger.info(
                                f"Iteration {self.iteration}: New best profit={trial.profit_value:.4f} "
                                f"(metrics: {trial.metrics})"
                            )
            
            if not improvement:
                self.no_improvement_count += 1
            
            # Check convergence
            converged, reason = self._check_convergence()
            if converged:
                logger.info(f"Converged: {reason}")
                break
            
            if self.config.verbose and self.iteration % 10 == 0:
                valid_trials = [t for t in self.trials if t.is_valid]
                logger.info(
                    f"Iteration {self.iteration}: {len(valid_trials)}/{len(self.trials)} valid trials, "
                    f"best={self.best_trial.profit_value:.4f if self.best_trial else 0}"
                )
        
        # Compile results
        valid_trials = [t for t in self.trials if t.is_valid]
        
        result = OptimizationResult(
            best_params=self.best_trial.params if self.best_trial else {},
            best_profit=self.best_trial.profit_value if self.best_trial else 0.0,
            best_metrics=self.best_trial.metrics if self.best_trial else {},
            all_trials=self.trials,
            valid_trials=valid_trials,
            iterations_completed=self.iteration,
            total_time_seconds=time.time() - start_time,
            convergence_achieved=self.no_improvement_count >= self.config.patience,
            early_stopped=self.iteration < self.config.max_iterations,
            best_train_profit=self.best_trial.train_profit if self.best_trial else None,
            best_test_profit=self.best_trial.test_profit if self.best_trial else None,
            overfit_score=self.best_trial.overfit_score if self.best_trial else None,
            method_used=self.config.method,
        )
        
        # Save results
        if self.config.output_dir:
            output_path = Path(self.config.output_dir) / f"optimization_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
            result.save(output_path)
            logger.info(f"Results saved to {output_path}")
        
        return result


# =============================================================================
# Convenience function for quick optimization
# =============================================================================

def create_fibonacci_adx_optimizer() -> MLProfitOptimizer:
    """Create optimizer configured for Fibonacci ADX strategy."""
    param_specs = [
        ParameterSpec(
            name="adx_period",
            min_value=7,
            max_value=28,
            default=14,
            integer=True,
            description="ADX calculation period",
        ),
        ParameterSpec(
            name="adx_threshold",
            min_value=15,
            max_value=40,
            default=25,
            step=1.0,
            description="Minimum ADX for trend confirmation",
        ),
        ParameterSpec(
            name="swing_lookback",
            min_value=20,
            max_value=100,
            default=50,
            integer=True,
            description="Bars for swing high/low identification",
        ),
        ParameterSpec(
            name="tolerance",
            min_value=0.005,
            max_value=0.03,
            default=0.01,
            step=0.001,
            description="Price tolerance near Fibonacci level",
        ),
    ]
    
    config = OptimizationConfig(
        profit_metric=ProfitMetric.RISK_ADJUSTED_RETURN,
        method=OptimizationMethod.BAYESIAN,
        max_iterations=50,
        batch_size=4,
        use_walk_forward=True,
        train_ratio=0.7,
        min_trades=20,
        verbose=True,
    )
    
    return MLProfitOptimizer(param_specs, config)
