"""
Walk-Forward Optimization and Monte Carlo Simulation for Backtesting.

Implements production-grade backtesting enhancements:
- Walk-forward optimization with rolling windows
- Monte Carlo simulation for robustness testing
- Bootstrap resampling for statistical significance
- Parameter sensitivity analysis
- Regime-specific performance attribution

Step 6 of Trade Enhancement Roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import logging
from typing import Any, Callable
import random

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives."""
    
    SHARPE_RATIO = auto()
    SORTINO_RATIO = auto()
    CALMAR_RATIO = auto()
    TOTAL_RETURN = auto()
    RISK_ADJUSTED_RETURN = auto()
    MAX_DRAWDOWN = auto()
    WIN_RATE = auto()
    PROFIT_FACTOR = auto()


class MonteCarloMethod(Enum):
    """Monte Carlo simulation methods."""
    
    BOOTSTRAP = auto()  # Resample historical returns
    BLOCK_BOOTSTRAP = auto()  # Preserve autocorrelation
    PARAMETRIC = auto()  # Assume distribution
    PATH_SIMULATION = auto()  # Geometric Brownian motion


@dataclass
class WalkForwardWindow:
    """Single walk-forward window."""
    
    in_sample_start: datetime
    in_sample_end: datetime
    out_sample_start: datetime
    out_sample_end: datetime
    optimal_params: dict[str, Any]
    in_sample_metric: float
    out_sample_metric: float
    
    @property
    def is_overfitted(self) -> bool:
        """Check if out-sample significantly underperforms in-sample."""
        if self.in_sample_metric == 0:
            return False
        degradation = 1 - (self.out_sample_metric / self.in_sample_metric)
        return degradation > 0.5  # >50% degradation


@dataclass
class WalkForwardResult:
    """Complete walk-forward optimization result."""
    
    windows: list[WalkForwardWindow]
    objective: OptimizationObjective
    aggregated_out_sample_metric: float
    parameter_stability: dict[str, float]  # Param -> coefficient of variation
    robustness_score: float  # 0-1 score
    optimal_stable_params: dict[str, Any]
    total_in_sample_periods: int
    total_out_sample_periods: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def window_count(self) -> int:
        return len(self.windows)
        
    @property
    def average_degradation(self) -> float:
        """Average performance degradation from in-sample to out-sample."""
        degradations = []
        for w in self.windows:
            if w.in_sample_metric != 0:
                degradations.append(1 - (w.out_sample_metric / w.in_sample_metric))
        return float(np.mean(degradations)) if degradations else 0


@dataclass
class MonteCarloPath:
    """Single Monte Carlo simulation path."""
    
    path_id: int
    returns: np.ndarray
    cumulative_return: float
    max_drawdown: float
    sharpe_ratio: float
    final_value: float
    
    
@dataclass
class MonteCarloResult:
    """Complete Monte Carlo simulation result."""
    
    method: MonteCarloMethod
    num_simulations: int
    paths: list[MonteCarloPath]
    confidence_intervals: dict[str, tuple[float, float]]  # Metric -> (low, high)
    expected_values: dict[str, float]
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    probability_of_profit: float
    probability_of_ruin: float  # P(drawdown > threshold)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def return_distribution(self) -> np.ndarray:
        """Get distribution of final returns."""
        return np.array([p.cumulative_return for p in self.paths])


@dataclass
class SensitivityResult:
    """Parameter sensitivity analysis result."""
    
    parameter_name: str
    base_value: Any
    test_values: list[Any]
    metrics_by_value: dict[Any, dict[str, float]]
    optimal_value: Any
    sensitivity_score: float  # Higher = more sensitive
    is_robust: bool  # Performance stable across values


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization."""
    
    in_sample_periods: int = 252  # Trading days
    out_sample_periods: int = 63  # Quarter
    step_size: int = 21  # Monthly steps
    min_windows: int = 4
    objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO
    anchored: bool = False  # If True, in-sample always starts from beginning


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    
    num_simulations: int = 1000
    method: MonteCarloMethod = MonteCarloMethod.BLOCK_BOOTSTRAP
    block_size: int = 21  # For block bootstrap
    horizon_days: int = 252  # Simulation horizon
    initial_capital: float = 100000
    ruin_threshold: float = 0.50  # 50% drawdown = ruin
    random_seed: int | None = None


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization Engine.
    
    Implements rolling window optimization to avoid overfitting.
    
    Example:
        >>> optimizer = WalkForwardOptimizer(config)
        >>> result = optimizer.optimize(
        ...     data=price_data,
        ...     parameter_space={"sma_fast": range(5, 20), "sma_slow": range(20, 50)},
        ...     strategy_fn=run_sma_strategy,
        ... )
    """
    
    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        """Initialize optimizer."""
        self.config = config or WalkForwardConfig()
        
    def generate_windows(
        self,
        data: pd.DataFrame,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate in-sample/out-sample window pairs.
        
        Returns:
            List of (in_sample_df, out_sample_df) tuples
        """
        windows = []
        total_periods = len(data)
        
        required_periods = self.config.in_sample_periods + self.config.out_sample_periods
        
        if total_periods < required_periods:
            logger.warning(f"Insufficient data: {total_periods} < {required_periods}")
            return windows
            
        start_idx = 0 if self.config.anchored else 0
        
        while True:
            if self.config.anchored:
                in_sample_start = 0
            else:
                in_sample_start = start_idx
                
            in_sample_end = in_sample_start + self.config.in_sample_periods
            out_sample_start = in_sample_end
            out_sample_end = out_sample_start + self.config.out_sample_periods
            
            if out_sample_end > total_periods:
                break
                
            in_sample = data.iloc[in_sample_start:in_sample_end]
            out_sample = data.iloc[out_sample_start:out_sample_end]
            
            windows.append((in_sample, out_sample))
            
            start_idx += self.config.step_size
            
        if len(windows) < self.config.min_windows:
            logger.warning(f"Only {len(windows)} windows (min: {self.config.min_windows})")
            
        return windows
        
    def optimize(
        self,
        data: pd.DataFrame,
        parameter_space: dict[str, list[Any]],
        strategy_fn: Callable[[pd.DataFrame, dict], dict[str, float]],
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.
        
        Args:
            data: Price data with DatetimeIndex
            parameter_space: Parameter -> list of values to test
            strategy_fn: Function(data, params) -> metrics dict
            
        Returns:
            WalkForwardResult with optimized parameters
        """
        windows = self.generate_windows(data)
        
        if not windows:
            raise ValueError("No valid windows generated")
            
        wf_windows: list[WalkForwardWindow] = []
        all_optimal_params: list[dict[str, Any]] = []
        
        objective_key = self._objective_to_key(self.config.objective)
        
        for in_sample, out_sample in windows:
            # Grid search on in-sample
            best_params = None
            best_metric = float("-inf")
            
            param_combinations = self._generate_param_combinations(parameter_space)
            
            for params in param_combinations:
                try:
                    metrics = strategy_fn(in_sample, params)
                    metric_value = metrics.get(objective_key, 0)
                    
                    if metric_value > best_metric:
                        best_metric = metric_value
                        best_params = params.copy()
                except Exception as e:
                    logger.debug(f"Strategy failed with params {params}: {e}")
                    continue
                    
            if best_params is None:
                continue
                
            # Validate on out-sample
            try:
                out_metrics = strategy_fn(out_sample, best_params)
                out_metric = out_metrics.get(objective_key, 0)
            except Exception:
                out_metric = 0
                
            wf_window = WalkForwardWindow(
                in_sample_start=in_sample.index[0],
                in_sample_end=in_sample.index[-1],
                out_sample_start=out_sample.index[0],
                out_sample_end=out_sample.index[-1],
                optimal_params=best_params,
                in_sample_metric=best_metric,
                out_sample_metric=out_metric,
            )
            
            wf_windows.append(wf_window)
            all_optimal_params.append(best_params)
            
        # Analyze parameter stability
        param_stability = self._calculate_param_stability(all_optimal_params)
        
        # Calculate stable params (median of optimal)
        stable_params = self._calculate_stable_params(all_optimal_params)
        
        # Calculate robustness score
        robustness = self._calculate_robustness(wf_windows)
        
        # Aggregate out-sample performance
        agg_out_sample = float(np.mean([w.out_sample_metric for w in wf_windows]))
        
        return WalkForwardResult(
            windows=wf_windows,
            objective=self.config.objective,
            aggregated_out_sample_metric=agg_out_sample,
            parameter_stability=param_stability,
            robustness_score=robustness,
            optimal_stable_params=stable_params,
            total_in_sample_periods=self.config.in_sample_periods * len(wf_windows),
            total_out_sample_periods=self.config.out_sample_periods * len(wf_windows),
        )
        
    def _objective_to_key(self, objective: OptimizationObjective) -> str:
        """Convert objective enum to metrics dict key."""
        mapping = {
            OptimizationObjective.SHARPE_RATIO: "sharpe_ratio",
            OptimizationObjective.SORTINO_RATIO: "sortino_ratio",
            OptimizationObjective.CALMAR_RATIO: "calmar_ratio",
            OptimizationObjective.TOTAL_RETURN: "total_return",
            OptimizationObjective.RISK_ADJUSTED_RETURN: "risk_adjusted_return",
            OptimizationObjective.MAX_DRAWDOWN: "max_drawdown",
            OptimizationObjective.WIN_RATE: "win_rate",
            OptimizationObjective.PROFIT_FACTOR: "profit_factor",
        }
        return mapping.get(objective, "sharpe_ratio")
        
    def _generate_param_combinations(
        self,
        parameter_space: dict[str, list[Any]],
    ) -> list[dict[str, Any]]:
        """Generate all parameter combinations (grid search)."""
        import itertools
        
        keys = list(parameter_space.keys())
        values = list(parameter_space.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
            
        return combinations
        
    def _calculate_param_stability(
        self,
        all_params: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Calculate coefficient of variation for each parameter."""
        if not all_params:
            return {}
            
        stability = {}
        
        for param_name in all_params[0].keys():
            values = [p[param_name] for p in all_params]
            
            # Only calculate for numeric parameters
            try:
                values = [float(v) for v in values]
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Coefficient of variation (lower = more stable)
                cv = std_val / mean_val if mean_val != 0 else float("inf")
                stability[param_name] = float(cv)
            except (TypeError, ValueError):
                # Non-numeric parameter
                unique_ratio = len(set(values)) / len(values)
                stability[param_name] = unique_ratio
                
        return stability
        
    def _calculate_stable_params(
        self,
        all_params: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate stable (median) parameters."""
        if not all_params:
            return {}
            
        stable = {}
        
        for param_name in all_params[0].keys():
            values = [p[param_name] for p in all_params]
            
            try:
                values = [float(v) for v in values]
                stable[param_name] = float(np.median(values))
            except (TypeError, ValueError):
                # Mode for non-numeric
                from collections import Counter
                stable[param_name] = Counter(values).most_common(1)[0][0]
                
        return stable
        
    def _calculate_robustness(self, windows: list[WalkForwardWindow]) -> float:
        """Calculate robustness score (0-1)."""
        if not windows:
            return 0.0
            
        # Factors:
        # 1. Low degradation from in-sample to out-sample
        # 2. Positive out-sample performance
        # 3. Low overfitting
        
        degradations = []
        positive_out = 0
        overfitted = 0
        
        for w in windows:
            if w.in_sample_metric != 0:
                deg = 1 - (w.out_sample_metric / w.in_sample_metric)
                degradations.append(max(0, deg))  # Only count degradation
                
            if w.out_sample_metric > 0:
                positive_out += 1
                
            if w.is_overfitted:
                overfitted += 1
                
        avg_degradation = np.mean(degradations) if degradations else 0
        pct_positive = positive_out / len(windows)
        pct_not_overfitted = 1 - (overfitted / len(windows))
        
        # Weight and combine
        robustness = 0.4 * (1 - avg_degradation) + 0.3 * pct_positive + 0.3 * pct_not_overfitted
        
        return max(0, min(1, robustness))


class MonteCarloSimulator:
    """
    Monte Carlo Simulation Engine.
    
    Simulates strategy performance under different market conditions.
    
    Example:
        >>> simulator = MonteCarloSimulator(config)
        >>> result = simulator.simulate(
        ...     returns=daily_returns,
        ... )
        >>> print(f"95% VaR: {result.var_95:.2%}")
    """
    
    def __init__(self, config: MonteCarloConfig | None = None) -> None:
        """Initialize simulator."""
        self.config = config or MonteCarloConfig()
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            random.seed(self.config.random_seed)
            
    def simulate(
        self,
        returns: pd.Series | np.ndarray,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.
        
        Args:
            returns: Historical return series
            
        Returns:
            MonteCarloResult with simulation analysis
        """
        returns = np.array(returns)
        
        if self.config.method == MonteCarloMethod.BOOTSTRAP:
            paths = self._bootstrap_simulation(returns)
        elif self.config.method == MonteCarloMethod.BLOCK_BOOTSTRAP:
            paths = self._block_bootstrap_simulation(returns)
        elif self.config.method == MonteCarloMethod.PARAMETRIC:
            paths = self._parametric_simulation(returns)
        else:
            paths = self._path_simulation(returns)
            
        return self._analyze_paths(paths)
        
    def _bootstrap_simulation(
        self,
        returns: np.ndarray,
    ) -> list[MonteCarloPath]:
        """Standard bootstrap resampling."""
        paths = []
        
        for i in range(self.config.num_simulations):
            # Resample with replacement
            sampled = np.random.choice(
                returns,
                size=self.config.horizon_days,
                replace=True,
            )
            
            path = self._create_path(i, sampled)
            paths.append(path)
            
        return paths
        
    def _block_bootstrap_simulation(
        self,
        returns: np.ndarray,
    ) -> list[MonteCarloPath]:
        """Block bootstrap to preserve autocorrelation."""
        paths = []
        n = len(returns)
        block_size = min(self.config.block_size, n)
        
        for i in range(self.config.num_simulations):
            sampled = []
            
            while len(sampled) < self.config.horizon_days:
                # Random block start
                start = np.random.randint(0, max(1, n - block_size))
                block = returns[start:start + block_size]
                sampled.extend(block)
                
            sampled = np.array(sampled[:self.config.horizon_days])
            path = self._create_path(i, sampled)
            paths.append(path)
            
        return paths
        
    def _parametric_simulation(
        self,
        returns: np.ndarray,
    ) -> list[MonteCarloPath]:
        """Parametric simulation assuming normal distribution."""
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        paths = []
        
        for i in range(self.config.num_simulations):
            sampled = np.random.normal(mu, sigma, self.config.horizon_days)
            path = self._create_path(i, sampled)
            paths.append(path)
            
        return paths
        
    def _path_simulation(
        self,
        returns: np.ndarray,
    ) -> list[MonteCarloPath]:
        """Geometric Brownian Motion simulation."""
        mu = np.mean(returns) * 252  # Annualized
        sigma = np.std(returns) * np.sqrt(252)  # Annualized
        dt = 1 / 252  # Daily
        
        paths = []
        
        for i in range(self.config.num_simulations):
            # GBM: dS = mu*S*dt + sigma*S*dW
            z = np.random.standard_normal(self.config.horizon_days)
            
            # Log returns
            log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
            
            path = self._create_path(i, log_returns)
            paths.append(path)
            
        return paths
        
    def _create_path(
        self,
        path_id: int,
        returns: np.ndarray,
    ) -> MonteCarloPath:
        """Create MonteCarloPath from return series."""
        # Cumulative return
        cumulative = np.cumprod(1 + returns)
        final_value = self.config.initial_capital * cumulative[-1]
        cumulative_return = cumulative[-1] - 1
        
        # Max drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = float(np.min(drawdowns))
        
        # Sharpe ratio (annualized)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = (mean_ret * 252) / (std_ret * np.sqrt(252)) if std_ret > 0 else 0
        
        return MonteCarloPath(
            path_id=path_id,
            returns=returns,
            cumulative_return=float(cumulative_return),
            max_drawdown=float(max_dd),
            sharpe_ratio=float(sharpe),
            final_value=float(final_value),
        )
        
    def _analyze_paths(self, paths: list[MonteCarloPath]) -> MonteCarloResult:
        """Analyze simulation paths and compute statistics."""
        returns = np.array([p.cumulative_return for p in paths])
        drawdowns = np.array([p.max_drawdown for p in paths])
        sharpes = np.array([p.sharpe_ratio for p in paths])
        final_values = np.array([p.final_value for p in paths])
        
        # Confidence intervals (95%)
        confidence_intervals = {
            "cumulative_return": (
                float(np.percentile(returns, 2.5)),
                float(np.percentile(returns, 97.5)),
            ),
            "max_drawdown": (
                float(np.percentile(drawdowns, 2.5)),
                float(np.percentile(drawdowns, 97.5)),
            ),
            "sharpe_ratio": (
                float(np.percentile(sharpes, 2.5)),
                float(np.percentile(sharpes, 97.5)),
            ),
            "final_value": (
                float(np.percentile(final_values, 2.5)),
                float(np.percentile(final_values, 97.5)),
            ),
        }
        
        # Expected values
        expected_values = {
            "cumulative_return": float(np.mean(returns)),
            "max_drawdown": float(np.mean(drawdowns)),
            "sharpe_ratio": float(np.mean(sharpes)),
            "final_value": float(np.mean(final_values)),
        }
        
        # VaR and CVaR (on returns, negative = loss)
        var_95 = float(np.percentile(returns, 5))
        cvar_95 = float(np.mean(returns[returns <= var_95]))
        
        # Probabilities
        prob_profit = float(np.mean(returns > 0))
        prob_ruin = float(np.mean(drawdowns < -self.config.ruin_threshold))
        
        return MonteCarloResult(
            method=self.config.method,
            num_simulations=self.config.num_simulations,
            paths=paths,
            confidence_intervals=confidence_intervals,
            expected_values=expected_values,
            var_95=var_95,
            cvar_95=cvar_95,
            probability_of_profit=prob_profit,
            probability_of_ruin=prob_ruin,
        )


class SensitivityAnalyzer:
    """
    Parameter Sensitivity Analysis.
    
    Tests how sensitive strategy performance is to parameter changes.
    """
    
    def __init__(self) -> None:
        """Initialize analyzer."""
        pass
        
    def analyze_parameter(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame, dict], dict[str, float]],
        base_params: dict[str, Any],
        param_name: str,
        test_values: list[Any],
        primary_metric: str = "sharpe_ratio",
    ) -> SensitivityResult:
        """
        Analyze sensitivity to a single parameter.
        
        Args:
            data: Price data
            strategy_fn: Strategy function
            base_params: Base parameter set
            param_name: Parameter to vary
            test_values: Values to test
            primary_metric: Metric to optimize
            
        Returns:
            SensitivityResult
        """
        metrics_by_value: dict[Any, dict[str, float]] = {}
        
        for value in test_values:
            params = base_params.copy()
            params[param_name] = value
            
            try:
                metrics = strategy_fn(data, params)
                metrics_by_value[value] = metrics
            except Exception as e:
                logger.debug(f"Strategy failed with {param_name}={value}: {e}")
                continue
                
        if not metrics_by_value:
            return SensitivityResult(
                parameter_name=param_name,
                base_value=base_params.get(param_name),
                test_values=test_values,
                metrics_by_value={},
                optimal_value=base_params.get(param_name),
                sensitivity_score=1.0,
                is_robust=False,
            )
            
        # Find optimal value
        optimal_value = max(
            metrics_by_value.keys(),
            key=lambda v: metrics_by_value[v].get(primary_metric, 0),
        )
        
        # Calculate sensitivity score
        metric_values = [m.get(primary_metric, 0) for m in metrics_by_value.values()]
        mean_metric = np.mean(metric_values)
        std_metric = np.std(metric_values)
        
        # CV as sensitivity measure (higher = more sensitive)
        sensitivity = std_metric / abs(mean_metric) if mean_metric != 0 else float("inf")
        
        # Robust if CV < 0.3 and all values positive (for positive metrics)
        is_robust = sensitivity < 0.3 and all(v > 0 for v in metric_values if v != 0)
        
        return SensitivityResult(
            parameter_name=param_name,
            base_value=base_params.get(param_name),
            test_values=test_values,
            metrics_by_value=metrics_by_value,
            optimal_value=optimal_value,
            sensitivity_score=float(min(sensitivity, 10)),  # Cap at 10
            is_robust=is_robust,
        )
        
    def full_sensitivity_analysis(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame, dict], dict[str, float]],
        base_params: dict[str, Any],
        parameter_ranges: dict[str, list[Any]],
        primary_metric: str = "sharpe_ratio",
    ) -> dict[str, SensitivityResult]:
        """Run sensitivity analysis on all parameters."""
        results = {}
        
        for param_name, test_values in parameter_ranges.items():
            result = self.analyze_parameter(
                data=data,
                strategy_fn=strategy_fn,
                base_params=base_params,
                param_name=param_name,
                test_values=test_values,
                primary_metric=primary_metric,
            )
            results[param_name] = result
            
        return results
