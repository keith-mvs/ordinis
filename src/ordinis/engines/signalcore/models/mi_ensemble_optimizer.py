"""
MI Ensemble Hyperparameter Optimizer.

Implements profit-maximized parameter tuning using Bayesian optimization.
Integrates with ProofBench for backtesting and validates across market regimes.

Author: Ordinis Quantitative Team
Date: 2025-12-25
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from ordinis.engines.proofbench.core.engine import ProofBenchEngine
from ordinis.engines.proofbench.core.config import ProofBenchEngineConfig
from ordinis.engines.signalcore.models.mi_ensemble import MIEnsembleModel
from ordinis.engines.signalcore.core.model import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizationObjective:
    """Defines profit maximization objective with constraints."""

    primary_metric: str = "total_return"  # Main optimization target
    constraints: dict[str, tuple[str, float]] = field(default_factory=lambda: {
        "sharpe_ratio": (">=", 1.0),      # Minimum Sharpe ratio
        "max_drawdown": ("<=", 0.25),     # Maximum 25% drawdown
        "win_rate": (">=", 0.45),         # Minimum 45% win rate
        "profit_factor": (">=", 1.2),     # Minimum profit factor
    })
    penalty_weight: float = 100.0  # Penalty multiplier for constraint violations
    
    def calculate_fitness(self, metrics: dict[str, float]) -> float:
        """Calculate fitness score with penalties for constraint violations.
        
        Args:
            metrics: Backtest performance metrics
            
        Returns:
            Fitness score (higher is better)
        """
        # Start with primary metric (total return as percentage)
        fitness = metrics.get(self.primary_metric, 0.0)
        
        # Apply constraint penalties
        penalty = 0.0
        for metric_name, (operator, threshold) in self.constraints.items():
            value = metrics.get(metric_name, 0.0)
            
            if operator == ">=" and value < threshold:
                penalty += (threshold - value) * self.penalty_weight
            elif operator == "<=" and value > threshold:
                penalty += (value - threshold) * self.penalty_weight
        
        return fitness - penalty


@dataclass
class ParameterSpace:
    """Defines searchable parameter space for MI Ensemble."""
    
    # Mutual Information parameters
    mi_lookback: tuple[int, int] = (63, 504)  # 3 months to 2 years
    mi_bins: tuple[int, int] = (5, 20)  # Discretization bins
    forward_period: tuple[int, int] = (1, 21)  # Forward return period
    
    # Weight constraints
    min_weight: tuple[float, float] = (0.0, 0.1)
    max_weight: tuple[float, float] = (0.3, 0.7)
    
    # Ensemble parameters
    recalc_frequency: tuple[int, int] = (5, 63)  # Days between recalc
    ensemble_threshold: tuple[float, float] = (0.1, 0.5)
    min_signals_agree: tuple[int, int] = (1, 4)  # Number of signals
    
    def suggest_parameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest parameter values for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        return {
            "mi_lookback": trial.suggest_int("mi_lookback", *self.mi_lookback),
            "mi_bins": trial.suggest_int("mi_bins", *self.mi_bins),
            "forward_period": trial.suggest_int("forward_period", *self.forward_period),
            "min_weight": trial.suggest_float("min_weight", *self.min_weight),
            "max_weight": trial.suggest_float("max_weight", *self.max_weight),
            "recalc_frequency": trial.suggest_int("recalc_frequency", *self.recalc_frequency),
            "ensemble_threshold": trial.suggest_float("ensemble_threshold", *self.ensemble_threshold),
            "min_signals_agree": trial.suggest_int("min_signals_agree", *self.min_signals_agree),
        }


@dataclass
class ValidationStrategy:
    """Time-series cross-validation strategy."""
    
    n_splits: int = 5  # Number of train/test splits
    test_size_days: int = 126  # ~6 months test period
    gap_days: int = 0  # Gap between train and test
    
    def generate_splits(
        self, 
        df: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate time-series train/test splits.
        
        Args:
            df: Full dataset with datetime index
            
        Returns:
            List of (train_df, test_df) tuples
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        total_days = len(df)
        test_days = self.test_size_days
        
        splits = []
        for i in range(self.n_splits):
            # Calculate split point
            test_start_idx = total_days - (self.n_splits - i) * test_days
            test_end_idx = test_start_idx + test_days
            
            if test_start_idx < 252:  # Need minimum history
                continue
                
            # Apply gap
            train_end_idx = test_start_idx - self.gap_days
            
            train_df = df.iloc[:train_end_idx]
            test_df = df.iloc[test_start_idx:test_end_idx]
            
            if len(train_df) > 0 and len(test_df) > 0:
                splits.append((train_df, test_df))
        
        return splits


class MIEnsembleOptimizer:
    """Bayesian optimizer for MI Ensemble parameters.
    
    Uses Optuna with TPE sampler for efficient hyperparameter search.
    Integrates with ProofBench for accurate profit estimation.
    
    Example:
        >>> optimizer = MIEnsembleOptimizer(
        ...     data=historical_df,
        ...     symbols=["AAPL", "MSFT"],
        ...     objective=OptimizationObjective(primary_metric="sharpe_ratio")
        ... )
        >>> study = optimizer.optimize(n_trials=100)
        >>> best_params = study.best_params
        >>> print(f"Best Sharpe: {study.best_value:.2f}")
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        symbols: list[str],
        objective: OptimizationObjective | None = None,
        param_space: ParameterSpace | None = None,
        validation: ValidationStrategy | None = None,
        initial_capital: float = 100000.0,
    ):
        """Initialize optimizer.
        
        Args:
            data: Historical price data (multi-symbol if multiple)
            symbols: List of symbols to trade
            objective: Optimization objective and constraints
            param_space: Parameter search space
            validation: Cross-validation strategy
            initial_capital: Starting capital for backtest
        """
        self.data = data
        self.symbols = symbols
        self.objective = objective or OptimizationObjective()
        self.param_space = param_space or ParameterSpace()
        self.validation = validation or ValidationStrategy()
        self.initial_capital = initial_capital
        
        # Initialize ProofBench
        self.proofbench_config = ProofBenchEngineConfig(
            initial_capital=initial_capital,
            commission=0.001,  # 10 bps
            slippage=0.0005,   # 5 bps
        )
        
        self._trial_count = 0
        self._best_score = -np.inf
        
    async def _backtest_with_params(
        self,
        params: dict[str, Any],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Run backtest with given parameters.
        
        Args:
            params: Model parameters to test
            train_data: Training data (for MI calculation)
            test_data: Test data (for performance evaluation)
            
        Returns:
            Dictionary of performance metrics
        """
        # Create model with suggested parameters
        config = ModelConfig(
            model_id=f"mi_ensemble_trial_{self._trial_count}",
            model_type="ensemble",
            parameters=params,
        )
        model = MIEnsembleModel(config)
        
        # Initialize ProofBench
        proofbench = ProofBenchEngine(self.proofbench_config)
        await proofbench.initialize()
        
        try:
            # Run backtest on test period
            results = await proofbench.run_backtest(
                model=model,
                data=test_data,
                symbols=self.symbols,
            )
            
            # Extract metrics
            metrics = {
                "total_return": results.total_return_pct,
                "sharpe_ratio": results.sharpe_ratio,
                "sortino_ratio": results.sortino_ratio,
                "max_drawdown": abs(results.max_drawdown_pct) / 100.0,
                "win_rate": results.win_rate,
                "profit_factor": results.profit_factor,
                "total_trades": results.total_trades,
                "avg_trade_pct": results.avg_trade_return_pct,
            }
            
            return metrics
            
        finally:
            await proofbench.shutdown()
    
    def _objective_function(self, trial: optuna.Trial) -> float:
        """Optuna objective function.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Fitness score to maximize
        """
        self._trial_count += 1
        
        # Suggest parameters
        params = self.param_space.suggest_parameters(trial)
        
        # Run cross-validation
        splits = self.validation.generate_splits(self.data)
        
        if len(splits) == 0:
            logger.warning("No valid splits generated")
            return -np.inf
        
        fold_scores = []
        fold_metrics = []
        
        for fold_idx, (train_df, test_df) in enumerate(splits):
            try:
                # Run backtest (async, but we'll use asyncio.run for simplicity)
                import asyncio
                metrics = asyncio.run(
                    self._backtest_with_params(params, train_df, test_df)
                )
                
                # Calculate fitness
                score = self.objective.calculate_fitness(metrics)
                fold_scores.append(score)
                fold_metrics.append(metrics)
                
                # Report intermediate value for pruning
                trial.report(score, fold_idx)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                    
            except Exception as e:
                logger.warning(f"Fold {fold_idx} failed: {e}")
                fold_scores.append(-np.inf)
        
        # Average score across folds
        avg_score = np.mean([s for s in fold_scores if s > -np.inf])
        
        if avg_score > self._best_score:
            self._best_score = avg_score
            logger.info(
                f"Trial {self._trial_count}: New best score {avg_score:.4f} "
                f"with params {params}"
            )
        
        # Store metrics in trial user attributes
        if fold_metrics:
            avg_metrics = {
                key: np.mean([m[key] for m in fold_metrics])
                for key in fold_metrics[0].keys()
            }
            for key, value in avg_metrics.items():
                trial.set_user_attr(f"avg_{key}", value)
        
        return avg_score
    
    def optimize(
        self,
        n_trials: int = 100,
        timeout: int | None = None,
        n_jobs: int = 1,
        study_name: str | None = None,
        storage: str | None = None,
    ) -> optuna.Study:
        """Run Bayesian optimization.
        
        Args:
            n_trials: Maximum number of trials
            timeout: Maximum optimization time in seconds
            n_jobs: Number of parallel jobs (1 = sequential)
            study_name: Name for the study (for persistence)
            storage: Optuna storage URL (e.g., sqlite:///optuna.db)
            
        Returns:
            Optuna study object with results
        """
        # Configure sampler (TPE = Tree-structured Parzen Estimator)
        sampler = TPESampler(
            n_startup_trials=10,  # Random trials before TPE
            multivariate=True,     # Consider parameter interactions
            seed=42,
        )
        
        # Configure pruner (stops unpromising trials early)
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
        )
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
        
        logger.info(f"Starting optimization with {n_trials} trials")
        
        # Run optimization
        study.optimize(
            self._objective_function,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )
        
        # Log results
        logger.info(f"Optimization complete!")
        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        # Print constraint satisfaction
        best_trial = study.best_trial
        logger.info("\nBest trial metrics:")
        for key, value in best_trial.user_attrs.items():
            if key.startswith("avg_"):
                metric_name = key[4:]  # Remove 'avg_' prefix
                logger.info(f"  {metric_name}: {value:.4f}")
        
        return study
    
    def plot_optimization_history(
        self,
        study: optuna.Study,
        save_path: Path | None = None,
    ) -> None:
        """Plot optimization history and parameter importance.
        
        Args:
            study: Completed Optuna study
            save_path: Optional path to save plots
        """
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
            )
            import plotly.io as pio
            
            # Optimization history
            fig1 = plot_optimization_history(study)
            fig1.update_layout(title="MI Ensemble Optimization History")
            
            # Parameter importance
            fig2 = plot_param_importances(study)
            fig2.update_layout(title="Parameter Importance")
            
            # Parallel coordinates
            fig3 = plot_parallel_coordinate(study)
            fig3.update_layout(title="Parameter Interactions")
            
            if save_path:
                save_path = Path(save_path)
                save_path.mkdir(parents=True, exist_ok=True)
                
                pio.write_html(fig1, save_path / "optimization_history.html")
                pio.write_html(fig2, save_path / "param_importance.html")
                pio.write_html(fig3, save_path / "param_interactions.html")
                
                logger.info(f"Plots saved to {save_path}")
            else:
                fig1.show()
                fig2.show()
                fig3.show()
                
        except ImportError:
            logger.warning("Plotly not installed. Skipping visualization.")


def run_optimization_pipeline(
    data: pd.DataFrame,
    symbols: list[str],
    n_trials: int = 100,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Complete optimization pipeline with results export.
    
    Args:
        data: Historical price data
        symbols: Trading symbols
        n_trials: Number of optimization trials
        output_dir: Directory for results (default: artifacts/optimization/)
        
    Returns:
        Dictionary containing study and best parameters
    """
    if output_dir is None:
        output_dir = Path("artifacts/optimization/mi_ensemble")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize optimizer with profit-focused objective
    objective = OptimizationObjective(
        primary_metric="total_return",
        constraints={
            "sharpe_ratio": (">=", 1.2),
            "max_drawdown": ("<=", 0.20),
            "win_rate": (">=", 0.48),
            "profit_factor": (">=", 1.3),
        }
    )
    
    optimizer = MIEnsembleOptimizer(
        data=data,
        symbols=symbols,
        objective=objective,
    )
    
    # Run optimization with persistence
    storage_path = output_dir / "optuna_study.db"
    study = optimizer.optimize(
        n_trials=n_trials,
        study_name="mi_ensemble_profit_max",
        storage=f"sqlite:///{storage_path}",
    )
    
    # Save results
    results_df = study.trials_dataframe()
    results_df.to_csv(output_dir / "trials.csv", index=False)
    
    # Save best parameters
    best_params_file = output_dir / "best_parameters.json"
    import json
    with open(best_params_file, "w") as f:
        json.dump(
            {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "n_trials": len(study.trials),
                "optimization_date": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    
    # Generate plots
    optimizer.plot_optimization_history(study, save_path=output_dir)
    
    logger.info(f"Results saved to {output_dir}")
    
    return {
        "study": study,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "output_dir": output_dir,
    }
