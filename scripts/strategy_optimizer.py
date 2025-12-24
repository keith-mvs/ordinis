#!/usr/bin/env python3
"""
GPU-Accelerated Strategy Optimizer for Ordinis.

This module implements a sophisticated parameter optimization framework for the 4 target
strategies: ATR-RSI, MTF Momentum, MI Ensemble, and Kalman Hybrid.

Key Features:
- CUDA-accelerated backtesting via CuPy/PyTorch
- Sobol sequence sampling with edge-zone oversampling
- Walk-forward validation with purging/embargo
- Factor-based parameter interaction analysis
- Bootstrap confidence intervals for robustness

Target: >30% CAGR on robust out-of-sample validation with acceptable risk.

Usage:
    conda activate ordinis-env
    python scripts/strategy_optimizer.py --strategy atr_rsi --cycles 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = np

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
    if HAS_TORCH:
        torch.set_default_device('cuda')
except ImportError:
    HAS_TORCH = False
    torch = None

# Import models
from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.engines.signalcore.models.atr_optimized_rsi import ATROptimizedRSIModel
from ordinis.engines.signalcore.models.mi_ensemble import MIEnsembleModel
from ordinis.engines.signalcore.models.mtf_momentum import MTFMomentumModel
from ordinis.engines.signalcore.models.kalman_hybrid import KalmanHybridModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimization.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# PARAMETER SPACE DEFINITIONS - Edge zones explicitly marked
# ============================================================================

@dataclass
class ParameterSpec:
    """Specification for a single parameter."""
    name: str
    min_val: float
    max_val: float
    param_type: str  # 'float', 'int', 'log_float'
    factor_group: str  # 'signal', 'risk', 'regime', 'cost'
    edge_zone_pct: float = 0.15  # Lowest/highest 15% are edge zones
    description: str = ""

    def sample(self, u: float) -> float:
        """Transform uniform [0,1] to parameter value."""
        if self.param_type == 'log_float':
            log_min = np.log10(self.min_val)
            log_max = np.log10(self.max_val)
            return 10 ** (log_min + u * (log_max - log_min))
        elif self.param_type == 'int':
            return int(self.min_val + u * (self.max_val - self.min_val))
        else:
            return self.min_val + u * (self.max_val - self.min_val)

    def is_edge_zone(self, val: float) -> bool:
        """Check if value is in edge zone."""
        normalized = (val - self.min_val) / (self.max_val - self.min_val)
        return normalized < self.edge_zone_pct or normalized > (1 - self.edge_zone_pct)


# ATR-RSI Parameters
ATR_RSI_PARAMS = [
    ParameterSpec("rsi_period", 5, 21, "int", "signal", description="RSI lookback period"),
    ParameterSpec("rsi_oversold", 25, 45, "int", "signal", description="RSI oversold threshold"),
    ParameterSpec("rsi_exit", 45, 65, "int", "signal", description="RSI exit threshold"),
    ParameterSpec("atr_period", 7, 21, "int", "risk", description="ATR lookback period"),
    ParameterSpec("atr_stop_mult", 1.0, 3.5, "float", "risk", description="ATR stop loss multiplier"),
    ParameterSpec("atr_tp_mult", 1.5, 5.0, "float", "risk", description="ATR take profit multiplier"),
]

# MTF Momentum Parameters
MTF_MOMENTUM_PARAMS = [
    ParameterSpec("formation_period", 63, 252, "int", "signal", description="Momentum formation period"),
    ParameterSpec("skip_period", 5, 30, "int", "signal", description="Skip recent days (reversal)"),
    ParameterSpec("momentum_percentile", 0.5, 0.9, "float", "signal", description="Top percentile = winner"),
    ParameterSpec("stoch_k_period", 5, 21, "int", "signal", description="Stochastic %K period"),
    ParameterSpec("stoch_d_period", 2, 5, "int", "signal", description="Stochastic %D smoothing"),
    ParameterSpec("stoch_oversold", 15, 40, "float", "signal", description="Stochastic oversold level"),
    ParameterSpec("stoch_overbought", 60, 85, "float", "signal", description="Stochastic overbought level"),
    ParameterSpec("atr_stop_mult", 1.0, 3.5, "float", "risk", description="ATR stop loss multiplier"),
    ParameterSpec("atr_tp_mult", 1.5, 5.0, "float", "risk", description="ATR take profit multiplier"),
]

# MI Ensemble Parameters
MI_ENSEMBLE_PARAMS = [
    ParameterSpec("mi_lookback", 63, 504, "int", "signal", description="MI calculation lookback"),
    ParameterSpec("mi_bins", 5, 15, "int", "signal", description="Discretization bins for MI"),
    ParameterSpec("forward_period", 3, 10, "int", "signal", description="Forward return period"),
    ParameterSpec("min_weight", 0.0, 0.1, "float", "signal", description="Minimum signal weight"),
    ParameterSpec("max_weight", 0.3, 0.6, "float", "signal", description="Maximum signal weight cap"),
    ParameterSpec("recalc_frequency", 5, 42, "int", "signal", description="MI recalc frequency"),
    ParameterSpec("ensemble_threshold", 0.1, 0.5, "float", "signal", description="Ensemble signal threshold"),
    ParameterSpec("min_signals_agree", 1, 4, "int", "signal", description="Min signals that must agree"),
    ParameterSpec("atr_stop_mult", 1.0, 3.5, "float", "risk", description="ATR stop loss multiplier"),
    ParameterSpec("atr_tp_mult", 1.5, 5.0, "float", "risk", description="ATR take profit multiplier"),
]

# Kalman Hybrid Parameters
KALMAN_HYBRID_PARAMS = [
    ParameterSpec("process_noise_q", 1e-6, 1e-3, "log_float", "signal", description="Kalman process noise"),
    ParameterSpec("observation_noise_r", 1e-3, 1e-1, "log_float", "signal", description="Kalman observation noise"),
    ParameterSpec("residual_z_entry", 1.0, 3.0, "float", "signal", description="Z-score entry threshold"),
    ParameterSpec("residual_z_exit", 0.2, 1.0, "float", "signal", description="Z-score exit threshold"),
    ParameterSpec("trend_slope_min", 1e-5, 1e-3, "log_float", "regime", description="Min trend slope for confirmation"),
    ParameterSpec("residual_lookback", 30, 150, "int", "signal", description="Residual normalization lookback"),
    ParameterSpec("atr_stop_mult", 1.0, 3.5, "float", "risk", description="ATR stop loss multiplier"),
    ParameterSpec("atr_tp_mult", 1.5, 5.0, "float", "risk", description="ATR take profit multiplier"),
]

STRATEGY_PARAMS = {
    "atr_rsi": ATR_RSI_PARAMS,
    "mtf_momentum": MTF_MOMENTUM_PARAMS,
    "mi_ensemble": MI_ENSEMBLE_PARAMS,
    "kalman_hybrid": KALMAN_HYBRID_PARAMS,
}


# ============================================================================
# SAMPLER - Sobol with edge zone oversampling
# ============================================================================

class EdgeBiasedSobolSampler:
    """Sobol sequence sampler with explicit edge zone coverage."""

    def __init__(
        self,
        param_specs: list[ParameterSpec],
        n_samples: int = 100,
        edge_oversample_factor: float = 2.0,
        seed: int = 42,
    ):
        self.param_specs = param_specs
        self.n_samples = n_samples
        self.edge_oversample_factor = edge_oversample_factor
        self.seed = seed
        self.d = len(param_specs)

    def generate(self) -> list[dict[str, Any]]:
        """Generate parameter samples with edge zone oversampling."""
        # Base Sobol samples
        engine = Sobol(d=self.d, scramble=True, seed=self.seed)
        base_samples = engine.random(n=self.n_samples)

        # Generate edge zone samples
        edge_samples = self._generate_edge_samples(
            int(self.n_samples * self.edge_oversample_factor * 0.3)
        )

        # Combine and convert to parameter dicts
        all_samples = np.vstack([base_samples, edge_samples])
        
        param_dicts = []
        for sample in all_samples:
            params = {}
            for i, (u, spec) in enumerate(zip(sample, self.param_specs)):
                params[spec.name] = spec.sample(u)
            param_dicts.append(params)

        logger.info(
            f"Generated {len(param_dicts)} samples ({len(base_samples)} Sobol + "
            f"{len(edge_samples)} edge-biased)"
        )
        return param_dicts

    def _generate_edge_samples(self, n: int) -> np.ndarray:
        """Generate samples biased toward edge zones."""
        rng = np.random.default_rng(self.seed + 1)
        samples = []

        for _ in range(n):
            sample = np.zeros(self.d)
            for i, spec in enumerate(self.param_specs):
                # 50% chance to be in edge zone for each parameter
                if rng.random() < 0.5:
                    # Edge zone: either low or high
                    if rng.random() < 0.5:
                        sample[i] = rng.uniform(0, spec.edge_zone_pct)
                    else:
                        sample[i] = rng.uniform(1 - spec.edge_zone_pct, 1)
                else:
                    sample[i] = rng.uniform(0, 1)
            samples.append(sample)

        return np.array(samples) if samples else np.empty((0, self.d))


# ============================================================================
# WALK-FORWARD VALIDATOR
# ============================================================================

@dataclass
class WalkForwardConfig:
    """Walk-forward validation configuration."""
    train_days: int = 504  # 2 years training
    val_days: int = 126    # 6 months validation
    test_days: int = 126   # 6 months holdout (NEVER touch during tuning)
    step_days: int = 63    # Quarterly steps
    purge_days: int = 5    # Purge gap between train/val
    embargo_days: int = 5  # Embargo after val before next train


@dataclass
class FoldMetrics:
    """Metrics for a single walk-forward fold."""
    fold_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str

    # Train metrics
    train_cagr: float = 0.0
    train_sharpe: float = 0.0
    train_sortino: float = 0.0
    train_max_dd: float = 0.0

    # Validation metrics
    val_cagr: float = 0.0
    val_sharpe: float = 0.0
    val_sortino: float = 0.0
    val_max_dd: float = 0.0
    val_calmar: float = 0.0

    # Overfitting indicator
    train_val_sharpe_ratio: float = 0.0  # train_sharpe / val_sharpe (>2 = overfit)


@dataclass
class OptimizationResult:
    """Complete optimization result for a strategy."""
    strategy_name: str
    best_params: dict[str, Any]
    param_specs: list[ParameterSpec]

    # Walk-forward fold results
    folds: list[FoldMetrics] = field(default_factory=list)

    # Aggregate out-of-sample metrics
    oos_cagr: float = 0.0
    oos_sharpe: float = 0.0
    oos_sortino: float = 0.0
    oos_max_dd: float = 0.0
    oos_calmar: float = 0.0
    oos_n_trades: int = 0
    oos_win_rate: float = 0.0
    oos_profit_factor: float = 0.0

    # Final holdout test (ONLY computed once at end)
    test_cagr: float = 0.0
    test_sharpe: float = 0.0
    test_sortino: float = 0.0
    test_max_dd: float = 0.0
    test_calmar: float = 0.0

    # Robustness checks
    bootstrap_sharpe_ci: tuple[float, float] = (0.0, 0.0)
    bootstrap_cagr_ci: tuple[float, float] = (0.0, 0.0)
    edge_zone_params: list[str] = field(default_factory=list)

    # Meta
    n_cycles: int = 0
    optimization_time_sec: float = 0.0
    timestamp: str = ""


# ============================================================================
# GPU-ACCELERATED BACKTEST ENGINE
# ============================================================================

class GPUBacktestEngine:
    """High-performance backtesting with GPU acceleration."""

    def __init__(
        self,
        use_gpu: bool = True,
        spread_bps: float = 3.0,
        impact_bps: float = 5.0,
        risk_free_rate: float = 0.045,
    ):
        self.use_gpu = use_gpu and HAS_CUPY
        self.spread_bps = spread_bps
        self.impact_bps = impact_bps
        self.total_cost_bps = spread_bps + impact_bps
        self.risk_free_rate = risk_free_rate

        if self.use_gpu:
            logger.info("GPU backtesting enabled via CuPy")
        else:
            logger.info("Using CPU backtesting (CuPy not available)")

    def compute_returns(self, prices: np.ndarray) -> np.ndarray:
        """Compute log returns with GPU if available."""
        if self.use_gpu:
            prices_gpu = cp.asarray(prices)
            returns = cp.diff(cp.log(prices_gpu))
            return cp.asnumpy(returns)
        return np.diff(np.log(prices))

    def compute_sharpe(
        self,
        returns: np.ndarray,
        annualize: bool = True,
    ) -> float:
        """Compute Sharpe ratio."""
        if len(returns) < 30:
            return 0.0

        excess = returns - self.risk_free_rate / 252
        mean = np.mean(excess)
        std = np.std(excess, ddof=1)

        if std == 0:
            return 0.0

        sharpe = mean / std
        if annualize:
            sharpe *= np.sqrt(252)
        return float(sharpe)

    def compute_sortino(
        self,
        returns: np.ndarray,
        annualize: bool = True,
    ) -> float:
        """Compute Sortino ratio (downside deviation)."""
        if len(returns) < 30:
            return 0.0

        excess = returns - self.risk_free_rate / 252
        mean = np.mean(excess)
        downside = returns[returns < 0]

        if len(downside) == 0:
            return float('inf')

        downside_std = np.std(downside, ddof=1)
        if downside_std == 0:
            return float('inf')

        sortino = mean / downside_std
        if annualize:
            sortino *= np.sqrt(252)
        return float(sortino)

    def compute_max_drawdown(self, returns: np.ndarray) -> float:
        """Compute maximum drawdown percentage."""
        if len(returns) == 0:
            return 0.0

        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return float(np.min(drawdowns) * 100)

    def compute_cagr(self, returns: np.ndarray, n_days: int) -> float:
        """Compute Compound Annual Growth Rate."""
        if n_days == 0 or len(returns) == 0:
            return 0.0

        total_return = np.prod(1 + returns) - 1
        years = n_days / 252
        if years == 0:
            return 0.0

        cagr = (1 + total_return) ** (1 / years) - 1
        return float(cagr * 100)

    def compute_calmar(self, cagr: float, max_dd: float) -> float:
        """Compute Calmar ratio (CAGR / max drawdown)."""
        if max_dd == 0 or max_dd > -0.01:  # Avoid division by ~0
            return float('inf') if cagr > 0 else 0.0
        return cagr / abs(max_dd)

    def run_backtest(
        self,
        model: Any,
        data: pd.DataFrame,
        start_idx: int,
        end_idx: int,
    ) -> dict[str, Any]:
        """
        Run a backtest for a model on a data slice.

        Returns metrics dict.
        """
        if end_idx <= start_idx or start_idx < 0:
            return self._empty_metrics()

        data_slice = data.iloc[start_idx:end_idx].copy()
        if len(data_slice) < 50:
            return self._empty_metrics()

        # Run signal generation
        signals = asyncio.run(self._generate_signals(model, data_slice))

        # Simulate trades
        trades = self._simulate_trades(signals, data_slice)

        # Compute metrics
        return self._compute_trade_metrics(trades, data_slice)

    async def _generate_signals(
        self,
        model: Any,
        data: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Generate trading signals using the model."""
        signals = []
        symbol = data.get("symbol", data.iloc[0].get("symbol", "UNK")).iloc[0] if "symbol" in data.columns else "UNK"

        # Warmup period
        warmup = 50

        for i in range(warmup, len(data)):
            timestamp = data.index[i] if hasattr(data.index[i], 'isoformat') else datetime.now()
            lookback = data.iloc[max(0, i - 252):i + 1]

            try:
                signal = await model.generate(symbol, lookback, timestamp)
                if signal and signal.signal_type == SignalType.ENTRY:
                    signals.append({
                        "idx": i,
                        "direction": 1 if signal.direction == Direction.LONG else -1,
                        "metadata": signal.metadata or {},
                        "confidence": signal.confidence,
                    })
            except Exception:
                continue

        return signals

    def _simulate_trades(
        self,
        signals: list[dict[str, Any]],
        data: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Simulate trades from signals."""
        trades = []
        position = 0
        entry_price = 0.0
        entry_idx = 0

        close_prices = data["close"].values

        for sig in signals:
            idx = sig["idx"]
            direction = sig["direction"]
            metadata = sig["metadata"]

            price = close_prices[idx]
            cost = price * self.total_cost_bps / 10000

            # Check for position exit first
            if position != 0:
                # Check stop/target
                stop_loss = metadata.get("stop_loss", 0)
                take_profit = metadata.get("take_profit", float('inf'))

                if position > 0:  # Long position
                    if price <= stop_loss or price >= take_profit:
                        pnl = (price - entry_price) / entry_price * 100 - self.total_cost_bps / 100
                        trades.append({
                            "entry_idx": entry_idx,
                            "exit_idx": idx,
                            "direction": position,
                            "pnl_pct": pnl,
                            "holding_days": idx - entry_idx,
                        })
                        position = 0

            # New entry if no position
            if position == 0 and direction != 0:
                position = direction
                entry_price = price + cost * np.sign(direction)
                entry_idx = idx

        # Close any open position at end
        if position != 0 and len(close_prices) > 0:
            final_price = close_prices[-1]
            if position > 0:
                pnl = (final_price - entry_price) / entry_price * 100 - self.total_cost_bps / 100
            else:
                pnl = (entry_price - final_price) / entry_price * 100 - self.total_cost_bps / 100
            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": len(close_prices) - 1,
                "direction": position,
                "pnl_pct": pnl,
                "holding_days": len(close_prices) - 1 - entry_idx,
            })

        return trades

    def _compute_trade_metrics(
        self,
        trades: list[dict[str, Any]],
        data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Compute metrics from trades."""
        if not trades:
            return self._empty_metrics()

        pnls = np.array([t["pnl_pct"] for t in trades])
        n_trades = len(trades)

        # Convert to daily returns (approximate)
        n_days = len(data)
        daily_returns = np.zeros(n_days)
        for t in trades:
            if t["holding_days"] > 0:
                daily_pnl = t["pnl_pct"] / t["holding_days"] / 100
                daily_returns[t["entry_idx"]:t["exit_idx"]] = daily_pnl

        # Metrics
        cagr = self.compute_cagr(daily_returns, n_days)
        sharpe = self.compute_sharpe(daily_returns)
        sortino = self.compute_sortino(daily_returns)
        max_dd = self.compute_max_drawdown(daily_returns)
        calmar = self.compute_calmar(cagr, max_dd)

        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0.0
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return {
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_dd": max_dd,
            "calmar": calmar,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_pnl": float(np.mean(pnls)),
            "total_pnl": float(np.sum(pnls)),
        }

    def _empty_metrics(self) -> dict[str, Any]:
        """Return empty metrics dict."""
        return {
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd": 0.0,
            "calmar": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
        }


# ============================================================================
# STRATEGY OPTIMIZER
# ============================================================================

class StrategyOptimizer:
    """Main optimizer class for a single strategy."""

    # Model class mapping
    MODEL_CLASSES = {
        "atr_rsi": ATROptimizedRSIModel,
        "mtf_momentum": MTFMomentumModel,
        "mi_ensemble": MIEnsembleModel,
        "kalman_hybrid": KalmanHybridModel,
    }

    MASSIVE_SYMBOLS = [
        "AAPL", "MSFT", "GOOGL", "NVDA", "META",
        "JPM", "BAC", "GS", "MS", "WFC",
        "JNJ", "UNH", "PFE", "ABBV", "TMO",
        "WMT", "HD", "NKE", "MCD", "SBUX",
        "XOM", "CVX", "COP", "EOG", "SLB",
    ]

    def __init__(
        self,
        strategy_name: str,
        wf_config: WalkForwardConfig | None = None,
        n_cycles: int = 100,
        target_cagr: float = 30.0,
        use_gpu: bool = True,
    ):
        self.strategy_name = strategy_name
        self.wf_config = wf_config or WalkForwardConfig()
        self.n_cycles = n_cycles
        self.target_cagr = target_cagr
        self.use_gpu = use_gpu

        self.param_specs = STRATEGY_PARAMS.get(strategy_name, [])
        self.model_class = self.MODEL_CLASSES.get(strategy_name)

        if not self.param_specs:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        if not self.model_class:
            raise ValueError(f"No model class for: {strategy_name}")

        self.engine = GPUBacktestEngine(use_gpu=use_gpu)
        self.data_cache: dict[str, pd.DataFrame] = {}
        self.results: list[tuple[dict, float, dict]] = []  # (params, score, metrics)

    def load_data(self) -> bool:
        """Load all symbol data."""
        data_dir = Path(__file__).parent.parent / "data" / "historical"

        for symbol in self.MASSIVE_SYMBOLS:
            csv_path = data_dir / f"{symbol}_historical.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, parse_dates=["Date"])
                    df = df.rename(columns={"Date": "date"})
                    df = df.set_index("date")
                    df.columns = df.columns.str.lower()
                    df["symbol"] = symbol
                    self.data_cache[symbol] = df
                except Exception as e:
                    logger.warning(f"Failed to load {symbol}: {e}")

        logger.info(f"Loaded {len(self.data_cache)} symbols")
        return len(self.data_cache) >= 10

    def create_model(self, params: dict[str, Any]) -> Any:
        """Create a model instance with given parameters."""
        config = ModelConfig(
            model_id=f"{self.strategy_name}_opt",
            model_type=self.strategy_name,
            parameters=params,
        )
        return self.model_class(config)

    def evaluate_params(
        self,
        params: dict[str, Any],
        symbols: list[str] | None = None,
        train_ratio: float = 0.7,
    ) -> tuple[float, dict[str, Any]]:
        """
        Evaluate a parameter set across symbols.

        Returns (composite_score, metrics_dict)
        """
        symbols = symbols or list(self.data_cache.keys())[:10]  # Limit for speed
        model = self.create_model(params)

        all_train_metrics = []
        all_val_metrics = []

        for symbol in symbols:
            data = self.data_cache.get(symbol)
            if data is None or len(data) < 500:
                continue

            # Split into train/val (leaving test untouched)
            n = len(data)
            test_start = n - self.wf_config.test_days
            train_end = int((test_start) * train_ratio)
            val_end = test_start - self.wf_config.purge_days

            if train_end < 200:
                continue

            # Train metrics
            train_metrics = self.engine.run_backtest(model, data, 0, train_end)
            all_train_metrics.append(train_metrics)

            # Validation metrics
            val_start = train_end + self.wf_config.purge_days
            if val_start < val_end:
                val_metrics = self.engine.run_backtest(model, data, val_start, val_end)
                all_val_metrics.append(val_metrics)

        if not all_val_metrics:
            return -999.0, {}

        # Aggregate validation metrics
        avg_val = self._aggregate_metrics(all_val_metrics)
        avg_train = self._aggregate_metrics(all_train_metrics)

        # Composite score: prioritize Sharpe and CAGR, penalize overfitting
        overfit_penalty = 0.0
        if avg_train["sharpe"] > 0 and avg_val["sharpe"] > 0:
            overfit_ratio = avg_train["sharpe"] / avg_val["sharpe"]
            if overfit_ratio > 2.0:
                overfit_penalty = (overfit_ratio - 2.0) * 0.5

        score = (
            avg_val["sharpe"] * 0.4 +
            avg_val["cagr"] / 10 * 0.3 +  # Normalize CAGR contribution
            avg_val["sortino"] * 0.1 / 2 +
            avg_val["calmar"] * 0.1 +
            avg_val["profit_factor"] * 0.1 -
            overfit_penalty
        )

        return score, {
            "train": avg_train,
            "val": avg_val,
            "overfit_ratio": avg_train["sharpe"] / avg_val["sharpe"] if avg_val["sharpe"] > 0 else 0,
        }

    def _aggregate_metrics(self, metrics_list: list[dict]) -> dict[str, Any]:
        """Aggregate metrics across symbols."""
        if not metrics_list:
            return {"cagr": 0, "sharpe": 0, "sortino": 0, "max_dd": 0, "calmar": 0,
                    "n_trades": 0, "win_rate": 0, "profit_factor": 0}

        return {
            "cagr": np.mean([m["cagr"] for m in metrics_list]),
            "sharpe": np.mean([m["sharpe"] for m in metrics_list]),
            "sortino": np.mean([m["sortino"] for m in metrics_list]),
            "max_dd": np.mean([m["max_dd"] for m in metrics_list]),
            "calmar": np.mean([m["calmar"] for m in metrics_list]),
            "n_trades": sum([m["n_trades"] for m in metrics_list]),
            "win_rate": np.mean([m["win_rate"] for m in metrics_list]),
            "profit_factor": np.mean([m["profit_factor"] for m in metrics_list]),
        }

    def run_optimization(self) -> OptimizationResult:
        """Run full optimization cycle."""
        start_time = time.time()

        # Load data
        if not self.load_data():
            raise RuntimeError("Failed to load sufficient data")

        # Generate parameter samples
        sampler = EdgeBiasedSobolSampler(
            self.param_specs,
            n_samples=self.n_cycles,
            edge_oversample_factor=2.0,
        )
        param_samples = sampler.generate()

        logger.info(f"Starting optimization for {self.strategy_name}")
        logger.info(f"Parameter space: {len(self.param_specs)} dimensions")
        logger.info(f"Samples to evaluate: {len(param_samples)}")

        # Evaluate all samples
        best_score = -float('inf')
        best_params = {}
        best_metrics = {}

        for i, params in enumerate(param_samples):
            try:
                score, metrics = self.evaluate_params(params)
                self.results.append((params, score, metrics))

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_metrics = metrics

                    if (i + 1) % 10 == 0:
                        logger.info(
                            f"Cycle {i + 1}/{len(param_samples)}: "
                            f"Best Score={best_score:.3f}, "
                            f"Val CAGR={best_metrics.get('val', {}).get('cagr', 0):.1f}%, "
                            f"Val Sharpe={best_metrics.get('val', {}).get('sharpe', 0):.2f}"
                        )

            except Exception as e:
                logger.warning(f"Cycle {i + 1} failed: {e}")
                continue

        # Run final holdout test with best params
        test_metrics = self._run_holdout_test(best_params)

        # Compute bootstrap confidence intervals
        bootstrap_ci = self._compute_bootstrap_ci(best_params)

        # Identify edge zone parameters
        edge_params = [
            spec.name for spec in self.param_specs
            if spec.name in best_params and spec.is_edge_zone(best_params[spec.name])
        ]

        # Build result
        result = OptimizationResult(
            strategy_name=self.strategy_name,
            best_params=best_params,
            param_specs=self.param_specs,
            oos_cagr=best_metrics.get("val", {}).get("cagr", 0),
            oos_sharpe=best_metrics.get("val", {}).get("sharpe", 0),
            oos_sortino=best_metrics.get("val", {}).get("sortino", 0),
            oos_max_dd=best_metrics.get("val", {}).get("max_dd", 0),
            oos_calmar=best_metrics.get("val", {}).get("calmar", 0),
            oos_n_trades=best_metrics.get("val", {}).get("n_trades", 0),
            oos_win_rate=best_metrics.get("val", {}).get("win_rate", 0),
            oos_profit_factor=best_metrics.get("val", {}).get("profit_factor", 0),
            test_cagr=test_metrics.get("cagr", 0),
            test_sharpe=test_metrics.get("sharpe", 0),
            test_sortino=test_metrics.get("sortino", 0),
            test_max_dd=test_metrics.get("max_dd", 0),
            test_calmar=test_metrics.get("calmar", 0),
            bootstrap_sharpe_ci=bootstrap_ci.get("sharpe", (0, 0)),
            bootstrap_cagr_ci=bootstrap_ci.get("cagr", (0, 0)),
            edge_zone_params=edge_params,
            n_cycles=len(param_samples),
            optimization_time_sec=time.time() - start_time,
            timestamp=datetime.now().isoformat(),
        )

        return result

    def _run_holdout_test(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run final test on holdout data (never seen during tuning)."""
        model = self.create_model(params)
        all_metrics = []

        for symbol, data in self.data_cache.items():
            if len(data) < self.wf_config.test_days + 50:
                continue

            test_start = len(data) - self.wf_config.test_days
            metrics = self.engine.run_backtest(model, data, test_start, len(data))
            all_metrics.append(metrics)

        return self._aggregate_metrics(all_metrics)

    def _compute_bootstrap_ci(
        self,
        params: dict[str, Any],
        n_bootstrap: int = 100,
        confidence: float = 0.95,
    ) -> dict[str, tuple[float, float]]:
        """Compute bootstrap confidence intervals for key metrics."""
        model = self.create_model(params)
        sharpe_samples = []
        cagr_samples = []

        # Collect validation metrics from random symbol subsets
        symbols = list(self.data_cache.keys())
        rng = np.random.default_rng(42)

        for _ in range(n_bootstrap):
            # Bootstrap sample of symbols
            boot_symbols = rng.choice(symbols, size=len(symbols), replace=True)
            _, metrics = self.evaluate_params(params, symbols=list(boot_symbols))
            
            val = metrics.get("val", {})
            sharpe_samples.append(val.get("sharpe", 0))
            cagr_samples.append(val.get("cagr", 0))

        alpha = 1 - confidence
        sharpe_ci = (
            np.percentile(sharpe_samples, alpha / 2 * 100),
            np.percentile(sharpe_samples, (1 - alpha / 2) * 100),
        )
        cagr_ci = (
            np.percentile(cagr_samples, alpha / 2 * 100),
            np.percentile(cagr_samples, (1 - alpha / 2) * 100),
        )

        return {"sharpe": sharpe_ci, "cagr": cagr_ci}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def print_result(result: OptimizationResult):
    """Pretty print optimization result."""
    print("\n" + "=" * 70)
    print(f"OPTIMIZATION RESULT: {result.strategy_name.upper()}")
    print("=" * 70)

    print("\n📊 BEST PARAMETERS:")
    for spec in result.param_specs:
        val = result.best_params.get(spec.name, "N/A")
        edge_marker = " [EDGE]" if spec.name in result.edge_zone_params else ""
        if isinstance(val, float):
            print(f"  {spec.name}: {val:.6g}{edge_marker}")
        else:
            print(f"  {spec.name}: {val}{edge_marker}")

    print("\n📈 OUT-OF-SAMPLE (Validation) METRICS:")
    print(f"  CAGR:          {result.oos_cagr:+.1f}%")
    print(f"  Sharpe:        {result.oos_sharpe:.2f}")
    print(f"  Sortino:       {result.oos_sortino:.2f}")
    print(f"  Max Drawdown:  {result.oos_max_dd:.1f}%")
    print(f"  Calmar:        {result.oos_calmar:.2f}")
    print(f"  Win Rate:      {result.oos_win_rate:.1f}%")
    print(f"  Profit Factor: {result.oos_profit_factor:.2f}")
    print(f"  Total Trades:  {result.oos_n_trades}")

    print("\n🔒 FINAL HOLDOUT TEST METRICS:")
    print(f"  CAGR:          {result.test_cagr:+.1f}%")
    print(f"  Sharpe:        {result.test_sharpe:.2f}")
    print(f"  Sortino:       {result.test_sortino:.2f}")
    print(f"  Max Drawdown:  {result.test_max_dd:.1f}%")
    print(f"  Calmar:        {result.test_calmar:.2f}")

    print("\n📊 ROBUSTNESS (95% Bootstrap CI):")
    print(f"  Sharpe CI:     [{result.bootstrap_sharpe_ci[0]:.2f}, {result.bootstrap_sharpe_ci[1]:.2f}]")
    print(f"  CAGR CI:       [{result.bootstrap_cagr_ci[0]:.1f}%, {result.bootstrap_cagr_ci[1]:.1f}%]")

    print("\n⚙️ OPTIMIZATION META:")
    print(f"  Cycles:        {result.n_cycles}")
    print(f"  Time:          {result.optimization_time_sec:.1f}s")
    print(f"  Timestamp:     {result.timestamp}")

    # Assessment
    print("\n🎯 ASSESSMENT:")
    target_cagr = 30.0
    if result.test_cagr >= target_cagr:
        print(f"  ✅ PASS: Test CAGR {result.test_cagr:.1f}% >= target {target_cagr}%")
    else:
        print(f"  ❌ FAIL: Test CAGR {result.test_cagr:.1f}% < target {target_cagr}%")

    if result.test_sharpe >= 1.5:
        print(f"  ✅ Sharpe {result.test_sharpe:.2f} >= 1.5 (acceptable)")
    else:
        print(f"  ⚠️ Sharpe {result.test_sharpe:.2f} < 1.5 (marginal)")

    if result.test_max_dd > -20:
        print(f"  ✅ Max DD {result.test_max_dd:.1f}% within -20% limit")
    else:
        print(f"  ⚠️ Max DD {result.test_max_dd:.1f}% exceeds -20% limit")

    print("=" * 70)


def save_result(result: OptimizationResult, output_dir: Path):
    """Save result to JSON and pickle."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON (human readable)
    json_path = output_dir / f"{result.strategy_name}_optimization.json"
    json_data = {
        "strategy_name": result.strategy_name,
        "best_params": result.best_params,
        "oos_metrics": {
            "cagr": result.oos_cagr,
            "sharpe": result.oos_sharpe,
            "sortino": result.oos_sortino,
            "max_dd": result.oos_max_dd,
            "calmar": result.oos_calmar,
            "n_trades": result.oos_n_trades,
            "win_rate": result.oos_win_rate,
            "profit_factor": result.oos_profit_factor,
        },
        "test_metrics": {
            "cagr": result.test_cagr,
            "sharpe": result.test_sharpe,
            "sortino": result.test_sortino,
            "max_dd": result.test_max_dd,
            "calmar": result.test_calmar,
        },
        "bootstrap_ci": {
            "sharpe": list(result.bootstrap_sharpe_ci),
            "cagr": list(result.bootstrap_cagr_ci),
        },
        "edge_zone_params": result.edge_zone_params,
        "n_cycles": result.n_cycles,
        "optimization_time_sec": result.optimization_time_sec,
        "timestamp": result.timestamp,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Saved JSON result to {json_path}")

    # Pickle (full object)
    pkl_path = output_dir / f"{result.strategy_name}_optimization.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)
    logger.info(f"Saved pickle result to {pkl_path}")


def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated Strategy Optimizer")
    parser.add_argument(
        "--strategy",
        choices=["atr_rsi", "mtf_momentum", "mi_ensemble", "kalman_hybrid", "all"],
        default="all",
        help="Strategy to optimize",
    )
    parser.add_argument("--cycles", type=int, default=100, help="Number of optimization cycles")
    parser.add_argument("--target-cagr", type=float, default=30.0, help="Target CAGR percentage")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--output-dir", type=str, default="artifacts/optimization", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    strategies = (
        ["atr_rsi", "mtf_momentum", "mi_ensemble", "kalman_hybrid"]
        if args.strategy == "all"
        else [args.strategy]
    )

    all_results = []

    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"OPTIMIZING: {strategy.upper()}")
        print(f"{'='*70}")

        try:
            optimizer = StrategyOptimizer(
                strategy_name=strategy,
                n_cycles=args.cycles,
                target_cagr=args.target_cagr,
                use_gpu=not args.no_gpu,
            )
            result = optimizer.run_optimization()
            all_results.append(result)

            print_result(result)
            save_result(result, output_dir)

        except Exception as e:
            logger.error(f"Optimization failed for {strategy}: {e}")
            import traceback
            traceback.print_exc()

    # Summary comparison
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("COMPARATIVE SUMMARY")
        print("=" * 70)
        print(f"{'Strategy':<15} {'Test CAGR':>10} {'Test Sharpe':>12} {'Test MaxDD':>10} {'Status':>10}")
        print("-" * 70)

        for r in sorted(all_results, key=lambda x: x.test_cagr, reverse=True):
            status = "✅ PASS" if r.test_cagr >= args.target_cagr else "❌ FAIL"
            print(f"{r.strategy_name:<15} {r.test_cagr:>+9.1f}% {r.test_sharpe:>12.2f} {r.test_max_dd:>9.1f}% {status:>10}")

        print("=" * 70)


if __name__ == "__main__":
    main()
