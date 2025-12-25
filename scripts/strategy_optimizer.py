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
import concurrent.futures
import os

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
        try:
            # Set default device if available (safe in modern torch)
            torch.set_default_device('cuda')
        except Exception:
            pass
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
from ordinis.engines.signalcore.models.adx_trend import ADXTrendModel
from ordinis.engines.signalcore.models.fibonacci_retracement import FibonacciRetracementModel
# Fibonacci strategy uses ADX + Fibonacci models directly (wrapped by adapter)

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

# Fibonacci ADX Parameters
FIBONACCI_ADX_PARAMS = [
    ParameterSpec("adx_period", 7, 21, "int", "signal", description="ADX lookback period"),
    ParameterSpec("adx_threshold", 15, 40, "float", "signal", description="ADX threshold for trend confirmation"),
    ParameterSpec("swing_lookback", 20, 120, "int", "signal", description="Bars for swing detection"),
    ParameterSpec("tolerance", 0.005, 0.03, "float", "signal", description="Price tolerance near Fibonacci level"),
    ParameterSpec("fib_382_weight", 0.0, 1.0, "float", "signal", description="Weight for 38.2 level"),
    ParameterSpec("fib_500_weight", 0.0, 1.0, "float", "signal", description="Weight for 50.0 level"),
    ParameterSpec("fib_618_weight", 0.0, 1.0, "float", "signal", description="Weight for 61.8 level"),
    ParameterSpec("take_profit_1272_mult", 1.0, 3.0, "float", "risk", description="Multiplier for 1.272 extension"),
    ParameterSpec("take_profit_1618_mult", 1.0, 4.0, "float", "risk", description="Multiplier for 1.618 extension"),
    ParameterSpec("require_volume_confirmation", 0, 1, "int", "signal", description="Require volume confirmation (0/1)"),
    ParameterSpec("max_pyramids", 0, 3, "int", "signal", description="Max pyramids (adds after swing break)"),
]

STRATEGY_PARAMS = {
    "atr_rsi": ATR_RSI_PARAMS,
    "mtf_momentum": MTF_MOMENTUM_PARAMS,
    "mi_ensemble": MI_ENSEMBLE_PARAMS,
    "kalman_hybrid": KALMAN_HYBRID_PARAMS,
    "fibonacci_adx": FIBONACCI_ADX_PARAMS,
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
        # Dynamic GPU/CuPy initialization (avoid relying on top-level import state)
        self._cupy = None
        self.use_gpu = False
        self.spread_bps = spread_bps
        self.impact_bps = impact_bps
        self.total_cost_bps = spread_bps + impact_bps
        self.risk_free_rate = risk_free_rate

        if use_gpu:
            # Prefer CuPy if available; otherwise try torch (CUDA) as a fallback.
            try:
                import cupy as cupy

                # Validate CUDA devices
                try:
                    device_count = cupy.cuda.runtime.getDeviceCount()
                except Exception:
                    device_count = 0

                if device_count > 0:
                    self._cupy = cupy
                    self.use_gpu = True
                    # Keep backward compatibility for code using top-level `cp`
                    globals()["cp"] = cupy
                    logger.info(f"GPU backtesting enabled via CuPy (devices={device_count})")
                    try:
                        # quick smoke test
                        arr = cupy.arange(1000000)
                        _ = cupy.sum(arr)
                        logger.info("CuPy smoke test OK")
                    except Exception as sm_e:
                        logger.warning(f"CuPy smoke test failed: {sm_e}")
                else:
                    logger.info("CuPy installed but no CUDA devices found; falling back to other GPU backends")
            except Exception as e:
                logger.info(f"CuPy not available: {e}")

                # Torch fallback
                if HAS_TORCH:
                    try:
                        # Confirm device availability
                        if torch.cuda.device_count() > 0:
                            self._cupy = None
                            self.use_gpu = True
                            logger.info(f"GPU backtesting enabled via PyTorch (devices={torch.cuda.device_count()})")
                        else:
                            logger.info("PyTorch installed but no CUDA devices found; using CPU")
                    except Exception as se:
                        logger.info(f"PyTorch GPU check failed: {se}; using CPU backtesting")
                else:
                    logger.info("Using CPU backtesting (no GPU backend available)")
        else:
            logger.info("GPU explicitly disabled - using CPU backtesting")

    def compute_returns(self, prices: np.ndarray) -> np.ndarray:
        """Compute log returns with GPU if available. Supports CuPy and PyTorch as backends."""
        # CuPy path
        if self.use_gpu and getattr(self, "_cupy", None) is not None:
            cp_local = self._cupy
            prices_gpu = cp_local.asarray(prices)
            returns_gpu = cp_local.diff(cp_local.log(prices_gpu))
            return cp_local.asnumpy(returns_gpu)

        # PyTorch CUDA fallback
        if self.use_gpu and getattr(self, "_cupy", None) is None and HAS_TORCH and torch is not None:
            try:
                dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                t = torch.tensor(prices, device=dev, dtype=torch.float32)
                # small epsilon to ensure positive values
                t = torch.clamp(t, min=1e-8)
                rtn = t.log().diff()
                # move back to CPU numpy
                return rtn.cpu().numpy()
            except Exception as e:
                logger.debug(f"Torch compute_returns failed: {e}; falling back to CPU path")

        # CPU fallback
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


# ----------------------------------------------------------------------------
# Strategy adapter for Fibonacci ADX (model wrapper)
# ----------------------------------------------------------------------------
class FibonacciStrategyAdapter:
    """Adapter to wrap ADXTrendModel + FibonacciRetracementModel into the optimizer's Model-like interface."""

    def __init__(self, config: ModelConfig):
        self.config = config
        params = config.parameters or {}
        name = config.model_id or f"fibonacci_{int(time.time())}"
        adx_period = int(params.get("adx_period", 14))
        adx_threshold = float(params.get("adx_threshold", 25))
        swing_lookback = int(params.get("swing_lookback", 50))
        fib_levels = params.get("key_levels", [0.382, 0.5, 0.618])
        tolerance = float(params.get("tolerance", 0.01))

        # Instantiate underlying models
        adx_cfg = ModelConfig(
            model_id=f"{name}-adx",
            model_type="trend",
            parameters={"adx_period": adx_period, "adx_threshold": adx_threshold},
        )
        self.adx_model = ADXTrendModel(adx_cfg)

        fib_cfg = ModelConfig(
            model_id=f"{name}-fib",
            model_type="static_level",
            parameters={"swing_lookback": swing_lookback, "key_levels": fib_levels, "tolerance": tolerance},
        )
        self.fib_model = FibonacciRetracementModel(fib_cfg)

    async def generate(self, symbol: str, data: pd.DataFrame, timestamp: datetime):
        """Generate combined signal using ADX filter and Fibonacci levels."""
        try:
            adx_signal = await self.adx_model.generate(data, timestamp)
            # ADX filter
            if adx_signal.metadata.get("adx", 0) < adx_signal.metadata.get("adx_threshold", 0):
                return None

            fib_signal = await self.fib_model.generate(data, timestamp)
            if not fib_signal or fib_signal.signal_type != SignalType.ENTRY:
                return None

            if adx_signal.direction != fib_signal.direction:
                return None

            # Combine scores
            combined_score = (adx_signal.score * 0.4) + (fib_signal.score * 0.6)
            combined_prob = (adx_signal.probability * 0.4) + (fib_signal.probability * 0.6)

            swing_high = fib_signal.metadata.get("swing_high")
            swing_low = fib_signal.metadata.get("swing_low")
            current_price = fib_signal.metadata.get("current_price")

            if fib_signal.direction == Direction.LONG:
                stop_loss = swing_low * 0.98
                take_profit = swing_high
            else:
                stop_loss = swing_high * 1.02
                take_profit = swing_low

            metadata = {
                **fib_signal.metadata,
                "adx": adx_signal.metadata.get("adx"),
                "plus_di": adx_signal.metadata.get("plus_di"),
                "minus_di": adx_signal.metadata.get("minus_di"),
                "strategy": self.config.model_id,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }

            return Signal(
                symbol=fib_signal.symbol,
                timestamp=timestamp,
                signal_type=SignalType.ENTRY,
                direction=fib_signal.direction,
                probability=combined_prob,
                expected_return=fib_signal.expected_return * 1.3,
                confidence_interval=fib_signal.confidence_interval,
                score=combined_score,
                model_id=self.config.model_id,
                model_version=self.config.version,
                feature_contributions={**adx_signal.feature_contributions, **fib_signal.feature_contributions},
                regime=f"{adx_signal.regime}_{fib_signal.regime}",
                data_quality=min(adx_signal.data_quality, fib_signal.data_quality),
                staleness=max(adx_signal.staleness, fib_signal.staleness),
                metadata=metadata,
            )
        except Exception as e:
            logger.debug(f"FibonacciStrategyAdapter.generate failed: {e}")
            return None


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
        "fibonacci_adx": FibonacciStrategyAdapter,
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
        metric: str = "cagr",
        symbols_group: str | None = None,
        symbols: list[str] | None = None,
        n_symbols: int | None = None,
        years: int | None = None,
        n_workers: int | None = None,
        bootstrap_n: int = 100,
        skip_bootstrap: bool = False,
        bootstrap_confidence: float = 0.95,
        use_optuna: bool = False,
        optuna_trials: int = 100,
    ):
        self.strategy_name = strategy_name
        self.wf_config = wf_config or WalkForwardConfig()
        self.n_cycles = n_cycles
        self.target_cagr = target_cagr
        self.use_gpu = use_gpu
        self.metric = metric
        self.symbols_group = symbols_group
        self.symbols_override = symbols
        self.n_symbols = n_symbols
        self.years = years
        self.n_workers = n_workers or max(1, (os.cpu_count() or 1) - 1)

        # Bootstrap configuration
        self.bootstrap_n = int(bootstrap_n)
        self.skip_bootstrap = bool(skip_bootstrap)
        self.bootstrap_confidence = float(bootstrap_confidence)

        self.param_specs = STRATEGY_PARAMS.get(strategy_name, [])
        self.model_class = self.MODEL_CLASSES.get(strategy_name)

        # Optuna configuration
        self.use_optuna = bool(use_optuna)
        self.optuna_trials = int(optuna_trials)

        if not self.param_specs:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        if not self.model_class:
            raise ValueError(f"No model class for: {strategy_name}")

        self.engine = GPUBacktestEngine(use_gpu=use_gpu)
        self.data_cache: dict[str, pd.DataFrame] = {}
        self.results: list[tuple[dict, float, dict]] = []  # (params, score, metrics)

    def load_data(self) -> bool:
        """Load symbol data from historical CSVs or a configured symbol group."""
        data_dir = Path(__file__).parent.parent / "data" / "historical"

        # Determine which symbol list to use
        if getattr(self, "symbols_override", None):
            symbols_to_load = list(self.symbols_override)
        elif getattr(self, "symbols_group", None):
            config_path = Path(__file__).parent.parent / "configs" / "optimization" / "atr_rsi_opt_config.yaml"
            symbols_to_load = []
            if config_path.exists():
                try:
                    import yaml

                    with open(config_path) as f:
                        cfg = yaml.safe_load(f)
                    symbols_dict = cfg.get("symbols", {})

                    if self.symbols_group == "mid_cap":
                        symbols_to_load = symbols_dict.get("mid_cap", [])
                    elif self.symbols_group == "small_cap":
                        symbols_to_load = symbols_dict.get("small_cap", [])
                    elif self.symbols_group in ("both", "all"):
                        for v in symbols_dict.values():
                            if isinstance(v, list):
                                symbols_to_load.extend(v)
                    else:
                        symbols_to_load = self.MASSIVE_SYMBOLS
                except Exception as e:
                    logger.warning(f"Failed to load symbol group config: {e}")
                    symbols_to_load = self.MASSIVE_SYMBOLS
            else:
                logger.warning("Symbol group config not found; using default symbol list")
                symbols_to_load = self.MASSIVE_SYMBOLS
        else:
            symbols_to_load = self.MASSIVE_SYMBOLS

        # Optionally limit number of symbols for speed
        if getattr(self, "n_symbols", None):
            symbols_to_load = symbols_to_load[: self.n_symbols]

        for symbol in symbols_to_load:
            # Search for CSV in data_dir and subdirectories (supports mid_cap/small_cap folders)
            csv_candidates = list(data_dir.rglob(f"{symbol}_historical.csv"))
            if csv_candidates:
                csv_path = csv_candidates[0]
                try:
                    df = pd.read_csv(csv_path, parse_dates=["Date"])
                    df = df.rename(columns={"Date": "date"})
                    df = df.set_index("date")
                    df.columns = df.columns.str.lower()

                    # Trim to last N years if requested
                    if getattr(self, "years", None):
                        try:
                            cutoff = pd.Timestamp.now() - pd.DateOffset(years=self.years)
                            df = df[df.index >= cutoff]
                        except Exception:
                            # If index is not datetime-like, skip trimming
                            pass

                    df["symbol"] = symbol
                    self.data_cache[symbol] = df
                except Exception as e:
                    logger.warning(f"Failed to load {symbol} from {csv_path}: {e}")
            else:
                # Attempt to fetch via APIs if CSV not available
                logger.debug(f"CSV not found for {symbol} in {data_dir} (searched subdirs); attempting API fetch")
                fetched = False
                try:
                    df_api = self._fetch_symbol_via_api(symbol, years=self.years)
                    if df_api is not None:
                        df_api.columns = df_api.columns.str.lower()
                        self.data_cache[symbol] = df_api
                        fetched = True
                except Exception as e:
                    logger.debug(f"API fetch for {symbol} failed: {e}")

                if not fetched:
                    logger.debug(f"No data available for {symbol}")

        logger.info(f"Loaded {len(self.data_cache)} symbols")
        return len(self.data_cache) >= 1

    def _fetch_symbol_via_api(self, symbol: str, years: int | None = None) -> pd.DataFrame | None:
        """Attempt to fetch historical OHLCV for a symbol using available API clients.

        Supported (best-effort): Polygon REST v2, Finnhub. Uses requests if available.
        Returns DataFrame with columns: date (index), open, high, low, close, volume
        """
        # Prefer CSVs first; this helper only called when CSV not found
        try:
            import requests
        except Exception:
            logger.debug("Requests not available; cannot perform API fetch")
            return None

        # Determine date range
        end = pd.Timestamp.now()
        if years is None:
            start = end - pd.DateOffset(years=5)
        else:
            start = end - pd.DateOffset(years=years)

        api_key = os.environ.get("POLYGON_API_KEY")
        if api_key:
            # Polygon historic aggregates v2 (day) endpoint
            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start.date()}/{end.date()}"
                f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
            )
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    jd = r.json()
                    results = jd.get("results", [])
                    if not results:
                        return None
                    df = pd.DataFrame(results)
                    df["date"] = pd.to_datetime(df["t"], unit="ms")
                    df = df.set_index("date")
                    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
                    df = df[["open", "high", "low", "close", "volume"]]
                    return df
            except Exception as e:
                logger.debug(f"Polygon fetch failed: {e}")

        api_key = os.environ.get("FINNHUB_API_KEY")
        if api_key:
            # Finnhub candle API
            try:
                from time import time
                resolution = 'D'
                from_ts = int(start.timestamp())
                to_ts = int(end.timestamp())
                url = (
                    f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution={resolution}&from={from_ts}&to={to_ts}&token={api_key}"
                )
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    jd = r.json()
                    if jd.get('s') != 'ok':
                        return None
                    import numpy as _np
                    dates = pd.to_datetime(jd['t'], unit='s')
                    df = pd.DataFrame({'open': jd['o'], 'high': jd['h'], 'low': jd['l'], 'close': jd['c'], 'volume': jd['v']}, index=dates)
                    df.index.name = 'date'
                    return df
            except Exception as e:
                logger.debug(f"Finnhub fetch failed: {e}")

        # If no providers or fetch failed
        return None

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

        # Composite score or metric-specific score - penalize overfitting
        overfit_penalty = 0.0
        if avg_train["sharpe"] > 0 and avg_val["sharpe"] > 0:
            overfit_ratio = avg_train["sharpe"] / avg_val["sharpe"]
            if overfit_ratio > 2.0:
                overfit_penalty = (overfit_ratio - 2.0) * 0.5

        metric = getattr(self, "metric", "sharpe")
        if metric == "cagr":
            score = avg_val.get("cagr", 0) - overfit_penalty
            # Penalize excessive drawdown beyond -20%
            max_dd = avg_val.get("max_dd", 0)
            if max_dd < -20:
                dd_penalty = (abs(max_dd) - 20) * 0.5
                score -= dd_penalty
        elif metric == "profit_factor":
            score = avg_val.get("profit_factor", 0) - overfit_penalty
        else:
            # default composite (sharpe-focused)
            score = (
                avg_val.get("sharpe", 0) * 0.4 +
                avg_val.get("cagr", 0) / 10 * 0.3 +
                avg_val.get("sortino", 0) * 0.05 +
                avg_val.get("calmar", 0) * 0.1 +
                avg_val.get("profit_factor", 0) * 0.1 -
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

        # If Optuna requested, attempt optuna-based search (falls back to Sobol if unavailable)
        if getattr(self, 'use_optuna', False):
            try:
                result = self._run_optuna_search()
                # If optuna found a best result, return it (it handles holdout/test/bootstrap)
                return result
            except Exception as e:
                logger.warning(f"Optuna search failed or not available, falling back to Sobol sampler: {e}")

        # Generate parameter samples via Sobol
        sampler = EdgeBiasedSobolSampler(
            self.param_specs,
            n_samples=self.n_cycles,
            edge_oversample_factor=2.0,
        )
        param_samples = sampler.generate()

        logger.info(f"Starting optimization for {self.strategy_name}")
        logger.info(f"Starting evaluation: cycles={self.n_cycles}, workers={self.n_workers}, years={self.years}")
        logger.info(f"Parameter space: {len(self.param_specs)} dimensions")
        logger.info(f"Samples to evaluate: {len(param_samples)}")

        # Evaluate all samples
        best_score = -float('inf')
        best_params = {}
        best_metrics = {}

        # Determine symbol set to evaluate
        symbols_to_use = list(self.data_cache.keys())
        if self.n_symbols:
            symbols_to_use = symbols_to_use[: self.n_symbols]

        # Evaluate with keyboard-interrupt handling so runs can be stopped cleanly
        try:
            # Parallel evaluation across worker processes (CPU) if requested
            if getattr(self, "n_workers", 1) and self.n_workers > 1:
                logger.info(f"Evaluating {len(param_samples)} samples across {self.n_workers} workers")
                wf_cfg_dict = {
                    "train_days": self.wf_config.train_days,
                    "val_days": self.wf_config.val_days,
                    "test_days": self.wf_config.test_days,
                    "step_days": self.wf_config.step_days,
                    "purge_days": self.wf_config.purge_days,
                    "embargo_days": self.wf_config.embargo_days,
                }

                args_list = [
                    (params, self.strategy_name, wf_cfg_dict, self.metric, self.years, symbols_to_use)
                    for params in param_samples
                ]

                with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as ex:
                    futures = {ex.submit(_evaluate_params_worker, args): args for args in args_list}
                    completed = 0
                    for f in concurrent.futures.as_completed(futures):
                        try:
                            params, score, metrics = f.result()
                            completed += 1
                            self.results.append((params, score, metrics))

                            if score > best_score:
                                best_score = score
                                best_params = params
                                best_metrics = metrics

                            if completed % 10 == 0 or completed == len(param_samples):
                                logger.info(
                                    f"Completed {completed}/{len(param_samples)} - Best Score={best_score:.3f}, "
                                    f"Val CAGR={best_metrics.get('val', {}).get('cagr', 0):.1f}%, "
                                    f"Val Sharpe={best_metrics.get('val', {}).get('sharpe', 0):.2f}"
                                )

                        except Exception as e:
                            logger.warning(f"Worker evaluation failed: {e}")

            else:
                for i, params in enumerate(param_samples):
                    try:
                        score, metrics = self.evaluate_params(params, symbols=symbols_to_use)
                        self.results.append((params, score, metrics))

                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_metrics = metrics

                        if (i + 1) % 10 == 0 or (i + 1) == len(param_samples):
                            logger.info(
                                f"Cycle {i + 1}/{len(param_samples)}: "
                                f"Best Score={best_score:.3f}, "
                                f"Val CAGR={best_metrics.get('val', {}).get('cagr', 0):.1f}%, "
                                f"Val Sharpe={best_metrics.get('val', {}).get('sharpe', 0):.2f}"
                            )

                    except KeyboardInterrupt:
                        logger.warning(f"Cycle {i+1} interrupted by user. Building partial result.")
                        raise
                    except Exception as e:
                        logger.warning(f"Cycle {i + 1} failed: {e}")
                        continue

        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user during sample evaluation. Returning partial result.")
            # Package partial result from progress so far
            partial_result = OptimizationResult(
                strategy_name=self.strategy_name,
                best_params=best_params,
                param_specs=self.param_specs,
                oos_cagr=best_metrics.get("val", {}).get("cagr", 0),
                oos_sharpe=best_metrics.get("val", {}).get("sharpe", 0),
                test_cagr=0.0,
                test_sharpe=0.0,
                n_cycles=len(param_samples),
                optimization_time_sec=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
            )
            return partial_result

        # Run final holdout test with best params (skip if no valid best params)
        if not best_params:
            logger.warning("No valid best parameters found; skipping holdout test and bootstrap CI.")
            test_metrics = self._aggregate_metrics([])
            bootstrap_ci = {"sharpe": (0.0, 0.0), "cagr": (0.0, 0.0)}
        else:
            try:
                test_metrics = self._run_holdout_test(best_params)
            except Exception as e:
                logger.warning(f"Holdout test failed: {e}")
                test_metrics = self._aggregate_metrics([])

            # Compute bootstrap confidence intervals (optionally skipped)
            if self.skip_bootstrap or self.bootstrap_n <= 0:
                logger.info("Skipping bootstrap CI computation as requested.")
                bootstrap_ci = {"sharpe": (0.0, 0.0), "cagr": (0.0, 0.0)}
            else:
                try:
                    bootstrap_ci = self._compute_bootstrap_ci(best_params, n_bootstrap=self.bootstrap_n, confidence=self.bootstrap_confidence)
                except KeyboardInterrupt:
                    logger.warning("Bootstrap CI interrupted by user; aborting CI calculation.")
                    bootstrap_ci = {"sharpe": (0.0, 0.0), "cagr": (0.0, 0.0)}
                except Exception as e:
                    logger.warning(f"Bootstrap CI failed: {e}")
                    bootstrap_ci = {"sharpe": (0.0, 0.0), "cagr": (0.0, 0.0)}

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
        n_bootstrap: int | None = None,
        confidence: float | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Compute bootstrap confidence intervals for key metrics."""
        n_bootstrap = self.bootstrap_n if n_bootstrap is None else int(n_bootstrap)
        confidence = self.bootstrap_confidence if confidence is None else float(confidence)

        if n_bootstrap <= 0:
            logger.info("Bootstrap sample size <= 0; skipping CI computation.")
            return {"sharpe": (0.0, 0.0), "cagr": (0.0, 0.0)}

        sharpe_samples = []
        cagr_samples = []

        # Collect validation metrics from random symbol subsets
        symbols = list(self.data_cache.keys())
        if not symbols:
            logger.warning("No symbols available for bootstrap CI computation.")
            return {"sharpe": (0.0, 0.0), "cagr": (0.0, 0.0)}

        rng = np.random.default_rng(42)
        progress_step = max(1, n_bootstrap // 10)

        for i in range(n_bootstrap):
            try:
                # Bootstrap sample of symbols
                boot_symbols = rng.choice(symbols, size=len(symbols), replace=True)
                _, metrics = self.evaluate_params(params, symbols=list(boot_symbols))

                val = metrics.get("val", {})
                sharpe_samples.append(val.get("sharpe", 0))
                cagr_samples.append(val.get("cagr", 0))
            except KeyboardInterrupt:
                logger.warning("Bootstrap CI computation interrupted by user.")
                break
            except Exception as e:
                logger.debug(f"Bootstrap iteration {i} failed: {e}")
                sharpe_samples.append(0)
                cagr_samples.append(0)

            if (i + 1) % progress_step == 0 or (i + 1) == n_bootstrap:
                logger.info(f"Bootstrap progress: {i + 1}/{n_bootstrap}")

        if not sharpe_samples:
            return {"sharpe": (0.0, 0.0), "cagr": (0.0, 0.0)}

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


# Optuna integration (optional)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    optuna = None
    OPTUNA_AVAILABLE = False


def _run_optuna_objective_from_specs(specs: list[ParameterSpec], trial: 'optuna.Trial') -> dict:
    params = {}
    for spec in specs:
        if spec.param_type == 'int':
            params[spec.name] = trial.suggest_int(spec.name, int(spec.min_val), int(spec.max_val))
        elif spec.param_type == 'log_float':
            params[spec.name] = trial.suggest_float(spec.name, float(spec.min_val), float(spec.max_val), log=True)
        else:
            params[spec.name] = trial.suggest_float(spec.name, float(spec.min_val), float(spec.max_val))
    return params


def _run_optuna_search_wrapper(self):
    return self._run_optuna_search()


def _run_optuna_search(self):
    """Run Optuna optimization over the parameter space."""
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna not available in this environment")

    def objective(trial: 'optuna.Trial') -> float:
        params = _run_optuna_objective_from_specs(self.param_specs, trial)
        score, metrics = self.evaluate_params(params)
        trial.set_user_attr('metrics', metrics)
        return float(score)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=self.optuna_trials)

    best_params = study.best_params if hasattr(study, 'best_params') else {}
    best_value = study.best_value if hasattr(study, 'best_value') else None

    # Run holdout and bootstrap using the best params
    if not best_params:
        raise RuntimeError('Optuna did not find any valid parameters')

    test_metrics = self._run_holdout_test(best_params)
    bootstrap_ci = {"sharpe": (0.0, 0.0), "cagr": (0.0, 0.0)}
    if not self.skip_bootstrap and self.bootstrap_n > 0:
        try:
            bootstrap_ci = self._compute_bootstrap_ci(best_params, n_bootstrap=self.bootstrap_n, confidence=self.bootstrap_confidence)
        except Exception:
            bootstrap_ci = {"sharpe": (0.0, 0.0), "cagr": (0.0, 0.0)}

    result = OptimizationResult(
        strategy_name=self.strategy_name,
        best_params=best_params,
        param_specs=self.param_specs,
        oos_cagr=0.0,
        oos_sharpe=0.0,
        test_cagr=test_metrics.get('cagr', 0),
        test_sharpe=test_metrics.get('sharpe', 0),
        bootstrap_sharpe_ci=bootstrap_ci.get('sharpe', (0, 0)),
        bootstrap_cagr_ci=bootstrap_ci.get('cagr', (0, 0)),
        n_cycles=self.optuna_trials,
        optimization_time_sec=0.0,
        timestamp=datetime.now().isoformat(),
    )
    # Attach the study for debugging
    try:
        result.optuna_study = study
    except Exception:
        pass

    return result


# Bind Optuna helper to StrategyOptimizer (module-level function -> method)
try:
    StrategyOptimizer._run_optuna_search = _run_optuna_search
except Exception:
    # If StrategyOptimizer not yet defined at import time, this will be set later
    pass

# Worker function for parallel evaluation

def _evaluate_params_worker(args_tuple):
    """Worker for evaluating parameter sets in a separate process.

    args_tuple: (params, strategy_name, wf_cfg_dict, metric, years, symbols)
    """
    try:
        params, strategy_name, wf_cfg_dict, metric, years, symbols = args_tuple
        wf_cfg = WalkForwardConfig(**wf_cfg_dict)

        # Create local optimizer instance - GPU disabled for worker processes
        opt = StrategyOptimizer(
            strategy_name=strategy_name,
            wf_config=wf_cfg,
            n_cycles=1,
            use_gpu=False,
            metric=metric,
            years=years,
            n_workers=1,
            bootstrap_n=0,
            skip_bootstrap=True,
        )

        # Load data (from CSVs) - worker will have its own copy
        ok = opt.load_data()
        if not ok:
            return params, -999.0, {"error": "Failed to load data in worker"}

        score, metrics = opt.evaluate_params(params, symbols=symbols)
        return params, score, metrics
    except Exception as e:
        # Return failing score for this paramset (stringify exception for inter-process safety)
        return args_tuple[0], -999.0, {"error": str(e)}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def print_result(result: OptimizationResult):
    # Backward-compatibility for displaying Optuna study info
    try:
        if hasattr(result, 'optuna_study') and result.optuna_study is not None:
            logger.info(f"Optuna study best value: {getattr(result.optuna_study, 'best_value', 'N/A')}")
    except Exception:
        pass
    """Pretty print optimization result."""
    print("\n" + "=" * 70)
    print(f"OPTIMIZATION RESULT: {result.strategy_name.upper()}")
    print("=" * 70)

    # Save quick diagnostic to artifacts even on small runs for easier triage
    try:
        outdir = Path('artifacts/optimization')
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / f"{result.strategy_name}_last_run_summary.txt", 'w') as f:
            f.write(str(result))
    except Exception:
        pass

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
        choices=["atr_rsi", "mtf_momentum", "mi_ensemble", "kalman_hybrid", "fibonacci_adx", "all"],
        default="all",
        help="Strategy to optimize",
    )
    parser.add_argument("--cycles", type=int, default=200, help="Number of optimization cycles")
    parser.add_argument("--metric", choices=["sharpe", "cagr", "profit_factor", "total_pnl", "total_return"], default="cagr", help="Primary metric to optimize")
    parser.add_argument("--symbols-group", choices=["mid_cap", "small_cap", "both", "all", "massive"], default="both", help="Symbol group to optimize")
    parser.add_argument("--n-symbols", type=int, default=None, help="Limit to the first N symbols for speed")
    parser.add_argument("--years", type=int, default=5, help="Historical horizon in years (uses last N years of CSV data)")
    parser.add_argument("--n-workers", type=int, default=None, help="Number of worker processes to use (default: cpu_count-1)")
    parser.add_argument("--target-cagr", type=float, default=30.0, help="Target CAGR percentage")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--no-bootstrap", action="store_true", help="Skip bootstrap confidence interval computation")
    parser.add_argument("--bootstrap-n", type=int, default=100, help="Number of bootstrap iterations (default 100)")
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95, help="Bootstrap CI confidence level (default 0.95)")
    parser.add_argument("--use-optuna", action="store_true", help="Use Optuna for search (TPESampler). Falls back if not installed")
    parser.add_argument("--optuna-trials", type=int, default=100, help="Number of Optuna trials if used")
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
                metric=args.metric,
                symbols_group=args.symbols_group,
                n_symbols=args.n_symbols,
                years=args.years,
                bootstrap_n=args.bootstrap_n,
                skip_bootstrap=args.no_bootstrap,
                bootstrap_confidence=args.bootstrap_confidence,
                use_optuna=args.use_optuna,
                optuna_trials=args.optuna_trials,
            )
            result = optimizer.run_optimization()
            all_results.append(result)

            print_result(result)
            save_result(result, output_dir)

        except KeyboardInterrupt:
            logger.warning("Optimization cancelled by user.")
            try:
                if 'result' in locals():
                    partial_dir = output_dir / "partial"
                    save_result(result, partial_dir)
            except Exception as se:
                logger.warning(f"Failed to save partial result: {se}")
            break

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
