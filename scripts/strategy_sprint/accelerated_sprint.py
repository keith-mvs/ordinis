"""
Accelerated Strategy Sprint Runner.

Runs all 8 strategies with GPU acceleration and AI-guided optimization
using Massive historical data.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ai_optimizer import (
    EVT_TAIL_PROFILE,
    GARCH_BREAKOUT_PROFILE,
    HMM_REGIME_PROFILE,
    KALMAN_TREND_PROFILE,
    MI_LEAD_LAG_PROFILE,
    MTF_MOMENTUM_PROFILE,
    NETWORK_REGIME_PROFILE,
    OU_PAIRS_PROFILE,
    AIOptimizerConfig,
    AIStrategyOptimizer,
    StrategyProfile,
)
from gpu_accelerator import GPUBacktestEngine, GPUConfig
from massive_data import VOLATILE_SYMBOLS, load_universe
from visualizer import generate_all_visualizations

logger = logging.getLogger(__name__)


@dataclass
class SprintConfig:
    """Configuration for accelerated sprint."""

    strategies: list[str] = None  # None = all
    symbols: list[str] = None  # None = VOLATILE_SYMBOLS
    use_gpu: bool = True
    use_ai: bool = True
    ai_iterations: int = 3
    parallel_symbols: int = 4
    output_dir: str = "artifacts/reports/strategy_sprint"
    # Walk-forward validation settings
    walk_forward: bool = True
    train_ratio: float = 0.7  # 70% train, 30% test

    def __post_init__(self):
        if self.strategies is None:
            self.strategies = ["garch", "kalman", "hmm", "ou_pairs", "evt", "mtf", "mi", "network"]
        if self.symbols is None:
            self.symbols = VOLATILE_SYMBOLS


class AcceleratedSprintRunner:
    """Orchestrates GPU-accelerated strategy sprint."""

    def __init__(self, config: SprintConfig | None = None):
        self.config = config or SprintConfig()
        self.gpu_engine: GPUBacktestEngine | None = None
        self.ai_optimizer: AIStrategyOptimizer | None = None
        self.results: dict[str, Any] = {}
        self.universe: dict[str, pd.DataFrame] = {}

    async def initialize(self) -> None:
        """Initialize engines and load data."""
        logger.info("Initializing Accelerated Sprint Runner...")

        # Load Massive data
        logger.info(f"Loading {len(self.config.symbols)} symbols from Massive...")
        self.universe = load_universe(self.config.symbols, min_days=500)
        logger.info(f"Loaded {len(self.universe)} symbols")

        # Initialize GPU engine
        if self.config.use_gpu:
            self.gpu_engine = GPUBacktestEngine(
                GPUConfig(
                    use_gpu=True,
                    parallel_workers=self.config.parallel_symbols,
                )
            )
            await self.gpu_engine.initialize()
            logger.info(f"GPU available: {self.gpu_engine.gpu_available}")

        # Initialize AI optimizer
        if self.config.use_ai:
            self.ai_optimizer = AIStrategyOptimizer(
                AIOptimizerConfig(
                    max_iterations=self.config.ai_iterations,
                    samples_per_iteration=6,
                )
            )
            ai_available = await self.ai_optimizer.initialize()
            logger.info(f"AI optimizer available: {ai_available}")

    async def close(self) -> None:
        """Clean up resources."""
        if self.gpu_engine:
            await self.gpu_engine.close()

    # =========================================================================
    # Strategy Implementations
    # =========================================================================

    def backtest_garch(
        self,
        df: pd.DataFrame,
        symbol: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """GARCH Breakout backtest with GPU acceleration."""
        threshold = params.get("breakout_threshold", 1.5)
        lookback = int(params.get("garch_lookback", 60))
        atr_stop = params.get("atr_stop_mult", 2.0)
        atr_tp = params.get("atr_tp_mult", 3.0)

        prices = df["close"].values
        returns = np.diff(prices) / prices[:-1]

        # Use GPU-accelerated volatility if available
        if self.gpu_engine:
            vol_short = self.gpu_engine.compute_ewma_volatility_gpu(returns, lookback)
            vol_long = self.gpu_engine.compute_volatility_gpu(returns, 252)
        else:
            vol_short = pd.Series(returns).ewm(span=lookback).std().values
            vol_long = pd.Series(returns).rolling(252).std().values

        # Calculate ATR
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        atr = pd.Series(tr).rolling(14).mean().values

        # Generate signals and simulate
        trades = []
        position = None

        start_idx = max(lookback, 252) + 30

        for i in range(start_idx, len(prices) - 1):
            if vol_long[i] > 0:
                ratio = vol_short[i] / vol_long[i]
            else:
                continue

            current_price = prices[i]
            current_atr = atr[i - 1] if i > 0 else 0

            # Exit logic
            if position is not None:
                exit_reason = None
                if position["direction"] == 1:
                    if current_price <= position["stop"]:
                        exit_reason = "stop"
                    elif current_price >= position["target"]:
                        exit_reason = "target"
                elif current_price >= position["stop"]:
                    exit_reason = "stop"
                elif current_price <= position["target"]:
                    exit_reason = "target"

                if position.get("bars", 0) >= 10:
                    exit_reason = "time"

                if exit_reason:
                    pnl_pct = (
                        (current_price - position["entry"])
                        / position["entry"]
                        * position["direction"]
                        * 100
                    )
                    trades.append(
                        {"pnl": pnl_pct, "exit": exit_reason, "bars": position.get("bars", 0)}
                    )
                    position = None
                else:
                    position["bars"] = position.get("bars", 0) + 1

            # Entry logic
            if position is None and ratio > threshold:
                direction = 1 if prices[i] > prices[i - 5] else -1
                position = {
                    "entry": current_price,
                    "direction": direction,
                    "stop": current_price - direction * current_atr * atr_stop,
                    "target": current_price + direction * current_atr * atr_tp,
                    "bars": 0,
                }

        return self._compute_metrics(trades, symbol)

    def backtest_kalman(
        self,
        df: pd.DataFrame,
        symbol: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Kalman Trend Filter backtest."""
        process_var = params.get("process_variance", 0.01)
        measure_var = params.get("measurement_variance", 0.1)
        trend_thresh = params.get("trend_threshold", 0.01)
        lookback = int(params.get("lookback", 30))

        prices = df["close"].values
        n = len(prices)

        # Kalman filter
        x = prices[0]  # State estimate
        P = 1.0  # Error covariance

        filtered = np.zeros(n)
        filtered[0] = x

        for i in range(1, n):
            # Predict
            x_pred = x
            P_pred = P + process_var

            # Update
            K = P_pred / (P_pred + measure_var)
            x = x_pred + K * (prices[i] - x_pred)
            P = (1 - K) * P_pred

            filtered[i] = x

        # Generate signals
        trades = []
        position = None

        for i in range(lookback + 10, n - 1):
            trend = (filtered[i] - filtered[i - lookback]) / filtered[i - lookback]

            # Exit
            if position is not None:
                if position["direction"] == 1 and trend < -trend_thresh:
                    pnl = (prices[i] - position["entry"]) / position["entry"] * 100
                    trades.append({"pnl": pnl, "bars": position.get("bars", 0)})
                    position = None
                elif position["direction"] == -1 and trend > trend_thresh:
                    pnl = (position["entry"] - prices[i]) / position["entry"] * 100
                    trades.append({"pnl": pnl, "bars": position.get("bars", 0)})
                    position = None
                elif position.get("bars", 0) >= 20:
                    pnl = (
                        (prices[i] - position["entry"])
                        / position["entry"]
                        * position["direction"]
                        * 100
                    )
                    trades.append({"pnl": pnl, "bars": position.get("bars", 0)})
                    position = None
                else:
                    position["bars"] = position.get("bars", 0) + 1

            # Entry
            if position is None:
                if trend > trend_thresh:
                    position = {"entry": prices[i], "direction": 1, "bars": 0}
                elif trend < -trend_thresh:
                    position = {"entry": prices[i], "direction": -1, "bars": 0}

        return self._compute_metrics(trades, symbol)

    def backtest_hmm(
        self,
        df: pd.DataFrame,
        symbol: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """HMM Regime backtest (simplified without hmmlearn)."""
        n_regimes = int(params.get("n_regimes", 3))
        lookback = int(params.get("lookback", 252))
        transition_thresh = params.get("transition_threshold", 0.7)

        returns = df["close"].pct_change().dropna().values
        n = len(returns)

        # Simplified regime detection using volatility clustering
        vol = pd.Series(returns).rolling(20).std().values

        # Quantile-based regime assignment
        regimes = np.zeros(n, dtype=int)
        for i in range(lookback, n):
            window_vol = vol[i - lookback : i]
            current_vol = vol[i]
            if np.isnan(current_vol) or np.all(np.isnan(window_vol)):
                continue
            percentile = np.nanpercentile(window_vol, [33, 66])
            if current_vol < percentile[0]:
                regimes[i] = 0  # Low vol
            elif current_vol < percentile[1]:
                regimes[i] = 1  # Medium vol
            else:
                regimes[i] = 2  # High vol

        # Trade based on regime changes
        trades = []
        position = None
        prices = df["close"].values

        for i in range(lookback + 10, n - 1):
            # Regime transition detection
            curr_regime = regimes[i]

            # Exit in high vol regime
            if position is not None:
                if curr_regime == 2:  # High vol = exit
                    # Calculate actual price-based PnL
                    pnl = (
                        (prices[i] - position["entry_price"])
                        / position["entry_price"]
                        * position["direction"]
                        * 100
                    )
                    trades.append({"pnl": pnl, "bars": position.get("bars", 0)})
                    position = None
                elif position.get("bars", 0) >= 15:
                    pnl = (
                        (prices[i] - position["entry_price"])
                        / position["entry_price"]
                        * position["direction"]
                        * 100
                    )
                    trades.append({"pnl": pnl, "bars": position.get("bars", 0)})
                    position = None
                else:
                    position["bars"] = position.get("bars", 0) + 1

            # Enter in low/medium vol with momentum
            if position is None and curr_regime < 2:
                momentum = np.sum(returns[i - 5 : i])
                if abs(momentum) > 0.01:
                    direction = 1 if momentum > 0 else -1
                    position = {"direction": direction, "entry_price": prices[i], "bars": 0}

        return self._compute_metrics(trades, symbol)

    def backtest_ou_pairs(
        self,
        symbol: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """OU Pairs trading (requires 2 cointegrated symbols)."""
        zscore_entry = params.get("zscore_entry", 2.0)
        zscore_exit = params.get("zscore_exit", 0.5)
        lookback = int(params.get("lookback", 60))

        # Find a pair for this symbol
        pair_map = {
            "NVDA": "GOOGL",
            "MS": "GS",
            "BAC": "JPM",
            "META": "GOOGL",
            "WFC": "BAC",
            "EOG": "SLB",
            "SLB": "COP",
            "GS": "MS",
        }

        pair_symbol = pair_map.get(symbol)
        if not pair_symbol or pair_symbol not in self.universe:
            return {
                "symbol": symbol,
                "trades": 0,
                "sharpe": 0,
                "total_return": 0,
                "win_rate": 0,
                "max_drawdown": 0,
            }

        df1 = self.universe[symbol]
        df2 = self.universe[pair_symbol]

        # Align dates
        common = df1.index.intersection(df2.index)
        p1 = df1.loc[common, "close"].values
        p2 = df2.loc[common, "close"].values

        n = len(p1)
        if n < lookback + 100:
            return {
                "symbol": symbol,
                "trades": 0,
                "sharpe": 0,
                "total_return": 0,
                "win_rate": 0,
                "max_drawdown": 0,
            }

        # Calculate spread
        log_p1 = np.log(p1)
        log_p2 = np.log(p2)

        # Rolling hedge ratio
        trades = []
        position = None

        for i in range(lookback + 10, n - 1):
            window1 = log_p1[i - lookback : i]
            window2 = log_p2[i - lookback : i]

            # OLS hedge ratio
            beta = np.cov(window1, window2)[0, 1] / np.var(window2)
            spread = log_p1[i] - beta * log_p2[i]
            spread_mean = np.mean(log_p1[i - lookback : i] - beta * log_p2[i - lookback : i])
            spread_std = np.std(log_p1[i - lookback : i] - beta * log_p2[i - lookback : i])

            if spread_std == 0:
                continue

            zscore = (spread - spread_mean) / spread_std

            # Exit
            if position is not None:
                if abs(zscore) < zscore_exit:
                    # Calculate realistic PnL from price changes (not log spread * 100)
                    # PnL = long leg return - beta * short leg return
                    long_ret = (p1[i] - position["entry_p1"]) / position["entry_p1"]
                    short_ret = (p2[i] - position["entry_p2"]) / position["entry_p2"]
                    pnl = (long_ret - position["beta"] * short_ret) * position["direction"] * 100
                    trades.append({"pnl": pnl, "bars": position.get("bars", 0)})
                    position = None
                elif position.get("bars", 0) >= 20:
                    long_ret = (p1[i] - position["entry_p1"]) / position["entry_p1"]
                    short_ret = (p2[i] - position["entry_p2"]) / position["entry_p2"]
                    pnl = (long_ret - position["beta"] * short_ret) * position["direction"] * 100
                    trades.append({"pnl": pnl, "bars": position.get("bars", 0)})
                    position = None
                else:
                    position["bars"] = position.get("bars", 0) + 1

            # Entry
            if position is None:
                if zscore > zscore_entry:
                    # Short spread when zscore high (spread will revert down)
                    position = {
                        "entry_p1": p1[i],
                        "entry_p2": p2[i],
                        "beta": beta,
                        "direction": -1,
                        "bars": 0,
                    }
                elif zscore < -zscore_entry:
                    # Long spread when zscore low (spread will revert up)
                    position = {
                        "entry_p1": p1[i],
                        "entry_p2": p2[i],
                        "beta": beta,
                        "direction": 1,
                        "bars": 0,
                    }

        result = self._compute_metrics(trades, symbol)
        result["pair"] = pair_symbol
        return result

    def backtest_evt(
        self,
        df: pd.DataFrame,
        symbol: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """EVT Tail Risk backtest - trades reversals after extreme moves."""
        threshold_pct = params.get("threshold_percentile", 95)
        lookback = int(params.get("lookback", 252))
        holding = int(params.get("holding_period", 5))
        min_exceed = int(params.get("min_exceedances", 20))

        returns = df["close"].pct_change().dropna().values
        prices = df["close"].values
        n = len(returns)

        trades = []
        position = None

        for i in range(lookback, n - holding):
            # Exit logic first
            if position is not None:
                if position["bars"] >= holding:
                    pnl = (
                        (prices[i] - position["entry"])
                        / position["entry"]
                        * position["direction"]
                        * 100
                    )
                    trades.append({"pnl": pnl, "bars": position["bars"]})
                    position = None
                else:
                    position["bars"] += 1
                continue

            # Compute tail thresholds from lookback window
            window = returns[i - lookback : i]
            upper_thresh = np.percentile(window, threshold_pct)
            lower_thresh = np.percentile(window, 100 - threshold_pct)

            # Count exceedances to check if we have enough tail events
            upper_exceed = np.sum(window > upper_thresh)
            lower_exceed = np.sum(window < lower_thresh)

            if upper_exceed < min_exceed or lower_exceed < min_exceed:
                continue  # Not enough tail data

            current_ret = returns[i]

            # Entry on extreme events (fade the move)
            if current_ret > upper_thresh:
                # Extreme up move - go short expecting reversion
                position = {"entry": prices[i], "direction": -1, "bars": 0}
            elif current_ret < lower_thresh:
                # Extreme down move - go long expecting bounce
                position = {"entry": prices[i], "direction": 1, "bars": 0}

        return self._compute_metrics(trades, symbol)

    def backtest_mtf(
        self,
        df: pd.DataFrame,
        symbol: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Multi-Timeframe Momentum backtest - requires alignment across timeframes."""
        short_period = int(params.get("short_period", 10))
        medium_period = int(params.get("medium_period", 30))
        long_period = int(params.get("long_period", 120))
        align_thresh = params.get("alignment_threshold", 0.7)

        prices = df["close"].values
        n = len(prices)

        # Compute momentum for each timeframe
        mom_short = np.zeros(n)
        mom_medium = np.zeros(n)
        mom_long = np.zeros(n)

        for i in range(long_period, n):
            mom_short[i] = (prices[i] - prices[i - short_period]) / prices[i - short_period]
            mom_medium[i] = (prices[i] - prices[i - medium_period]) / prices[i - medium_period]
            mom_long[i] = (prices[i] - prices[i - long_period]) / prices[i - long_period]

        trades = []
        position = None

        for i in range(long_period + 10, n - 1):
            # Count aligned timeframes
            bullish = sum([mom_short[i] > 0, mom_medium[i] > 0, mom_long[i] > 0])
            bearish = sum([mom_short[i] < 0, mom_medium[i] < 0, mom_long[i] < 0])

            # Exit logic
            if position is not None:
                should_exit = False
                if position["direction"] == 1 and bullish < 2:
                    should_exit = True  # Lost alignment
                elif position["direction"] == -1 and bearish < 2:
                    should_exit = True
                elif position["bars"] >= 20:
                    should_exit = True

                if should_exit:
                    pnl = (
                        (prices[i] - position["entry"])
                        / position["entry"]
                        * position["direction"]
                        * 100
                    )
                    trades.append({"pnl": pnl, "bars": position["bars"]})
                    position = None
                else:
                    position["bars"] += 1

            # Entry logic
            if position is None:
                if bullish / 3 >= align_thresh:
                    position = {"entry": prices[i], "direction": 1, "bars": 0}
                elif bearish / 3 >= align_thresh:
                    position = {"entry": prices[i], "direction": -1, "bars": 0}

        return self._compute_metrics(trades, symbol)

    def backtest_mi(
        self,
        df: pd.DataFrame,
        symbol: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Mutual Information Lead-Lag backtest - uses volume as leading indicator."""
        mi_lookback = int(params.get("mi_lookback", 60))
        mi_threshold = params.get("mi_threshold", 0.15)
        lag_range = int(params.get("lag_range", 5))
        smoothing = int(params.get("signal_smoothing", 3))

        prices = df["close"].values
        returns = np.diff(prices) / prices[:-1]

        # Use volume changes as potential leading indicator
        if "volume" in df.columns:
            vol = df["volume"].values
            vol_change = np.diff(vol) / (vol[:-1] + 1)
        else:
            # Fallback: use price momentum as proxy
            vol_change = returns.copy()

        n = len(returns)

        def compute_mi(x, y, bins=10):
            """Compute mutual information between two series."""
            try:
                c_xy = np.histogram2d(x, y, bins=bins)[0]
                c_x = np.histogram(x, bins=bins)[0]
                c_y = np.histogram(y, bins=bins)[0]

                # Normalize
                p_xy = c_xy / c_xy.sum()
                p_x = c_x / c_x.sum()
                p_y = c_y / c_y.sum()

                # MI = sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
                mi = 0
                for i in range(bins):
                    for j in range(bins):
                        if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                            mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
                return mi
            except:
                return 0

        trades = []
        position = None
        signal_ema = 0

        for i in range(mi_lookback + lag_range + 10, n - 1):
            # Find best lag
            best_mi = 0
            best_lag = 1
            for lag in range(1, lag_range + 1):
                vol_window = vol_change[i - mi_lookback - lag : i - lag]
                ret_window = returns[i - mi_lookback : i]
                if len(vol_window) == len(ret_window):
                    mi = compute_mi(vol_window, ret_window)
                    if mi > best_mi:
                        best_mi = mi
                        best_lag = lag

            # Generate signal if MI is high enough
            signal = 0
            if best_mi > mi_threshold:
                # Use lagged volume change as predictor
                signal = np.sign(vol_change[i - best_lag]) if i >= best_lag else 0

            # EMA smoothing
            alpha = 2 / (smoothing + 1)
            signal_ema = alpha * signal + (1 - alpha) * signal_ema

            # Exit
            if position is not None:
                should_exit = False
                if position["direction"] == 1 and signal_ema < 0:
                    should_exit = True
                elif position["direction"] == -1 and signal_ema > 0:
                    should_exit = True
                elif position["bars"] >= 15:
                    should_exit = True

                if should_exit:
                    pnl = (
                        (prices[i] - position["entry"])
                        / position["entry"]
                        * position["direction"]
                        * 100
                    )
                    trades.append({"pnl": pnl, "bars": position["bars"]})
                    position = None
                else:
                    position["bars"] += 1

            # Entry
            if position is None and abs(signal_ema) > 0.3:
                direction = 1 if signal_ema > 0 else -1
                position = {"entry": prices[i], "direction": direction, "bars": 0}

        return self._compute_metrics(trades, symbol)

    def backtest_network(
        self,
        df: pd.DataFrame,
        symbol: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Network Correlation Regime backtest - uses correlation clustering."""
        corr_lookback = int(params.get("corr_lookback", 60))
        edge_thresh = params.get("edge_threshold", 0.5)
        density_high = params.get("density_high", 0.7)
        density_low = params.get("density_low", 0.3)

        prices = df["close"].values
        n = len(prices)

        # For single symbol, use rolling autocorrelations at different lags as proxy for "network"
        # In production, this would use actual cross-asset correlations
        trades = []
        position = None

        for i in range(corr_lookback + 20, n - 1):
            # Compute rolling autocorrelations at various lags as network proxy
            returns_window = (
                np.diff(prices[i - corr_lookback : i]) / prices[i - corr_lookback : i - 1]
            )

            if len(returns_window) < 10:
                continue

            # Calculate correlation matrix between lagged series
            lags = [1, 2, 3, 5, 10]
            edges = 0
            total_pairs = 0

            for l1 in lags:
                for l2 in lags:
                    if l1 < l2 and len(returns_window) > max(l1, l2):
                        s1 = returns_window[l1:]
                        s2 = returns_window[:-l2] if l2 > 0 else returns_window
                        min_len = min(len(s1), len(s2))
                        if min_len > 5:
                            corr = np.corrcoef(s1[:min_len], s2[:min_len])[0, 1]
                            if not np.isnan(corr) and abs(corr) > edge_thresh:
                                edges += 1
                            total_pairs += 1

            # Network density
            density = edges / total_pairs if total_pairs > 0 else 0.5

            # Exit
            if position is not None:
                should_exit = False
                # Exit risk-off when density drops
                if position["direction"] == -1 and density < density_high:
                    should_exit = True
                # Exit risk-on when density rises
                elif position["direction"] == 1 and density > density_low + 0.2:
                    should_exit = True
                elif position["bars"] >= 20:
                    should_exit = True

                if should_exit:
                    pnl = (
                        (prices[i] - position["entry"])
                        / position["entry"]
                        * position["direction"]
                        * 100
                    )
                    trades.append({"pnl": pnl, "bars": position["bars"]})
                    position = None
                else:
                    position["bars"] += 1

            # Entry
            if position is None:
                if density > density_high:
                    # High correlation = risk-off, go defensive (short bias)
                    position = {"entry": prices[i], "direction": -1, "bars": 0}
                elif density < density_low:
                    # Low correlation = opportunity, go long
                    position = {"entry": prices[i], "direction": 1, "bars": 0}

        return self._compute_metrics(trades, symbol)

    def _compute_metrics(
        self,
        trades: list[dict],
        symbol: str,
        txn_cost_bps: float = 10.0,  # 10 bps = 0.1% round-trip
    ) -> dict[str, Any]:
        """Compute strategy metrics from trades with transaction costs."""
        if not trades:
            return {
                "symbol": symbol,
                "trades": 0,
                "total_return": 0,
                "annual_return": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "sharpe": 0,
            }

        # Apply transaction costs (entry + exit = round-trip)
        txn_cost_pct = txn_cost_bps / 100  # 10 bps = 0.1%
        pnls = [t["pnl"] - txn_cost_pct for t in trades]  # Subtract txn cost from each trade
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_return = sum(pnls)

        # Annualized return: estimate years from trade count and avg holding period
        avg_bars = np.mean([t.get("bars", 5) for t in trades])
        total_days = len(trades) * avg_bars
        years = total_days / 252 if total_days > 0 else 1
        annual_return = total_return / years if years > 0 else total_return

        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Drawdown
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

        # Sharpe (annualized)
        if len(pnls) > 1:
            trades_per_year = 252 / max(avg_bars, 1)
            sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(trades_per_year)
        else:
            sharpe = 0

        return {
            "symbol": symbol,
            "trades": len(trades),
            "total_return": total_return,
            "annual_return": annual_return,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
        }

    # =========================================================================
    # Sprint Orchestration
    # =========================================================================

    async def run_strategy(
        self,
        strategy_name: str,
        profile: StrategyProfile,
        backtest_fn,
    ) -> dict[str, Any]:
        """Run single strategy with optimization across all symbols."""
        logger.info(f"\n{'='*60}")
        logger.info(f"STRATEGY: {strategy_name.upper()}")
        logger.info(f"{'='*60}")

        all_results = []
        best_params_per_symbol = {}

        for symbol in self.universe.keys():
            logger.info(f"\n--- {symbol} ---")
            df = self.universe[symbol]

            # Create backtest wrapper
            if strategy_name == "ou_pairs":

                def bt_fn(params):
                    return self.backtest_ou_pairs(symbol, params)
            else:

                def bt_fn(params):
                    return backtest_fn(df, symbol, params)

            # Default params
            default_params = {
                name: defn.get("default", 0) for name, defn in profile.param_definitions.items()
            }

            if self.config.use_ai and self.ai_optimizer:
                # AI-guided optimization
                opt_result = await self.ai_optimizer.run_optimization(
                    profile,
                    bt_fn,
                    default_params,
                )

                best_params = opt_result["best_params"]
                best_result = bt_fn(best_params)
                best_result["optimized"] = True
                best_result["ai_used"] = opt_result["ai_available"]
                best_params_per_symbol[symbol] = best_params
            else:
                # Just run with defaults
                best_result = bt_fn(default_params)
                best_result["optimized"] = False
                best_params_per_symbol[symbol] = default_params

            all_results.append(best_result)

            logger.info(f"  Trades: {best_result['trades']}")
            logger.info(
                f"  Return: {best_result['total_return']:.2f}% (annual: {best_result.get('annual_return', 0):.2f}%)"
            )
            logger.info(f"  Sharpe: {best_result['sharpe']:.2f}")
            logger.info(f"  Win Rate: {best_result['win_rate']:.1f}%")

        # Aggregate
        results_df = pd.DataFrame(all_results)

        return {
            "strategy": strategy_name,
            "results": results_df.to_dict("records"),
            "best_params": best_params_per_symbol,
            "avg_sharpe": results_df["sharpe"].mean(),
            "avg_return": results_df["total_return"].mean(),
            "avg_annual_return": results_df["annual_return"].mean()
            if "annual_return" in results_df
            else 0,
            "avg_win_rate": results_df["win_rate"].mean(),
            "total_trades": results_df["trades"].sum(),
        }

    async def run_walk_forward(
        self,
        strategy_name: str,
        profile: StrategyProfile,
        backtest_fn,
    ) -> dict[str, Any]:
        """Run walk-forward validation: train on first 70%, test on last 30%."""
        logger.info(f"\n{'='*60}")
        logger.info(f"WALK-FORWARD: {strategy_name.upper()}")
        logger.info(f"{'='*60}")

        train_results = []
        test_results = []
        best_params_per_symbol = {}

        train_ratio = self.config.train_ratio

        for symbol in self.universe.keys():
            logger.info(f"\n--- {symbol} ---")
            df_full = self.universe[symbol]
            n = len(df_full)
            split_idx = int(n * train_ratio)

            df_train = df_full.iloc[:split_idx].copy()
            df_test = df_full.iloc[split_idx:].copy()

            logger.info(f"  Train: {len(df_train)} days, Test: {len(df_test)} days")

            # Create backtest wrappers for train and test
            if strategy_name == "ou_pairs":
                # OU pairs needs special handling with the pair
                def train_bt_fn(params):
                    return self._backtest_ou_pairs_on_df(df_train, symbol, params)

                def test_bt_fn(params):
                    return self._backtest_ou_pairs_on_df(df_test, symbol, params)
            else:

                def train_bt_fn(params, df=df_train, sym=symbol, fn=backtest_fn):
                    return fn(df, sym, params)

                def test_bt_fn(params, df=df_test, sym=symbol, fn=backtest_fn):
                    return fn(df, sym, params)

            # Default params
            default_params = {
                name: defn.get("default", 0) for name, defn in profile.param_definitions.items()
            }

            # Optimize on TRAIN set
            if self.config.use_ai and self.ai_optimizer:
                opt_result = await self.ai_optimizer.run_optimization(
                    profile,
                    train_bt_fn,
                    default_params,
                )
                best_params = opt_result["best_params"]
            else:
                best_params = default_params

            # Evaluate on TRAIN (in-sample)
            train_result = train_bt_fn(best_params)
            train_result["phase"] = "train"
            train_results.append(train_result)

            # Evaluate on TEST (out-of-sample) with same params
            test_result = test_bt_fn(best_params)
            test_result["phase"] = "test"
            test_results.append(test_result)

            best_params_per_symbol[symbol] = best_params

            logger.info(
                f"  TRAIN: Sharpe={train_result['sharpe']:.2f}, Return={train_result['total_return']:.1f}%"
            )
            logger.info(
                f"  TEST:  Sharpe={test_result['sharpe']:.2f}, Return={test_result['total_return']:.1f}%"
            )

            # Flag overfitting
            if train_result["sharpe"] > 0.5 and test_result["sharpe"] < 0:
                logger.warning(f"  ⚠️ OVERFIT DETECTED: Train Sharpe >> Test Sharpe")

        # Aggregate
        train_df = pd.DataFrame(train_results)
        test_df = pd.DataFrame(test_results)

        return {
            "strategy": strategy_name,
            "train_results": train_df.to_dict("records"),
            "test_results": test_df.to_dict("records"),
            "best_params": best_params_per_symbol,
            "train_avg_sharpe": train_df["sharpe"].mean(),
            "test_avg_sharpe": test_df["sharpe"].mean(),
            "train_avg_annual_return": train_df["annual_return"].mean()
            if "annual_return" in train_df
            else 0,
            "test_avg_annual_return": test_df["annual_return"].mean()
            if "annual_return" in test_df
            else 0,
            "train_total_trades": train_df["trades"].sum(),
            "test_total_trades": test_df["trades"].sum(),
            "overfit_ratio": train_df["sharpe"].mean() / (test_df["sharpe"].mean() + 0.01),
        }

    def _backtest_ou_pairs_on_df(
        self,
        df: pd.DataFrame,
        symbol: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """OU pairs backtest on a specific dataframe subset."""
        # Simplified: just use the provided df as the primary asset
        # and create a synthetic pair from a lagged version
        zscore_entry = params.get("zscore_entry", 2.0)
        zscore_exit = params.get("zscore_exit", 0.5)
        lookback = int(params.get("lookback", 60))

        if len(df) < lookback + 50:
            return self._compute_metrics([], symbol)

        prices = df["close"].values
        # Create synthetic pair from MA-smoothed version
        ma = pd.Series(prices).rolling(10).mean().values

        p1 = prices
        p2 = ma

        n = len(p1)
        trades = []
        position = None

        for i in range(lookback + 10, n - 1):
            if np.isnan(p2[i]):
                continue

            # Compute spread and zscore
            spread = np.log(p1[i - lookback : i]) - np.log(p2[i - lookback : i])
            spread = spread[~np.isnan(spread)]
            if len(spread) < 20:
                continue

            current_spread = np.log(p1[i]) - np.log(p2[i])
            zscore = (current_spread - np.mean(spread)) / (np.std(spread) + 1e-10)
            beta = 1.0  # Simplified

            # Exit
            if position is not None:
                if abs(zscore) < zscore_exit:
                    long_ret = (p1[i] - position["entry_p1"]) / position["entry_p1"]
                    short_ret = (p2[i] - position["entry_p2"]) / position["entry_p2"]
                    pnl = (long_ret - beta * short_ret) * position["direction"] * 100
                    trades.append({"pnl": pnl, "bars": position.get("bars", 0)})
                    position = None
                elif position.get("bars", 0) >= 20:
                    long_ret = (p1[i] - position["entry_p1"]) / position["entry_p1"]
                    short_ret = (p2[i] - position["entry_p2"]) / position["entry_p2"]
                    pnl = (long_ret - beta * short_ret) * position["direction"] * 100
                    trades.append({"pnl": pnl, "bars": position.get("bars", 0)})
                    position = None
                else:
                    position["bars"] = position.get("bars", 0) + 1

            # Entry
            if position is None:
                if zscore > zscore_entry:
                    position = {"entry_p1": p1[i], "entry_p2": p2[i], "direction": -1, "bars": 0}
                elif zscore < -zscore_entry:
                    position = {"entry_p1": p1[i], "entry_p2": p2[i], "direction": 1, "bars": 0}

        return self._compute_metrics(trades, symbol)

    async def run_sprint(self) -> dict[str, Any]:
        """Run full accelerated sprint."""
        sprint_start = time.perf_counter()

        logger.info("=" * 60)
        logger.info("ACCELERATED STRATEGY SPRINT")
        logger.info(f"Started: {datetime.now().isoformat()}")
        logger.info(f"Strategies: {self.config.strategies}")
        logger.info(f"Symbols: {list(self.universe.keys())}")
        logger.info(f"GPU: {self.gpu_engine.gpu_available if self.gpu_engine else False}")
        logger.info(f"AI Optimization: {self.config.use_ai}")
        logger.info(f"Walk-Forward: {self.config.walk_forward}")
        logger.info("=" * 60)

        strategy_results = {}

        # Map strategies to backtests and profiles
        strategy_map = {
            "garch": (self.backtest_garch, GARCH_BREAKOUT_PROFILE),
            "kalman": (self.backtest_kalman, KALMAN_TREND_PROFILE),
            "hmm": (self.backtest_hmm, HMM_REGIME_PROFILE),
            "ou_pairs": (None, OU_PAIRS_PROFILE),  # Special handling
            "evt": (self.backtest_evt, EVT_TAIL_PROFILE),
            "mtf": (self.backtest_mtf, MTF_MOMENTUM_PROFILE),
            "mi": (self.backtest_mi, MI_LEAD_LAG_PROFILE),
            "network": (self.backtest_network, NETWORK_REGIME_PROFILE),
        }

        for strategy in self.config.strategies:
            if strategy in strategy_map:
                backtest_fn, profile = strategy_map[strategy]
                if self.config.walk_forward:
                    result = await self.run_walk_forward(strategy, profile, backtest_fn)
                else:
                    result = await self.run_strategy(strategy, profile, backtest_fn)
                strategy_results[strategy] = result
            else:
                logger.warning(f"Strategy '{strategy}' not implemented in accelerated runner")

        sprint_time = time.perf_counter() - sprint_start

        # Save results
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Summary
        summary = {
            "timestamp": timestamp,
            "sprint_time_seconds": sprint_time,
            "gpu_used": self.gpu_engine.gpu_available if self.gpu_engine else False,
            "ai_used": self.config.use_ai,
            "symbols_tested": list(self.universe.keys()),
            "strategies": {},
        }

        for name, result in strategy_results.items():
            if self.config.walk_forward:
                summary["strategies"][name] = {
                    "train_avg_sharpe": float(result.get("train_avg_sharpe", 0)),
                    "test_avg_sharpe": float(result.get("test_avg_sharpe", 0)),
                    "train_avg_annual_return": float(result.get("train_avg_annual_return", 0)),
                    "test_avg_annual_return": float(result.get("test_avg_annual_return", 0)),
                    "train_total_trades": int(result.get("train_total_trades", 0)),
                    "test_total_trades": int(result.get("test_total_trades", 0)),
                    "overfit_ratio": float(result.get("overfit_ratio", 1)),
                }
                # Save detailed results
                train_df = pd.DataFrame(result.get("train_results", []))
                test_df = pd.DataFrame(result.get("test_results", []))
                train_df.to_csv(output_dir / f"{name}_train_{timestamp}.csv", index=False)
                test_df.to_csv(output_dir / f"{name}_test_{timestamp}.csv", index=False)
            else:
                summary["strategies"][name] = {
                    "avg_sharpe": float(result["avg_sharpe"]),
                    "avg_return": float(result["avg_return"]),
                    "avg_win_rate": float(result["avg_win_rate"]),
                    "total_trades": int(result["total_trades"]),
                }
                # Save detailed results
                results_df = pd.DataFrame(result["results"])
                results_df.to_csv(output_dir / f"{name}_results_{timestamp}.csv", index=False)

        # Save summary
        with open(output_dir / f"sprint_summary_{timestamp}.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Print final summary
        logger.info("\n" + "=" * 60)
        logger.info("SPRINT COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Time: {sprint_time:.1f}s")

        for name, result in strategy_results.items():
            logger.info(f"\n{name.upper()}:")
            if self.config.walk_forward:
                train_sharpe = result.get("train_avg_sharpe", 0)
                test_sharpe = result.get("test_avg_sharpe", 0)
                train_ret = result.get("train_avg_annual_return", 0)
                test_ret = result.get("test_avg_annual_return", 0)
                overfit = result.get("overfit_ratio", 1)
                logger.info(
                    f"  Train Sharpe: {float(train_sharpe):.3f} | Test Sharpe: {float(test_sharpe):.3f}"
                )
                logger.info(
                    f"  Train Return: {float(train_ret):.2f}% | Test Return: {float(test_ret):.2f}%"
                )
                logger.info(
                    f"  Overfit Ratio: {float(overfit):.2f} {'⚠️ OVERFIT' if overfit > 2 else '✓'}"
                )
            else:
                logger.info(f"  Avg Sharpe: {float(result['avg_sharpe']):.3f}")
                annual = result.get("avg_annual_return", result["avg_return"])
                logger.info(f"  Avg Annual Return: {float(annual):.2f}%")
                logger.info(f"  Total Trades: {int(result['total_trades'])}")

        # Generate visualizations
        logger.info("\n" + "-" * 40)
        logger.info("Generating visualizations...")
        try:
            viz_results = await generate_all_visualizations(
                {
                    "sprint_time": sprint_time,
                    "detailed_results": strategy_results,
                }
            )
            for viz_name, viz_path in viz_results.items():
                logger.info(f"  ✓ {viz_name}: {viz_path}")
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")

        return {
            "success": True,
            "sprint_time": sprint_time,
            "summary": summary,
            "detailed_results": strategy_results,
        }


async def main():
    """Run accelerated strategy sprint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Configure sprint - ALL 8 strategies with walk-forward validation
    config = SprintConfig(
        strategies=["garch", "kalman", "hmm", "ou_pairs", "evt", "mtf", "mi", "network"],
        symbols=VOLATILE_SYMBOLS,
        use_gpu=True,
        use_ai=True,
        ai_iterations=2,  # Faster for demo
        walk_forward=True,  # Enable out-of-sample testing
        train_ratio=0.7,  # 70% train, 30% test
    )

    runner = AcceleratedSprintRunner(config)

    try:
        await runner.initialize()
        results = await runner.run_sprint()

        if results["success"]:
            logger.info("\n✓ Sprint completed successfully!")
            logger.info(f"Results saved to: {config.output_dir}")

        return results

    finally:
        await runner.close()


if __name__ == "__main__":
    asyncio.run(main())
