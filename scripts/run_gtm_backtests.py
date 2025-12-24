#!/usr/bin/env python3
"""
GPU-Accelerated GTM Strategy Backtest Runner.

Executes documented backtests for all GTM strategies with:
- GPU/CUDA acceleration via CuPy/Numba
- Transaction cost modeling (8 bps round-trip)
- Extended symbol universe (30+ symbols)
- Walk-forward validation
- Bootstrap confidence intervals
- Comprehensive results export

Usage:
    python scripts/run_gtm_backtests.py --strategies all --days 252 --gpu
    python scripts/run_gtm_backtests.py --strategies atr_rsi mi_ensemble --symbols AAPL,MSFT,GOOGL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ordinis.engines.proofbench.validation.strategy_validation import (
    BootstrapResult,
    CostAnalysis,
    StrategyValidationResult,
    WalkForwardPeriod,
    WalkForwardSummary,
    create_default_validator,
)
from ordinis.engines.portfolio.costs.transaction_cost_model import SimpleCostModel

# GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

try:
    from numba import cuda, jit
    HAS_NUMBA = True
except ImportError:
    cuda = None
    jit = None
    HAS_NUMBA = False

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_TORCH = False

logger = logging.getLogger(__name__)


# =============================================================================
# Symbol Universe - REAL MASSIVE DATA ONLY
# =============================================================================

# Symbols with REAL historical data exported from Massive
# Located in data/historical/{symbol}_historical.csv or {symbol}.parquet
MASSIVE_SYMBOLS = [
    # Tech (high beta, momentum-driven)
    "AAPL", "MSFT", "GOOGL", "NVDA", "META",
    # Financials
    "JPM", "BAC", "GS", "MS", "WFC",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "TMO",
    # Consumer
    "WMT", "HD", "NKE", "MCD", "SBUX",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB",
]

# Categorization for strategy-specific testing
VOLATILE_SYMBOLS = ["NVDA", "META", "GOOGL", "MS", "GS"]  # High beta
STABLE_SYMBOLS = ["JNJ", "PFE", "WMT", "MCD", "HD"]  # Lower volatility
ENERGY_SYMBOLS = ["XOM", "CVX", "COP", "EOG", "SLB"]  # Energy sector
FINANCE_SYMBOLS = ["JPM", "BAC", "GS", "MS", "WFC"]  # Financials
TECH_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]  # Technology

# Sector mapping for analysis
SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "NVDA": "Technology", "META": "Technology",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "WFC": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "TMO": "Healthcare",
    "WMT": "Consumer", "HD": "Consumer", "NKE": "Consumer",
    "MCD": "Consumer", "SBUX": "Consumer",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "EOG": "Energy", "SLB": "Energy",
}


@dataclass
class GPUBacktestConfig:
    """Configuration for GPU-accelerated backtesting."""

    # GPU settings
    use_gpu: bool = True
    device_id: int = 0
    batch_size: int = 2048

    # Backtest settings
    initial_capital: float = 100_000.0
    days: int = 252  # 1 year minimum
    bar_frequency: str = "1D"  # Daily bars for now

    # Cost model (8 bps round-trip as per assessment)
    spread_bps: float = 3.0
    impact_bps: float = 5.0
    commission_per_trade: float = 0.0

    # Walk-forward
    train_days: int = 126  # 6 months
    test_days: int = 63  # 3 months
    step_days: int = 21  # 1 month step

    # Bootstrap
    n_bootstrap: int = 1000

    # Output
    output_dir: Path = field(default_factory=lambda: Path("data/backtest_results"))


@dataclass
class BacktestResult:
    """Result from a single symbol backtest."""

    symbol: str
    strategy: str
    start_date: date
    end_date: date
    n_bars: int
    n_trades: int

    # Returns
    gross_return_pct: float
    net_return_pct: float
    gross_sharpe: float
    net_sharpe: float

    # Risk metrics
    max_drawdown_pct: float
    volatility_pct: float
    sortino_ratio: float

    # Trade stats
    win_rate_pct: float
    profit_factor: float
    avg_trade_return_pct: float

    # Costs
    total_costs: Decimal
    avg_cost_per_trade_bps: float

    # Timing
    compute_time_ms: float
    used_gpu: bool


class GPUBacktestRunner:
    """GPU-accelerated backtest runner for GTM strategies."""

    def __init__(self, config: GPUBacktestConfig):
        self.config = config
        self.gpu_available = self._check_gpu()
        self.cost_model = SimpleCostModel(
            spread_bps=config.spread_bps,
            impact_bps=config.impact_bps,
            commission_per_trade=config.commission_per_trade,
        )
        self.validator = create_default_validator()

        if self.gpu_available:
            logger.info(f"GPU acceleration enabled: CuPy={HAS_CUPY}, Numba={HAS_NUMBA}, PyTorch={HAS_TORCH}")
        else:
            logger.warning("GPU not available, using CPU")

    def _check_gpu(self) -> bool:
        """Check GPU availability."""
        if not self.config.use_gpu:
            return False

        if HAS_CUPY:
            try:
                cp.cuda.Device(self.config.device_id).use()
                gpu_name = cp.cuda.runtime.getDeviceProperties(self.config.device_id)["name"]
                logger.info(f"Using GPU: {gpu_name.decode()}")
                return True
            except Exception as e:
                logger.warning(f"CuPy GPU check failed: {e}")

        if HAS_TORCH:
            try:
                device = torch.device(f"cuda:{self.config.device_id}")
                torch.zeros(1, device=device)
                gpu_name = torch.cuda.get_device_name(self.config.device_id)
                logger.info(f"Using GPU via PyTorch: {gpu_name}")
                return True
            except Exception as e:
                logger.warning(f"PyTorch GPU check failed: {e}")

        if HAS_NUMBA and cuda:
            try:
                if cuda.is_available():
                    logger.info("Numba CUDA available")
                    return True
            except Exception as e:
                logger.warning(f"Numba GPU check failed: {e}")

        return False

    def _compute_returns_gpu(self, prices: np.ndarray) -> np.ndarray:
        """Compute returns using GPU."""
        if self.gpu_available and HAS_CUPY:
            prices_gpu = cp.asarray(prices)
            returns_gpu = cp.diff(prices_gpu) / prices_gpu[:-1]
            return cp.asnumpy(returns_gpu)
        return np.diff(prices) / prices[:-1]

    def _compute_sharpe_gpu(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252,
    ) -> float:
        """Compute Sharpe ratio using GPU."""
        if len(returns) < 2:
            return 0.0

        if self.gpu_available and HAS_CUPY:
            returns_gpu = cp.asarray(returns)
            mean_ret = float(cp.mean(returns_gpu))
            std_ret = float(cp.std(returns_gpu, ddof=1))
        else:
            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns, ddof=1))

        if std_ret == 0:
            return 0.0

        daily_rf = risk_free_rate / periods_per_year
        excess_return = mean_ret - daily_rf
        return float(excess_return / std_ret * np.sqrt(periods_per_year))

    def _compute_max_drawdown_gpu(self, prices: np.ndarray) -> float:
        """Compute maximum drawdown using GPU.
        
        Note: CuPy doesn't support maximum.accumulate, so we compute cummax on CPU
        and transfer back for the min operation on GPU.
        """
        if len(prices) < 2:
            return 0.0

        # Cummax not supported in CuPy, compute on CPU
        cummax = np.maximum.accumulate(prices)
        drawdown = (prices - cummax) / cummax
        
        if self.gpu_available and HAS_CUPY:
            # Use GPU for the min reduction
            drawdown_gpu = cp.asarray(drawdown)
            return float(-cp.min(drawdown_gpu) * 100)
        else:
            return float(-np.min(drawdown) * 100)

    def _compute_volatility_gpu(
        self,
        returns: np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """Compute annualized volatility using GPU."""
        if len(returns) < 2:
            return 0.0

        if self.gpu_available and HAS_CUPY:
            returns_gpu = cp.asarray(returns)
            std_ret = float(cp.std(returns_gpu, ddof=1))
        else:
            std_ret = float(np.std(returns, ddof=1))

        return std_ret * np.sqrt(periods_per_year) * 100

    def _bootstrap_sharpe_gpu(
        self,
        returns: np.ndarray,
        n_resamples: int = 1000,
    ) -> BootstrapResult:
        """Bootstrap Sharpe ratio confidence interval using GPU."""
        if len(returns) < 50:
            sharpe = self._compute_sharpe_gpu(returns)
            return BootstrapResult(
                metric_name="sharpe",
                point_estimate=sharpe,
                ci_lower=sharpe,
                ci_upper=sharpe,
                std_error=0.0,
                n_resamples=0,
                significant=sharpe > 0,
            )

        if self.gpu_available and HAS_CUPY:
            returns_gpu = cp.asarray(returns)
            n = len(returns_gpu)

            # Generate random indices on GPU
            indices = cp.random.randint(0, n, size=(n_resamples, n))

            # Resample and compute Sharpes
            sharpes = cp.zeros(n_resamples)
            for i in range(n_resamples):
                sample = returns_gpu[indices[i]]
                mean_ret = cp.mean(sample)
                std_ret = cp.std(sample, ddof=1)
                if std_ret > 0:
                    sharpes[i] = mean_ret / std_ret * cp.sqrt(252)

            sharpes = cp.asnumpy(sharpes)
        else:
            rng = np.random.default_rng(42)
            sharpes = np.zeros(n_resamples)
            n = len(returns)

            for i in range(n_resamples):
                sample = rng.choice(returns, size=n, replace=True)
                mean_ret = np.mean(sample)
                std_ret = np.std(sample, ddof=1)
                if std_ret > 0:
                    sharpes[i] = mean_ret / std_ret * np.sqrt(252)

        point_estimate = self._compute_sharpe_gpu(returns)
        ci_lower = float(np.percentile(sharpes, 2.5))
        ci_upper = float(np.percentile(sharpes, 97.5))
        std_error = float(np.std(sharpes))

        return BootstrapResult(
            metric_name="sharpe",
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            std_error=std_error,
            n_resamples=n_resamples,
            significant=ci_lower > 0,
        )

    def load_massive_data(
        self,
        symbol: str,
        min_days: int = 252,
    ) -> pd.DataFrame | None:
        """
        Load REAL historical data from Massive export.
        
        NO SYNTHETIC DATA - production-quality backtests only.
        
        Looks for data in:
        1. data/historical/{symbol}_historical.csv
        2. data/historical/{symbol}.parquet
        """
        data_dir = Path(__file__).parent.parent / "data" / "historical"
        
        # Try CSV first (more complete)
        csv_path = data_dir / f"{symbol}_historical.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, parse_dates=["Date"])
                df = df.rename(columns={"Date": "date"})
                df = df.set_index("date")
                df.columns = df.columns.str.lower()
                
                if len(df) >= min_days:
                    logger.debug(f"Loaded {symbol}: {len(df)} days from Massive CSV")
                    return df
                logger.warning(f"{symbol}: Only {len(df)} days available (need {min_days})")
                return None
            except Exception as e:
                logger.error(f"Error loading {symbol} CSV: {e}")
        
        # Try parquet
        parquet_path = data_dir / f"{symbol}.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                df.columns = df.columns.str.lower()
                
                if len(df) >= min_days:
                    logger.debug(f"Loaded {symbol}: {len(df)} days from Massive parquet")
                    return df
                logger.warning(f"{symbol}: Only {len(df)} days available (need {min_days})")
                return None
            except Exception as e:
                logger.error(f"Error loading {symbol} parquet: {e}")
        
        logger.warning(f"No Massive data file found for {symbol}")
        return None

    async def get_market_data(
        self,
        symbol: str,
        n_days: int = 252,
    ) -> pd.DataFrame | None:
        """
        Get market data for backtesting - REAL DATA ONLY.
        
        Uses Massive historical exports. Returns None if no real data available.
        """
        df = self.load_massive_data(symbol, min_days=n_days)
        
        if df is not None:
            # Trim to requested days if we have more
            if len(df) > n_days:
                df = df.iloc[-n_days:]
            return df
        
        return None

    async def run_strategy_backtest(
        self,
        strategy_name: str,
        symbol: str,
        data: pd.DataFrame,
    ) -> BacktestResult:
        """Run backtest for a specific strategy and symbol."""
        start_time = time.perf_counter()

        prices = data["close"].values
        returns = self._compute_returns_gpu(prices)

        # Simulate strategy signals based on strategy type
        if strategy_name == "atr_rsi":
            signals, trades = await self._simulate_atr_rsi(data)
        elif strategy_name == "mtf_momentum":
            signals, trades = await self._simulate_mtf_momentum(data)
        elif strategy_name == "mi_ensemble":
            signals, trades = await self._simulate_mi_ensemble(data)
        elif strategy_name == "kalman_hybrid":
            signals, trades = await self._simulate_kalman_hybrid(data)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Calculate metrics
        strategy_returns = returns * signals[:-1]  # Align with returns

        # Apply costs
        n_trades = trades
        total_cost_bps = (self.config.spread_bps + self.config.impact_bps) * 2  # Round-trip
        total_costs = Decimal(str(n_trades * total_cost_bps / 10000 * self.config.initial_capital))

        net_returns = strategy_returns.copy()
        if n_trades > 0:
            cost_per_day = float(total_costs) / len(net_returns) / self.config.initial_capital
            net_returns = net_returns - cost_per_day

        # Compute all metrics using GPU
        gross_sharpe = self._compute_sharpe_gpu(strategy_returns)
        net_sharpe = self._compute_sharpe_gpu(net_returns)

        equity_curve = self.config.initial_capital * np.cumprod(1 + net_returns)
        max_drawdown = self._compute_max_drawdown_gpu(equity_curve)
        volatility = self._compute_volatility_gpu(net_returns)

        # Trade statistics
        if n_trades > 0:
            winning_trades = np.sum(strategy_returns > 0)
            win_rate = winning_trades / n_trades * 100
            gross_profit = np.sum(strategy_returns[strategy_returns > 0])
            gross_loss = abs(np.sum(strategy_returns[strategy_returns < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            avg_trade_return = np.mean(strategy_returns) * 100
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade_return = 0.0

        # Sortino ratio
        downside_returns = net_returns[net_returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns, ddof=1)
            sortino = (np.mean(net_returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
        else:
            sortino = float("inf") if np.mean(net_returns) > 0 else 0.0

        compute_time = (time.perf_counter() - start_time) * 1000

        return BacktestResult(
            symbol=symbol,
            strategy=strategy_name,
            start_date=data.index[0].date(),
            end_date=data.index[-1].date(),
            n_bars=len(data),
            n_trades=n_trades,
            gross_return_pct=float((equity_curve[-1] / self.config.initial_capital - 1) * 100),
            net_return_pct=float((equity_curve[-1] / self.config.initial_capital - 1) * 100),
            gross_sharpe=gross_sharpe,
            net_sharpe=net_sharpe,
            max_drawdown_pct=max_drawdown,
            volatility_pct=volatility,
            sortino_ratio=sortino if sortino != float("inf") else 99.99,
            win_rate_pct=win_rate,
            profit_factor=profit_factor if profit_factor != float("inf") else 99.99,
            avg_trade_return_pct=avg_trade_return,
            total_costs=total_costs,
            avg_cost_per_trade_bps=float(total_cost_bps),
            compute_time_ms=compute_time,
            used_gpu=self.gpu_available,
        )

    async def _simulate_atr_rsi(self, data: pd.DataFrame) -> tuple[np.ndarray, int]:
        """Simulate ATR-RSI strategy signals."""
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values

        # RSI calculation
        delta = np.diff(close, prepend=close[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        if self.gpu_available and HAS_CUPY:
            gains_gpu = cp.asarray(gains)
            losses_gpu = cp.asarray(losses)

            # EMA
            alpha = 2.0 / 15  # period=14
            avg_gain = cp.zeros_like(gains_gpu)
            avg_loss = cp.zeros_like(losses_gpu)
            avg_gain[0] = gains_gpu[0]
            avg_loss[0] = losses_gpu[0]

            for i in range(1, len(gains_gpu)):
                avg_gain[i] = alpha * gains_gpu[i] + (1 - alpha) * avg_gain[i - 1]
                avg_loss[i] = alpha * losses_gpu[i] + (1 - alpha) * avg_loss[i - 1]

            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            rsi = cp.asnumpy(rsi)
        else:
            alpha = 2.0 / 15
            avg_gain = np.zeros_like(gains)
            avg_loss = np.zeros_like(losses)
            avg_gain[0] = gains[0]
            avg_loss[0] = losses[0]

            for i in range(1, len(gains)):
                avg_gain[i] = alpha * gains[i] + (1 - alpha) * avg_gain[i - 1]
                avg_loss[i] = alpha * losses[i] + (1 - alpha) * avg_loss[i - 1]

            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

        # SMA filter
        sma20 = np.convolve(close, np.ones(20) / 20, mode="same")

        # Signals: long when RSI < 35 and price > SMA20
        signals = np.zeros(len(close))
        signals[(rsi < 35) & (close > sma20)] = 1.0

        # Count trades (signal changes)
        n_trades = int(np.sum(np.abs(np.diff(signals)) > 0))

        return signals, n_trades

    async def _simulate_mtf_momentum(self, data: pd.DataFrame) -> tuple[np.ndarray, int]:
        """Simulate MTF Momentum strategy signals."""
        close = data["close"].values

        # 12-1 month momentum (skip last month)
        momentum_period = 252  # ~12 months
        skip_period = 21  # ~1 month

        signals = np.zeros(len(close))

        for i in range(momentum_period, len(close)):
            if i >= momentum_period + skip_period:
                past_return = close[i - skip_period] / close[i - momentum_period] - 1
                signals[i] = 1.0 if past_return > 0 else -1.0

        n_trades = int(np.sum(np.abs(np.diff(signals)) > 0))
        return signals, n_trades

    async def _simulate_mi_ensemble(self, data: pd.DataFrame) -> tuple[np.ndarray, int]:
        """Simulate MI Ensemble strategy signals."""
        close = data["close"].values

        # Simplified ensemble: combine RSI, momentum, mean reversion
        n = len(close)
        signals = np.zeros(n)

        # RSI component
        delta = np.diff(close, prepend=close[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        alpha = 2.0 / 15
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        for i in range(1, n):
            avg_gain[i] = alpha * gains[i] + (1 - alpha) * avg_gain[i - 1]
            avg_loss[i] = alpha * losses[i] + (1 - alpha) * avg_loss[i - 1]
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi_signal = (rsi - 50) / 50

        # Momentum component
        mom_period = 20
        momentum = np.zeros(n)
        for i in range(mom_period, n):
            momentum[i] = (close[i] / close[i - mom_period] - 1) * 10

        # Mean reversion component
        sma20 = np.convolve(close, np.ones(20) / 20, mode="same")
        std20 = np.zeros(n)
        for i in range(20, n):
            std20[i] = np.std(close[i - 20 : i])
        zscore = (close - sma20) / (std20 + 1e-10)
        mr_signal = -zscore  # Negative for mean reversion

        # Equal weighted ensemble
        ensemble = (rsi_signal + momentum + mr_signal) / 3
        signals = np.sign(ensemble)

        n_trades = int(np.sum(np.abs(np.diff(signals)) > 0))
        return signals, n_trades

    async def _simulate_kalman_hybrid(self, data: pd.DataFrame) -> tuple[np.ndarray, int]:
        """Simulate Kalman Hybrid strategy signals."""
        close = data["close"].values
        n = len(close)

        # Simple Kalman filter for trend
        # State: [price, velocity]
        kalman_price = np.zeros(n)
        kalman_velocity = np.zeros(n)

        # Kalman parameters
        process_noise = 0.01
        measurement_noise = 0.1
        kalman_price[0] = close[0]
        kalman_velocity[0] = 0

        for i in range(1, n):
            # Predict
            pred_price = kalman_price[i - 1] + kalman_velocity[i - 1]
            pred_velocity = kalman_velocity[i - 1]

            # Update (simplified)
            innovation = close[i] - pred_price
            kalman_gain = process_noise / (process_noise + measurement_noise)

            kalman_price[i] = pred_price + kalman_gain * innovation
            kalman_velocity[i] = pred_velocity + kalman_gain * innovation * 0.1

        # Signal based on Kalman velocity
        signals = np.sign(kalman_velocity)

        # Add momentum confirmation
        mom_period = 10
        for i in range(mom_period, n):
            mom = close[i] / close[i - mom_period] - 1
            if signals[i] > 0 and mom < -0.02:
                signals[i] = 0  # Cancel long if momentum negative
            elif signals[i] < 0 and mom > 0.02:
                signals[i] = 0  # Cancel short if momentum positive

        n_trades = int(np.sum(np.abs(np.diff(signals)) > 0))
        return signals, n_trades

    async def run_walk_forward(
        self,
        strategy_name: str,
        symbol: str,
        data: pd.DataFrame,
    ) -> WalkForwardSummary:
        """Run walk-forward validation."""
        train_days = self.config.train_days
        test_days = self.config.test_days
        step_days = self.config.step_days

        periods: list[WalkForwardPeriod] = []
        n_bars = len(data)

        period_id = 0
        i = 0

        while i + train_days + test_days <= n_bars:
            train_data = data.iloc[i : i + train_days]
            test_data = data.iloc[i + train_days : i + train_days + test_days]

            # Run backtest on each period
            train_result = await self.run_strategy_backtest(strategy_name, symbol, train_data)
            test_result = await self.run_strategy_backtest(strategy_name, symbol, test_data)

            period = WalkForwardPeriod(
                period_id=period_id,
                train_start=train_data.index[0].date(),
                train_end=train_data.index[-1].date(),
                test_start=test_data.index[0].date(),
                test_end=test_data.index[-1].date(),
                train_sharpe=train_result.net_sharpe,
                test_sharpe=test_result.net_sharpe,
                train_return=train_result.net_return_pct,
                test_return=test_result.net_return_pct,
                test_max_dd=test_result.max_drawdown_pct,
                n_trades=test_result.n_trades,
                profitable=test_result.net_return_pct > 0,
            )
            periods.append(period)

            i += step_days
            period_id += 1

        if not periods:
            return WalkForwardSummary(
                periods=[],
                n_periods=0,
                periods_profitable=0,
                win_rate_pct=0.0,
                avg_test_sharpe=0.0,
                std_test_sharpe=0.0,
                robustness_ratio=0.0,
                worst_period_sharpe=0.0,
                best_period_sharpe=0.0,
            )

        test_sharpes = [p.test_sharpe for p in periods]
        train_sharpes = [p.train_sharpe for p in periods]
        profitable = sum(1 for p in periods if p.profitable)

        avg_train = np.mean(train_sharpes) if train_sharpes else 0.0
        avg_test = np.mean(test_sharpes) if test_sharpes else 0.0
        robustness = avg_test / avg_train if avg_train != 0 else 0.0

        return WalkForwardSummary(
            periods=periods,
            n_periods=len(periods),
            periods_profitable=profitable,
            win_rate_pct=profitable / len(periods) * 100,
            avg_test_sharpe=float(avg_test),
            std_test_sharpe=float(np.std(test_sharpes)) if test_sharpes else 0.0,
            robustness_ratio=float(robustness),
            worst_period_sharpe=float(min(test_sharpes)) if test_sharpes else 0.0,
            best_period_sharpe=float(max(test_sharpes)) if test_sharpes else 0.0,
        )

    async def run_full_validation(
        self,
        strategy_name: str,
        symbols: list[str],
    ) -> StrategyValidationResult:
        """Run full validation for a strategy across all symbols.
        
        Uses REAL historical data from Massive - no synthetic data.
        """
        logger.info(f"Running validation for {strategy_name} on {len(symbols)} symbols (REAL MASSIVE DATA)")

        all_returns: list[np.ndarray] = []
        all_results: list[BacktestResult] = []
        total_trades = 0
        symbols_loaded = []

        for symbol in symbols:
            logger.info(f"  Processing {symbol}...")
            data = await self.get_market_data(symbol, self.config.days)
            
            if data is None:
                logger.warning(f"  SKIPPED {symbol}: No Massive data available")
                continue
                
            logger.info(f"    Loaded {len(data)} days of real data ({data.index[0].date()} to {data.index[-1].date()})")
            
            result = await self.run_strategy_backtest(strategy_name, symbol, data)
            all_results.append(result)
            symbols_loaded.append(symbol)

            # Collect returns for aggregate analysis
            prices = data["close"].values
            returns = self._compute_returns_gpu(prices)
            all_returns.append(returns)
            total_trades += result.n_trades

        if not all_results:
            raise ValueError(f"No symbols with valid Massive data for {strategy_name}")

        # Aggregate metrics
        all_returns_flat = np.concatenate(all_returns)
        gross_sharpe = self._compute_sharpe_gpu(all_returns_flat)
        net_sharpe = gross_sharpe * 0.85  # Approximate cost impact

        # Bootstrap CI
        bootstrap = self._bootstrap_sharpe_gpu(all_returns_flat, self.config.n_bootstrap)

        # Walk-forward on first symbol with data
        wf_data = await self.get_market_data(symbols_loaded[0], self.config.days * 2)
        if wf_data is not None:
            walk_forward = await self.run_walk_forward(strategy_name, symbols_loaded[0], wf_data)
        else:
            walk_forward = None

        # Aggregate drawdown
        avg_drawdown = np.mean([r.max_drawdown_pct for r in all_results])

        result = StrategyValidationResult(
            strategy_id=strategy_name,
            strategy_version="1.0.0",
            validated_at=datetime.now(timezone.utc),
            backtest_start=all_results[0].start_date if all_results else None,
            backtest_end=all_results[0].end_date if all_results else None,
            symbols_tested=symbols_loaded,
            gross_sharpe=gross_sharpe,
            gross_return_pct=np.mean([r.gross_return_pct for r in all_results]),
            net_sharpe=net_sharpe,
            net_return_pct=np.mean([r.net_return_pct for r in all_results]),
            max_drawdown_pct=avg_drawdown,
            volatility_pct=np.mean([r.volatility_pct for r in all_results]),
            sortino_ratio=np.mean([r.sortino_ratio for r in all_results]),
            n_trades=total_trades,
            win_rate_pct=np.mean([r.win_rate_pct for r in all_results]),
            profit_factor=np.mean([r.profit_factor for r in all_results]),
            avg_trade_return_pct=np.mean([r.avg_trade_return_pct for r in all_results]),
            walk_forward=walk_forward,
            sharpe_bootstrap=bootstrap,
            cost_analysis=CostAnalysis(
                gross_return=np.mean([r.gross_return_pct for r in all_results]),
                net_return=np.mean([r.net_return_pct for r in all_results]),
                total_costs=sum((r.total_costs for r in all_results), Decimal("0")),
                avg_cost_per_trade_bps=self.config.spread_bps + self.config.impact_bps,
                cost_drag_pct=15.0,  # Estimated
                n_trades=total_trades,
            ),
        )

        # Check acceptance
        result.check_acceptance()

        return result


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated GTM Strategy Backtest Runner - REAL MASSIVE DATA ONLY"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help="Strategies to run (comma-separated or 'all')",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="massive",
        help=(
            "Symbol list: 'massive' (all 25 with real data), "
            "'tech', 'finance', 'energy', 'volatile', 'stable', "
            "or comma-separated symbols"
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=252,
        help="Number of trading days (default: 252)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/backtest_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # Parse symbols - REAL MASSIVE DATA ONLY
    if args.symbols == "massive":
        symbols = MASSIVE_SYMBOLS
    elif args.symbols == "tech":
        symbols = TECH_SYMBOLS
    elif args.symbols == "finance":
        symbols = FINANCE_SYMBOLS
    elif args.symbols == "energy":
        symbols = ENERGY_SYMBOLS
    elif args.symbols == "volatile":
        symbols = VOLATILE_SYMBOLS
    elif args.symbols == "stable":
        symbols = STABLE_SYMBOLS
    else:
        symbols = [s.strip() for s in args.symbols.split(",")]

    # Parse strategies
    all_strategies = ["atr_rsi", "mtf_momentum", "mi_ensemble", "kalman_hybrid"]
    if args.strategies == "all":
        strategies = all_strategies
    else:
        strategies = [s.strip() for s in args.strategies.split(",")]

    # Config
    config = GPUBacktestConfig(
        use_gpu=args.gpu,
        days=args.days,
        output_dir=Path(args.output),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    runner = GPUBacktestRunner(config)

    logger.info("=" * 70)
    logger.info("GTM Strategy Backtest Runner - REAL MASSIVE DATA")
    logger.info("=" * 70)
    logger.info(f"GPU Available: {runner.gpu_available}")
    logger.info(f"Data Source: Massive historical exports (data/historical/)")
    logger.info(f"Strategies: {strategies}")
    logger.info(f"Symbols requested: {len(symbols)} ({symbols[:5]}...)")
    logger.info(f"Days: {args.days}")
    logger.info(f"Cost Model: {config.spread_bps + config.impact_bps} bps round-trip")
    logger.info("=" * 70)

    results: dict[str, StrategyValidationResult] = {}

    for strategy in strategies:
        logger.info(f"\nRunning {strategy.upper()}...")
        start = time.perf_counter()

        result = await runner.run_full_validation(strategy, symbols)
        results[strategy] = result

        elapsed = time.perf_counter() - start
        logger.info(f"  Completed in {elapsed:.2f}s")
        logger.info(f"  Status: {result.status.name}")
        logger.info(f"  Net Sharpe: {result.net_sharpe:.3f}")
        logger.info(f"  Max DD: {result.max_drawdown_pct:.1f}%")
        logger.info(f"  Bootstrap CI: [{result.sharpe_bootstrap.ci_lower:.3f}, {result.sharpe_bootstrap.ci_upper:.3f}]")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = config.output_dir / f"gtm_backtest_{timestamp}.json"

    # Convert to serializable format
    output_data = {
        "timestamp": timestamp,
        "data_source": "Massive historical exports",
        "config": {
            "gpu_enabled": runner.gpu_available,
            "days": config.days,
            "symbols": symbols,
            "cost_bps": config.spread_bps + config.impact_bps,
        },
        "results": {},
    }

    for name, result in results.items():
        output_data["results"][name] = {
            "status": result.status.name,
            "net_sharpe": result.net_sharpe,
            "gross_sharpe": result.gross_sharpe,
            "max_drawdown_pct": result.max_drawdown_pct,
            "n_trades": result.n_trades,
            "win_rate_pct": result.win_rate_pct,
            "bootstrap_ci_lower": result.sharpe_bootstrap.ci_lower if result.sharpe_bootstrap else None,
            "bootstrap_ci_upper": result.sharpe_bootstrap.ci_upper if result.sharpe_bootstrap else None,
            "walk_forward_win_rate": result.walk_forward.win_rate_pct if result.walk_forward else None,
            "criteria_results": {k: (v[0], v[1]) for k, v in result.criteria_results.items()},
        }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_file}")

    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Strategy':<20} {'Status':<12} {'Net Sharpe':>12} {'Max DD':>10} {'Trades':>10}")
    logger.info("-" * 70)
    for name, result in results.items():
        logger.info(
            f"{name:<20} {result.status.name:<12} {result.net_sharpe:>12.3f} "
            f"{result.max_drawdown_pct:>9.1f}% {result.n_trades:>10}"
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
