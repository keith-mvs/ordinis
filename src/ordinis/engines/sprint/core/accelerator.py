"""
GPU-Accelerated Backtest Engine.

Uses NVIDIA GPU for numerical acceleration via CuPy and Numba.
Falls back to CPU when GPU is not available.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from typing import Any, Callable

import numpy as np

# Try GPU acceleration libraries
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

try:
    from numba import cuda, jit, prange

    HAS_NUMBA = True
except ImportError:
    cuda = None
    jit = None
    prange = range
    HAS_NUMBA = False

logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """GPU acceleration configuration."""

    use_gpu: bool = True
    device_id: int = 0
    batch_size: int = 1024
    parallel_workers: int = 4


class GPUBacktestEngine:
    """GPU-accelerated backtest engine."""

    def __init__(self, config: GPUConfig | None = None):
        self.config = config or GPUConfig()
        self._executor = None

        # Check GPU availability
        self.gpu_available = self._check_gpu()
        if self.gpu_available:
            logger.info(f"GPU acceleration enabled (CuPy: {HAS_CUPY}, Numba: {HAS_NUMBA})")
        else:
            logger.warning("GPU not available, using CPU fallback")

    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        if not self.config.use_gpu:
            return False

        if HAS_CUPY:
            try:
                cp.cuda.Device(self.config.device_id).use()
                return True
            except Exception:
                pass

        if HAS_NUMBA:
            try:
                return cuda.is_available()
            except Exception:
                pass

        return False

    async def initialize(self) -> None:
        """Initialize engine resources."""
        self._executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)

    async def close(self) -> None:
        """Close resources."""
        if self._executor:
            self._executor.shutdown(wait=False)

    # =========================================================================
    # GPU-Accelerated Computations
    # =========================================================================

    def compute_returns_gpu(self, prices: np.ndarray) -> np.ndarray:
        """Compute returns using GPU if available."""
        if self.gpu_available and HAS_CUPY:
            prices_gpu = cp.asarray(prices)
            returns_gpu = cp.diff(prices_gpu) / prices_gpu[:-1]
            return cp.asnumpy(returns_gpu)
        return np.diff(prices) / prices[:-1]

    def compute_volatility_gpu(
        self,
        returns: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """Compute rolling volatility using GPU."""
        if self.gpu_available and HAS_CUPY:
            returns_gpu = cp.asarray(returns)
            n = len(returns_gpu)
            result = cp.zeros(n)
            for i in range(window, n):
                result[i] = cp.std(returns_gpu[i - window : i])
            return cp.asnumpy(result)

        # CPU fallback
        result = np.zeros(len(returns))
        for i in range(window, len(returns)):
            result[i] = np.std(returns[i - window : i])
        return result

    def compute_ewma_volatility_gpu(
        self,
        returns: np.ndarray,
        span: int = 60,
    ) -> np.ndarray:
        """Compute EWMA volatility using GPU."""
        alpha = 2.0 / (span + 1)

        if self.gpu_available and HAS_CUPY:
            returns_gpu = cp.asarray(returns)
            sq_returns = returns_gpu**2

            n = len(sq_returns)
            ewma_var = cp.zeros(n)
            ewma_var[0] = sq_returns[0]

            for i in range(1, n):
                ewma_var[i] = alpha * sq_returns[i] + (1 - alpha) * ewma_var[i - 1]

            return cp.asnumpy(cp.sqrt(ewma_var))

        # CPU fallback
        sq_returns = returns**2
        ewma_var = np.zeros(len(returns))
        ewma_var[0] = sq_returns[0]
        for i in range(1, len(sq_returns)):
            ewma_var[i] = alpha * sq_returns[i] + (1 - alpha) * ewma_var[i - 1]
        return np.sqrt(ewma_var)

    def compute_covariance_matrix_gpu(
        self,
        returns_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute covariance matrix using GPU."""
        if self.gpu_available and HAS_CUPY:
            returns_gpu = cp.asarray(returns_matrix)
            cov_gpu = cp.cov(returns_gpu, rowvar=False)
            return cp.asnumpy(cov_gpu)
        return np.cov(returns_matrix, rowvar=False)

    def compute_correlation_matrix_gpu(
        self,
        returns_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute correlation matrix using GPU."""
        if self.gpu_available and HAS_CUPY:
            returns_gpu = cp.asarray(returns_matrix)
            corr_gpu = cp.corrcoef(returns_gpu, rowvar=False)
            return cp.asnumpy(corr_gpu)
        return np.corrcoef(returns_matrix, rowvar=False)

    def batch_backtest(
        self,
        backtest_fn: Callable,
        param_sets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run multiple backtests in parallel using thread pool."""
        if self._executor is None:
            return [backtest_fn(params) for params in param_sets]

        futures = [self._executor.submit(backtest_fn, params) for params in param_sets]
        return [f.result() for f in futures]

    def run_backtest(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        initial_capital: float = 100_000.0,
        position_size: float = 0.95,
        commission: float = 0.001,
        always_invested: bool = True,
    ) -> dict[str, Any]:
        """
        Run a vectorized backtest on price data with signals.

        Args:
            prices: Price series (close prices)
            signals: Signal series (-1, 0, 1) for short/flat/long
            initial_capital: Starting capital
            position_size: Fraction of capital per position
            commission: Commission per trade (fraction)
            always_invested: If True, hold long when signal is 0 (avoid cash drag)

        Returns:
            Dictionary with equity_curve, trades, and metrics
        """
        n = len(prices)
        if len(signals) != n:
            # Align signals to prices
            signals = np.zeros(n)

        # If always_invested, convert 0 signals to 1 (hold long as default)
        if always_invested:
            signals = np.where(signals == 0, 1, signals)

        # Compute returns
        returns = np.zeros(n)
        returns[1:] = np.diff(prices) / prices[:-1]

        # Position tracking
        position = 0
        equity = initial_capital
        equity_curve = np.zeros(n)
        equity_curve[0] = equity
        trades = []

        entry_price = 0.0
        entry_idx = 0

        for i in range(1, n):
            signal = int(signals[i])

            # Check for position change
            if signal != position:
                # Close existing position
                if position != 0:
                    exit_price = prices[i]
                    if position == 1:
                        pnl = (exit_price / entry_price - 1) * equity * position_size
                    else:  # position == -1
                        pnl = (entry_price / exit_price - 1) * equity * position_size

                    # Apply commission
                    pnl -= abs(pnl) * commission
                    equity += pnl

                    trades.append(
                        {
                            "entry_idx": entry_idx,
                            "exit_idx": i,
                            "direction": position,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl": pnl,
                        }
                    )

                # Open new position
                if signal != 0:
                    entry_price = prices[i]
                    entry_idx = i
                    # Apply entry commission
                    equity -= abs(equity * position_size * commission)

                position = signal

            # Update equity with P&L from position
            if position != 0:
                daily_pnl = position * returns[i] * equity * position_size
                equity += daily_pnl

            equity_curve[i] = equity

        # Close any open position at the end
        if position != 0:
            exit_price = prices[-1]
            if position == 1:
                pnl = (exit_price / entry_price - 1) * initial_capital * position_size
            else:
                pnl = (entry_price / exit_price - 1) * initial_capital * position_size
            pnl -= abs(pnl) * commission
            trades.append(
                {
                    "entry_idx": entry_idx,
                    "exit_idx": n - 1,
                    "direction": position,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                }
            )

        return {
            "equity_curve": equity_curve,
            "trades": trades,
            "final_equity": equity,
        }


# Numba JIT-compiled functions for CPU acceleration
if HAS_NUMBA:

    @jit(nopython=True, parallel=True, cache=True)
    def compute_rolling_sharpe_numba(
        returns: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Compute rolling Sharpe ratio using Numba JIT."""
        n = len(returns)
        result = np.zeros(n)
        sqrt_window = np.sqrt(window)

        for i in prange(window, n):
            window_returns = returns[i - window : i]
            mean_ret = np.mean(window_returns)
            std_ret = np.std(window_returns)
            if std_ret > 1e-10:
                result[i] = mean_ret / std_ret * sqrt_window

        return result

    @jit(nopython=True, cache=True)
    def compute_max_drawdown_numba(equity_curve: np.ndarray) -> float:
        """Compute maximum drawdown using Numba JIT."""
        n = len(equity_curve)
        peak = equity_curve[0]
        max_dd = 0.0

        for i in range(1, n):
            peak = max(equity_curve[i], peak)
            dd = (peak - equity_curve[i]) / peak
            max_dd = max(dd, max_dd)

        return max_dd
else:

    def compute_rolling_sharpe_numba(returns: np.ndarray, window: int) -> np.ndarray:
        """CPU fallback for rolling Sharpe."""
        n = len(returns)
        result = np.zeros(n)
        sqrt_window = np.sqrt(window)
        for i in range(window, n):
            window_returns = returns[i - window : i]
            mean_ret = np.mean(window_returns)
            std_ret = np.std(window_returns)
            if std_ret > 1e-10:
                result[i] = mean_ret / std_ret * sqrt_window
        return result

    def compute_max_drawdown_numba(equity_curve: np.ndarray) -> float:
        """CPU fallback for max drawdown."""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        return np.max(drawdown)
