"""
GPU-Accelerated Backtest Engine with AI Optimization.

Uses NVIDIA GPU for numerical acceleration and Mistral/NVIDIA models
for intelligent parameter optimization.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

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

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """GPU acceleration configuration."""

    use_gpu: bool = True
    device_id: int = 0
    batch_size: int = 1024
    parallel_workers: int = 4
    use_ai_optimization: bool = True
    ai_model: str = "mistral-small-2503"  # Fast Mistral model
    ai_provider: str = "github"  # Use GitHub Models (free tier)


@dataclass
class OptimizationResult:
    """Result from AI-guided optimization."""

    best_params: dict[str, Any]
    best_score: float
    all_results: list[dict[str, Any]]
    optimization_time: float
    ai_suggestions: list[str] = field(default_factory=list)


class GPUBacktestEngine:
    """GPU-accelerated backtest engine with AI optimization."""

    def __init__(self, config: GPUConfig | None = None):
        self.config = config or GPUConfig()
        self._helix = None
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
        """Initialize engine with AI provider."""
        if self.config.use_ai_optimization:
            try:
                from ordinis.ai.helix import Helix, HelixConfig

                helix_config = HelixConfig(
                    github_token=os.getenv("GITHUB_TOKEN"),
                    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
                    nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
                )
                self._helix = Helix(helix_config)
                await self._helix.initialize()
                logger.info("AI optimization enabled via Helix")
            except Exception as e:
                logger.warning(f"AI optimization unavailable: {e}")
                self._helix = None

        # Thread pool for parallel CPU work
        self._executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)

    async def close(self) -> None:
        """Close resources."""
        if self._executor:
            self._executor.shutdown(wait=False)
        if self._helix:
            await self._helix.close()

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
            # Rolling std using convolution trick
            n = len(returns_gpu)
            result = cp.zeros(n)
            for i in range(window, n):
                result[i] = cp.std(returns_gpu[i - window : i])
            return cp.asnumpy(result)

        # CPU fallback with numpy
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

            # EWMA via scan
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

    # =========================================================================
    # Parallel Backtest Execution
    # =========================================================================

    async def run_parallel_backtests(
        self,
        backtest_fn: Callable,
        param_grid: list[dict[str, Any]],
        data: pd.DataFrame,
        symbol: str,
    ) -> list[dict[str, Any]]:
        """Run multiple backtests in parallel."""
        loop = asyncio.get_event_loop()

        async def run_one(params: dict[str, Any]) -> dict[str, Any]:
            result = await loop.run_in_executor(
                self._executor,
                backtest_fn,
                data,
                symbol,
                params,
            )
            result["params"] = params
            return result

        # Run all in parallel
        tasks = [run_one(p) for p in param_grid]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Backtest failed: {r}")
            else:
                valid_results.append(r)

        return valid_results

    # =========================================================================
    # AI-Guided Optimization
    # =========================================================================

    async def get_optimization_suggestions(
        self,
        strategy_name: str,
        current_params: dict[str, Any],
        current_metrics: dict[str, float],
        param_ranges: dict[str, tuple[float, float]],
    ) -> list[dict[str, Any]]:
        """Use AI to suggest parameter improvements."""
        if not self._helix:
            return self._generate_grid_params(param_ranges)

        prompt = f"""You are a quantitative trading strategy optimizer.

Strategy: {strategy_name}
Current Parameters: {current_params}
Current Metrics:
- Sharpe Ratio: {current_metrics.get('sharpe', 0):.3f}
- Win Rate: {current_metrics.get('win_rate', 0):.1f}%
- Total Return: {current_metrics.get('total_return', 0):.2f}%
- Max Drawdown: {current_metrics.get('max_drawdown', 0):.2f}%
- Profit Factor: {current_metrics.get('profit_factor', 0):.2f}

Parameter Ranges:
{param_ranges}

Based on the current performance, suggest 5 different parameter combinations to test.
Focus on improving Sharpe ratio while controlling drawdown.

Return ONLY a JSON array of parameter dictionaries, no explanation:
[{{"param1": value1, "param2": value2, ...}}, ...]
"""
        try:
            from ordinis.ai.helix.models import ChatMessage

            response = await self._helix.chat(
                messages=[ChatMessage(role="user", content=prompt)],
                model_id=self.config.ai_model,
            )

            # Parse JSON from response
            import json

            content = response.content.strip()
            # Extract JSON array
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                suggestions = json.loads(content[start:end])
                return suggestions
        except Exception as e:
            logger.warning(f"AI suggestion failed: {e}")

        return self._generate_grid_params(param_ranges)

    def _generate_grid_params(
        self,
        param_ranges: dict[str, tuple[float, float]],
        n_samples: int = 10,
    ) -> list[dict[str, Any]]:
        """Generate parameter grid for optimization."""
        import itertools

        param_values = {}
        for name, (low, high) in param_ranges.items():
            param_values[name] = np.linspace(low, high, 3).tolist()

        # Generate combinations
        keys = list(param_values.keys())
        values = list(param_values.values())
        combinations = list(itertools.product(*values))

        return [dict(zip(keys, combo)) for combo in combinations[:n_samples]]

    async def optimize_strategy(
        self,
        strategy_name: str,
        backtest_fn: Callable,
        data: pd.DataFrame,
        symbol: str,
        initial_params: dict[str, Any],
        param_ranges: dict[str, tuple[float, float]],
        n_iterations: int = 3,
        metric: str = "sharpe",
    ) -> OptimizationResult:
        """Run AI-guided optimization loop."""
        start_time = time.perf_counter()

        all_results = []
        best_params = initial_params.copy()
        best_score = float("-inf")
        ai_suggestions = []

        # Initial backtest
        initial_result = backtest_fn(data, symbol, initial_params)
        all_results.append({"params": initial_params, **initial_result})
        best_score = initial_result.get(metric, 0)

        logger.info(f"Initial {metric}: {best_score:.4f}")

        for iteration in range(n_iterations):
            logger.info(f"Optimization iteration {iteration + 1}/{n_iterations}")

            # Get current metrics
            current_metrics = {
                "sharpe": initial_result.get("sharpe", 0),
                "win_rate": initial_result.get("win_rate", 0),
                "total_return": initial_result.get("total_return", 0),
                "max_drawdown": initial_result.get("max_drawdown", 0),
                "profit_factor": initial_result.get("profit_factor", 0),
            }

            # Get suggestions from AI
            suggestions = await self.get_optimization_suggestions(
                strategy_name,
                best_params,
                current_metrics,
                param_ranges,
            )

            if suggestions:
                ai_suggestions.append(f"Iteration {iteration + 1}: {len(suggestions)} params")

            # Run parallel backtests
            results = await self.run_parallel_backtests(
                backtest_fn,
                suggestions,
                data,
                symbol,
            )

            all_results.extend(results)

            # Find best
            for r in results:
                score = r.get(metric, 0)
                if score > best_score:
                    best_score = score
                    best_params = r.get("params", best_params)
                    logger.info(f"  New best {metric}: {score:.4f}")

        optimization_time = time.perf_counter() - start_time

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_time=optimization_time,
            ai_suggestions=ai_suggestions,
        )


# =============================================================================
# Numba JIT-compiled functions for maximum speed
# =============================================================================

if HAS_NUMBA:

    @jit(nopython=True, parallel=True)
    def compute_rolling_sharpe_numba(
        returns: np.ndarray,
        window: int,
        risk_free: float = 0.0,
    ) -> np.ndarray:
        """Compute rolling Sharpe ratio with Numba acceleration."""
        n = len(returns)
        result = np.zeros(n)

        for i in prange(window, n):
            window_returns = returns[i - window : i]
            mean_ret = np.mean(window_returns) - risk_free / 252
            std_ret = np.std(window_returns)
            if std_ret > 0:
                result[i] = mean_ret / std_ret * np.sqrt(252)

        return result

    @jit(nopython=True)
    def compute_max_drawdown_numba(equity: np.ndarray) -> float:
        """Compute maximum drawdown with Numba acceleration."""
        peak = equity[0]
        max_dd = 0.0

        for i in range(len(equity)):
            peak = max(equity[i], peak)
            dd = (peak - equity[i]) / peak
            max_dd = max(dd, max_dd)

        return max_dd

    @jit(nopython=True, parallel=True)
    def vectorized_backtest_signals(
        prices: np.ndarray,
        vol_short: np.ndarray,
        vol_long: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Generate trading signals with Numba acceleration."""
        n = len(prices)
        signals = np.zeros(n)

        for i in prange(1, n):
            if vol_long[i] > 0:
                ratio = vol_short[i] / vol_long[i]
                if ratio > threshold:
                    # Direction from momentum
                    if prices[i] > prices[i - 1]:
                        signals[i] = 1.0
                    else:
                        signals[i] = -1.0

        return signals

else:
    # CPU fallbacks without Numba
    def compute_rolling_sharpe_numba(returns, window, risk_free=0.0):
        n = len(returns)
        result = np.zeros(n)
        for i in range(window, n):
            window_returns = returns[i - window : i]
            mean_ret = np.mean(window_returns) - risk_free / 252
            std_ret = np.std(window_returns)
            if std_ret > 0:
                result[i] = mean_ret / std_ret * np.sqrt(252)
        return result

    def compute_max_drawdown_numba(equity):
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            peak = max(val, peak)
            dd = (peak - val) / peak
            max_dd = max(dd, max_dd)
        return max_dd

    def vectorized_backtest_signals(prices, vol_short, vol_long, threshold):
        n = len(prices)
        signals = np.zeros(n)
        for i in range(1, n):
            if vol_long[i] > 0:
                ratio = vol_short[i] / vol_long[i]
                if ratio > threshold:
                    if prices[i] > prices[i - 1]:
                        signals[i] = 1.0
                    else:
                        signals[i] = -1.0
        return signals


# =============================================================================
# Demo / Test
# =============================================================================


async def demo():
    """Demonstrate GPU acceleration capabilities."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    logger.info("=" * 60)
    logger.info("GPU ACCELERATOR DEMO")
    logger.info("=" * 60)

    engine = GPUBacktestEngine(GPUConfig(use_ai_optimization=False))
    await engine.initialize()

    # Generate test data
    np.random.seed(42)
    n = 10000
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))

    # Benchmark GPU vs CPU
    logger.info(f"\nBenchmarking with {n:,} data points...")

    # Returns
    start = time.perf_counter()
    returns_gpu = engine.compute_returns_gpu(prices)
    gpu_time = time.perf_counter() - start

    start = time.perf_counter()
    returns_cpu = np.diff(prices) / prices[:-1]
    cpu_time = time.perf_counter() - start

    logger.info(f"Returns: GPU={gpu_time*1000:.2f}ms, CPU={cpu_time*1000:.2f}ms")

    # Volatility
    start = time.perf_counter()
    vol_gpu = engine.compute_ewma_volatility_gpu(returns_gpu)
    gpu_time = time.perf_counter() - start

    start = time.perf_counter()
    vol_cpu = pd.Series(returns_cpu).ewm(span=60).std().values
    cpu_time = time.perf_counter() - start

    logger.info(f"EWMA Vol: GPU={gpu_time*1000:.2f}ms, CPU={cpu_time*1000:.2f}ms")

    # Rolling Sharpe (Numba)
    start = time.perf_counter()
    sharpe = compute_rolling_sharpe_numba(returns_gpu, 252)
    numba_time = time.perf_counter() - start
    logger.info(f"Rolling Sharpe (Numba): {numba_time*1000:.2f}ms")

    # Max Drawdown (Numba)
    equity = 100 * np.exp(np.cumsum(returns_gpu))
    start = time.perf_counter()
    max_dd = compute_max_drawdown_numba(equity)
    numba_time = time.perf_counter() - start
    logger.info(f"Max Drawdown (Numba): {numba_time*1000:.2f}ms, DD={max_dd*100:.2f}%")

    await engine.close()

    logger.info("\n✓ GPU Accelerator ready for strategy sprint")

    return {
        "gpu_available": engine.gpu_available,
        "cupy": HAS_CUPY,
        "numba": HAS_NUMBA,
    }


if __name__ == "__main__":
    asyncio.run(demo())
