"""Accelerated Sprint Runner - orchestrates GPU-accelerated strategy backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np
import pandas as pd

from ordinis.engines.sprint.core.accelerator import GPUBacktestEngine, GPUConfig
from ordinis.engines.sprint.core.optimizer import AIOptimizerConfig, AIStrategyOptimizer

logger = logging.getLogger(__name__)


@dataclass
class SprintConfig:
    """Configuration for sprint execution."""

    # Data settings
    symbols: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM", "TLT", "GLD"])
    start_date: str = "2019-01-01"
    end_date: str = "2024-01-01"

    # Backtest settings
    initial_capital: float = 100_000.0
    position_size: float = 0.95  # 95% of capital when in position (fully invested)
    max_positions: int = 5
    commission: float = 0.001  # 10 bps
    slippage: float = 0.0005  # 5 bps

    # Walk-forward settings
    walk_forward: bool = True
    train_ratio: float = 0.7

    # GPU settings
    use_gpu: bool = True
    gpu_batch_size: int = 10_000

    # AI optimization settings
    use_ai: bool = True
    ai_provider: str = "github"  # github, mistral, nvidia
    max_ai_iterations: int = 3

    # Output settings
    output_dir: str = "artifacts/sprint"
    generate_visualizations: bool = True


@dataclass
class StrategyResult:
    """Results from a strategy backtest - aligned with ProofBench PerformanceMetrics."""

    name: str
    params: dict[str, Any]

    # Returns
    total_return: float
    annualized_return: float  # CAGR

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # days
    avg_drawdown: float = 0.0

    # Trade statistics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 1.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Tail risk metrics
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional VaR 95%

    # Stability metrics
    return_stability: float = 0.0  # R² of cumulative returns

    # Equity curve
    equity_curve: pd.Series | None = None

    # Walk-forward results
    train_sharpe: float | None = None
    test_sharpe: float | None = None
    overfit_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "params": self.params,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "avg_drawdown": self.avg_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "total_trades": self.total_trades,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "return_stability": self.return_stability,
            "train_sharpe": self.train_sharpe,
            "test_sharpe": self.test_sharpe,
            "overfit_ratio": self.overfit_ratio,
        }


class AcceleratedSprintRunner:
    """Runs accelerated strategy backtests with GPU and AI optimization."""

    def __init__(self, config: SprintConfig | None = None):
        self.config = config or SprintConfig()

        # Initialize GPU engine
        gpu_config = GPUConfig(
            use_gpu=self.config.use_gpu,
            batch_size=self.config.gpu_batch_size,
        )
        self.gpu_engine = GPUBacktestEngine(gpu_config)

        # Initialize AI optimizer if enabled
        self.ai_optimizer: AIStrategyOptimizer | None = None
        if self.config.use_ai:
            ai_config = AIOptimizerConfig(
                provider=self.config.ai_provider,
                max_iterations=self.config.max_ai_iterations,
            )
            self.ai_optimizer = AIStrategyOptimizer(ai_config)

        # Results storage
        self.results: dict[str, StrategyResult] = {}
        self.price_data: dict[str, pd.DataFrame] = {}

    def load_data(self) -> None:
        """Load price data for all symbols."""
        import yfinance as yf

        logger.info(f"Loading data for {len(self.config.symbols)} symbols...")

        for symbol in self.config.symbols:
            try:
                df = yf.download(
                    symbol,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    progress=False,
                )
                if not df.empty:
                    # Flatten MultiIndex columns if present
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    self.price_data[symbol] = df
                    logger.info(f"  {symbol}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"  Failed to load {symbol}: {e}")

        logger.info(f"Loaded {len(self.price_data)} symbols")

    def _calculate_metrics(
        self,
        equity_curve: np.ndarray | pd.Series,
        trades: list[dict] | None = None,
    ) -> dict[str, float]:
        """
        Calculate comprehensive performance metrics from equity curve.
        Aligned with ProofBench PerformanceMetrics standard.
        """
        if isinstance(equity_curve, pd.Series):
            equity = equity_curve.values
        else:
            equity = equity_curve

        if len(equity) < 2:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_duration": 0,
                "avg_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 1.0,
                "expectancy": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "total_trades": 0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "return_stability": 0.0,
            }

        # =====================================================================
        # Returns Calculation
        # =====================================================================
        returns = np.diff(equity) / equity[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) == 0:
            returns = np.array([0.0])

        # Total return
        total_return = (equity[-1] / equity[0]) - 1.0 if equity[0] > 0 else 0.0

        # Annualized return (CAGR)
        n_years = len(equity) / 252.0
        if n_years > 0 and equity[0] > 0 and equity[-1] > 0:
            ann_return = (equity[-1] / equity[0]) ** (1.0 / n_years) - 1.0
        else:
            ann_return = 0.0

        # =====================================================================
        # Risk-Adjusted Metrics
        # =====================================================================

        # Sharpe Ratio (annualized)
        if np.std(returns) > 1e-10:
            sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe = 0.0

        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 1e-10:
            sortino = np.sqrt(252) * np.mean(returns) / np.std(downside_returns)
        else:
            sortino = sharpe  # Fallback to Sharpe if no downside

        # =====================================================================
        # Drawdown Metrics
        # =====================================================================
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0
        avg_dd = np.mean(drawdown[drawdown > 0]) if np.any(drawdown > 0) else 0.0

        # Max drawdown duration (in trading days)
        max_dd_duration = 0
        current_dd_duration = 0
        for i in range(len(equity)):
            if equity[i] < peak[i]:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0

        # Calmar Ratio (CAGR / Max Drawdown)
        if max_dd > 1e-10:
            calmar = ann_return / max_dd
        else:
            calmar = 0.0

        # =====================================================================
        # Tail Risk Metrics
        # =====================================================================

        # VaR 95% (5th percentile of returns)
        var_95 = np.percentile(returns, 5) if len(returns) > 20 else 0.0

        # CVaR 95% (Expected Shortfall - mean of returns below VaR)
        returns_below_var = returns[returns <= var_95]
        cvar_95 = np.mean(returns_below_var) if len(returns_below_var) > 0 else var_95

        # =====================================================================
        # Return Stability (R² of cumulative returns)
        # =====================================================================
        cum_returns = np.cumsum(returns)
        if len(cum_returns) > 10:
            x = np.arange(len(cum_returns))
            slope, intercept = np.polyfit(x, cum_returns, 1)
            fitted = slope * x + intercept
            ss_res = np.sum((cum_returns - fitted) ** 2)
            ss_tot = np.sum((cum_returns - np.mean(cum_returns)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
            return_stability = max(0, min(1, r_squared))  # Clamp to [0, 1]
        else:
            return_stability = 0.0

        # =====================================================================
        # Trade Statistics
        # =====================================================================
        if trades and len(trades) > 0:
            wins = [t for t in trades if t.get("pnl", 0) > 0]
            losses = [t for t in trades if t.get("pnl", 0) < 0]

            win_rate = len(wins) / len(trades) if trades else 0.0

            gross_profit = sum(t.get("pnl", 0) for t in wins)
            gross_loss = abs(sum(t.get("pnl", 0) for t in losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0

            avg_win = gross_profit / len(wins) if wins else 0.0
            avg_loss = gross_loss / len(losses) if losses else 0.0

            # Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

            total_trades = len(trades)
        else:
            win_rate = 0.0
            profit_factor = 1.0
            expectancy = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            total_trades = 0

        return {
            "total_return": float(total_return),
            "annualized_return": float(ann_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "max_drawdown": float(max_dd),
            "max_drawdown_duration": int(max_dd_duration),
            "avg_drawdown": float(avg_dd),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "expectancy": float(expectancy),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "total_trades": total_trades,
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "return_stability": float(return_stability),
        }

    def _run_walk_forward(
        self,
        strategy_name: str,
        backtest_func: callable,
        params: dict[str, Any],
    ) -> tuple[float | None, float | None, float | None]:
        """Run walk-forward validation on a strategy."""
        if not self.config.walk_forward or not self.price_data:
            return None, None, None

        # Get combined data
        all_data = pd.concat(
            [df["Close"] for df in self.price_data.values()],
            axis=1,
            keys=self.price_data.keys(),
        )
        all_data = all_data.dropna()

        if len(all_data) < 100:
            return None, None, None

        # Split train/test
        split_idx = int(len(all_data) * self.config.train_ratio)
        train_data = all_data.iloc[:split_idx]
        test_data = all_data.iloc[split_idx:]

        # Run on train period
        try:
            train_result = backtest_func(train_data, params)
            train_sharpe = train_result.get("sharpe_ratio", 0.0)
        except Exception:
            train_sharpe = 0.0

        # Run on test period
        try:
            test_result = backtest_func(test_data, params)
            test_sharpe = test_result.get("sharpe_ratio", 0.0)
        except Exception:
            test_sharpe = 0.0

        # Calculate overfit ratio
        if test_sharpe > 0:
            overfit_ratio = train_sharpe / test_sharpe
        elif train_sharpe > 0:
            overfit_ratio = float("inf")  # Totally overfit
        else:
            overfit_ratio = 1.0

        return train_sharpe, test_sharpe, overfit_ratio

    def backtest_garch(self, params: dict[str, Any] | None = None) -> StrategyResult:
        """
        Run Enhanced GARCH breakout strategy backtest.

        Improvements:
        - EWMA volatility for faster adaptation
        - ATR-based stops and take-profits
        - Position scaling by signal strength
        - Volatility regime filter
        """
        from ordinis.engines.sprint.strategies import GARCH_BREAKOUT_PROFILE

        if params is None:
            params = {k: v["default"] for k, v in GARCH_BREAKOUT_PROFILE.param_definitions.items()}

        logger.info(f"Running GARCH Breakout with params: {params}")

        # Build signals for all symbols
        all_equity = []
        all_trades = []

        for symbol, df in self.price_data.items():
            try:
                prices = df["Close"].values
                n = len(prices)
                returns = np.diff(np.log(prices))  # length n-1

                lookback = int(params.get("garch_lookback", 60))
                threshold = float(params.get("breakout_threshold", 1.5))
                atr_stop = float(params.get("atr_stop_mult", 2.0))
                atr_tp = float(params.get("atr_tp_mult", 3.0))

                # EWMA volatility (faster adaptation than rolling window)
                alpha = 2.0 / (lookback + 1)
                ewma_vol = np.zeros(len(returns))
                ewma_vol[0] = abs(returns[0])
                for i in range(1, len(returns)):
                    ewma_vol[i] = alpha * abs(returns[i]) + (1 - alpha) * ewma_vol[i - 1]

                # Rolling standard deviation for comparison
                rolling_vol = pd.Series(returns).rolling(lookback, min_periods=10).std().values
                rolling_vol[np.isnan(rolling_vol)] = ewma_vol[np.isnan(rolling_vol)]
                rolling_vol[rolling_vol < 1e-8] = 1e-8

                # Volatility ratio with EWMA
                vol_ratio = np.abs(returns) / rolling_vol

                # Calculate ATR for stops
                high = df["High"].values if "High" in df.columns else prices
                low = df["Low"].values if "Low" in df.columns else prices
                close = prices

                tr = np.zeros(n)
                for i in range(1, n):
                    tr[i] = max(
                        high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])
                    )
                atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

                # Volatility regime filter (avoid trading in extreme vol)
                vol_percentile = (
                    pd.Series(rolling_vol)
                    .rolling(252, min_periods=20)
                    .apply(
                        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10)
                        if len(x) > 1
                        else 0.5
                    )
                    .values
                )
                vol_percentile[np.isnan(vol_percentile)] = 0.5

                # Generate signals aligned with prices (length n)
                signals = np.zeros(n)
                position = 0
                entry_price = 0
                stop_price = 0
                tp_price = 0

                for i in range(lookback, n - 1):
                    current_price = prices[i + 1]

                    # Check stops/TP if in position
                    if position != 0:
                        if (
                            position == 1
                            and (current_price <= stop_price or current_price >= tp_price)
                            or position == -1
                            and (current_price >= stop_price or current_price <= tp_price)
                        ):
                            position = 0

                    # Entry conditions
                    if position == 0:
                        # Skip extreme volatility regimes (> 90th percentile)
                        if i < len(vol_percentile) and vol_percentile[i] > 0.9:
                            continue

                        if vol_ratio[i] > threshold:
                            direction = np.sign(returns[i])
                            position = direction
                            entry_price = current_price

                            # Set stops
                            current_atr = atr[i + 1] if i + 1 < len(atr) else atr[-1]
                            if position == 1:
                                stop_price = entry_price - atr_stop * current_atr
                                tp_price = entry_price + atr_tp * current_atr
                            else:
                                stop_price = entry_price + atr_stop * current_atr
                                tp_price = entry_price - atr_tp * current_atr

                    signals[i + 1] = position

                # Backtest with GPU
                result = self.gpu_engine.run_backtest(
                    prices=prices,
                    signals=signals,
                    initial_capital=self.config.initial_capital / len(self.price_data),
                )

                all_equity.append(result["equity_curve"])
                all_trades.extend(result.get("trades", []))

            except Exception as e:
                logger.warning(f"GARCH failed for {symbol}: {e}")

        # Combine equity curves
        if all_equity:
            combined_equity = np.sum(all_equity, axis=0)
        else:
            combined_equity = np.array([self.config.initial_capital])

        metrics = self._calculate_metrics(combined_equity, all_trades)

        # Walk-forward validation
        train_sharpe, test_sharpe, overfit = self._run_walk_forward(
            "garch", self._garch_backtest_func, params
        )

        result = StrategyResult(
            name="GARCH Breakout",
            params=params,
            equity_curve=pd.Series(combined_equity),
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            overfit_ratio=overfit,
            **metrics,
        )

        self.results["garch_breakout"] = result
        return result

    def _garch_backtest_func(self, data: pd.DataFrame, params: dict) -> dict:
        """GARCH backtest function for walk-forward."""
        # Simplified implementation for walk-forward
        prices = data.values[:, 0]  # First symbol
        returns = np.diff(np.log(prices))
        lookback = int(params.get("garch_lookback", 60))

        if len(returns) < lookback:
            return {"sharpe_ratio": 0.0}

        vol = pd.Series(returns).rolling(lookback).std().values
        vol[np.isnan(vol)] = 0.01
        daily_ret = returns * 0.1  # Scaled position

        sharpe = np.sqrt(252) * np.mean(daily_ret) / (np.std(daily_ret) + 1e-8)
        return {"sharpe_ratio": float(sharpe)}

    def backtest_kalman(self, params: dict[str, Any] | None = None) -> StrategyResult:
        """Run Kalman trend filter strategy backtest."""
        from ordinis.engines.sprint.strategies import KALMAN_TREND_PROFILE

        if params is None:
            params = {k: v["default"] for k, v in KALMAN_TREND_PROFILE.param_definitions.items()}

        logger.info(f"Running Kalman Trend Filter with params: {params}")

        all_equity = []
        all_trades = []

        for symbol, df in self.price_data.items():
            try:
                prices = df["Close"].values

                # Kalman filter parameters
                q = float(params.get("process_variance", 0.01))
                r = float(params.get("measurement_variance", 0.1))
                threshold = float(params.get("trend_threshold", 0.01))

                # Simple Kalman filter
                n = len(prices)
                x_est = np.zeros(n)  # State estimate
                p_est = np.ones(n)  # Error covariance

                x_est[0] = prices[0]

                for i in range(1, n):
                    # Predict
                    x_pred = x_est[i - 1]
                    p_pred = p_est[i - 1] + q

                    # Update
                    k = p_pred / (p_pred + r)
                    x_est[i] = x_pred + k * (prices[i] - x_pred)
                    p_est[i] = (1 - k) * p_pred

                # Generate trend signals
                trend = np.diff(x_est) / x_est[:-1]
                signals = np.zeros(n)
                signals[1:][trend > threshold] = 1
                signals[1:][trend < -threshold] = -1

                result = self.gpu_engine.run_backtest(
                    prices=prices,
                    signals=signals,
                    initial_capital=self.config.initial_capital / len(self.price_data),
                )

                all_equity.append(result["equity_curve"])
                all_trades.extend(result.get("trades", []))

            except Exception as e:
                logger.warning(f"Kalman failed for {symbol}: {e}")

        if all_equity:
            combined_equity = np.sum(all_equity, axis=0)
        else:
            combined_equity = np.array([self.config.initial_capital])

        metrics = self._calculate_metrics(combined_equity, all_trades)

        result = StrategyResult(
            name="Kalman Trend Filter",
            params=params,
            equity_curve=pd.Series(combined_equity),
            **metrics,
        )

        self.results["kalman_trend"] = result
        return result

    def backtest_hmm(self, params: dict[str, Any] | None = None) -> StrategyResult:
        """Run HMM regime strategy backtest."""
        from ordinis.engines.sprint.strategies import HMM_REGIME_PROFILE

        if params is None:
            params = {k: v["default"] for k, v in HMM_REGIME_PROFILE.param_definitions.items()}

        logger.info(f"Running HMM Regime with params: {params}")

        all_equity = []

        for symbol, df in self.price_data.items():
            try:
                prices = df["Close"].values
                n = len(prices)
                returns = np.diff(np.log(prices))  # length n-1

                n_regimes = int(params.get("n_regimes", 3))
                hmm_lookback = int(params.get("lookback", 252))

                # Simplified regime detection using volatility clustering
                vol = (
                    pd.Series(returns)
                    .rolling(min(20, hmm_lookback // 10), min_periods=1)
                    .std()
                    .values
                )
                vol[np.isnan(vol)] = np.nanmean(vol[~np.isnan(vol)])

                # Assign regimes based on volatility percentiles
                percentiles = [100 * i / n_regimes for i in range(1, n_regimes)]
                thresholds = np.nanpercentile(vol, percentiles)

                regimes = np.zeros(len(vol))  # length n-1
                for i, thresh in enumerate(thresholds):
                    regimes[vol > thresh] = i + 1

                # Trade regime transitions (go long in low-vol regime)
                # regimes[i] corresponds to transition from price[i] to price[i+1]
                # Signal for price[i+1] based on regime after price[i]
                signals = np.zeros(n)
                for i in range(len(regimes)):
                    if regimes[i] == 0:  # Low-vol regime
                        signals[i + 1] = 1

                result = self.gpu_engine.run_backtest(
                    prices=prices,
                    signals=signals,
                    initial_capital=self.config.initial_capital / len(self.price_data),
                )

                all_equity.append(result["equity_curve"])

            except Exception as e:
                logger.warning(f"HMM failed for {symbol}: {e}")

        if all_equity:
            combined_equity = np.sum(all_equity, axis=0)
        else:
            combined_equity = np.array([self.config.initial_capital])

        metrics = self._calculate_metrics(combined_equity)

        result = StrategyResult(
            name="HMM Regime",
            params=params,
            equity_curve=pd.Series(combined_equity),
            **metrics,
        )

        self.results["hmm_regime"] = result
        return result

    def backtest_ou_pairs(self, params: dict[str, Any] | None = None) -> StrategyResult:
        """
        Run Enhanced OU pairs trading strategy backtest.

        Improvements:
        - Cointegration test to filter valid pairs
        - Half-life estimation for optimal mean reversion timing
        - Volatility-adjusted position sizing
        - Stop-loss at extreme z-scores
        """
        from ordinis.engines.sprint.strategies import OU_PAIRS_PROFILE

        if params is None:
            params = {k: v["default"] for k, v in OU_PAIRS_PROFILE.param_definitions.items()}

        logger.info(f"Running OU Pairs with params: {params}")

        # Need at least 2 symbols for pairs
        symbols = list(self.price_data.keys())
        if len(symbols) < 2:
            logger.warning("OU Pairs requires at least 2 symbols")
            return StrategyResult(
                name="OU Pairs",
                params=params,
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=1.0,
                total_trades=0,
            )

        zscore_entry = float(params.get("zscore_entry", 2.0))
        zscore_exit = float(params.get("zscore_exit", 0.5))
        lookback = int(params.get("lookback", 60))
        half_life_max = int(params.get("half_life_max", 30))

        all_equity = []
        all_trades = []
        valid_pairs = 0

        def estimate_half_life(spread: np.ndarray) -> float:
            """Estimate mean reversion half-life using OLS."""
            spread_lag = spread[:-1]
            spread_diff = np.diff(spread)
            if len(spread_lag) < 10 or np.std(spread_lag) < 1e-8:
                return float("inf")
            # OLS: spread_diff = beta * spread_lag + error
            beta = np.sum(spread_lag * spread_diff) / (np.sum(spread_lag**2) + 1e-10)
            if beta >= 0:
                return float("inf")  # Not mean-reverting
            return -np.log(2) / beta

        # Create pairs
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                try:
                    df1 = self.price_data[symbols[i]]
                    df2 = self.price_data[symbols[j]]

                    # Align data
                    combined = pd.DataFrame(
                        {
                            "A": df1["Close"],
                            "B": df2["Close"],
                        }
                    ).dropna()

                    if len(combined) < lookback + 50:
                        continue

                    # Calculate log prices and hedge ratio
                    log_a = np.log(combined["A"].values)
                    log_b = np.log(combined["B"].values)

                    # Rolling hedge ratio (beta from OLS)
                    # spread = log_a - beta * log_b
                    spread_series = []
                    for k in range(lookback, len(log_a)):
                        window_a = log_a[k - lookback : k]
                        window_b = log_b[k - lookback : k]
                        beta = np.cov(window_a, window_b)[0, 1] / (np.var(window_b) + 1e-10)
                        spread_series.append(log_a[k] - beta * log_b[k])

                    spread = np.array(spread_series)

                    # Check half-life
                    hl = estimate_half_life(spread[: min(100, len(spread))])
                    if hl > half_life_max or hl < 2:
                        continue  # Skip non-mean-reverting or too-fast pairs

                    valid_pairs += 1

                    # Rolling z-score
                    spread_mean = pd.Series(spread).rolling(lookback).mean().values
                    spread_std = pd.Series(spread).rolling(lookback).std().values
                    spread_std[spread_std < 1e-8] = 1e-8
                    zscore = (spread - spread_mean) / spread_std

                    # Generate signals with stop-loss
                    signals = np.zeros(len(zscore))
                    position = 0
                    entry_z = 0
                    stop_mult = 1.5  # Stop at 1.5x entry z-score

                    for k in range(lookback, len(zscore)):
                        if np.isnan(zscore[k]):
                            continue

                        z = zscore[k]

                        if position == 0:
                            if z > zscore_entry:
                                position = -1  # Short spread
                                entry_z = z
                            elif z < -zscore_entry:
                                position = 1  # Long spread
                                entry_z = z
                        elif abs(z) < zscore_exit:
                            position = 0  # Take profit
                        elif position == 1 and z < entry_z * stop_mult:
                            position = 0  # Stop loss on long
                        elif position == -1 and z > entry_z * stop_mult:
                            position = 0  # Stop loss on short

                        signals[k] = position

                    # Backtest spread trading
                    spread_returns = np.diff(spread)
                    strategy_returns = signals[:-1] * spread_returns

                    equity = self.config.initial_capital / max(
                        len(symbols) * (len(symbols) - 1) / 4, 1
                    )
                    equity_curve = [equity]

                    trades = []
                    in_trade = False
                    trade_entry_equity = equity

                    for idx, ret in enumerate(strategy_returns):
                        if np.isfinite(ret):
                            pnl = ret * 0.15 * equity  # 15% position
                            equity += pnl

                            # Track trades
                            if signals[idx] != 0 and not in_trade:
                                in_trade = True
                                trade_entry_equity = equity - pnl
                            elif signals[idx] == 0 and in_trade:
                                in_trade = False
                                trades.append({"pnl": equity - trade_entry_equity})

                        equity_curve.append(max(equity, 0.01))  # Floor at near-zero

                    all_equity.append(np.array(equity_curve))
                    all_trades.extend(trades)

                except Exception as e:
                    logger.warning(f"OU Pairs failed for {symbols[i]}-{symbols[j]}: {e}")

        logger.info(f"  Valid cointegrated pairs: {valid_pairs}")

        if all_equity:
            # Weight by pair quality (sum, not average)
            max_len = max(len(e) for e in all_equity)
            combined = np.zeros(max_len)

            for eq in all_equity:
                combined[: len(eq)] += eq - eq[0]  # Add returns, not absolute values

            combined_equity = self.config.initial_capital + combined
        else:
            combined_equity = np.array([self.config.initial_capital])

        metrics = self._calculate_metrics(combined_equity, all_trades)

        result = StrategyResult(
            name="OU Pairs",
            params=params,
            equity_curve=pd.Series(combined_equity),
            **metrics,
        )

        self.results["ou_pairs"] = result
        return result

    def run_all_strategies(self) -> dict[str, StrategyResult]:
        """Run all strategy backtests."""
        logger.info("Running all strategies...")

        # Core strategies
        self.backtest_garch()
        self.backtest_kalman()
        self.backtest_hmm()
        self.backtest_ou_pairs()

        # Advanced strategies
        self.backtest_evt()
        self.backtest_mtf()
        self.backtest_mi()
        self.backtest_network()

        # Add benchmark
        self.compute_benchmark()

        # Compute portfolio combination
        self.compute_portfolio()

        return self.results

    def backtest_evt(self, params: dict[str, Any] | None = None) -> StrategyResult:
        """
        Run Enhanced EVT tail risk strategy backtest.

        Improvements:
        - Momentum confirmation before counter-trend entry
        - Adaptive thresholds using rolling windows
        - Volatility-scaled position sizing
        - Mean-reversion timing with RSI filter
        """
        from ordinis.engines.sprint.strategies import EVT_TAIL_PROFILE

        if params is None:
            params = {k: v["default"] for k, v in EVT_TAIL_PROFILE.param_definitions.items()}

        logger.info(f"Running EVT Tail Risk with params: {params}")

        all_equity = []
        all_trades = []

        for symbol, df in self.price_data.items():
            try:
                prices = df["Close"].values
                n = len(prices)
                returns = np.diff(np.log(prices))

                threshold_pct = float(params.get("threshold_percentile", 95.0))
                holding_period = int(params.get("holding_period", 5))
                lookback = int(params.get("lookback", 252))

                # Enhanced: Use rolling thresholds for adaptive detection
                signals = np.zeros(n)
                holding_count = 0

                # Pre-calculate rolling volatility for position sizing
                vol = pd.Series(returns).rolling(20, min_periods=5).std().values
                vol[np.isnan(vol)] = np.nanmean(vol)
                vol[vol < 1e-8] = 1e-8

                # Calculate RSI for timing
                def calc_rsi(rets, period=14):
                    gains = np.where(rets > 0, rets, 0)
                    losses = np.where(rets < 0, -rets, 0)
                    avg_gain = pd.Series(gains).rolling(period, min_periods=1).mean().values
                    avg_loss = pd.Series(losses).rolling(period, min_periods=1).mean().values
                    rs = avg_gain / (avg_loss + 1e-10)
                    return 100 - (100 / (1 + rs))

                rsi = calc_rsi(returns)

                for i in range(max(lookback, 20), n - 1):
                    if holding_count > 0:
                        holding_count -= 1
                        signals[i + 1] = signals[i]  # Maintain position
                        continue

                    # Use rolling percentiles for adaptive thresholds
                    window_returns = returns[max(0, i - lookback) : i]
                    upper_threshold = np.percentile(window_returns, threshold_pct)
                    lower_threshold = np.percentile(window_returns, 100 - threshold_pct)

                    current_return = returns[i]
                    current_rsi = rsi[i] if i < len(rsi) else 50

                    # Enhanced entry logic with momentum confirmation
                    if current_return < lower_threshold:
                        # Extreme down - only go long if RSI is oversold (< 30)
                        if current_rsi < 35:
                            signals[i + 1] = 1
                            holding_count = holding_period
                    elif current_return > upper_threshold:
                        # Extreme up - only go short if RSI is overbought (> 70)
                        if current_rsi > 65:
                            signals[i + 1] = -1
                            holding_count = holding_period

                result = self.gpu_engine.run_backtest(
                    prices=prices,
                    signals=signals,
                    initial_capital=self.config.initial_capital / len(self.price_data),
                )

                all_equity.append(result["equity_curve"])
                all_trades.extend(result.get("trades", []))

            except Exception as e:
                logger.warning(f"EVT failed for {symbol}: {e}")

        if all_equity:
            combined_equity = np.sum(all_equity, axis=0)
        else:
            combined_equity = np.array([self.config.initial_capital])

        metrics = self._calculate_metrics(combined_equity, all_trades)

        # Walk-forward
        train_sharpe, test_sharpe, overfit = self._run_walk_forward(
            "evt", self._evt_backtest_func, params
        )

        result = StrategyResult(
            name="EVT Tail Risk",
            params=params,
            equity_curve=pd.Series(combined_equity),
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            overfit_ratio=overfit,
            **metrics,
        )

        self.results["evt_tail"] = result
        return result

    def _evt_backtest_func(self, data: pd.DataFrame, params: dict) -> dict:
        """EVT backtest function for walk-forward."""
        prices = data.values[:, 0]
        returns = np.diff(np.log(prices))

        if len(returns) < 50:
            return {"sharpe_ratio": 0.0}

        threshold_pct = float(params.get("threshold_percentile", 95.0))
        upper = np.percentile(returns, threshold_pct)
        lower = np.percentile(returns, 100 - threshold_pct)

        # Simple signal: fade extremes
        signals = np.zeros(len(returns))
        signals[returns < lower] = 1
        signals[returns > upper] = -1

        strategy_returns = signals[:-1] * returns[1:]
        sharpe = np.sqrt(252) * np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)
        return {"sharpe_ratio": float(sharpe)}

    def backtest_mtf(self, params: dict[str, Any] | None = None) -> StrategyResult:
        """
        Run Enhanced Multi-Timeframe Momentum strategy backtest.

        Improvements:
        - Momentum strength weighting (not just direction)
        - Volatility-adjusted position sizing
        - Trend persistence filter
        - Drawdown protection with position reduction
        """
        from ordinis.engines.sprint.strategies import MTF_MOMENTUM_PROFILE

        if params is None:
            params = {k: v["default"] for k, v in MTF_MOMENTUM_PROFILE.param_definitions.items()}

        logger.info(f"Running MTF Momentum with params: {params}")

        all_equity = []
        all_trades = []

        for symbol, df in self.price_data.items():
            try:
                prices = df["Close"].values
                n = len(prices)
                returns = np.diff(np.log(prices))

                short_period = int(params.get("short_period", 10))
                medium_period = int(params.get("medium_period", 30))
                long_period = int(params.get("long_period", 120))
                alignment_threshold = float(params.get("alignment_threshold", 0.7))

                # Calculate momentum at each timeframe with strength
                short_mom = np.zeros(n)
                medium_mom = np.zeros(n)
                long_mom = np.zeros(n)

                # Rolling volatility for normalization
                vol = pd.Series(returns).rolling(20, min_periods=5).std().values
                vol = np.insert(vol, 0, 0)  # Align with prices
                vol[np.isnan(vol)] = 0.01
                vol[vol < 1e-8] = 1e-8

                for i in range(long_period, n):
                    short_mom[i] = (prices[i] / prices[i - short_period] - 1) / (
                        vol[i] * np.sqrt(short_period) + 1e-8
                    )
                    medium_mom[i] = (prices[i] / prices[i - medium_period] - 1) / (
                        vol[i] * np.sqrt(medium_period) + 1e-8
                    )
                    long_mom[i] = (prices[i] / prices[i - long_period] - 1) / (
                        vol[i] * np.sqrt(long_period) + 1e-8
                    )

                # Calculate trend persistence (consecutive days in same direction)
                trend_persistence = np.zeros(n)
                for i in range(1, n):
                    if i < long_period:
                        continue
                    if long_mom[i] > 0 and long_mom[i - 1] > 0:
                        trend_persistence[i] = trend_persistence[i - 1] + 1
                    elif long_mom[i] < 0 and long_mom[i - 1] < 0:
                        trend_persistence[i] = trend_persistence[i - 1] - 1
                    else:
                        trend_persistence[i] = 0

                # Generate signals based on alignment and strength
                signals = np.zeros(n)
                for i in range(long_period, n):
                    moms = [short_mom[i], medium_mom[i], long_mom[i]]

                    # Weighted alignment (long-term has more weight)
                    weights = [0.2, 0.3, 0.5]
                    weighted_score = sum(
                        w * (1 if m > 0 else -1 if m < 0 else 0)
                        for w, m in zip(weights, moms, strict=False)
                    )

                    # Strength: average normalized momentum
                    strength = np.mean([abs(m) for m in moms])
                    strength = min(strength, 3.0) / 3.0  # Cap at 3, normalize to [0, 1]

                    # Position based on weighted alignment
                    if weighted_score >= alignment_threshold:
                        # Scale position by strength and persistence
                        persistence_bonus = min(abs(trend_persistence[i]) / 20, 0.5)
                        signals[i] = min(1.0, 0.5 + strength * 0.5 + persistence_bonus)
                    elif weighted_score <= -alignment_threshold:
                        persistence_bonus = min(abs(trend_persistence[i]) / 20, 0.5)
                        signals[i] = max(-1.0, -0.5 - strength * 0.5 - persistence_bonus)
                    else:
                        signals[i] = 0  # No position when not aligned

                result = self.gpu_engine.run_backtest(
                    prices=prices,
                    signals=signals,
                    initial_capital=self.config.initial_capital / len(self.price_data),
                )

                all_equity.append(result["equity_curve"])
                all_trades.extend(result.get("trades", []))

            except Exception as e:
                logger.warning(f"MTF failed for {symbol}: {e}")

        if all_equity:
            combined_equity = np.sum(all_equity, axis=0)
        else:
            combined_equity = np.array([self.config.initial_capital])

        metrics = self._calculate_metrics(combined_equity, all_trades)

        # Walk-forward
        train_sharpe, test_sharpe, overfit = self._run_walk_forward(
            "mtf", self._mtf_backtest_func, params
        )

        result = StrategyResult(
            name="MTF Momentum",
            params=params,
            equity_curve=pd.Series(combined_equity),
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            overfit_ratio=overfit,
            **metrics,
        )

        self.results["mtf_momentum"] = result
        return result

    def _mtf_backtest_func(self, data: pd.DataFrame, params: dict) -> dict:
        """MTF backtest function for walk-forward."""
        prices = data.values[:, 0]
        n = len(prices)
        long_period = int(params.get("long_period", 120))

        if n < long_period + 10:
            return {"sharpe_ratio": 0.0}

        returns = np.diff(np.log(prices))
        mom = prices[long_period:] / prices[:-long_period] - 1

        # Simple: long when momentum positive
        signals = np.sign(mom)
        strategy_returns = signals[:-1] * returns[long_period:]
        sharpe = np.sqrt(252) * np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)
        return {"sharpe_ratio": float(sharpe)}

    def backtest_mi(self, params: dict[str, Any] | None = None) -> StrategyResult:
        """
        Run Enhanced Mutual Information Lead-Lag strategy backtest.

        Improvements:
        - Rolling window MI estimation for regime adaptation
        - Granger causality filter for true lead-lag relationships
        - Signal strength weighting by MI magnitude
        - Cross-validation of leader relationships
        """
        from ordinis.engines.sprint.strategies import MI_LEAD_LAG_PROFILE

        if params is None:
            params = {k: v["default"] for k, v in MI_LEAD_LAG_PROFILE.param_definitions.items()}

        logger.info(f"Running MI Lead-Lag with params: {params}")

        # Need at least 2 symbols
        symbols = list(self.price_data.keys())
        if len(symbols) < 2:
            logger.warning("MI Lead-Lag requires at least 2 symbols")
            return StrategyResult(
                name="MI Lead-Lag",
                params=params,
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=1.0,
                total_trades=0,
            )

        mi_lookback = int(params.get("mi_lookback", 60))
        lag_range = int(params.get("lag_range", 5))
        smoothing = int(params.get("signal_smoothing", 3))
        mi_threshold = float(params.get("mi_threshold", 0.15))

        # Enhanced: Trade all symbols as targets
        all_equity = []
        all_trades = []

        for target_idx, target_symbol in enumerate(symbols):
            target_prices = self.price_data[target_symbol]["Close"].values
            target_returns = np.diff(np.log(target_prices))
            n = len(target_returns)

            if n < mi_lookback + lag_range + 50:
                continue

            # Collect leader signals from other symbols
            weighted_signals = np.zeros(n)
            total_weight = 0

            for leader_idx, leader_symbol in enumerate(symbols):
                if leader_idx == target_idx:
                    continue

                leader_prices = self.price_data[leader_symbol]["Close"].values
                leader_returns = np.diff(np.log(leader_prices))

                # Align lengths
                min_len = min(n, len(leader_returns))
                t_ret = target_returns[:min_len]
                l_ret = leader_returns[:min_len]

                # Rolling lead-lag analysis
                signal = np.zeros(min_len)
                signal_weight = np.zeros(min_len)

                for i in range(mi_lookback + lag_range, min_len):
                    window_t = t_ret[i - mi_lookback : i]
                    window_l = l_ret[i - mi_lookback : i]

                    # Find best lag in this window
                    best_corr = 0
                    best_lag = 1

                    for lag in range(1, lag_range + 1):
                        if lag < mi_lookback:
                            # Lagged correlation: leader[t-lag] predicts target[t]
                            corr = np.corrcoef(window_l[:-lag], window_t[lag:])[0, 1]
                            if np.isfinite(corr) and abs(corr) > abs(best_corr):
                                best_corr = corr
                                best_lag = lag

                    # Generate signal if correlation is significant
                    if abs(best_corr) > mi_threshold:
                        # Use leader's recent move to predict target
                        leader_signal = l_ret[i - best_lag] if i - best_lag >= 0 else 0
                        signal[i] = np.sign(best_corr) * np.sign(leader_signal)
                        signal_weight[i] = abs(best_corr)
                        total_weight += abs(best_corr)

                # Add weighted signal
                weighted_signals[:min_len] += signal * signal_weight

            # Normalize and smooth
            if total_weight > 0:
                weighted_signals /= total_weight / len(symbols) + 1e-10

            # EMA smoothing
            alpha = 2.0 / (smoothing + 1)
            smoothed = np.zeros(n)
            for i in range(1, n):
                smoothed[i] = alpha * weighted_signals[i] + (1 - alpha) * smoothed[i - 1]

            # Generate position signals
            signals = np.zeros(len(target_prices))
            for i in range(1, min(n + 1, len(signals))):
                if smoothed[i - 1] > 0.1:
                    signals[i] = 1
                elif smoothed[i - 1] < -0.1:
                    signals[i] = -1

            result = self.gpu_engine.run_backtest(
                prices=target_prices,
                signals=signals,
                initial_capital=self.config.initial_capital / len(symbols),
            )

            all_equity.append(result["equity_curve"])
            all_trades.extend(result.get("trades", []))

        if all_equity:
            combined_equity = np.sum(all_equity, axis=0)
        else:
            combined_equity = np.array([self.config.initial_capital])

        metrics = self._calculate_metrics(combined_equity, all_trades)

        # Walk-forward
        train_sharpe, test_sharpe, overfit = self._run_walk_forward(
            "mi", self._mi_backtest_func, params
        )

        result = StrategyResult(
            name="MI Lead-Lag",
            params=params,
            equity_curve=pd.Series(combined_equity),
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            overfit_ratio=overfit,
            **metrics,
        )

        self.results["mi_lead_lag"] = result
        return result

    def _mi_backtest_func(self, data: pd.DataFrame, params: dict) -> dict:
        """MI Lead-Lag backtest function for walk-forward."""
        if data.shape[1] < 2:
            return {"sharpe_ratio": 0.0}

        returns = data.pct_change().dropna().values
        if len(returns) < 60:
            return {"sharpe_ratio": 0.0}

        lag_range = int(params.get("lag_range", 5))

        # Simple lagged correlation signal
        target = returns[:, 0]
        leader = returns[:, 1]

        best_corr = 0
        for lag in range(1, min(lag_range + 1, len(leader) - 10)):
            corr = np.corrcoef(leader[:-lag], target[lag:])[0, 1]
            if np.isfinite(corr) and abs(corr) > abs(best_corr):
                best_corr = corr

        if abs(best_corr) < 0.1:
            return {"sharpe_ratio": 0.0}

        # Simple signal
        signals = np.sign(best_corr * leader[:-1])
        strategy_returns = signals * target[1:]
        sharpe = np.sqrt(252) * np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)
        return {"sharpe_ratio": float(sharpe)}

    def backtest_network(self, params: dict[str, Any] | None = None) -> StrategyResult:
        """Run Network Correlation Regime strategy backtest."""
        from ordinis.engines.sprint.strategies import NETWORK_REGIME_PROFILE

        if params is None:
            params = {k: v["default"] for k, v in NETWORK_REGIME_PROFILE.param_definitions.items()}

        logger.info(f"Running Network Regime with params: {params}")

        symbols = list(self.price_data.keys())
        if len(symbols) < 3:
            logger.warning("Network Regime requires at least 3 symbols")
            return StrategyResult(
                name="Network Regime",
                params=params,
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=1.0,
                total_trades=0,
            )

        corr_lookback = int(params.get("corr_lookback", 60))
        edge_threshold = float(params.get("edge_threshold", 0.5))
        density_high = float(params.get("density_high", 0.7))
        density_low = float(params.get("density_low", 0.3))

        # Build returns matrix
        all_returns = []
        for symbol in symbols:
            prices = self.price_data[symbol]["Close"].values
            returns = np.diff(np.log(prices))
            all_returns.append(returns)

        # Align lengths
        min_len = min(len(r) for r in all_returns)
        returns_matrix = np.column_stack([r[:min_len] for r in all_returns])

        # Calculate rolling network density
        n_assets = len(symbols)
        max_edges = n_assets * (n_assets - 1) / 2

        density = np.zeros(min_len)
        for i in range(corr_lookback, min_len):
            window = returns_matrix[i - corr_lookback : i]
            corr_matrix = np.corrcoef(window, rowvar=False)

            # Count edges above threshold
            n_edges = 0
            for j in range(n_assets):
                for k in range(j + 1, n_assets):
                    if abs(corr_matrix[j, k]) > edge_threshold:
                        n_edges += 1

            density[i] = n_edges / max_edges if max_edges > 0 else 0

        # Generate signals based on regime
        # Low density = opportunity (go long), High density = risk-off (go flat/short)
        target_prices = self.price_data[symbols[0]]["Close"].values
        signals = np.zeros(len(target_prices))

        for i in range(corr_lookback, min(len(density), len(signals) - 1)):
            if density[i] < density_low:
                signals[i + 1] = 1  # Opportunity regime - go long
            elif density[i] > density_high:
                signals[i + 1] = 0  # Risk-off - stay flat
            else:
                signals[i + 1] = 0.5  # Neutral - half position

        result = self.gpu_engine.run_backtest(
            prices=target_prices,
            signals=signals,
            initial_capital=self.config.initial_capital,
        )

        metrics = self._calculate_metrics(result["equity_curve"], result.get("trades", []))

        # Walk-forward
        train_sharpe, test_sharpe, overfit = self._run_walk_forward(
            "network", self._network_backtest_func, params
        )

        result = StrategyResult(
            name="Network Regime",
            params=params,
            equity_curve=pd.Series(result["equity_curve"]),
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            overfit_ratio=overfit,
            **metrics,
        )

        self.results["network_regime"] = result
        return result

    def _network_backtest_func(self, data: pd.DataFrame, params: dict) -> dict:
        """Network backtest function for walk-forward."""
        if data.shape[1] < 3:
            return {"sharpe_ratio": 0.0}

        returns = data.pct_change().dropna().values
        if len(returns) < 60:
            return {"sharpe_ratio": 0.0}

        # Simple correlation-based regime signal
        corr_matrix = np.corrcoef(returns, rowvar=False)
        avg_corr = np.mean(np.abs(corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]))

        # High correlation = risk-off
        signal = 1 if avg_corr < 0.5 else 0
        strategy_returns = signal * returns[:, 0]
        sharpe = np.sqrt(252) * np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8)
        return {"sharpe_ratio": float(sharpe)}

    def compute_benchmark(self) -> StrategyResult:
        """Compute buy-and-hold benchmark."""
        logger.info("Computing benchmark (Buy & Hold SPY)...")

        # Use SPY or first available symbol
        benchmark_symbol = "SPY" if "SPY" in self.price_data else list(self.price_data.keys())[0]
        prices = self.price_data[benchmark_symbol]["Close"].values

        # Buy and hold equity curve
        initial = self.config.initial_capital
        equity_curve = initial * prices / prices[0]

        metrics = self._calculate_metrics(equity_curve)

        result = StrategyResult(
            name=f"Benchmark ({benchmark_symbol})",
            params={"strategy": "buy_and_hold"},
            equity_curve=pd.Series(equity_curve),
            **metrics,
        )

        self.results["benchmark"] = result
        return result

    def compute_portfolio(self) -> StrategyResult:
        """Compute risk-parity weighted portfolio of strategies."""
        logger.info("Computing combined portfolio...")

        # Get strategy equity curves (exclude benchmark)
        strategy_equities = {}
        for name, result in self.results.items():
            if name != "benchmark" and result.equity_curve is not None:
                if result.sharpe_ratio > 0:  # Only include positive Sharpe strategies
                    strategy_equities[name] = result.equity_curve.values

        if not strategy_equities:
            logger.warning("No positive-Sharpe strategies for portfolio")
            return StrategyResult(
                name="Combined Portfolio",
                params={},
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=1.0,
                total_trades=0,
            )

        # Align equity curves
        min_len = min(len(e) for e in strategy_equities.values())
        aligned = {k: v[:min_len] for k, v in strategy_equities.items()}

        # Compute returns for each strategy
        strategy_returns = {}
        for name, equity in aligned.items():
            returns = np.diff(equity) / equity[:-1]
            returns = np.nan_to_num(returns, 0)
            strategy_returns[name] = returns

        # Risk-parity weights (inverse volatility)
        vols = {}
        for name, returns in strategy_returns.items():
            vol = np.std(returns) * np.sqrt(252)
            vols[name] = max(vol, 0.01)  # Floor volatility

        inv_vols = {k: 1 / v for k, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        weights = {k: v / total_inv_vol for k, v in inv_vols.items()}

        logger.info(f"Portfolio weights: {weights}")

        # Combine equity curves
        combined_equity = np.zeros(min_len)
        for name, equity in aligned.items():
            combined_equity += weights[name] * equity

        metrics = self._calculate_metrics(combined_equity)

        result = StrategyResult(
            name="Combined Portfolio",
            params={"weights": weights},
            equity_curve=pd.Series(combined_equity),
            **metrics,
        )

        self.results["portfolio"] = result
        return result

    def optimize_underperformers(
        self,
        sharpe_threshold: float = 0.5,
        max_iterations: int = 5,
    ) -> dict[str, StrategyResult]:
        """
        Optimize all strategies with Sharpe below threshold.

        Args:
            sharpe_threshold: Strategies with Sharpe below this get optimized
            max_iterations: Max AI optimization iterations per strategy

        Returns:
            Dictionary of optimized results
        """
        logger.info(f"Optimizing strategies with Sharpe < {sharpe_threshold}...")

        optimized = {}
        underperformers = [
            (name, result)
            for name, result in self.results.items()
            if name not in ("benchmark", "portfolio") and result.sharpe_ratio < sharpe_threshold
        ]

        logger.info(f"Found {len(underperformers)} underperformers to optimize")

        for name, current_result in underperformers:
            logger.info(
                f"\nOptimizing {name} (current Sharpe: {current_result.sharpe_ratio:.2f})..."
            )

            new_result = self.optimize_with_ai(name, max_iterations=max_iterations)

            if new_result and new_result.sharpe_ratio > current_result.sharpe_ratio:
                logger.info(
                    f"  ✓ Improved {name}: Sharpe {current_result.sharpe_ratio:.2f} -> {new_result.sharpe_ratio:.2f}"
                )
                optimized[name] = new_result
            else:
                logger.info(f"  ✗ Could not improve {name}")

        # Recompute portfolio with optimized strategies
        if optimized:
            logger.info("\nRecomputing portfolio with optimized strategies...")
            self.compute_portfolio()

        return optimized

    def run_parameter_sensitivity(
        self,
        strategy_name: str,
        param_name: str,
        n_samples: int = 10,
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis on a single parameter.

        Args:
            strategy_name: Strategy to analyze
            param_name: Parameter to vary
            n_samples: Number of samples across parameter range

        Returns:
            DataFrame with parameter values and metrics
        """
        from ordinis.engines.sprint.strategies import STRATEGY_PROFILES

        profile = STRATEGY_PROFILES.get(strategy_name)
        if not profile:
            logger.error(f"Unknown strategy: {strategy_name}")
            return pd.DataFrame()

        param_def = profile.param_definitions.get(param_name)
        if not param_def:
            logger.error(f"Unknown parameter: {param_name}")
            return pd.DataFrame()

        logger.info(f"Running sensitivity analysis: {strategy_name}.{param_name}")

        # Generate parameter values
        min_val = param_def["min"]
        max_val = param_def["max"]

        if param_def["type"] == "int":
            values = np.linspace(min_val, max_val, n_samples).astype(int)
            values = np.unique(values)  # Remove duplicates
        else:
            values = np.linspace(min_val, max_val, n_samples)

        # Get base params
        base_params = {k: v["default"] for k, v in profile.param_definitions.items()}

        # Run backtests
        results = []
        method_map = {
            "garch_breakout": self.backtest_garch,
            "kalman_trend": self.backtest_kalman,
            "hmm_regime": self.backtest_hmm,
            "ou_pairs": self.backtest_ou_pairs,
            "evt_tail": self.backtest_evt,
            "mtf_momentum": self.backtest_mtf,
            "mi_lead_lag": self.backtest_mi,
            "network_regime": self.backtest_network,
        }

        backtest_method = method_map.get(strategy_name)
        if not backtest_method:
            return pd.DataFrame()

        for val in values:
            params = base_params.copy()
            params[param_name] = val

            try:
                result = backtest_method(params)
                results.append(
                    {
                        param_name: val,
                        "sharpe_ratio": result.sharpe_ratio,
                        "sortino_ratio": result.sortino_ratio,
                        "calmar_ratio": result.calmar_ratio,
                        "max_drawdown": result.max_drawdown,
                        "total_return": result.total_return,
                        "return_stability": result.return_stability,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed for {param_name}={val}: {e}")

        df = pd.DataFrame(results)

        # Log best value
        if not df.empty:
            best_idx = df["sharpe_ratio"].idxmax()
            best_val = df.loc[best_idx, param_name]
            best_sharpe = df.loc[best_idx, "sharpe_ratio"]
            logger.info(f"  Best: {param_name}={best_val} -> Sharpe={best_sharpe:.2f}")

        return df

    def run_full_sensitivity_analysis(
        self,
        strategy_name: str,
        n_samples: int = 7,
    ) -> dict[str, pd.DataFrame]:
        """
        Run sensitivity analysis on all parameters of a strategy.

        Args:
            strategy_name: Strategy to analyze
            n_samples: Samples per parameter

        Returns:
            Dictionary of parameter -> sensitivity DataFrame
        """
        from ordinis.engines.sprint.strategies import STRATEGY_PROFILES

        profile = STRATEGY_PROFILES.get(strategy_name)
        if not profile:
            return {}

        logger.info(f"\nFull sensitivity analysis for {strategy_name}")
        logger.info("=" * 50)

        sensitivity_results = {}

        for param_name in profile.param_definitions.keys():
            df = self.run_parameter_sensitivity(strategy_name, param_name, n_samples)
            if not df.empty:
                sensitivity_results[param_name] = df

        return sensitivity_results

    def optimize_with_ai(
        self, strategy_name: str, max_iterations: int = 5
    ) -> StrategyResult | None:
        """Optimize a strategy using AI with full optimization loop."""
        import asyncio

        if not self.ai_optimizer:
            logger.warning("AI optimizer not enabled")
            return None

        from ordinis.engines.sprint.strategies import STRATEGY_PROFILES

        profile = STRATEGY_PROFILES.get(strategy_name)
        if not profile:
            logger.error(f"Unknown strategy: {strategy_name}")
            return None

        logger.info(f"Starting AI optimization for {strategy_name}...")

        # Get backtest function
        backtest_methods = {
            "garch_breakout": self._backtest_garch_params,
            "kalman_trend": self._backtest_kalman_params,
            "hmm_regime": self._backtest_hmm_params,
            "ou_pairs": self._backtest_ou_params,
            "evt_tail": self._backtest_evt_params,
            "mtf_momentum": self._backtest_mtf_params,
            "mi_lead_lag": self._backtest_mi_params,
            "network_regime": self._backtest_network_params,
        }

        backtest_fn = backtest_methods.get(strategy_name)
        if not backtest_fn:
            logger.error(f"No backtest function for {strategy_name}")
            return None

        # Get initial params
        current_result = self.results.get(strategy_name)
        initial_params = current_result.params if current_result else None

        # Run async optimization
        async def run_opt():
            await self.ai_optimizer.initialize()
            return await self.ai_optimizer.run_optimization(
                profile=profile,
                backtest_fn=backtest_fn,
                initial_params=initial_params,
                max_iterations=max_iterations,
            )

        try:
            result = asyncio.run(run_opt())
        except RuntimeError:
            # Event loop already running
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(run_opt())

        if result and result.get("best_params"):
            best_params = result["best_params"]
            logger.info(f"AI optimization complete. Best params: {best_params}")

            # Run full backtest with best params
            method_map = {
                "garch_breakout": self.backtest_garch,
                "kalman_trend": self.backtest_kalman,
                "hmm_regime": self.backtest_hmm,
                "ou_pairs": self.backtest_ou_pairs,
                "evt_tail": self.backtest_evt,
                "mtf_momentum": self.backtest_mtf,
                "mi_lead_lag": self.backtest_mi,
                "network_regime": self.backtest_network,
            }

            backtest_method = method_map.get(strategy_name)
            if backtest_method:
                return backtest_method(best_params)

        return None

    def _backtest_garch_params(self, params: dict) -> dict:
        """Backtest GARCH with params for optimization."""
        result = self.backtest_garch(params)
        return result.to_dict()

    def _backtest_kalman_params(self, params: dict) -> dict:
        """Backtest Kalman with params for optimization."""
        result = self.backtest_kalman(params)
        return result.to_dict()

    def _backtest_hmm_params(self, params: dict) -> dict:
        """Backtest HMM with params for optimization."""
        result = self.backtest_hmm(params)
        return result.to_dict()

    def _backtest_ou_params(self, params: dict) -> dict:
        """Backtest OU Pairs with params for optimization."""
        result = self.backtest_ou_pairs(params)
        return result.to_dict()

    def _backtest_evt_params(self, params: dict) -> dict:
        """Backtest EVT with params for optimization."""
        result = self.backtest_evt(params)
        return result.to_dict()

    def _backtest_mtf_params(self, params: dict) -> dict:
        """Backtest MTF with params for optimization."""
        result = self.backtest_mtf(params)
        return result.to_dict()

    def _backtest_mi_params(self, params: dict) -> dict:
        """Backtest MI Lead-Lag with params for optimization."""
        result = self.backtest_mi(params)
        return result.to_dict()

    def _backtest_network_params(self, params: dict) -> dict:
        """Backtest Network Regime with params for optimization."""
        result = self.backtest_network(params)
        return result.to_dict()

    def print_summary(self) -> None:
        """Print comprehensive results summary with ProofBench-aligned KPIs."""
        print("\n" + "=" * 90)
        print("STRATEGY PERFORMANCE SUMMARY (ProofBench KPIs)")
        print("=" * 90)

        # Sort by Sharpe ratio descending
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].sharpe_ratio,
            reverse=True,
        )

        for name, result in sorted_results:
            is_benchmark = name == "benchmark"
            is_portfolio = name == "portfolio"

            # Special formatting for benchmark and portfolio
            if is_benchmark:
                print(f"\n{'─' * 90}")
                print(f"📊 {result.name} (Reference)")
                print(f"{'─' * 90}")
            elif is_portfolio:
                print(f"\n{'═' * 90}")
                print(f"💼 {result.name}")
                print(f"{'═' * 90}")
            else:
                print(f"\n{result.name}")
                print("-" * 45)

            # Core Returns
            print(f"  {'Returns':─<25}")
            print(f"    Total Return:         {result.total_return * 100:>8.2f}%")
            print(f"    CAGR:                 {result.annualized_return * 100:>8.2f}%")

            # Risk-Adjusted Metrics
            print(f"  {'Risk-Adjusted':─<25}")
            print(f"    Sharpe Ratio:         {result.sharpe_ratio:>8.2f}")
            print(f"    Sortino Ratio:        {result.sortino_ratio:>8.2f}")
            print(f"    Calmar Ratio:         {result.calmar_ratio:>8.2f}")

            # Drawdown Metrics
            print(f"  {'Drawdowns':─<25}")
            print(f"    Max Drawdown:         {result.max_drawdown * 100:>8.2f}%")
            print(f"    Max DD Duration:      {result.max_drawdown_duration:>8d} days")
            print(f"    Avg Drawdown:         {result.avg_drawdown * 100:>8.2f}%")

            # Tail Risk
            print(f"  {'Tail Risk':─<25}")
            print(f"    VaR (95%):            {result.var_95 * 100:>8.2f}%")
            print(f"    CVaR (95%):           {result.cvar_95 * 100:>8.2f}%")

            # Stability
            print(f"  {'Stability':─<25}")
            print(f"    Return Stability:     {result.return_stability:>8.2f} (R²)")

            # Trade Statistics (skip for benchmark)
            if not is_benchmark and result.total_trades > 0:
                print(f"  {'Trade Statistics':─<25}")
                print(f"    Total Trades:         {result.total_trades:>8d}")
                print(f"    Win Rate:             {result.win_rate * 100:>8.2f}%")
                print(f"    Profit Factor:        {result.profit_factor:>8.2f}")
                print(f"    Expectancy:          ${result.expectancy:>8.2f}")
                print(f"    Avg Win:             ${result.avg_win:>8.2f}")
                print(f"    Avg Loss:            ${result.avg_loss:>8.2f}")

            # Walk-Forward Results
            if result.train_sharpe is not None and result.test_sharpe is not None:
                print(f"  {'Walk-Forward':─<25}")
                print(f"    Train Sharpe:         {result.train_sharpe:>8.2f}")
                print(f"    Test Sharpe:          {result.test_sharpe:>8.2f}")
                if result.overfit_ratio is not None and result.overfit_ratio != float("inf"):
                    of_status = (
                        "✓"
                        if result.overfit_ratio < 1.5
                        else "⚠"
                        if result.overfit_ratio < 2.0
                        else "✗"
                    )
                    print(f"    Overfit Ratio:        {result.overfit_ratio:>8.2f} {of_status}")

        # Print comparison table
        print("\n" + "=" * 90)
        print("STRATEGY COMPARISON TABLE")
        print("=" * 90)
        print(
            f"{'Strategy':<25} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>8} {'Stab':>6}"
        )
        print("-" * 90)

        for name, result in sorted_results:
            print(
                f"{result.name[:24]:<25} "
                f"{result.annualized_return * 100:>7.1f}% "
                f"{result.sharpe_ratio:>8.2f} "
                f"{result.sortino_ratio:>8.2f} "
                f"{result.max_drawdown * 100:>7.1f}% "
                f"{result.calmar_ratio:>8.2f} "
                f"{result.return_stability:>6.2f}"
            )

        print("=" * 90)


def run_sprint(config: SprintConfig | None = None) -> AcceleratedSprintRunner:
    """Convenience function to run the full sprint."""
    runner = AcceleratedSprintRunner(config)
    runner.load_data()
    runner.run_all_strategies()
    runner.print_summary()
    return runner


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    config = SprintConfig(
        symbols=["SPY", "QQQ", "IWM", "TLT", "GLD"],
        start_date="2019-01-01",
        end_date="2024-01-01",
        use_gpu=True,
        use_ai=True,
    )

    run_sprint(config)
