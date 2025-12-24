#!/usr/bin/env python3
"""
Real Strategy Backtest Runner - Uses ACTUAL ML Models.

This script runs backtests using the REAL model implementations from
signalcore, not simplified stubs. Includes S&P 500 benchmark comparison.

Key differences from run_gtm_backtests.py:
1. Uses actual Model classes (ATROptimizedRSIModel, MIEnsembleStrategy, etc.)
2. Compares performance to S&P 500 benchmark (SPY)
3. Calculates alpha/beta vs benchmark
4. Requires minimum 2-year data for valid statistical tests

Usage:
    conda activate ordinis-env
    python scripts/run_real_strategy_backtests.py --gpu --days 500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    torch = None

# Import REAL models
from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.engines.signalcore.models.atr_optimized_rsi import ATROptimizedRSIModel
from ordinis.engines.signalcore.models.mi_ensemble import MIEnsembleModel
from ordinis.engines.signalcore.models.mtf_momentum import MTFMomentumModel
from ordinis.engines.signalcore.models.kalman_hybrid import KalmanHybridModel

# Validation harness

logger = logging.getLogger(__name__)

# Real Massive symbols with sufficient data
MASSIVE_SYMBOLS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "NVDA", "META",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "TMO",
    # Consumer
    "WMT", "HD", "NKE", "MCD", "SBUX",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB",
]

# S&P 500 proxy (benchmark)
BENCHMARK_SYMBOL = "SPY"


@dataclass
class BacktestConfig:
    """Configuration for real model backtests."""
    
    use_gpu: bool = True
    device_id: int = 0
    days: int = 500  # Minimum 2 years for statistical validity
    initial_capital: float = 100_000.0
    spread_bps: float = 3.0  # 3 bps bid-ask
    impact_bps: float = 5.0  # 5 bps market impact
    risk_free_rate: float = 0.045  # Current T-bill rate
    
    # Walk-forward
    train_days: int = 252  # 1 year training
    test_days: int = 63    # 3 months test
    step_days: int = 21    # Monthly steps
    
    # Position sizing
    max_position_pct: float = 0.05  # 5% per position
    max_sector_pct: float = 0.25   # 25% per sector


@dataclass
class StrategyResult:
    """Results from a real strategy backtest."""
    
    strategy_name: str
    symbol: str
    
    # Returns
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    benchmark_return_pct: float = 0.0
    benchmark_cagr_pct: float = 0.0
    
    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Alpha/Beta
    alpha_pct: float = 0.0  # Annualized alpha
    beta: float = 0.0
    information_ratio: float = 0.0
    
    # Risk
    max_drawdown_pct: float = 0.0
    volatility_pct: float = 0.0
    downside_volatility_pct: float = 0.0
    
    # Trading
    n_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return_pct: float = 0.0
    
    # Costs
    total_costs: float = 0.0
    cost_drag_pct: float = 0.0
    
    # Statistical
    t_stat: float = 0.0  # vs benchmark
    p_value: float = 0.0
    
    # Time
    start_date: str = ""
    end_date: str = ""
    n_days: int = 0


@dataclass
class AggregateResult:
    """Aggregate results across all symbols for a strategy."""
    
    strategy_name: str
    n_symbols: int = 0
    
    # Aggregate returns
    portfolio_return_pct: float = 0.0
    portfolio_cagr_pct: float = 0.0
    benchmark_cagr_pct: float = 0.0
    excess_cagr_pct: float = 0.0
    
    # Aggregate risk-adjusted
    avg_sharpe: float = 0.0
    median_sharpe: float = 0.0
    std_sharpe: float = 0.0
    
    # Aggregate alpha
    avg_alpha_pct: float = 0.0
    avg_beta: float = 0.0
    avg_ir: float = 0.0
    
    # Risk
    avg_max_dd_pct: float = 0.0
    worst_max_dd_pct: float = 0.0
    
    # Trading
    total_trades: int = 0
    avg_win_rate: float = 0.0
    avg_profit_factor: float = 0.0
    
    # Bootstrap CI
    sharpe_ci_lower: float = 0.0
    sharpe_ci_upper: float = 0.0
    alpha_ci_lower: float = 0.0
    alpha_ci_upper: float = 0.0
    
    # Pass/Fail
    beats_benchmark: bool = False
    sharpe_significant: bool = False
    alpha_significant: bool = False
    
    # Individual results
    per_symbol_results: list[StrategyResult] = field(default_factory=list)


class RealBacktestRunner:
    """Runs backtests using actual ML model implementations."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.gpu_available = self._check_gpu()
        self.benchmark_data: pd.DataFrame | None = None
        
        # Initialize models
        self.models = self._initialize_models()
        
    def _check_gpu(self) -> bool:
        """Check GPU availability."""
        if not self.config.use_gpu:
            return False
            
        if HAS_CUPY:
            try:
                cp.cuda.Device(self.config.device_id).use()
                gpu_name = cp.cuda.runtime.getDeviceProperties(self.config.device_id)["name"]
                logger.info(f"GPU enabled: {gpu_name.decode()}")
                return True
            except Exception as e:
                logger.warning(f"CuPy GPU failed: {e}")
                
        if HAS_TORCH and torch.cuda.is_available():
            logger.info(f"GPU enabled via PyTorch: {torch.cuda.get_device_name(0)}")
            return True
            
        return False
    
    def _initialize_models(self) -> dict:
        """Initialize all real strategy models with TUNED parameters."""
        models = {}
        
        # ATR-Optimized RSI - Tuned for more signals
        # Lower oversold threshold, faster RSI = more trades
        models["atr_rsi"] = ATROptimizedRSIModel(ModelConfig(
            model_id="atr_rsi_v1",
            model_type="atr_optimized_rsi",
            parameters={
                "rsi_period": 10,        # Faster RSI (was 14)
                "rsi_oversold": 40,      # Higher threshold = more signals (was 35)
                "rsi_overbought": 60,    # Lower threshold = more signals (was 65)
                "rsi_exit": 50,
                "atr_period": 14,
                "atr_stop_mult": 2.0,    # Wider stop (was 1.5)
                "atr_tp_mult": 3.0,      # Bigger targets (was 2.0)
                "use_optimized": True,
            },
        ))
        
        # MTF Momentum - FIXED PARAMETERS (use actual model params!)
        # Relaxed conditions for more signals
        models["mtf_momentum"] = MTFMomentumModel(ModelConfig(
            model_id="mtf_momentum_v1",
            model_type="mtf_momentum",
            parameters={
                "formation_period": 126,     # 6 months (was 252)
                "skip_period": 10,           # 2 weeks (was 21)
                "momentum_percentile": 0.6,  # Top 40% = winner (was 0.8)
                "stoch_k_period": 10,        # Faster stochastic (was 14)
                "stoch_d_period": 3,
                "stoch_oversold": 40.0,      # Higher (was 30)
                "stoch_overbought": 60.0,    # Lower (was 70)
                "atr_period": 14,
                "atr_stop_mult": 2.0,
                "atr_tp_mult": 3.0,
            },
        ))
        
        # MI Ensemble - FIXED PARAMETERS
        # Lower threshold, fewer signals need to agree
        models["mi_ensemble"] = MIEnsembleModel(ModelConfig(
            model_id="mi_ensemble_v1",
            model_type="mi_ensemble",
            parameters={
                "mi_lookback": 126,          # 6 months (was 252)
                "mi_bins": 8,                # Fewer bins for faster MI
                "forward_period": 5,
                "min_weight": 0.0,
                "max_weight": 0.4,           # Lower cap
                "recalc_frequency": 10,      # More frequent recalc
                "ensemble_threshold": 0.15,  # Lower threshold for signals (was 0.3)
                "min_signals_agree": 2,
                "atr_period": 14,
                "atr_stop_mult": 2.0,
                "atr_tp_mult": 3.0,
            },
        ))
        
        # Kalman Hybrid - FIXED PARAMETERS (correct param names!)
        # Lower z-score threshold, smaller trend slope requirement
        models["kalman_hybrid"] = KalmanHybridModel(ModelConfig(
            model_id="kalman_hybrid_v1",
            model_type="kalman_hybrid",
            parameters={
                "process_noise_q": 1e-4,      # More responsive (was 1e-5)
                "observation_noise_r": 1e-2,
                "residual_z_entry": 1.5,      # Lower threshold (was 2.0)
                "residual_z_exit": 0.3,       # Faster exit (was 0.5)
                "trend_slope_min": 0.00005,   # Smaller requirement (was 0.0001)
                "residual_lookback": 50,      # Shorter (was 100)
                "atr_period": 14,
                "atr_stop_mult": 2.0,
                "atr_tp_mult": 3.0,
            },
        ))
        
        return models
    
    def load_data(self, symbol: str, min_days: int = 252) -> pd.DataFrame | None:
        """Load real Massive data for a symbol."""
        data_dir = Path(__file__).parent.parent / "data" / "historical"
        
        # Try CSV first
        csv_path = data_dir / f"{symbol}_historical.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, parse_dates=["Date"])
                df = df.rename(columns={"Date": "date"})
                df = df.set_index("date")
                df.columns = df.columns.str.lower()
                
                if len(df) >= min_days:
                    logger.debug(f"Loaded {symbol}: {len(df)} days")
                    return df
                logger.warning(f"{symbol}: Only {len(df)} days (need {min_days})")
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
        
        # Try parquet
        parquet_path = data_dir / f"{symbol}.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                df.columns = df.columns.str.lower()
                
                if len(df) >= min_days:
                    logger.debug(f"Loaded {symbol}: {len(df)} days (parquet)")
                    return df
            except Exception as e:
                logger.error(f"Error loading {symbol} parquet: {e}")
        
        return None
    
    def load_benchmark(self, n_days: int) -> pd.DataFrame | None:
        """Load SPY as benchmark."""
        if self.benchmark_data is not None:
            return self.benchmark_data
            
        df = self.load_data(BENCHMARK_SYMBOL, min_days=n_days)
        if df is not None:
            # Ensure we have the right columns
            if "close" in df.columns:
                self.benchmark_data = df
                return df
        
        # Try loading from any available data
        logger.warning(f"No {BENCHMARK_SYMBOL} data, using market average as proxy")
        return None
    
    def compute_returns(self, prices: np.ndarray) -> np.ndarray:
        """Compute returns with GPU acceleration."""
        if self.gpu_available and HAS_CUPY:
            prices_gpu = cp.asarray(prices)
            returns_gpu = cp.diff(prices_gpu) / prices_gpu[:-1]
            return cp.asnumpy(returns_gpu)
        return np.diff(prices) / prices[:-1]
    
    def compute_alpha_beta(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        risk_free_rate: float = 0.045,
    ) -> tuple[float, float, float]:
        """Compute alpha, beta, and information ratio."""
        if len(strategy_returns) != len(benchmark_returns):
            min_len = min(len(strategy_returns), len(benchmark_returns))
            strategy_returns = strategy_returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]
        
        if len(strategy_returns) < 30:
            return 0.0, 1.0, 0.0
        
        # Beta via regression
        cov = np.cov(strategy_returns, benchmark_returns)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
        
        # Alpha (annualized)
        daily_rf = risk_free_rate / 252
        strategy_mean = np.mean(strategy_returns)
        benchmark_mean = np.mean(benchmark_returns)
        
        daily_alpha = strategy_mean - daily_rf - beta * (benchmark_mean - daily_rf)
        annual_alpha = daily_alpha * 252 * 100  # Percent
        
        # Information Ratio
        tracking_error = np.std(strategy_returns - benchmark_returns, ddof=1)
        ir = (strategy_mean - benchmark_mean) / tracking_error * np.sqrt(252) if tracking_error > 0 else 0.0
        
        return annual_alpha, beta, ir
    
    def compute_cagr(self, initial: float, final: float, years: float) -> float:
        """Compute CAGR."""
        if initial <= 0 or final <= 0 or years <= 0:
            return 0.0
        return ((final / initial) ** (1 / years) - 1) * 100
    
    async def generate_signals(
        self,
        model,
        symbol: str,
        data: pd.DataFrame,
    ) -> tuple[np.ndarray, int]:
        """Generate signals using the REAL model."""
        n = len(data)
        signals = np.zeros(n)
        positions = np.zeros(n)  # Track actual position
        trades = 0
        current_position = 0
        
        # Validate model can handle data
        is_valid, reason = model.validate(data)
        if not is_valid:
            logger.warning(f"{symbol}: {reason}")
            return signals, 0
        
        # Determine warmup period based on model requirements
        # MI Ensemble needs 252 + 5 + 50 = 307 bars
        # Other models need less, but we use min 100 for safety
        warmup = max(100, getattr(model.config, 'min_data_points', 100))
        if hasattr(model, 'mi_config'):
            warmup = max(warmup, model.mi_config.mi_lookback + model.mi_config.forward_period + 50)
        
        # Generate signals bar by bar (realistic simulation)
        for i in range(warmup, n):  # Start after model-specific warmup
            try:
                # Get data up to current bar (no lookahead)
                current_data = data.iloc[:i+1].copy()
                timestamp = data.index[i]
                
                # Generate signal using real model
                signal = await model.generate(symbol, current_data, timestamp)
                
                if signal is not None:
                    if signal.signal_type == SignalType.ENTRY:
                        if signal.direction == Direction.LONG and current_position <= 0:
                            signals[i] = 1.0
                            if current_position != 1:
                                trades += 1
                                current_position = 1
                        elif signal.direction == Direction.SHORT and current_position >= 0:
                            signals[i] = -1.0
                            if current_position != -1:
                                trades += 1
                                current_position = -1
                    elif signal.signal_type == SignalType.EXIT:
                        if current_position != 0:
                            trades += 1
                            current_position = 0
                        signals[i] = 0.0
                else:
                    # Maintain position
                    signals[i] = current_position
                    
                positions[i] = current_position
                
            except Exception as e:
                logger.debug(f"{symbol} bar {i}: {e}")
                signals[i] = current_position  # Maintain position on error
                positions[i] = current_position
        
        return signals, trades
    
    async def run_symbol_backtest(
        self,
        strategy_name: str,
        symbol: str,
        data: pd.DataFrame,
        benchmark_returns: np.ndarray | None,
    ) -> StrategyResult:
        """Run backtest for one strategy on one symbol."""
        model = self.models.get(strategy_name)
        if model is None:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Generate signals
        signals, n_trades = await self.generate_signals(model, symbol, data)
        
        # Calculate returns
        prices = data["close"].values
        returns = self.compute_returns(prices)
        
        # Align signals with returns (signals[t] applied to returns[t])
        signals_aligned = signals[:-1]  # Remove last signal (no forward return)
        strategy_returns = returns * signals_aligned
        
        # Apply transaction costs
        total_cost_bps = (self.config.spread_bps + self.config.impact_bps) * 2  # Round-trip
        total_costs = n_trades * total_cost_bps / 10000 * self.config.initial_capital
        
        net_returns = strategy_returns.copy()
        if n_trades > 0 and len(net_returns) > 0:
            cost_per_bar = total_costs / len(net_returns) / self.config.initial_capital
            net_returns = net_returns - cost_per_bar
        
        # Calculate equity curve
        equity = self.config.initial_capital * np.cumprod(1 + net_returns)
        benchmark_equity = self.config.initial_capital * np.cumprod(1 + benchmark_returns) if benchmark_returns is not None else None
        
        # Total return
        final_equity = equity[-1] if len(equity) > 0 else self.config.initial_capital
        total_return = (final_equity / self.config.initial_capital - 1) * 100
        
        # CAGR
        years = len(data) / 252
        cagr = self.compute_cagr(self.config.initial_capital, final_equity, years)
        
        # Benchmark metrics
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            benchmark_equity_final = self.config.initial_capital * np.prod(1 + benchmark_returns[:len(net_returns)])
            benchmark_return = (benchmark_equity_final / self.config.initial_capital - 1) * 100
            benchmark_cagr = self.compute_cagr(self.config.initial_capital, benchmark_equity_final, years)
        else:
            benchmark_return = 0.0
            benchmark_cagr = 0.0
        
        # Risk metrics
        if len(net_returns) > 1:
            volatility = np.std(net_returns, ddof=1) * np.sqrt(252) * 100
            
            # Sharpe
            daily_rf = self.config.risk_free_rate / 252
            excess_returns = net_returns - daily_rf
            sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0
            
            # Sortino
            downside_returns = net_returns[net_returns < 0]
            downside_vol = np.std(downside_returns, ddof=1) * np.sqrt(252) * 100 if len(downside_returns) > 0 else volatility
            sortino = np.mean(excess_returns) / (np.std(downside_returns, ddof=1)) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0.0
            
            # Max drawdown
            cummax = np.maximum.accumulate(equity)
            drawdown = (equity - cummax) / cummax
            max_dd = -np.min(drawdown) * 100
            
            # Calmar
            calmar = cagr / max_dd if max_dd > 0 else 0.0
        else:
            volatility = 0.0
            sharpe = 0.0
            sortino = 0.0
            max_dd = 0.0
            calmar = 0.0
            downside_vol = 0.0
        
        # Alpha/Beta
        if benchmark_returns is not None and len(benchmark_returns) >= len(net_returns):
            alpha, beta, ir = self.compute_alpha_beta(
                net_returns,
                benchmark_returns[:len(net_returns)],
                self.config.risk_free_rate,
            )
        else:
            alpha, beta, ir = 0.0, 1.0, 0.0
        
        # Trade statistics
        if n_trades > 0:
            trade_returns = strategy_returns[strategy_returns != 0]
            winning = np.sum(trade_returns > 0)
            win_rate = winning / len(trade_returns) * 100 if len(trade_returns) > 0 else 0.0
            
            gross_profit = np.sum(trade_returns[trade_returns > 0])
            gross_loss = abs(np.sum(trade_returns[trade_returns < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            
            avg_trade = np.mean(trade_returns) * 100 if len(trade_returns) > 0 else 0.0
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade = 0.0
        
        # Statistical significance (t-test vs benchmark)
        if benchmark_returns is not None and len(net_returns) > 30:
            from scipy import stats
            excess_vs_bench = net_returns - benchmark_returns[:len(net_returns)]
            t_stat, p_value = stats.ttest_1samp(excess_vs_bench, 0)
        else:
            t_stat, p_value = 0.0, 1.0
        
        cost_drag = (total_costs / final_equity * 100) if final_equity > 0 else 0.0
        
        return StrategyResult(
            strategy_name=strategy_name,
            symbol=symbol,
            total_return_pct=total_return,
            cagr_pct=cagr,
            benchmark_return_pct=benchmark_return,
            benchmark_cagr_pct=benchmark_cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            alpha_pct=alpha,
            beta=beta,
            information_ratio=ir,
            max_drawdown_pct=max_dd,
            volatility_pct=volatility,
            downside_volatility_pct=downside_vol,
            n_trades=n_trades,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            avg_trade_return_pct=avg_trade,
            total_costs=total_costs,
            cost_drag_pct=cost_drag,
            t_stat=t_stat,
            p_value=p_value,
            start_date=str(data.index[0].date()) if hasattr(data.index[0], "date") else str(data.index[0]),
            end_date=str(data.index[-1].date()) if hasattr(data.index[-1], "date") else str(data.index[-1]),
            n_days=len(data),
        )
    
    async def run_strategy_full(
        self,
        strategy_name: str,
        symbols: list[str],
    ) -> AggregateResult:
        """Run full backtest for a strategy across all symbols."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {strategy_name.upper()} with REAL model")
        logger.info(f"{'='*60}")
        
        # Load benchmark first
        benchmark_df = self.load_benchmark(self.config.days)
        benchmark_returns = None
        if benchmark_df is not None:
            benchmark_returns = self.compute_returns(benchmark_df["close"].values[-self.config.days:])
        
        results: list[StrategyResult] = []
        
        for symbol in symbols:
            data = self.load_data(symbol, min_days=self.config.days)
            if data is None:
                logger.warning(f"Skipping {symbol}: no data")
                continue
            
            # Trim to requested days
            if len(data) > self.config.days:
                data = data.iloc[-self.config.days:]
            
            logger.info(f"  {symbol}: {len(data)} days ({data.index[0].date()} to {data.index[-1].date()})")
            
            result = await self.run_symbol_backtest(
                strategy_name,
                symbol,
                data,
                benchmark_returns,
            )
            results.append(result)
            
            # Log per-symbol result
            status = "✅" if result.sharpe_ratio > 1.0 and result.cagr_pct > result.benchmark_cagr_pct else "❌"
            logger.info(f"    {status} CAGR: {result.cagr_pct:.1f}% | Sharpe: {result.sharpe_ratio:.2f} | Alpha: {result.alpha_pct:.1f}% | Trades: {result.n_trades}")
        
        if not results:
            return AggregateResult(strategy_name=strategy_name)
        
        # Aggregate metrics
        sharpes = [r.sharpe_ratio for r in results]
        alphas = [r.alpha_pct for r in results]
        cagrs = [r.cagr_pct for r in results]
        benchmark_cagrs = [r.benchmark_cagr_pct for r in results]
        
        # Portfolio return (equal-weight)
        portfolio_cagr = np.mean(cagrs)
        benchmark_cagr = np.mean(benchmark_cagrs) if benchmark_cagrs else 0.0
        
        # Bootstrap CI for Sharpe
        sharpe_samples = np.array(sharpes)
        rng = np.random.default_rng(42)
        bootstrap_sharpes = [np.mean(rng.choice(sharpe_samples, size=len(sharpe_samples), replace=True)) for _ in range(1000)]
        sharpe_ci_lower = np.percentile(bootstrap_sharpes, 2.5)
        sharpe_ci_upper = np.percentile(bootstrap_sharpes, 97.5)
        
        # Bootstrap CI for Alpha
        alpha_samples = np.array(alphas)
        bootstrap_alphas = [np.mean(rng.choice(alpha_samples, size=len(alpha_samples), replace=True)) for _ in range(1000)]
        alpha_ci_lower = np.percentile(bootstrap_alphas, 2.5)
        alpha_ci_upper = np.percentile(bootstrap_alphas, 97.5)
        
        return AggregateResult(
            strategy_name=strategy_name,
            n_symbols=len(results),
            portfolio_return_pct=np.mean([r.total_return_pct for r in results]),
            portfolio_cagr_pct=portfolio_cagr,
            benchmark_cagr_pct=benchmark_cagr,
            excess_cagr_pct=portfolio_cagr - benchmark_cagr,
            avg_sharpe=np.mean(sharpes),
            median_sharpe=np.median(sharpes),
            std_sharpe=np.std(sharpes),
            avg_alpha_pct=np.mean(alphas),
            avg_beta=np.mean([r.beta for r in results]),
            avg_ir=np.mean([r.information_ratio for r in results]),
            avg_max_dd_pct=np.mean([r.max_drawdown_pct for r in results]),
            worst_max_dd_pct=np.max([r.max_drawdown_pct for r in results]),
            total_trades=sum(r.n_trades for r in results),
            avg_win_rate=np.mean([r.win_rate_pct for r in results if r.n_trades > 0]),
            avg_profit_factor=np.mean([r.profit_factor for r in results if r.profit_factor > 0 and r.profit_factor < 100]),
            sharpe_ci_lower=sharpe_ci_lower,
            sharpe_ci_upper=sharpe_ci_upper,
            alpha_ci_lower=alpha_ci_lower,
            alpha_ci_upper=alpha_ci_upper,
            beats_benchmark=portfolio_cagr > benchmark_cagr,
            sharpe_significant=sharpe_ci_lower > 0,
            alpha_significant=alpha_ci_lower > 0,
            per_symbol_results=results,
        )


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real Strategy Backtest Runner")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--days", type=int, default=500, help="Days of history to use")
    parser.add_argument("--strategies", type=str, default="all", help="Comma-separated strategy list")
    parser.add_argument("--symbols", type=str, default="massive", help="Symbol list or 'massive' for all")
    parser.add_argument("--output", type=str, default="data/backtest_results", help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    
    config = BacktestConfig(
        use_gpu=args.gpu,
        days=args.days,
    )
    
    runner = RealBacktestRunner(config)
    
    # Select strategies
    all_strategies = ["atr_rsi", "mtf_momentum", "mi_ensemble", "kalman_hybrid"]
    strategies = all_strategies if args.strategies == "all" else args.strategies.split(",")
    
    # Select symbols
    symbols = MASSIVE_SYMBOLS if args.symbols == "massive" else args.symbols.split(",")
    
    logger.info("="*70)
    logger.info("REAL STRATEGY BACKTEST - Using Actual ML Models")
    logger.info("="*70)
    logger.info(f"GPU: {'Enabled' if runner.gpu_available else 'Disabled'}")
    logger.info(f"Days: {config.days}")
    logger.info(f"Strategies: {strategies}")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Benchmark: {BENCHMARK_SYMBOL}")
    logger.info("="*70)
    
    all_results: dict[str, AggregateResult] = {}
    
    for strategy in strategies:
        result = await runner.run_strategy_full(strategy, symbols)
        all_results[strategy] = result
    
    # Print summary
    print("\n" + "="*80)
    print("STRATEGY COMPARISON - Real Models vs S&P 500 Benchmark")
    print("="*80)
    print(f"{'Strategy':<15} {'CAGR':>8} {'Bench':>8} {'Excess':>8} {'Sharpe':>8} {'Alpha':>8} {'MaxDD':>8} {'Trades':>8}")
    print("-"*80)
    
    best_strategy = None
    best_excess = -float("inf")
    
    for name, result in all_results.items():
        status = "✅" if result.beats_benchmark and result.sharpe_significant else "❌"
        print(f"{name:<15} {result.portfolio_cagr_pct:>7.1f}% {result.benchmark_cagr_pct:>7.1f}% "
              f"{result.excess_cagr_pct:>+7.1f}% {result.avg_sharpe:>8.2f} {result.avg_alpha_pct:>+7.1f}% "
              f"{result.avg_max_dd_pct:>7.1f}% {result.total_trades:>8} {status}")
        
        if result.excess_cagr_pct > best_excess:
            best_excess = result.excess_cagr_pct
            best_strategy = name
    
    print("="*80)
    print(f"\n🏆 BEST GTM CANDIDATE: {best_strategy.upper() if best_strategy else 'NONE'}")
    print(f"   Excess CAGR vs Benchmark: {best_excess:+.1f}%")
    print()
    
    # Detailed results
    for name, result in all_results.items():
        print(f"\n--- {name.upper()} ---")
        print(f"Portfolio CAGR: {result.portfolio_cagr_pct:.2f}%")
        print(f"Benchmark CAGR: {result.benchmark_cagr_pct:.2f}%")
        print(f"Excess CAGR: {result.excess_cagr_pct:+.2f}%")
        print(f"Sharpe: {result.avg_sharpe:.3f} [{result.sharpe_ci_lower:.3f}, {result.sharpe_ci_upper:.3f}]")
        print(f"Alpha: {result.avg_alpha_pct:.2f}% [{result.alpha_ci_lower:.2f}%, {result.alpha_ci_upper:.2f}%]")
        print(f"Beta: {result.avg_beta:.2f}")
        print(f"Max DD: {result.worst_max_dd_pct:.1f}%")
        print(f"Win Rate: {result.avg_win_rate:.1f}%")
        print(f"Profit Factor: {result.avg_profit_factor:.2f}")
        print(f"Status: {'✅ PASS' if result.beats_benchmark and result.sharpe_significant else '❌ FAIL'}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"real_backtest_{timestamp}.json"
    
    results_dict = {
        "timestamp": timestamp,
        "config": {
            "days": config.days,
            "gpu": runner.gpu_available,
            "symbols": symbols,
            "benchmark": BENCHMARK_SYMBOL,
            "cost_bps": config.spread_bps + config.impact_bps,
        },
        "results": {},
    }
    
    for name, result in all_results.items():
        results_dict["results"][name] = {
            "n_symbols": result.n_symbols,
            "portfolio_cagr_pct": float(result.portfolio_cagr_pct),
            "benchmark_cagr_pct": float(result.benchmark_cagr_pct),
            "excess_cagr_pct": float(result.excess_cagr_pct),
            "avg_sharpe": float(result.avg_sharpe),
            "sharpe_ci_lower": float(result.sharpe_ci_lower),
            "sharpe_ci_upper": float(result.sharpe_ci_upper),
            "avg_alpha_pct": float(result.avg_alpha_pct),
            "alpha_ci_lower": float(result.alpha_ci_lower),
            "alpha_ci_upper": float(result.alpha_ci_upper),
            "avg_max_dd_pct": float(result.avg_max_dd_pct),
            "total_trades": result.total_trades,
            "beats_benchmark": bool(result.beats_benchmark),
            "sharpe_significant": bool(result.sharpe_significant),
            "alpha_significant": bool(result.alpha_significant),
        }
    
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
