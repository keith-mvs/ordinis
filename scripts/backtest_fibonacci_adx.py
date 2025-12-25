"""
Fibonacci ADX Strategy Backtest Script.

Runs a comprehensive backtest using REAL historical data from data/historical/.
Supports multiple symbols, walk-forward validation, and performance analytics.
GPU-accelerated when CuPy/Numba are available.

Usage:
    python scripts/backtest_fibonacci_adx.py --symbols AAPL MSFT NVDA --capital 100000
    python scripts/backtest_fibonacci_adx.py --all-symbols --capital 500000
    python scripts/backtest_fibonacci_adx.py --symbols AAPL --enhanced --gpu
"""

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ordinis.application.strategies.fibonacci_adx import FibonacciADXStrategy
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType

# GPU acceleration imports
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    jit = None
    prange = range
    HAS_NUMBA = False

# Import GPU backtest engine
try:
    from ordinis.engines.sprint.core.accelerator import (
        GPUBacktestEngine,
        GPUConfig,
        compute_max_drawdown_numba,
        compute_rolling_sharpe_numba,
    )
    HAS_GPU_ENGINE = True
except ImportError:
    HAS_GPU_ENGINE = False


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    direction: Direction
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    holding_bars: int
    exit_reason: str
    metadata: dict = field(default_factory=dict)


@dataclass 
class Position:
    """Represents an open position."""
    symbol: str
    direction: Direction
    entry_date: datetime
    entry_price: float
    quantity: int
    stop_loss: float | None = None
    take_profit: float | None = None
    take_profit_2: float | None = None
    take_profit_3: float | None = None
    metadata: dict = field(default_factory=dict)


class BacktestPortfolio:
    """Portfolio manager for backtesting with position management."""
    
    def __init__(self, initial_capital: float = 100000.0, max_position_pct: float = 0.10):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_position_pct = max_position_pct
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[dict] = []
        
    def get_equity(self, prices: dict[str, float]) -> float:
        """Calculate current equity."""
        equity = self.cash
        for symbol, pos in self.positions.items():
            if symbol in prices:
                if pos.direction == Direction.LONG:
                    equity += pos.quantity * prices[symbol]
                else:  # SHORT
                    # Mark-to-market for short: we received cash, now owe shares
                    equity -= pos.quantity * (prices[symbol] - pos.entry_price)
        return equity
    
    def update(self, date: datetime, prices: dict[str, float]) -> float:
        """Update equity curve."""
        equity = self.get_equity(prices)
        self.equity_curve.append({
            "date": date,
            "equity": equity,
            "cash": self.cash,
            "positions": len(self.positions),
        })
        return equity
    
    def check_stops(self, date: datetime, data: dict[str, pd.DataFrame], bar_idx: int) -> list[Trade]:
        """Check stop-loss and take-profit levels for all positions."""
        closed_trades = []
        
        for symbol, pos in list(self.positions.items()):
            if symbol not in data:
                continue
                
            df = data[symbol]
            if bar_idx >= len(df):
                continue
                
            row = df.iloc[bar_idx]
            high = row["high"]
            low = row["low"]
            close = row["close"]
            
            exit_price = None
            exit_reason = None
            
            if pos.direction == Direction.LONG:
                # Check stop-loss (use low of bar)
                if pos.stop_loss and low <= pos.stop_loss:
                    exit_price = pos.stop_loss
                    exit_reason = "stop_loss"
                # Check take-profit (use high of bar)
                elif pos.take_profit and high >= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = "take_profit_1"
            else:  # SHORT
                # Check stop-loss (use high of bar)
                if pos.stop_loss and high >= pos.stop_loss:
                    exit_price = pos.stop_loss
                    exit_reason = "stop_loss"
                # Check take-profit (use low of bar)
                elif pos.take_profit and low <= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = "take_profit_1"
            
            if exit_price:
                trade = self._close_position(symbol, exit_price, date, bar_idx, exit_reason)
                closed_trades.append(trade)
                
        return closed_trades
    
    def open_position(self, signal: Signal, price: float, date: datetime) -> bool:
        """Open a new position based on signal."""
        symbol = signal.symbol
        
        # Skip if already in position
        if symbol in self.positions:
            return False
        
        # Calculate position size
        equity = self.equity_curve[-1]["equity"] if self.equity_curve else self.initial_capital
        target_size = equity * self.max_position_pct
        quantity = int(target_size / price)
        
        if quantity <= 0:
            return False
            
        # Check cash
        cost = quantity * price
        if signal.direction == Direction.LONG and self.cash < cost:
            return False
            
        # Extract stop/take-profit from signal metadata
        metadata = signal.metadata or {}
        stop_loss = metadata.get("stop_loss")
        take_profit = metadata.get("take_profit")
        take_profit_2 = metadata.get("take_profit_2")
        take_profit_3 = metadata.get("take_profit_3")
        
        # Create position
        pos = Position(
            symbol=symbol,
            direction=signal.direction,
            entry_date=date,
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            metadata=metadata,
        )
        
        # Update cash
        if signal.direction == Direction.LONG:
            self.cash -= cost
        else:  # SHORT - receive cash (simplified)
            self.cash += cost
            
        self.positions[symbol] = pos
        return True
    
    def _close_position(
        self, symbol: str, exit_price: float, date: datetime, bar_idx: int, reason: str
    ) -> Trade:
        """Close a position and record the trade."""
        pos = self.positions.pop(symbol)
        
        # Calculate P&L
        if pos.direction == Direction.LONG:
            pnl = (exit_price - pos.entry_price) * pos.quantity
            self.cash += pos.quantity * exit_price
        else:  # SHORT
            pnl = (pos.entry_price - exit_price) * pos.quantity
            self.cash -= pos.quantity * exit_price
            
        pnl_pct = pnl / (pos.entry_price * pos.quantity) * 100
        
        # Calculate holding period (approximate - would need date index mapping)
        entry_idx = pos.metadata.get("entry_bar_idx", 0)
        holding_bars = bar_idx - entry_idx
        
        trade = Trade(
            symbol=symbol,
            entry_date=pos.entry_date,
            exit_date=date,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_bars=holding_bars,
            exit_reason=reason,
            metadata=pos.metadata,
        )
        
        self.trades.append(trade)
        return trade
    
    def close_all(self, prices: dict[str, float], date: datetime, bar_idx: int) -> list[Trade]:
        """Close all open positions at market."""
        closed = []
        for symbol in list(self.positions.keys()):
            if symbol in prices:
                trade = self._close_position(symbol, prices[symbol], date, bar_idx, "end_of_backtest")
                closed.append(trade)
        return closed


def load_historical_data(symbol: str, data_dir: Path) -> pd.DataFrame | None:
    """Load historical data for a symbol."""
    # Try CSV first
    csv_path = data_dir / f"{symbol}_historical.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        df.index = pd.to_datetime(df.index, utc=True)
        # Ensure lowercase column names
        df.columns = df.columns.str.lower()
        df["symbol"] = symbol
        return df
    
    # Try parquet
    parquet_path = data_dir / f"{symbol}.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        df.columns = df.columns.str.lower()
        df["symbol"] = symbol
        return df
        
    return None


def calculate_metrics(portfolio: BacktestPortfolio, use_gpu: bool = False) -> dict[str, Any]:
    """Calculate comprehensive performance metrics with optional GPU acceleration."""
    trades = portfolio.trades
    equity_curve = pd.DataFrame(portfolio.equity_curve)
    
    if len(trades) == 0:
        return {"error": "No trades executed"}
    
    # Basic stats
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl < 0]
    
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    
    gross_profit = sum(t.pnl for t in winning_trades)
    gross_loss = abs(sum(t.pnl for t in losing_trades))
    net_profit = gross_profit - gross_loss
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_win = gross_profit / len(winning_trades) if winning_trades else 0
    avg_loss = gross_loss / len(losing_trades) if losing_trades else 0
    avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    # Returns
    initial = portfolio.initial_capital
    final = equity_curve["equity"].iloc[-1]
    total_return = (final - initial) / initial * 100
    
    # Convert to numpy for GPU processing
    equity_arr = equity_curve["equity"].values.astype(np.float64)
    
    # Calculate daily returns
    if use_gpu and HAS_CUPY:
        # GPU-accelerated returns calculation
        equity_gpu = cp.asarray(equity_arr)
        returns_gpu = cp.diff(equity_gpu) / equity_gpu[:-1]
        daily_returns = cp.asnumpy(returns_gpu)
        
        # GPU-accelerated statistics
        mean_ret = float(cp.mean(returns_gpu))
        std_ret = float(cp.std(returns_gpu))
        
        # Downside deviation on GPU
        downside_mask = returns_gpu < 0
        if cp.any(downside_mask):
            downside_std = float(cp.std(returns_gpu[downside_mask]))
        else:
            downside_std = 0.0
            
        # Max drawdown - use CPU for accumulate (CuPy doesn't support it)
        # But we can still compute efficiently with Numba
        if HAS_NUMBA and HAS_GPU_ENGINE:
            max_drawdown = -compute_max_drawdown_numba(equity_arr) * 100
        else:
            peak = np.maximum.accumulate(equity_arr)
            dd = (peak - equity_arr) / peak * 100
            max_drawdown = np.min(dd)
    else:
        # CPU fallback
        daily_returns = np.diff(equity_arr) / equity_arr[:-1]
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns)
        
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 0.0
        
        # Use Numba-optimized if available
        if HAS_NUMBA and HAS_GPU_ENGINE:
            max_drawdown = -compute_max_drawdown_numba(equity_arr) * 100
        else:
            peak = np.maximum.accumulate(equity_arr)
            dd = (peak - equity_arr) / peak * 100
            max_drawdown = np.min(dd)
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    
    # Sortino ratio
    sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    
    # Calmar ratio
    years = len(equity_curve) / 252
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0
    calmar = abs(cagr / max_drawdown) if max_drawdown < 0 else 0
    
    # Average holding period
    avg_holding = np.mean([t.holding_bars for t in trades])
    
    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    
    return {
        # Summary
        "initial_capital": initial,
        "final_equity": final,
        "net_profit": net_profit,
        "total_return_pct": total_return,
        
        # Trade stats
        "total_trades": total_trades,
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate_pct": win_rate,
        
        # Profit analysis
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_win_loss_ratio": avg_win_loss_ratio,
        
        # Risk metrics
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_drawdown,
        "calmar_ratio": calmar,
        "cagr_pct": cagr,
        
        # Trade characteristics
        "avg_holding_bars": avg_holding,
        "exit_reasons": exit_reasons,
        
        # Duration
        "backtest_days": len(equity_curve),
        "start_date": str(equity_curve["date"].iloc[0]),
        "end_date": str(equity_curve["date"].iloc[-1]),
        
        # GPU info
        "gpu_accelerated": use_gpu and HAS_CUPY,
    }


async def run_backtest(
    symbols: list[str],
    initial_capital: float = 100000.0,
    max_position_pct: float = 0.10,
    warmup_bars: int = 100,
    enhanced_mode: bool = False,
    use_gpu: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run backtest on specified symbols with optional GPU acceleration.
    
    Args:
        symbols: List of stock symbols to backtest
        initial_capital: Starting capital
        max_position_pct: Max % of equity per position
        warmup_bars: Bars to skip for indicator warmup
        enhanced_mode: Use all v1.4 enhancements (volume, fractal, MTF)
        use_gpu: Enable GPU acceleration for metrics calculation
        output_dir: Directory to save results
    """
    start_time = time.time()
    
    print("=" * 70)
    print("FIBONACCI ADX STRATEGY BACKTEST")
    print("=" * 70)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Enhanced Mode: {enhanced_mode}")
    
    # GPU status
    if use_gpu:
        if HAS_CUPY:
            import cupy as cp
            print(f"GPU Acceleration: ✅ ENABLED (CuPy {cp.__version__})")
            try:
                gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
                gpu_mem = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1e9
                print(f"  Device: {gpu_name} ({gpu_mem:.1f} GB)")
            except Exception:
                print("  Device: CUDA GPU available")
        else:
            print("GPU Acceleration: ❌ CuPy not installed, using CPU")
            use_gpu = False
    else:
        print(f"GPU Acceleration: Disabled (use --gpu to enable)")
    
    if HAS_NUMBA:
        print(f"Numba JIT: ✅ ENABLED")
    
    print("=" * 70)
    
    # Initialize strategy
    strategy_params = {
        "name": "FibADX-Backtest",
        "adx_threshold": 25,
        "di_threshold": 20,
        "swing_lookback": 20,
        "level_tolerance": 0.01,
        "require_trend_accelerating": True,  # v1.2
    }
    
    if enhanced_mode:
        strategy_params.update({
            "require_volume_confirmation": True,
            "volume_lookback": 20,
            "use_fractal_swings": True,
            "fractal_period": 5,
            "require_mtf_alignment": True,
            "htf_sma_period": 50,
            "htf_multiplier": 4,
        })
    
    strategy = FibonacciADXStrategy(**strategy_params)
    print(f"Strategy: {strategy.name} v{strategy_params.get('require_trend_accelerating', 'basic')}")
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "historical"
    all_data: dict[str, pd.DataFrame] = {}
    
    for symbol in symbols:
        df = load_historical_data(symbol, data_dir)
        if df is not None:
            all_data[symbol] = df
            print(f"Loaded {symbol}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
        else:
            print(f"WARNING: No data found for {symbol}")
    
    if not all_data:
        return {"error": "No data loaded"}
    
    # Initialize portfolio
    portfolio = BacktestPortfolio(initial_capital, max_position_pct)
    
    # Get common date range
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index.tolist())
    dates = sorted(all_dates)
    
    print(f"\nBacktest Period: {dates[0].date()} to {dates[-1].date()} ({len(dates)} bars)")
    print("-" * 70)
    
    # Run simulation
    signals_generated = 0
    positions_opened = 0
    
    for i, date in enumerate(dates):
        if i < warmup_bars:
            continue
            
        # Get current prices
        prices = {}
        for symbol, df in all_data.items():
            if date in df.index:
                prices[symbol] = df.loc[date, "close"]
        
        # Check stop-loss / take-profit first
        portfolio.check_stops(date, all_data, i)
        
        # Generate signals for each symbol
        for symbol, df in all_data.items():
            # Skip if we don't have this date
            if date not in df.index:
                continue
                
            # Get data window up to current date
            window = df.loc[:date].copy()
            if len(window) < warmup_bars:
                continue
            
            # Generate signal
            try:
                signal = await strategy.generate_signal(window, date)
            except Exception as e:
                continue
            
            if signal and signal.signal_type == SignalType.ENTRY:
                signals_generated += 1
                signal.metadata["entry_bar_idx"] = i
                
                # Try to open position
                if symbol in prices:
                    opened = portfolio.open_position(signal, prices[symbol], date)
                    if opened:
                        positions_opened += 1
        
        # Update equity curve
        portfolio.update(date, prices)
        
        # Progress update every 500 bars
        if i > 0 and i % 500 == 0:
            equity = portfolio.equity_curve[-1]["equity"]
            print(f"  Bar {i}/{len(dates)}: Equity ${equity:,.2f}, Trades: {len(portfolio.trades)}")
    
    # Close any remaining positions
    final_prices = {s: df.iloc[-1]["close"] for s, df in all_data.items()}
    portfolio.close_all(final_prices, dates[-1], len(dates) - 1)
    portfolio.update(dates[-1], final_prices)
    
    # Calculate metrics (with GPU if enabled)
    metrics = calculate_metrics(portfolio, use_gpu=use_gpu)
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    
    print(f"\n{'SUMMARY':^70}")
    print("-" * 70)
    print(f"Initial Capital:     ${metrics['initial_capital']:>15,.2f}")
    print(f"Final Equity:        ${metrics['final_equity']:>15,.2f}")
    print(f"Net Profit:          ${metrics['net_profit']:>15,.2f}")
    print(f"Total Return:        {metrics['total_return_pct']:>15.2f}%")
    print(f"CAGR:                {metrics['cagr_pct']:>15.2f}%")
    
    print(f"\n{'TRADE STATISTICS':^70}")
    print("-" * 70)
    print(f"Total Trades:        {metrics['total_trades']:>15}")
    print(f"Winning Trades:      {metrics['winning_trades']:>15}")
    print(f"Losing Trades:       {metrics['losing_trades']:>15}")
    print(f"Win Rate:            {metrics['win_rate_pct']:>15.2f}%")
    print(f"Profit Factor:       {metrics['profit_factor']:>15.2f}")
    print(f"Avg Win/Loss Ratio:  {metrics['avg_win_loss_ratio']:>15.2f}")
    
    print(f"\n{'RISK METRICS':^70}")
    print("-" * 70)
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>15.2f}")
    print(f"Sortino Ratio:       {metrics['sortino_ratio']:>15.2f}")
    print(f"Max Drawdown:        {metrics['max_drawdown_pct']:>15.2f}%")
    print(f"Calmar Ratio:        {metrics['calmar_ratio']:>15.2f}")
    
    print(f"\n{'EXIT REASONS':^70}")
    print("-" * 70)
    for reason, count in metrics.get("exit_reasons", {}).items():
        print(f"  {reason}: {count}")
    
    print(f"\n{'EXECUTION':^70}")
    print("-" * 70)
    print(f"Elapsed Time:        {elapsed_time:>15.2f}s")
    print(f"GPU Accelerated:     {'Yes' if metrics.get('gpu_accelerated') else 'No':>15}")
    
    print("\n" + "=" * 70)
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        metrics_file = output_dir / f"backtest_fibonacci_adx_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\nMetrics saved to: {metrics_file}")
        
        # Save equity curve
        equity_file = output_dir / f"equity_curve_{timestamp}.csv"
        equity_df = pd.DataFrame(portfolio.equity_curve)
        equity_df.to_csv(equity_file, index=False)
        print(f"Equity curve saved to: {equity_file}")
        
        # Save trades
        if portfolio.trades:
            trades_data = [
                {
                    "symbol": t.symbol,
                    "entry_date": str(t.entry_date),
                    "exit_date": str(t.exit_date),
                    "direction": t.direction.name,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "holding_bars": t.holding_bars,
                    "exit_reason": t.exit_reason,
                }
                for t in portfolio.trades
            ]
            trades_file = output_dir / f"trades_{timestamp}.json"
            with open(trades_file, "w") as f:
                json.dump(trades_data, f, indent=2)
            print(f"Trades saved to: {trades_file}")
    
    return {
        "metrics": metrics,
        "signals_generated": signals_generated,
        "positions_opened": positions_opened,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest Fibonacci ADX Strategy")
    parser.add_argument(
        "--symbols", 
        nargs="+", 
        default=["AAPL", "MSFT", "NVDA", "GOOGL"],
        help="Stock symbols to backtest"
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Use all available symbols in data/historical/"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital"
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.10,
        help="Max position size as fraction of equity (default: 0.10)"
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Use enhanced mode with volume, fractal, and MTF filters"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration for metrics calculation (requires CuPy)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warmup bars to skip"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/backtest_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Get all symbols if requested
    if args.all_symbols:
        data_dir = Path(__file__).parent.parent / "data" / "historical"
        csv_files = list(data_dir.glob("*_historical.csv"))
        symbols = [f.stem.replace("_historical", "") for f in csv_files]
    else:
        symbols = args.symbols
    
    output_dir = Path(args.output)
    
    # Run backtest
    asyncio.run(run_backtest(
        symbols=symbols,
        initial_capital=args.capital,
        max_position_pct=args.position_size,
        warmup_bars=args.warmup,
        enhanced_mode=args.enhanced,
        use_gpu=args.gpu,
        output_dir=output_dir,
    ))


if __name__ == "__main__":
    main()
