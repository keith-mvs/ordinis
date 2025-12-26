#!/usr/bin/env python3
"""
GPU-Accelerated Fibonacci ADX Backtest - Daily Timeframe
Walk-Forward Validation with ML Optimization

This is the PROPER timeframe for Fibonacci ADX strategy.
"""

import sys
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print(f"CuPy: {cp.__version__}")
    print(f"CUDA Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    meminfo = cp.cuda.runtime.memGetInfo()
    print(f"VRAM: {meminfo[1] / 1e9:.1f} GB")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available - using CPU")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ordinis.application.strategies.fibonacci_adx import FibonacciADXStrategy

# Stock universe - diversified selection
SMALL_CAP = ["CPRX", "PAYO", "ESLT", "PRGS", "PGNY", "CALX", "SPSC", "GERN", "AMBA", "SCYX"]
MID_CAP = ["WSM", "TRGP", "RJF", "HOLX", "EME", "FFIV", "AVY", "NRG", "ULTA", "GNRC"]
LARGE_CAP = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]

def fetch_daily_data(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch daily OHLCV data from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval="1d")
        if df.empty or len(df) < 50:
            return None
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]
        return df
    except Exception as e:
        print(f"  [{symbol}] Error: {e}")
        return None


def gpu_accelerated_backtest(df: pd.DataFrame, params: dict) -> dict:
    """Run backtest with GPU acceleration where possible."""
    if GPU_AVAILABLE:
        # Move price data to GPU for calculations
        close_gpu = cp.asarray(df["close"].values)
        high_gpu = cp.asarray(df["high"].values)
        low_gpu = cp.asarray(df["low"].values)
        
        # GPU-accelerated returns calculation
        returns_gpu = cp.diff(close_gpu) / close_gpu[:-1]
        returns = cp.asnumpy(returns_gpu)
    else:
        returns = df["close"].pct_change().dropna().values
    
    # Initialize strategy with params
    strategy = FibonacciADXStrategy(
        adx_period=params.get("adx_period", 14),
        adx_threshold=params.get("adx_threshold", 25),
        fib_lookback=params.get("fib_lookback", 50),
        fib_tolerance=params.get("tolerance", 0.02),
    )
    
    # Track positions and P&L
    capital = 100000.0
    position = 0.0
    entry_price = 0.0
    trades = []
    equity_curve = [capital]
    
    position_size = params.get("position_size", 0.10)
    profit_target = params.get("profit_target", 0.05)
    stop_loss = params.get("stop_loss", 0.02)
    min_bars = params.get("min_bars_between", 5)
    bars_since_trade = min_bars
    
    for i in range(params.get("fib_lookback", 50), len(df)):
        bars_since_trade += 1
        current_price = df["close"].iloc[i]
        
        # Check exit conditions if in position
        if position != 0:
            pnl_pct = (current_price - entry_price) / entry_price * np.sign(position)
            
            # Exit on profit target or stop loss
            if pnl_pct >= profit_target or pnl_pct <= -stop_loss:
                pnl = position * (current_price - entry_price)
                capital += pnl
                trades.append({
                    "entry": entry_price,
                    "exit": current_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "bars_held": bars_since_trade
                })
                position = 0.0
                bars_since_trade = 0
        
        # Check entry conditions
        if position == 0 and bars_since_trade >= min_bars:
            lookback_df = df.iloc[i - params.get("fib_lookback", 50):i + 1].copy()
            
            try:
                signal = strategy.generate_signal(lookback_df, df.index[i])
                
                if signal is not None and signal.is_actionable(min_probability=0.55):
                    # Enter position
                    shares = (capital * position_size) / current_price
                    if signal.direction.value == "long":
                        position = shares
                    else:
                        position = -shares
                    entry_price = current_price
                    bars_since_trade = 0
            except Exception:
                pass
        
        equity_curve.append(capital + position * (current_price - entry_price) if position != 0 else capital)
    
    # Close any remaining position
    if position != 0:
        final_price = df["close"].iloc[-1]
        pnl = position * (final_price - entry_price)
        capital += pnl
        trades.append({
            "entry": entry_price,
            "exit": final_price,
            "pnl": pnl,
            "pnl_pct": (final_price - entry_price) / entry_price * np.sign(position),
            "bars_held": bars_since_trade
        })
    
    # Calculate metrics
    equity_curve = np.array(equity_curve)
    if GPU_AVAILABLE:
        equity_gpu = cp.asarray(equity_curve)
        returns_gpu = cp.diff(equity_gpu) / equity_gpu[:-1]
        
        # GPU-accelerated metrics
        mean_ret = float(cp.mean(returns_gpu))
        std_ret = float(cp.std(returns_gpu))
        cummax = cp.zeros_like(equity_gpu)
        cummax[0] = equity_gpu[0]
        for j in range(1, len(equity_gpu)):
            cummax[j] = max(cummax[j-1], equity_gpu[j])
        drawdowns = (cummax - equity_gpu) / cummax
        max_dd = float(cp.max(drawdowns))
    else:
        eq_returns = np.diff(equity_curve) / equity_curve[:-1]
        mean_ret = np.mean(eq_returns)
        std_ret = np.std(eq_returns)
        cummax = np.maximum.accumulate(equity_curve)
        drawdowns = (cummax - equity_curve) / cummax
        max_dd = np.max(drawdowns)
    
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    total_return = (capital - 100000) / 100000
    
    win_trades = [t for t in trades if t["pnl"] > 0]
    win_rate = len(win_trades) / len(trades) if trades else 0
    
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "final_capital": capital,
        "trades": trades
    }


def walk_forward_test(symbol: str, train_months: int = 12, test_months: int = 3) -> dict:
    """Walk-forward optimization and testing."""
    
    # Fetch 5 years of data for walk-forward
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    df = fetch_daily_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if df is None or len(df) < 500:
        return {"symbol": symbol, "status": "insufficient_data"}
    
    results = []
    
    # Walk-forward windows
    train_bars = train_months * 21  # ~21 trading days per month
    test_bars = test_months * 21
    window_size = train_bars + test_bars
    
    # Default parameters (will be optimized)
    best_params = {
        "adx_period": 14,
        "adx_threshold": 25,
        "fib_lookback": 50,
        "tolerance": 0.02,
        "position_size": 0.10,
        "profit_target": 0.05,
        "stop_loss": 0.02,
        "min_bars_between": 5
    }
    
    walk_forward_results = []
    
    i = 0
    while i + window_size <= len(df):
        train_df = df.iloc[i:i + train_bars]
        test_df = df.iloc[i + train_bars:i + window_size]
        
        # Simple parameter optimization on training set
        best_train_return = -float("inf")
        param_grid = [
            {"adx_threshold": t, "fib_lookback": l, "profit_target": p, "stop_loss": s}
            for t in [20, 25, 30]
            for l in [30, 50, 70]
            for p in [0.03, 0.05, 0.07]
            for s in [0.015, 0.02, 0.025]
        ]
        
        # Sample subset for speed
        sampled_params = random.sample(param_grid, min(20, len(param_grid)))
        
        for params in sampled_params:
            test_params = {**best_params, **params}
            try:
                result = gpu_accelerated_backtest(train_df, test_params)
                if result["total_return"] > best_train_return and result["num_trades"] >= 3:
                    best_train_return = result["total_return"]
                    best_params = test_params
            except Exception:
                continue
        
        # Test on out-of-sample data
        try:
            test_result = gpu_accelerated_backtest(test_df, best_params)
            walk_forward_results.append({
                "period": f"{test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}",
                "train_return": best_train_return,
                "test_return": test_result["total_return"],
                "test_sharpe": test_result["sharpe"],
                "test_max_dd": test_result["max_drawdown"],
                "test_trades": test_result["num_trades"],
                "test_win_rate": test_result["win_rate"],
                "params": best_params.copy()
            })
        except Exception:
            pass
        
        # Move forward by test period
        i += test_bars
    
    if not walk_forward_results:
        return {"symbol": symbol, "status": "no_valid_periods"}
    
    # Aggregate walk-forward results
    avg_test_return = np.mean([r["test_return"] for r in walk_forward_results])
    avg_sharpe = np.mean([r["test_sharpe"] for r in walk_forward_results])
    avg_max_dd = np.mean([r["test_max_dd"] for r in walk_forward_results])
    total_trades = sum([r["test_trades"] for r in walk_forward_results])
    avg_win_rate = np.mean([r["test_win_rate"] for r in walk_forward_results])
    
    return {
        "symbol": symbol,
        "status": "success",
        "num_periods": len(walk_forward_results),
        "avg_test_return": avg_test_return,
        "avg_sharpe": avg_sharpe,
        "avg_max_dd": avg_max_dd,
        "total_trades": total_trades,
        "avg_win_rate": avg_win_rate,
        "periods": walk_forward_results
    }


def main():
    print("=" * 80)
    print("GPU-ACCELERATED FIBONACCI ADX BACKTEST - DAILY TIMEFRAME")
    print("Walk-Forward Validation")
    print("=" * 80)
    print()
    
    all_results = {
        "small_cap": [],
        "mid_cap": [],
        "large_cap": []
    }
    
    categories = [
        ("SMALL CAP", SMALL_CAP, "small_cap"),
        ("MID CAP", MID_CAP, "mid_cap"),
        ("LARGE CAP", LARGE_CAP, "large_cap")
    ]
    
    for cat_name, symbols, cat_key in categories:
        print(f"\n{'=' * 80}")
        print(f"  {cat_name} STOCKS - Walk-Forward Test")
        print(f"{'=' * 80}")
        
        for symbol in symbols:
            print(f"\n[{symbol}] Running walk-forward test...")
            result = walk_forward_test(symbol)
            all_results[cat_key].append(result)
            
            if result["status"] == "success":
                status = "✓" if result["avg_test_return"] > 0 else "✗"
                print(f"  {status} Avg Return: {result['avg_test_return']*100:.2f}% | "
                      f"Sharpe: {result['avg_sharpe']:.2f} | "
                      f"MaxDD: {result['avg_max_dd']*100:.1f}% | "
                      f"Trades: {result['total_trades']} | "
                      f"Win Rate: {result['avg_win_rate']*100:.1f}%")
            else:
                print(f"  ⚠ {result['status']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY BY CATEGORY")
    print("=" * 80)
    
    for cat_name, _, cat_key in categories:
        valid_results = [r for r in all_results[cat_key] if r["status"] == "success"]
        if valid_results:
            avg_ret = np.mean([r["avg_test_return"] for r in valid_results])
            avg_sharpe = np.mean([r["avg_sharpe"] for r in valid_results])
            avg_dd = np.mean([r["avg_max_dd"] for r in valid_results])
            winners = sum(1 for r in valid_results if r["avg_test_return"] > 0)
            
            print(f"\n{cat_name}:")
            print(f"  Avg Return: {avg_ret*100:.2f}%")
            print(f"  Avg Sharpe: {avg_sharpe:.2f}")
            print(f"  Avg MaxDD:  {avg_dd*100:.1f}%")
            print(f"  Win Rate:   {winners}/{len(valid_results)} ({winners/len(valid_results)*100:.1f}%)")
    
    # Overall
    all_valid = [r for cat in all_results.values() for r in cat if r["status"] == "success"]
    if all_valid:
        print("\n" + "=" * 80)
        print("OVERALL STATISTICS")
        print("=" * 80)
        
        overall_avg_ret = np.mean([r["avg_test_return"] for r in all_valid])
        overall_sharpe = np.mean([r["avg_sharpe"] for r in all_valid])
        overall_dd = np.mean([r["avg_max_dd"] for r in all_valid])
        overall_winners = sum(1 for r in all_valid if r["avg_test_return"] > 0)
        
        print(f"\nTotal Symbols Tested: {len(all_valid)}")
        print(f"Overall Avg Return: {overall_avg_ret*100:.2f}%")
        print(f"Overall Avg Sharpe: {overall_sharpe:.2f}")
        print(f"Overall Avg MaxDD:  {overall_dd*100:.1f}%")
        print(f"Overall Win Rate:   {overall_winners}/{len(all_valid)} ({overall_winners/len(all_valid)*100:.1f}%)")
        
        # Best and worst performers
        sorted_results = sorted(all_valid, key=lambda x: x["avg_test_return"], reverse=True)
        print(f"\nTop 5 Performers:")
        for r in sorted_results[:5]:
            print(f"  {r['symbol']}: {r['avg_test_return']*100:.2f}% return, {r['avg_sharpe']:.2f} Sharpe")
        
        print(f"\nBottom 5 Performers:")
        for r in sorted_results[-5:]:
            print(f"  {r['symbol']}: {r['avg_test_return']*100:.2f}% return, {r['avg_sharpe']:.2f} Sharpe")
    
    # Save results
    output_path = Path("data/backtest_results")
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = output_path / f"daily_walkforward_{timestamp}.json"
    
    # Convert for JSON serialization
    def serialize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
