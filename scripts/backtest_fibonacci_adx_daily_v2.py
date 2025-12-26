#!/usr/bin/env python3
"""
GPU-Accelerated Fibonacci ADX Backtest - Daily Timeframe (Simplified)
Direct backtest with 3-year historical data per symbol.
"""

import sys
import json
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ordinis.application.strategies.fibonacci_adx import FibonacciADXStrategy

# Stock universe
SMALL_CAP = ["CPRX", "PAYO", "ESLT", "PRGS", "PGNY", "CALX", "SPSC", "GERN", "AMBA", "SCYX"]
MID_CAP = ["WSM", "TRGP", "RJF", "HOLX", "EME", "FFIV", "AVY", "NRG", "ULTA", "GNRC"]
LARGE_CAP = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]


def fetch_daily_data(symbol: str, years: int = 3) -> pd.DataFrame | None:
    """Fetch daily OHLCV data."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date.strftime("%Y-%m-%d"), 
                           end=end_date.strftime("%Y-%m-%d"), 
                           interval="1d")
        if df.empty or len(df) < 100:
            return None
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]
        return df
    except Exception as e:
        print(f"  [{symbol}] Error: {e}")
        return None


def backtest_daily(df: pd.DataFrame, params: dict) -> dict:
    """Run backtest on daily data with proper position management."""
    
    strategy = FibonacciADXStrategy(
        name="FibADX_Backtest",
        adx_period=params.get("adx_period", 14),
        adx_threshold=params.get("adx_threshold", 25),
        swing_lookback=params.get("fib_lookback", 50),
        tolerance=params.get("tolerance", 0.02),
    )
    
    capital = 100000.0
    position = 0.0
    entry_price = 0.0
    entry_idx = 0
    trades = []
    equity_curve = [capital]
    signals_generated = 0
    
    position_size = params.get("position_size", 0.10)
    profit_target = params.get("profit_target", 0.05)
    stop_loss = params.get("stop_loss", 0.02)
    max_hold_days = params.get("max_hold_days", 20)
    
    lookback = params.get("fib_lookback", 50)
    
    for i in range(lookback, len(df)):
        current_price = df["close"].iloc[i]
        current_idx = i
        
        # Track equity
        if position != 0:
            unrealized = position * (current_price - entry_price)
            equity_curve.append(capital + unrealized)
        else:
            equity_curve.append(capital)
        
        # Exit logic
        if position != 0:
            pnl_pct = (current_price - entry_price) / entry_price * np.sign(position)
            days_held = current_idx - entry_idx
            
            exit_signal = False
            exit_reason = ""
            
            if pnl_pct >= profit_target:
                exit_signal = True
                exit_reason = "profit_target"
            elif pnl_pct <= -stop_loss:
                exit_signal = True
                exit_reason = "stop_loss"
            elif days_held >= max_hold_days:
                exit_signal = True
                exit_reason = "max_hold"
            
            if exit_signal:
                pnl = position * (current_price - entry_price)
                capital += pnl
                trades.append({
                    "entry_date": df.index[entry_idx].strftime("%Y-%m-%d"),
                    "exit_date": df.index[current_idx].strftime("%Y-%m-%d"),
                    "entry": entry_price,
                    "exit": current_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "days_held": days_held,
                    "exit_reason": exit_reason
                })
                position = 0.0
        
        # Entry logic
        if position == 0:
            lookback_df = df.iloc[i - lookback:i + 1].copy()
            
            try:
                signal = strategy.generate_signal(lookback_df, df.index[i])
                
                if signal is not None:
                    signals_generated += 1
                    
                    if signal.is_actionable(min_probability=0.5):  # Lower threshold
                        shares = (capital * position_size) / current_price
                        if signal.direction.value == "long":
                            position = shares
                        else:
                            position = -shares
                        entry_price = current_price
                        entry_idx = current_idx
            except Exception as e:
                pass
    
    # Close any remaining position
    if position != 0:
        final_price = df["close"].iloc[-1]
        pnl = position * (final_price - entry_price)
        capital += pnl
        trades.append({
            "entry_date": df.index[entry_idx].strftime("%Y-%m-%d"),
            "exit_date": df.index[-1].strftime("%Y-%m-%d"),
            "entry": entry_price,
            "exit": final_price,
            "pnl": pnl,
            "pnl_pct": (final_price - entry_price) / entry_price * np.sign(position),
            "days_held": len(df) - 1 - entry_idx,
            "exit_reason": "end_of_data"
        })
    
    # Calculate metrics
    equity_curve = np.array(equity_curve)
    
    if GPU_AVAILABLE and len(equity_curve) > 1:
        equity_gpu = cp.asarray(equity_curve)
        returns_gpu = cp.diff(equity_gpu) / equity_gpu[:-1]
        mean_ret = float(cp.mean(returns_gpu))
        std_ret = float(cp.std(returns_gpu))
        
        # Max drawdown
        cummax = cp.zeros_like(equity_gpu)
        cummax[0] = equity_gpu[0]
        for j in range(1, len(equity_gpu)):
            cummax[j] = max(float(cummax[j-1]), float(equity_gpu[j]))
        cummax = cp.asarray(cummax)
        drawdowns = (cummax - equity_gpu) / cummax
        max_dd = float(cp.max(drawdowns))
    else:
        eq_returns = np.diff(equity_curve) / equity_curve[:-1]
        mean_ret = np.mean(eq_returns) if len(eq_returns) > 0 else 0
        std_ret = np.std(eq_returns) if len(eq_returns) > 0 else 1
        cummax = np.maximum.accumulate(equity_curve)
        drawdowns = (cummax - equity_curve) / cummax
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    total_return = (capital - 100000) / 100000
    
    win_trades = [t for t in trades if t["pnl"] > 0]
    win_rate = len(win_trades) / len(trades) if trades else 0
    
    avg_win = np.mean([t["pnl_pct"] for t in win_trades]) if win_trades else 0
    lose_trades = [t for t in trades if t["pnl"] <= 0]
    avg_loss = np.mean([t["pnl_pct"] for t in lose_trades]) if lose_trades else 0
    
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "num_trades": len(trades),
        "signals_generated": signals_generated,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "final_capital": capital,
        "bars_processed": len(df),
        "trades": trades[:10]  # First 10 trades for reference
    }


def main():
    print("=" * 80)
    print("GPU-ACCELERATED FIBONACCI ADX BACKTEST - DAILY TIMEFRAME")
    print("3-Year Historical Data | Simplified Backtest")
    print("=" * 80)
    print()
    
    # Default daily params (tuned for daily data)
    params = {
        "adx_period": 14,
        "adx_threshold": 20,  # Lower threshold for more signals
        "fib_lookback": 50,
        "tolerance": 0.025,   # Slightly wider tolerance
        "position_size": 0.10,
        "profit_target": 0.05,
        "stop_loss": 0.025,
        "max_hold_days": 15
    }
    
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
    
    total_bars = 0
    start_time = datetime.now()
    
    for cat_name, symbols, cat_key in categories:
        print(f"\n{'=' * 80}")
        print(f"  {cat_name} STOCKS")
        print(f"{'=' * 80}")
        
        for symbol in symbols:
            print(f"\n[{symbol}] Fetching data and backtesting...", end=" ")
            
            df = fetch_daily_data(symbol, years=3)
            if df is None:
                print("⚠ insufficient data")
                all_results[cat_key].append({"symbol": symbol, "status": "no_data"})
                continue
            
            result = backtest_daily(df, params)
            result["symbol"] = symbol
            result["status"] = "success"
            all_results[cat_key].append(result)
            total_bars += result["bars_processed"]
            
            status = "✓" if result["total_return"] > 0 else "✗"
            print(f"\n  {status} Return: {result['total_return']*100:.2f}% | "
                  f"Sharpe: {result['sharpe']:.2f} | "
                  f"MaxDD: {result['max_drawdown']*100:.1f}% | "
                  f"Trades: {result['num_trades']} | "
                  f"Signals: {result['signals_generated']} | "
                  f"Win Rate: {result['win_rate']*100:.1f}%")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY BY CATEGORY")
    print("=" * 80)
    
    for cat_name, _, cat_key in categories:
        valid_results = [r for r in all_results[cat_key] if r.get("status") == "success"]
        if valid_results:
            avg_ret = np.mean([r["total_return"] for r in valid_results])
            avg_sharpe = np.mean([r["sharpe"] for r in valid_results])
            avg_dd = np.mean([r["max_drawdown"] for r in valid_results])
            total_trades = sum([r["num_trades"] for r in valid_results])
            winners = sum(1 for r in valid_results if r["total_return"] > 0)
            
            print(f"\n{cat_name}:")
            print(f"  Avg Return: {avg_ret*100:.2f}%")
            print(f"  Avg Sharpe: {avg_sharpe:.2f}")
            print(f"  Avg MaxDD:  {avg_dd*100:.1f}%")
            print(f"  Total Trades: {total_trades}")
            print(f"  Win Rate:   {winners}/{len(valid_results)} ({winners/len(valid_results)*100:.1f}%)")
    
    # Overall
    all_valid = [r for cat in all_results.values() for r in cat if r.get("status") == "success"]
    if all_valid:
        print("\n" + "=" * 80)
        print("OVERALL STATISTICS")
        print("=" * 80)
        
        overall_avg_ret = np.mean([r["total_return"] for r in all_valid])
        overall_sharpe = np.mean([r["sharpe"] for r in all_valid])
        overall_dd = np.mean([r["max_drawdown"] for r in all_valid])
        overall_trades = sum([r["num_trades"] for r in all_valid])
        overall_winners = sum(1 for r in all_valid if r["total_return"] > 0)
        
        print(f"\nTotal Symbols Tested: {len(all_valid)}")
        print(f"Total Bars Processed: {total_bars:,}")
        print(f"Processing Time: {elapsed:.2f}s")
        print(f"Throughput: {total_bars/elapsed:,.0f} bars/second")
        print(f"\nOverall Avg Return: {overall_avg_ret*100:.2f}%")
        print(f"Overall Avg Sharpe: {overall_sharpe:.2f}")
        print(f"Overall Avg MaxDD:  {overall_dd*100:.1f}%")
        print(f"Overall Total Trades: {overall_trades}")
        print(f"Overall Win Rate:   {overall_winners}/{len(all_valid)} ({overall_winners/len(all_valid)*100:.1f}%)")
        
        # Best and worst
        sorted_results = sorted(all_valid, key=lambda x: x["total_return"], reverse=True)
        print(f"\nTop 5 Performers:")
        for r in sorted_results[:5]:
            print(f"  {r['symbol']}: {r['total_return']*100:.2f}% return, "
                  f"{r['sharpe']:.2f} Sharpe, {r['num_trades']} trades")
        
        print(f"\nBottom 5 Performers:")
        for r in sorted_results[-5:]:
            print(f"  {r['symbol']}: {r['total_return']*100:.2f}% return, "
                  f"{r['sharpe']:.2f} Sharpe, {r['num_trades']} trades")
    
    # Save results
    output_path = Path("data/backtest_results")
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = output_path / f"daily_backtest_{timestamp}.json"
    
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
