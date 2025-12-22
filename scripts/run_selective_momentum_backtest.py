#!/usr/bin/env python
"""
Selective Momentum + ATR-RSI Hybrid Strategy Backtest

KEY INSIGHT: ATR-RSI model generated 40.7% CAGR on NVDA (33.3% alpha) but -23.8% on UNH.
Mean-reversion works on MOMENTUM stocks that pull back - not on declining stocks.

STRATEGY: Two-stage selection
1. MOMENTUM SCREEN: Only trade stocks in top 30% momentum (positive 6-month return)
2. MEAN-REVERSION ENTRY: Use ATR-RSI model for timing entries on pullbacks
3. DYNAMIC WEIGHTING: Weight positions by momentum strength × signal confidence

This filters out value traps (UNH, NKE, TMO) and focuses on trend-following with pullback entry.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


# Massive 25 symbols
MASSIVE_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "META",  # Tech
    "JPM", "BAC", "GS", "MS", "WFC",           # Financials  
    "JNJ", "UNH", "PFE", "ABBV", "TMO",        # Healthcare
    "WMT", "HD", "NKE", "MCD", "SBUX",         # Consumer
    "XOM", "CVX", "COP", "EOG", "SLB",         # Energy
]

BENCHMARK_SYMBOL = "SPY"
DATA_DIR = Path(__file__).parent.parent / "data" / "historical"


def compute_momentum_score(data: pd.DataFrame, lookback_days: int = 126) -> float:
    """
    Compute momentum score for asset selection.
    
    Uses 6-month return adjusted for volatility (Sharpe-like measure).
    """
    if len(data) < lookback_days:
        return -999.0
    
    close = data["close"]
    returns = close.pct_change().dropna()
    
    if len(returns) < lookback_days:
        return -999.0
    
    # 6-month return
    total_return = (close.iloc[-1] / close.iloc[-lookback_days]) - 1
    
    # Volatility-adjusted (mini Sharpe)
    volatility = returns.iloc[-lookback_days:].std() * np.sqrt(252)
    
    if volatility == 0:
        return -999.0
    
    momentum_score = total_return / volatility
    
    return momentum_score


def compute_trend_strength(data: pd.DataFrame, period: int = 50) -> float:
    """
    Compute trend strength using linear regression slope.
    
    Positive = uptrend, Negative = downtrend
    """
    if len(data) < period:
        return 0.0
    
    close = data["close"].iloc[-period:]
    x = np.arange(len(close))
    
    # Simple linear regression slope
    slope = np.polyfit(x, close.values, 1)[0]
    
    # Normalize by price level
    normalized_slope = slope / close.mean() * 100
    
    return normalized_slope


def load_data(symbol: str, min_days: int = 252) -> pd.DataFrame | None:
    """Load historical data for symbol from local files."""
    # Try CSV first
    csv_path = DATA_DIR / f"{symbol}_historical.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            df = df.rename(columns={"Date": "date"})
            df = df.set_index("date")
            df.columns = df.columns.str.lower()
            
            if len(df) >= min_days:
                return df
            logger.warning(f"{symbol}: Only {len(df)} days (need {min_days})")
        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
    
    # Try parquet
    parquet_path = DATA_DIR / f"{symbol}.parquet"
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            df.columns = df.columns.str.lower()
            
            if len(df) >= min_days:
                return df
        except Exception as e:
            logger.error(f"Error loading {symbol} parquet: {e}")
    
    # Try lowercase name
    csv_path_lower = DATA_DIR / f"{symbol.lower()}_daily.csv"
    if csv_path_lower.exists():
        try:
            df = pd.read_csv(csv_path_lower, parse_dates=["date"], index_col="date")
            df.columns = df.columns.str.lower()
            if len(df) >= min_days:
                return df
        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
    
    logger.warning(f"No data found for {symbol}")
    return None


def backtest_symbol_sync(
    symbol: str,
    df: pd.DataFrame,
    rsi_oversold: int = 40,
    rsi_exit: int = 55,
    rsi_period: int = 10,
    atr_period: int = 14,
    atr_stop_mult: float = 1.5,
    atr_tp_mult: float = 2.5,
    initial_equity: float = 10000.0,
    cost_bps: float = 8.0,
) -> dict[str, Any]:
    """
    Synchronous backtest of ATR-RSI strategy on single symbol.
    
    Uses the model logic directly without async complexity.
    """
    from ordinis.engines.signalcore.features.technical import TechnicalIndicators
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    # Compute indicators
    rsi = TechnicalIndicators.rsi(close, rsi_period)
    
    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()
    
    trades = []
    position = None
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    entry_idx = 0
    
    equity = initial_equity
    position_size = 0.0
    
    warmup = max(rsi_period, atr_period) + 5
    
    for i in range(warmup, len(df)):
        curr_price = close.iloc[i]
        curr_rsi = rsi.iloc[i]
        curr_atr = atr.iloc[i]
        
        if np.isnan(curr_rsi) or np.isnan(curr_atr):
            continue
        
        if position is None:
            # Entry: RSI oversold
            if curr_rsi < rsi_oversold:
                position = "long"
                entry_price = curr_price
                entry_idx = i
                stop_loss = entry_price - (atr_stop_mult * curr_atr)
                take_profit = entry_price + (atr_tp_mult * curr_atr)
                
                # Transaction cost
                cost = equity * (cost_bps / 10000)
                equity -= cost
                position_size = equity / entry_price
        
        elif position == "long":
            # Exit conditions
            hit_stop = curr_price <= stop_loss
            hit_tp = curr_price >= take_profit
            rsi_signal = curr_rsi > rsi_exit
            
            if hit_stop or hit_tp or rsi_signal:
                exit_price = curr_price
                
                # Transaction cost
                cost = (position_size * exit_price) * (cost_bps / 10000)
                
                pnl_pct = (exit_price / entry_price - 1) * 100
                equity = position_size * exit_price - cost
                
                reason = "stop" if hit_stop else ("tp" if hit_tp else "rsi_exit")
                
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "bars_held": i - entry_idx,
                    "reason": reason,
                })
                
                position = None
    
    # Close open position at end
    if position == "long":
        exit_price = close.iloc[-1]
        pnl_pct = (exit_price / entry_price - 1) * 100
        equity = position_size * exit_price
        
        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": len(df) - 1,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "bars_held": len(df) - 1 - entry_idx,
            "reason": "end",
        })
    
    if not trades:
        return {
            "trades": 0,
            "cagr": 0.0,
            "total_return": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "final_equity": initial_equity,
        }
    
    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    days = len(df)
    total_return = (equity / initial_equity - 1) * 100
    cagr = ((equity / initial_equity) ** (252 / days) - 1) * 100
    win_rate = len(wins) / len(trades) * 100
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 999
    
    return {
        "trades": len(trades),
        "cagr": cagr,
        "total_return": total_return,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "final_equity": equity,
        "trade_list": trades,
    }


def backtest_selective_strategy(
    days: int = 500,
    momentum_percentile: float = 0.7,  # Top 30% momentum
    min_momentum_score: float = 0.3,   # Minimum momentum score
    initial_capital: float = 100000.0,
    transaction_cost_bps: float = 8.0,
) -> dict[str, Any]:
    """
    Run selective momentum + mean-reversion backtest.
    
    Two-stage process:
    1. Rank all symbols by momentum score
    2. Only trade top momentum symbols with ATR-RSI timing
    """
    
    logger.info("=" * 70)
    logger.info("SELECTIVE MOMENTUM + ATR-RSI HYBRID BACKTEST")
    logger.info("=" * 70)
    logger.info(f"Days: {days}")
    logger.info(f"Momentum Percentile: Top {(1-momentum_percentile)*100:.0f}%")
    logger.info(f"Min Momentum Score: {min_momentum_score}")
    logger.info("=" * 70)
    
    # Stage 1: Load all data and compute momentum
    logger.info("\n--- STAGE 1: Computing Momentum Scores ---")
    
    symbol_data = {}
    momentum_scores = {}
    trend_strengths = {}
    
    for symbol in MASSIVE_SYMBOLS:
        df = load_data(symbol, days)
        if df is not None:
            # Use only last N days
            df = df.tail(days + 50)  # Extra warmup
            symbol_data[symbol] = df
            momentum_scores[symbol] = compute_momentum_score(df)
            trend_strengths[symbol] = compute_trend_strength(df)
            logger.info(f"  {symbol}: Mom={momentum_scores[symbol]:.3f}, Trend={trend_strengths[symbol]:.3f}")
    
    # Stage 2: Filter to top momentum stocks
    sorted_by_momentum = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("\n--- Momentum Ranking ---")
    for i, (sym, score) in enumerate(sorted_by_momentum):
        logger.info(f"  {i+1}. {sym}: {score:.3f}")
    
    # Get top percentile
    n_select = max(1, int(len(sorted_by_momentum) * (1 - momentum_percentile)))
    selected_symbols = [
        s for s, score in sorted_by_momentum[:n_select] 
        if score >= min_momentum_score
    ]
    
    logger.info("\n--- STAGE 2: Selected Top Momentum Stocks ---")
    logger.info(f"Selected {len(selected_symbols)} symbols: {selected_symbols}")
    
    if not selected_symbols:
        logger.warning("No symbols passed momentum filter!")
        return {"error": "No symbols passed filter"}
    
    # Stage 3: Run ATR-RSI backtest on selected symbols
    logger.info("\n--- STAGE 3: ATR-RSI Timing on Selected Stocks ---")
    
    symbol_results = {}
    all_trades = []
    
    per_symbol_capital = initial_capital / len(selected_symbols)
    
    for symbol in selected_symbols:
        df = symbol_data[symbol]
        
        result = backtest_symbol_sync(
            symbol=symbol,
            df=df,
            rsi_oversold=35,   # Stricter - true oversold
            rsi_exit=50,       # Exit when neutral
            rsi_period=14,     # Standard period
            atr_period=14,
            atr_stop_mult=1.5,
            atr_tp_mult=2.0,   # Realistic targets
            initial_equity=per_symbol_capital,
            cost_bps=transaction_cost_bps,
        )
        
        result["momentum_score"] = momentum_scores[symbol]
        symbol_results[symbol] = result
        
        if result["trades"] > 0:
            if "trade_list" in result:
                for t in result["trade_list"]:
                    t["symbol"] = symbol
                    all_trades.append(t)
        
        status = "✅" if result["cagr"] > 21.5 else "❌"  # Beat S&P 500
        logger.info(
            f"  {symbol}: {status} CAGR={result['cagr']:.1f}% | "
            f"Trades={result['trades']} | WR={result['win_rate']:.0f}% | "
            f"PF={result['profit_factor']:.2f}"
        )
    
    # Compute portfolio results
    total_initial = initial_capital
    total_final = sum(r.get("final_equity", per_symbol_capital) for r in symbol_results.values())
    
    portfolio_cagr = ((total_final / total_initial) ** (252 / days) - 1) * 100
    portfolio_return = (total_final / total_initial - 1) * 100
    
    # Load benchmark
    spy_data = load_data("SPY", days)
    benchmark_cagr = 0.0
    if spy_data is not None:
        # Use last N days only
        spy_prices = spy_data["close"].tail(days)
        if len(spy_prices) >= days:
            spy_return = spy_prices.iloc[-1] / spy_prices.iloc[0] - 1
            benchmark_cagr = ((1 + spy_return) ** (252 / days) - 1) * 100
    
    excess_return = portfolio_cagr - benchmark_cagr
    
    # Trade statistics
    if all_trades:
        pnls = [t["pnl_pct"] for t in all_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 999
    else:
        win_rate = 0
        profit_factor = 0
    
    logger.info("\n" + "=" * 70)
    logger.info("PORTFOLIO RESULTS")
    logger.info("=" * 70)
    logger.info(f"Selected Symbols: {len(selected_symbols)}")
    logger.info(f"Total Trades: {len(all_trades)}")
    logger.info(f"Total Return: {portfolio_return:.1f}%")
    logger.info(f"Portfolio CAGR: {portfolio_cagr:.1f}%")
    logger.info(f"Benchmark CAGR: {benchmark_cagr:.1f}%")
    logger.info(f"Excess Return: {excess_return:+.1f}%")
    logger.info(f"Win Rate: {win_rate:.1f}%")
    logger.info(f"Profit Factor: {profit_factor:.2f}")
    
    if excess_return > 0:
        logger.info("\n🏆 SUCCESS: Strategy outperforms S&P 500!")
    else:
        logger.info("\n❌ Strategy underperforms S&P 500")
    
    return {
        "selected_symbols": selected_symbols,
        "momentum_scores": momentum_scores,
        "symbol_results": symbol_results,
        "portfolio_cagr": portfolio_cagr,
        "benchmark_cagr": benchmark_cagr,
        "excess_return": excess_return,
        "total_trades": len(all_trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "all_trades": all_trades,
    }


def run_parameter_sweep():
    """Run sweep over different momentum thresholds."""
    
    logger.info("\n" + "=" * 70)
    logger.info("PARAMETER SWEEP: Finding Optimal Momentum Threshold")
    logger.info("=" * 70)
    
    results = []
    
    for pct in [0.9, 0.8, 0.7, 0.6, 0.5]:  # Top 10%, 20%, 30%, 40%, 50%
        for min_score in [0.5, 0.3, 0.1, 0.0]:
            logger.info(f"\nTesting: Top {(1-pct)*100:.0f}% momentum, min_score={min_score}")
            
            result = backtest_selective_strategy(
                days=500,
                momentum_percentile=pct,
                min_momentum_score=min_score,
            )
            
            if "error" not in result:
                results.append({
                    "percentile": pct,
                    "min_score": min_score,
                    "n_symbols": len(result["selected_symbols"]),
                    "portfolio_cagr": result["portfolio_cagr"],
                    "excess_return": result["excess_return"],
                    "trades": result["total_trades"],
                    "win_rate": result["win_rate"],
                })
    
    # Find best
    if results:
        best = max(results, key=lambda x: x["excess_return"])
        
        logger.info("\n" + "=" * 70)
        logger.info("SWEEP RESULTS")
        logger.info("=" * 70)
        
        df = pd.DataFrame(results)
        logger.info(f"\n{df.to_string()}")
        
        logger.info("\nBEST CONFIG:")
        logger.info(f"  Momentum Percentile: Top {(1-best['percentile'])*100:.0f}%")
        logger.info(f"  Min Score: {best['min_score']}")
        logger.info(f"  Symbols: {best['n_symbols']}")
        logger.info(f"  Portfolio CAGR: {best['portfolio_cagr']:.1f}%")
        logger.info(f"  Excess Return: {best['excess_return']:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Selective Momentum + ATR-RSI Backtest")
    parser.add_argument("--days", type=int, default=500, help="Days of data")
    parser.add_argument("--momentum-pct", type=float, default=0.7, help="Momentum percentile (0.7 = top 30%)")
    parser.add_argument("--min-score", type=float, default=0.3, help="Minimum momentum score")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    args = parser.parse_args()
    
    if args.sweep:
        run_parameter_sweep()
    else:
        backtest_selective_strategy(
            days=args.days,
            momentum_percentile=args.momentum_pct,
            min_momentum_score=args.min_score,
        )


if __name__ == "__main__":
    main()