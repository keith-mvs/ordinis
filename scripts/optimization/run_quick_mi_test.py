#!/usr/bin/env python3
"""
Quick MI Ensemble test using existing data without Yahoo Finance.

This script demonstrates the optimizer working with synthetic data
since Yahoo Finance is currently blocking requests.

Usage:
    python scripts/optimization/run_quick_mi_test.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from ordinis.engines.signalcore.models.mi_ensemble_optimizer import (
        MIEnsembleOptimizer,
        ParameterSpace,
        OptimizationObjective,
    )
    MI_PARAM_SPACE = ParameterSpace()
    DEFAULT_CONSTRAINTS = OptimizationObjective()
except ImportError:
    # Fallback if module not available
    MI_PARAM_SPACE = None
    DEFAULT_CONSTRAINTS = None
    MIEnsembleOptimizer = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_price_data(
    symbol: str,
    n_days: int = 504,  # ~2 years
    base_price: float = 25.0,
    volatility: float = 0.02,
    trend: float = 0.0005,
) -> pd.DataFrame:
    """Generate realistic synthetic price data."""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generate returns with drift and volatility
    returns = np.random.normal(trend, volatility, n_days)
    prices = base_price * (1 + returns).cumprod()
    
    # Add some volume
    volume = np.random.randint(100_000, 1_000_000, n_days)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'High': prices * (1 + np.random.uniform(0, 0.03, n_days)),
        'Low': prices * (1 + np.random.uniform(-0.03, 0, n_days)),
        'Close': prices,
        'Volume': volume,
        'Symbol': symbol,
    })
    
    return df.set_index('Date')


def calculate_mutual_information(price_data: pd.DataFrame, lookback: int = 126) -> pd.Series:
    """Calculate simplified MI score."""
    returns = price_data['Close'].pct_change()
    
    # Rolling correlation as proxy for MI
    mi_score = returns.rolling(lookback).apply(
        lambda x: abs(x.autocorr(1)) if len(x) > 1 else 0,
        raw=False
    )
    
    return mi_score.fillna(0)


def run_single_backtest(
    data: dict[str, pd.DataFrame],
    params: dict,
) -> dict:
    """Run a single backtest with given parameters."""
    
    portfolio_value = 100_000.0
    cash = portfolio_value
    positions = {}
    trades = []
    equity_curve = []
    
    # Extract parameters
    mi_lookback = params['mi_lookback']
    forward_period = params['forward_period']
    ensemble_threshold = params['ensemble_threshold']
    
    # Get all dates across all symbols
    all_dates = sorted(set.union(*[set(df.index) for df in data.values()]))
    
    for i, date in enumerate(all_dates[mi_lookback:]):
        # Update positions
        for symbol in list(positions.keys()):
            if symbol in data and date in data[symbol].index:
                current_price = data[symbol].loc[date, 'Close']
                positions[symbol]['current_value'] = positions[symbol]['shares'] * current_price
        
        portfolio_value = cash + sum(pos['current_value'] for pos in positions.values())
        equity_curve.append({'date': date, 'equity': portfolio_value})
        
        # Generate signals
        signals = {}
        for symbol, df in data.items():
            if date not in df.index:
                continue
                
            # Calculate MI score
            mi_score = calculate_mutual_information(
                df.loc[:date],
                lookback=mi_lookback
            ).iloc[-1]
            
            if mi_score > ensemble_threshold:
                signals[symbol] = {'score': mi_score, 'direction': 'long'}
        
        # Execute trades (simplified)
        if len(signals) > 0 and len(positions) < 5:  # Max 5 positions
            for symbol, signal in list(signals.items())[:5-len(positions)]:
                if symbol not in positions:
                    price = data[symbol].loc[date, 'Close']
                    position_size = portfolio_value * 0.15  # 15% per position
                    shares = int(position_size / price)
                    
                    if shares > 0 and cash >= shares * price:
                        positions[symbol] = {
                            'shares': shares,
                            'entry_price': price,
                            'entry_date': date,
                            'current_value': shares * price,
                        }
                        cash -= shares * price
                        trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                        })
        
        # Exit trades (simplified - hold for forward_period days)
        for symbol in list(positions.keys()):
            if symbol in data and date in data[symbol].index:
                hold_days = (date - positions[symbol]['entry_date']).days
                if hold_days >= forward_period:
                    price = data[symbol].loc[date, 'Close']
                    shares = positions[symbol]['shares']
                    cash += shares * price
                    
                    trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'shares': shares,
                        'price': price,
                        'pnl': shares * (price - positions[symbol]['entry_price']),
                    })
                    del positions[symbol]
    
    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve).set_index('date')
    returns = equity_df['equity'].pct_change().dropna()
    
    total_return = (portfolio_value - 100_000) / 100_000
    
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
    
    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
    total_trades = len([t for t in trades if t['action'] == 'SELL'])
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'final_equity': portfolio_value,
    }


def main():
    logger.info("=" * 80)
    logger.info("QUICK MI ENSEMBLE TEST - SYNTHETIC DATA")
    logger.info("=" * 80)
    
    # Generate synthetic data for 10 stocks
    logger.info("\nGenerating synthetic price data for 10 stocks...")
    symbols = [f"STOCK{i:02d}" for i in range(1, 11)]
    
    data = {}
    for symbol in symbols:
        data[symbol] = generate_synthetic_price_data(
            symbol,
            n_days=504,
            base_price=np.random.uniform(10, 45),
            volatility=np.random.uniform(0.015, 0.025),
            trend=np.random.uniform(-0.0005, 0.001),
        )
        logger.info(f"  {symbol}: {len(data[symbol])} days, ${data[symbol]['Close'].iloc[0]:.2f} -> ${data[symbol]['Close'].iloc[-1]:.2f}")
    
    # Initialize optimizer
    logger.info("\nInitializing optimizer...")
    optimizer = MIEnsembleOptimizer(
        n_trials=20,  # Quick test
        n_splits=3,
        test_size_days=63,
        use_nvidia_enhancement=False,  # Faster without AI
    )
    
    # Run optimization
    logger.info("\nRunning optimization (20 trials)...")
    logger.info("This will take ~2-5 minutes...\n")
    
    def objective_wrapper(params):
        """Wrapper for optimizer."""
        result = run_single_backtest(data, params)
        
        # Calculate fitness
        total_return = result['total_return']
        sharpe = result['sharpe']
        max_dd = result['max_drawdown']
        win_rate = result['win_rate']
        
        # Constraints
        violations = 0
        if sharpe < 0.8:
            violations += (0.8 - sharpe)
        if max_dd > 0.35:
            violations += (max_dd - 0.35)
        if win_rate < 0.42:
            violations += (0.42 - win_rate)
        
        fitness = total_return - 80.0 * violations
        
        return {
            'fitness': fitness,
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            **result,
        }
    
    # Override the optimizer's objective
    optimizer.objective = objective_wrapper
    
    # Run optimization
    best_params, best_score, study = optimizer.optimize(
        param_space=MI_PARAM_SPACE,
        direction='maximize',
    )
    
    # Results
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"\nBest Score: {best_score:.4f}")
    logger.info("\nBest Parameters:")
    for param, value in best_params.items():
        logger.info(f"  {param:25s}: {value}")
    
    # Run final backtest with best params
    logger.info("\nRunning final backtest with best parameters...")
    final_result = run_single_backtest(data, best_params)
    
    logger.info("\nFinal Performance Metrics:")
    logger.info(f"  Total Return:        {final_result['total_return']*100:.2f}%")
    logger.info(f"  Sharpe Ratio:        {final_result['sharpe']:.2f}")
    logger.info(f"  Max Drawdown:        {final_result['max_drawdown']*100:.2f}%")
    logger.info(f"  Win Rate:            {final_result['win_rate']*100:.2f}%")
    logger.info(f"  Total Trades:        {final_result['total_trades']}")
    logger.info(f"  Final Equity:        ${final_result['final_equity']:,.2f}")
    
    # Save results
    output_dir = Path("artifacts/optimization/quick_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'best_params': best_params,
        'best_score': best_score,
        'final_metrics': final_result,
        'timestamp': datetime.now().isoformat(),
    }
    
    output_file = output_dir / "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
