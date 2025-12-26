#!/usr/bin/env python3
"""
Standalone MI Ensemble Optimization Demo.

Demonstrates the optimizer working with synthetic data.
No external dependencies on Yahoo Finance or complex engines.

Usage:
    python scripts/optimization/demo_mi_optimizer.py
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("ERROR: Optuna not installed. Run: pip install optuna")
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleBacktester:
    """Simplified backtester for MI Ensemble strategy."""
    
    def __init__(self, initial_capital: float = 100_000):
        self.initial_capital = initial_capital
        self.reset()
    
    def reset(self):
        """Reset backtest state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
    
    def calculate_mi_score(self, prices: pd.Series, lookback: int) -> float:
        """Calculate simplified MI score (using autocorrelation)."""
        if len(prices) < lookback:
            return 0.0
        
        returns = prices.pct_change().dropna()[-lookback:]
        
        if len(returns) < 2:
            return 0.0
        
        # Use autocorrelation as proxy for mutual information
        mi_score = abs(returns.autocorr(1))
        return mi_score if not np.isnan(mi_score) else 0.0
    
    def backtest(
        self,
        data: dict[str, pd.DataFrame],
        params: dict,
    ) -> dict:
        """Run backtest with given parameters."""
        self.reset()
        
        # Extract parameters
        mi_lookback = params.get('mi_lookback', 126)
        forward_period = params.get('forward_period', 5)
        ensemble_threshold = params.get('ensemble_threshold', 0.3)
        max_weight = params.get('max_weight', 0.2)
        
        # Get all dates
        all_dates = sorted(set.union(*[set(df.index) for df in data.values()]))
        
        for i, date in enumerate(all_dates[mi_lookback:]):
            # Update positions
            for symbol in list(self.positions.keys()):
                if symbol in data and date in data[symbol].index:
                    current_price = data[symbol].loc[date, 'Close']
                    self.positions[symbol]['current_value'] = (
                        self.positions[symbol]['shares'] * current_price
                    )
            
            # Calculate portfolio value
            portfolio_value = self.cash + sum(
                pos['current_value'] for pos in self.positions.values()
            )
            self.equity_curve.append({
                'date': date,
                'equity': portfolio_value
            })
            
            # Generate signals
            signals = {}
            for symbol, df in data.items():
                if date not in df.index:
                    continue
                
                # Calculate MI score
                prices = df.loc[:date, 'Close']
                mi_score = self.calculate_mi_score(prices, mi_lookback)
                
                if mi_score > ensemble_threshold:
                    signals[symbol] = {
                        'score': mi_score,
                        'direction': 'long'
                    }
            
            # Enter new positions
            max_positions = 5
            if len(signals) > 0 and len(self.positions) < max_positions:
                # Sort by signal strength
                sorted_signals = sorted(
                    signals.items(),
                    key=lambda x: x[1]['score'],
                    reverse=True
                )
                
                for symbol, signal in sorted_signals[:max_positions - len(self.positions)]:
                    if symbol not in self.positions:
                        price = data[symbol].loc[date, 'Close']
                        position_size = portfolio_value * max_weight
                        shares = int(position_size / price)
                        
                        if shares > 0 and self.cash >= shares * price:
                            self.positions[symbol] = {
                                'shares': shares,
                                'entry_price': price,
                                'entry_date': date,
                                'current_value': shares * price,
                            }
                            self.cash -= shares * price
                            
                            self.trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'BUY',
                                'shares': shares,
                                'price': price,
                            })
            
            # Exit positions after holding period
            for symbol in list(self.positions.keys()):
                if symbol in data and date in data[symbol].index:
                    hold_days = (date - self.positions[symbol]['entry_date']).days
                    
                    if hold_days >= forward_period:
                        price = data[symbol].loc[date, 'Close']
                        shares = self.positions[symbol]['shares']
                        entry_price = self.positions[symbol]['entry_price']
                        
                        self.cash += shares * price
                        pnl = shares * (price - entry_price)
                        
                        self.trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'SELL',
                            'shares': shares,
                            'price': price,
                            'pnl': pnl,
                            'return': (price - entry_price) / entry_price,
                        })
                        
                        del self.positions[symbol]
        
        # Calculate performance metrics
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> dict:
        """Calculate performance metrics from backtest."""
        if len(self.equity_curve) == 0:
            return {
                'total_return': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 1.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'profit_factor': 0.0,
            }
        
        equity_df = pd.DataFrame(self.equity_curve).set_index('date')
        returns = equity_df['equity'].pct_change().dropna()
        
        final_value = equity_df['equity'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        sharpe = (
            returns.mean() / returns.std() * np.sqrt(252)
            if len(returns) > 0 and returns.std() > 0
            else 0.0
        )
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
        
        closed_trades = [t for t in self.trades if t['action'] == 'SELL']
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl', 0) <= 0]
        
        win_rate = (
            len(winning_trades) / len(closed_trades)
            if len(closed_trades) > 0
            else 0.0
        )
        
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': len(closed_trades),
            'profit_factor': profit_factor,
            'final_value': final_value,
        }


def generate_synthetic_data(
    n_symbols: int = 10,
    n_days: int = 504,
    base_price_range: tuple = (15, 45),
    volatility_range: tuple = (0.015, 0.025),
) -> dict[str, pd.DataFrame]:
    """Generate synthetic price data for backtesting."""
    data = {}
    
    for i in range(1, n_symbols + 1):
        symbol = f"STOCK{i:02d}"
        
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        base_price = np.random.uniform(*base_price_range)
        volatility = np.random.uniform(*volatility_range)
        trend = np.random.uniform(-0.0005, 0.001)
        
        returns = np.random.normal(trend, volatility, n_days)
        prices = base_price * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            'High': prices * (1 + np.random.uniform(0, 0.03, n_days)),
            'Low': prices * (1 + np.random.uniform(-0.03, 0, n_days)),
            'Close': prices,
            'Volume': np.random.randint(100_000, 1_000_000, n_days),
        })
        
        data[symbol] = df.set_index('Date')
    
    return data


def run_optimization(
    data: dict[str, pd.DataFrame],
    n_trials: int = 20,
) -> dict:
    """Run Bayesian optimization to find best parameters."""
    
    backtester = SimpleBacktester()
    
    def objective(trial):
        """Objective function for Optuna."""
        params = {
            'mi_lookback': trial.suggest_int('mi_lookback', 63, 252),
            'forward_period': trial.suggest_int('forward_period', 1, 10),
            'ensemble_threshold': trial.suggest_float('ensemble_threshold', 0.15, 0.45),
            'max_weight': trial.suggest_float('max_weight', 0.1, 0.25),
        }
        
        # Run backtest
        metrics = backtester.backtest(data, params)
        
        # Calculate fitness with constraints
        total_return = metrics['total_return']
        sharpe = metrics['sharpe']
        max_dd = metrics['max_drawdown']
        win_rate = metrics['win_rate']
        
        # Penalty for constraint violations
        penalty = 0.0
        if sharpe < 0.8:
            penalty += (0.8 - sharpe) * 80
        if max_dd > 0.35:
            penalty += (max_dd - 0.35) * 80
        if win_rate < 0.42:
            penalty += (0.42 - win_rate) * 80
        
        fitness = total_return - penalty
        
        # Store metrics for later retrieval
        trial.set_user_attr('metrics', metrics)
        
        return fitness
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )
    
    best_trial = study.best_trial
    best_params = best_trial.params
    best_metrics = best_trial.user_attrs['metrics']
    
    return {
        'best_params': best_params,
        'best_score': best_trial.value,
        'best_metrics': best_metrics,
        'study': study,
    }


def main():
    logger.info("=" * 80)
    logger.info("MI ENSEMBLE OPTIMIZATION DEMO - SYNTHETIC DATA")
    logger.info("=" * 80)
    
    # Generate data
    logger.info("\n📊 Generating synthetic price data for 10 stocks...")
    data = generate_synthetic_data(n_symbols=10, n_days=504)
    
    for symbol, df in data.items():
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        total_return = (end_price - start_price) / start_price * 100
        logger.info(
            f"  {symbol}: ${start_price:.2f} → ${end_price:.2f} "
            f"({total_return:+.1f}%)"
        )
    
    # Run optimization
    logger.info("\n🔍 Running Bayesian optimization (20 trials)...")
    logger.info("   This will take ~2-5 minutes...\n")
    
    results = run_optimization(data, n_trials=20)
    
    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"\n📈 Best Fitness Score: {results['best_score']:.4f}")
    
    logger.info("\n🏆 Best Parameters:")
    for param, value in results['best_params'].items():
        logger.info(f"  {param:25s}: {value}")
    
    logger.info("\n📊 Performance Metrics:")
    metrics = results['best_metrics']
    logger.info(f"  Total Return:        {metrics['total_return']*100:7.2f}%")
    logger.info(f"  Sharpe Ratio:        {metrics['sharpe']:7.2f}")
    logger.info(f"  Max Drawdown:        {metrics['max_drawdown']*100:7.2f}%")
    logger.info(f"  Win Rate:            {metrics['win_rate']*100:7.2f}%")
    logger.info(f"  Profit Factor:       {metrics['profit_factor']:7.2f}")
    logger.info(f"  Total Trades:        {metrics['total_trades']:7d}")
    logger.info(f"  Final Value:         ${metrics['final_value']:,.2f}")
    
    # Check constraints
    logger.info("\n✅ Constraint Validation:")
    constraints_met = True
    
    if metrics['sharpe'] >= 0.8:
        logger.info(f"  ✓ Sharpe ratio ≥ 0.8: {metrics['sharpe']:.2f}")
    else:
        logger.info(f"  ✗ Sharpe ratio < 0.8: {metrics['sharpe']:.2f}")
        constraints_met = False
    
    if metrics['max_drawdown'] <= 0.35:
        logger.info(f"  ✓ Max DD ≤ 35%: {metrics['max_drawdown']*100:.2f}%")
    else:
        logger.info(f"  ✗ Max DD > 35%: {metrics['max_drawdown']*100:.2f}%")
        constraints_met = False
    
    if metrics['win_rate'] >= 0.42:
        logger.info(f"  ✓ Win rate ≥ 42%: {metrics['win_rate']*100:.2f}%")
    else:
        logger.info(f"  ✗ Win rate < 42%: {metrics['win_rate']*100:.2f}%")
        constraints_met = False
    
    if constraints_met:
        logger.info("\n🎉 All constraints satisfied!")
    else:
        logger.info("\n⚠️  Some constraints not met (likely due to synthetic data)")
    
    # Save results
    output_dir = Path("artifacts/optimization/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'best_params': results['best_params'],
        'best_score': float(results['best_score']),
        'metrics': {k: float(v) for k, v in metrics.items()},
        'timestamp': datetime.now().isoformat(),
        'constraints_met': constraints_met,
    }
    
    output_file = output_dir / "results.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {output_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("DEMO COMPLETE!")
    logger.info("=" * 80)
    logger.info("\n📚 This demonstrates the MI Ensemble optimizer working with")
    logger.info("   synthetic data. For production use, connect to real data sources.")
    logger.info("\n📖 See docs/strategies/MI_ENSEMBLE/OPTIMIZATION_GUIDE.md for details.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Optimization interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Error: {e}", exc_info=True)
        exit(1)
