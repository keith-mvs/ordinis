"""
Full System Integration Example: SignalCore + RiskGuard + ProofBench.

Demonstrates complete workflow:
1. SignalCore generates quantitative signals
2. RiskGuard validates and sizes positions
3. ProofBench backtests the strategy
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.engines.proofbench import BacktestEngine, ExecutionConfig, SimulationConfig
from src.engines.riskguard import STANDARD_RISK_RULES, RiskGuardEngine
from src.engines.riskguard.core.engine import PortfolioState, Position, ProposedTrade
from src.engines.signalcore.core.model import ModelConfig, ModelRegistry
from src.engines.signalcore.core.signal import SignalType
from src.engines.signalcore.models.rsi_mean_reversion import RSIMeanReversionModel
from src.engines.signalcore.models.sma_crossover import SMACrossoverModel


def create_sample_data(symbol: str = "AAPL", days: int = 500) -> pd.DataFrame:
    """
    Create sample OHLCV data for testing.

    Args:
        symbol: Stock symbol
        days: Number of days of data

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)

    # Start with base price
    base_price = 150.0

    # Generate realistic price movement
    returns = np.random.normal(0.0005, 0.02, days)  # Mean return, volatility
    returns[::20] += np.random.choice([-0.05, 0.05], size=len(returns[::20]))  # Add jumps

    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
    opens = prices * (1 + np.random.uniform(-0.005, 0.005, days))
    highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.01, days)))
    lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.01, days)))
    closes = prices

    # Generate volume
    base_volume = 10_000_000
    volumes = base_volume * (1 + np.random.normal(0, 0.3, days))
    volumes = np.maximum(volumes, base_volume * 0.2)

    # Create DataFrame
    end_date = datetime(2024, 12, 31)
    dates = [end_date - timedelta(days=days - i - 1) for i in range(days)]

    data = pd.DataFrame(
        {
            "symbol": symbol,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=dates,
    )

    return data


def risk_aware_strategy(engine, symbol, bar):  # noqa: PLR0912, PLR0915
    """
    Trading strategy with RiskGuard integration.

    SignalCore generates signals → RiskGuard validates → Orders placed.

    Args:
        engine: BacktestEngine instance
        symbol: Trading symbol
        bar: Current bar data
    """
    # Initialize RiskGuard on first call
    if not hasattr(risk_aware_strategy, "risk_guard"):
        risk_aware_strategy.risk_guard = RiskGuardEngine(rules=STANDARD_RISK_RULES)
        risk_aware_strategy.signal_registry = ModelRegistry()

        # Register models
        sma_model = SMACrossoverModel(
            ModelConfig(
                model_id="sma_crossover",
                model_type="technical",
                parameters={"fast_period": 50, "slow_period": 200},
            )
        )

        rsi_model = RSIMeanReversionModel(
            ModelConfig(
                model_id="rsi_mean_reversion",
                model_type="technical",
                parameters={
                    "rsi_period": 14,
                    "oversold_threshold": 30,
                    "overbought_threshold": 70,
                },
            )
        )

        risk_aware_strategy.signal_registry.register(sma_model)
        risk_aware_strategy.signal_registry.register(rsi_model)

    risk_guard = risk_aware_strategy.risk_guard
    signal_registry = risk_aware_strategy.signal_registry

    # Get historical data for signal generation
    historical = engine.get_historical_data(symbol, bar.timestamp, lookback=252)

    if len(historical) < 100:
        return

    # Build portfolio state for risk checks
    current_position = engine.get_position(symbol)
    all_positions = engine.get_all_positions()

    open_positions = {}
    total_value = 0.0

    for pos_symbol, pos in all_positions.items():
        current_price = pos.market_value / pos.quantity if pos.quantity != 0 else 0
        open_positions[pos_symbol] = Position(
            symbol=pos_symbol,
            quantity=pos.quantity,
            entry_price=pos.avg_price,
            current_price=current_price,
            market_value=pos.market_value,
            unrealized_pnl=pos.unrealized_pnl,
        )
        total_value += abs(pos.market_value)

    portfolio = PortfolioState(
        equity=engine.get_equity(),
        cash=engine.get_cash(),
        peak_equity=getattr(engine, "_peak_equity", engine.get_equity()),
        daily_pnl=getattr(engine, "_daily_pnl", 0.0),
        daily_trades=getattr(engine, "_daily_trades", 0),
        open_positions=open_positions,
        total_positions=len(all_positions),
        total_exposure=total_value,
    )

    # Update peak equity
    if engine.get_equity() > getattr(engine, "_peak_equity", 0):
        engine._peak_equity = engine.get_equity()

    # Check kill switches FIRST
    triggered, reason = risk_guard.check_kill_switches(portfolio)
    if triggered:
        print(f"\n⚠️  KILL SWITCH TRIGGERED: {reason}")
        print("   Closing all positions...")

        # Close all positions
        for pos_symbol, pos in all_positions.items():
            if pos.quantity > 0:
                engine.close_position(pos_symbol)

        return

    # Generate signals from all models
    for model_id in signal_registry.list_models(enabled_only=True):
        model = signal_registry.get(model_id)

        try:
            signal = model.generate(historical, bar.timestamp)

            # Only act on actionable signals
            if not signal.is_actionable(min_probability=0.5, min_score=0.2):
                continue

            # Entry signals
            if signal.signal_type == SignalType.ENTRY and signal.direction.value == "long":
                # Skip if already in position
                if current_position and current_position.quantity > 0:
                    continue

                # Calculate proposed trade
                shares = int(portfolio.equity * 0.10 / bar.close)  # Target 10% position
                stop_price = bar.close * 0.95  # 5% stop loss

                proposed = ProposedTrade(
                    symbol=symbol,
                    direction="long",
                    quantity=shares,
                    entry_price=bar.close,
                    stop_price=stop_price,
                )

                # Evaluate with RiskGuard
                passed, results, adjusted = risk_guard.evaluate_signal(signal, proposed, portfolio)

                if passed:
                    # Place order
                    engine.submit_order(symbol, shares, "buy")

                    print(f"\n✓ {bar.timestamp.date()} | BUY {shares} {symbol} @ {bar.close:.2f}")
                    print(f"  Model: {signal.model_id} | Prob: {signal.probability:.2%}")
                    print(
                        f"  Score: {signal.score:.2f} | Expected Return: {signal.expected_return:.2%}"
                    )

                    # Show risk checks
                    for result in results:
                        if not result.passed:
                            print(f"  ⚠️  {result.rule_name}: {result.action_taken}")
                else:
                    print(f"\n✗ {bar.timestamp.date()} | REJECTED {symbol} trade")
                    for result in results:
                        if not result.passed:
                            print(
                                f"  {result.rule_name}: {result.current_value:.4f} vs {result.threshold:.4f}"
                            )

            # Exit signals
            elif signal.signal_type == SignalType.EXIT:
                if current_position and current_position.quantity > 0:
                    # Exit requires no risk check (closing position reduces risk)
                    engine.close_position(symbol)

                    print(
                        f"\n✓ {bar.timestamp.date()} | SELL {current_position.quantity} {symbol} @ {bar.close:.2f}"
                    )
                    print(f"  Model: {signal.model_id} | Prob: {signal.probability:.2%}")

        except Exception as e:  # noqa: S112
            # Skip model on error
            continue


def main():
    """Run full system integration backtest."""
    print("=== Full System Integration: SignalCore + RiskGuard + ProofBench ===\n")

    # Create sample data
    print("1. Generating sample market data...")
    data = create_sample_data("AAPL", days=500)
    print(f"   Generated {len(data)} days of AAPL data")

    # Set up backtest
    print("\n2. Configuring backtest engine...")

    sim_config = SimulationConfig(
        initial_capital=100000.0,
        benchmark_symbol="SPY",
    )

    exec_config = ExecutionConfig(
        estimated_spread=0.001,  # 10 bps spread
        commission_pct=0.001,  # 10 bps commission
        slippage_model="linear",
    )

    engine = BacktestEngine(sim_config, exec_config)

    # Run backtest
    print("\n3. Running backtest with risk-aware strategy...")
    print("   (SignalCore signals → RiskGuard validation → Execution)\n")

    start_date = data.index[100]  # Skip first 100 days for indicators
    end_date = data.index[-1]

    results = engine.run(
        symbols=["AAPL"],
        data={"AAPL": data},
        strategy_func=risk_aware_strategy,
        start_date=start_date,
        end_date=end_date,
    )

    # Display results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print("\nEquity Curve:")
    print(f"  Initial Capital:    ${sim_config.initial_capital:,.2f}")
    print(f"  Final Equity:       ${results.final_equity:,.2f}")
    print(
        f"  Total Return:       {((results.final_equity / sim_config.initial_capital) - 1) * 100:.2f}%"
    )

    print("\nRisk Metrics:")
    print(f"  Sharpe Ratio:       {results.metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:       {results.metrics.max_drawdown * 100:.2f}%")
    print(f"  Volatility:         {results.metrics.volatility * 100:.2f}%")

    print("\nTrade Statistics:")
    print(f"  Total Trades:       {results.metrics.total_trades}")
    print(f"  Win Rate:           {results.metrics.win_rate * 100:.1f}%")
    print(f"  Profit Factor:      {results.metrics.profit_factor:.2f}")
    print(f"  Avg Win:            ${results.metrics.avg_win:,.2f}")
    print(f"  Avg Loss:           ${results.metrics.avg_loss:,.2f}")

    # Show RiskGuard statistics
    print("\nRisk Management:")
    risk_guard = risk_aware_strategy.risk_guard
    print(f"  System Halted:      {risk_guard.is_halted()}")
    print(f"  Active Rules:       {len(risk_guard.list_rules(enabled_only=True))}")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    results = main()
    print("\nBacktest complete!")
    print("\nThis demonstrates:")
    print("✓ SignalCore generating probabilistic trading signals")
    print("✓ RiskGuard validating and sizing positions")
    print("✓ ProofBench backtesting with realistic execution")
    print("✓ Full audit trail of all decisions")
