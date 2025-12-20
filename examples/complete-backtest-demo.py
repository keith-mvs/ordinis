"""
Complete Backtesting Demo with NVIDIA AI.

Demonstrates a full end-to-end trading workflow:
1. Load market data
2. Generate AI-powered strategy (Cortex)
3. Create signals with AI interpretation (SignalCore)
4. Evaluate with AI risk analysis (RiskGuard)
5. Run backtest simulation (ProofBench)
6. Analyze results with AI narration

This is a WORKING example you can run immediately.
"""

import os
from pathlib import Path

# Add src to path for imports
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engines.cortex import CortexEngine
from engines.proofbench import (
    LLMPerformanceNarrator,
    SimulationConfig,
    SimulationEngine,
)
from engines.riskguard import (
    STANDARD_RISK_RULES,
    LLMEnhancedRiskGuard,
    RiskGuardEngine,
)
from engines.riskguard.core.engine import PortfolioState, ProposedTrade
from engines.signalcore.core.model import ModelConfig
from engines.signalcore.core.signal import Direction, SignalType
from engines.signalcore.models import LLMEnhancedModel, RSIMeanReversionModel


def generate_sample_market_data(
    symbol: str, start_date: str, end_date: str, trend: str = "neutral"
) -> pd.DataFrame:
    """
    Generate realistic simulated market data for backtesting.

    Args:
        symbol: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        trend: Market trend ("up", "down", "neutral", "volatile")

    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)

    # Base price
    base_price = 100.0

    # Generate price movement based on trend
    if trend == "up":
        trend_component = np.linspace(0, 20, n)
        volatility = 0.02
    elif trend == "down":
        trend_component = np.linspace(0, -20, n)
        volatility = 0.02
    elif trend == "volatile":
        trend_component = np.sin(np.linspace(0, 4 * np.pi, n)) * 10
        volatility = 0.04
    else:  # neutral
        trend_component = np.sin(np.linspace(0, 2 * np.pi, n)) * 5
        volatility = 0.015

    # Random walk
    returns = np.random.normal(0, volatility, n)
    price = base_price + trend_component + np.cumsum(returns)

    # Generate OHLCV
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": price * (1 + np.random.uniform(-0.005, 0.005, n)),
            "high": price * (1 + np.random.uniform(0.005, 0.015, n)),
            "low": price * (1 + np.random.uniform(-0.015, -0.005, n)),
            "close": price,
            "volume": np.random.randint(500000, 2000000, n),
        }
    )

    # Ensure high >= close >= low
    data["high"] = data[["high", "close"]].max(axis=1)
    data["low"] = data[["low", "close"]].min(axis=1)

    # Set timestamp as index (required by SimulationEngine)
    data.set_index("timestamp", inplace=True)

    return data


def run_complete_backtest(nvidia_api_key: str | None = None):  # noqa: PLR0915
    """
    Run complete backtesting workflow with AI enhancement.

    Args:
        nvidia_api_key: NVIDIA API key (optional, uses fallback if None)
    """
    print("\n" + "=" * 80)
    print("COMPLETE AI-POWERED BACKTESTING DEMO")
    print("=" * 80)

    if nvidia_api_key:
        print("\n[OK] NVIDIA AI Enabled - Full AI insights activated!")
    else:
        print("\n[INFO] Running with rule-based fallbacks (set NVIDIA_API_KEY for AI)")

    # ========== Phase 1: Strategy Generation ==========

    print("\n" + "=" * 80)
    print("PHASE 1: AI STRATEGY GENERATION")
    print("=" * 80)

    cortex = CortexEngine(
        nvidia_api_key=nvidia_api_key, usd_code_enabled=True, embeddings_enabled=True
    )

    market_context = {
        "regime": "mean_reverting",
        "volatility": "medium",
        "trend_strength": 0.4,
    }

    print("\n[1.1] Market Analysis:")
    print(f"  Regime: {market_context['regime']}")
    print(f"  Volatility: {market_context['volatility']}")

    hypothesis = cortex.generate_hypothesis(market_context)

    print("\n[1.2] AI-Generated Strategy:")
    print(f"  Name: {hypothesis.name}")
    print(f"  Type: {hypothesis.strategy_type}")
    print(f"  Confidence: {hypothesis.confidence:.1%}")
    print(f"  Expected Sharpe: {hypothesis.expected_sharpe:.2f}")
    print(f"  Win Rate Expectation: {hypothesis.expected_win_rate:.1%}")

    # ========== Phase 2: Data Preparation ==========

    print("\n" + "=" * 80)
    print("PHASE 2: MARKET DATA PREPARATION")
    print("=" * 80)

    symbol = "DEMO"
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    print("\n[2.1] Generating Market Data:")
    print(f"  Symbol: {symbol}")
    print(f"  Period: {start_date} to {end_date}")
    print("  Pattern: Mean-reverting with volatility")

    market_data = generate_sample_market_data(
        symbol=symbol, start_date=start_date, end_date=end_date, trend="volatile"
    )

    print("\n[2.2] Data Statistics:")
    print(f"  Total Bars: {len(market_data)}")
    print(f"  Start Price: ${market_data['close'].iloc[0]:.2f}")
    print(f"  End Price: ${market_data['close'].iloc[-1]:.2f}")
    print(f"  Price Range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}")

    # ========== Phase 3: Signal Model Setup ==========

    print("\n" + "=" * 80)
    print("PHASE 3: AI-ENHANCED SIGNAL MODEL")
    print("=" * 80)

    # Create RSI mean reversion model (matches our strategy)
    model_config = ModelConfig(
        model_id="rsi-mean-reversion-demo",
        model_type="mean_reversion",
        parameters={
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
        },
    )

    base_model = RSIMeanReversionModel(model_config)

    # Enhance with AI
    signal_model = LLMEnhancedModel(
        base_model=base_model, nvidia_api_key=nvidia_api_key, llm_enabled=True
    )

    print("\n[3.1] Signal Model:")
    print(f"  Model: {model_config.model_id}")
    print(f"  Type: {model_config.model_type}")
    print(f"  AI Enhanced: {signal_model.llm_enabled}")
    print("  Parameters:")
    for key, value in model_config.parameters.items():
        print(f"    - {key}: {value}")

    # ========== Phase 4: Risk Management Setup ==========

    print("\n" + "=" * 80)
    print("PHASE 4: AI-ENHANCED RISK MANAGEMENT")
    print("=" * 80)

    base_riskguard = RiskGuardEngine(rules=STANDARD_RISK_RULES.copy())
    riskguard = LLMEnhancedRiskGuard(
        base_engine=base_riskguard, nvidia_api_key=nvidia_api_key, llm_enabled=True
    )

    print("\n[4.1] Risk Management:")
    print(f"  Total Rules: {len(riskguard.list_rules())}")
    print(f"  AI Explanations: {'Enabled' if riskguard.llm_enabled else 'Disabled'}")
    print("\n[4.2] Key Risk Rules:")
    for rule in list(riskguard.list_rules())[:3]:
        print(f"  - {rule.name}: {rule.threshold}")

    # ========== Phase 5: Run Backtest ==========

    print("\n" + "=" * 80)
    print("PHASE 5: BACKTEST EXECUTION")
    print("=" * 80)

    sim_config = SimulationConfig(initial_capital=100000.0, bar_frequency="1d", risk_free_rate=0.02)

    sim_engine = SimulationEngine(config=sim_config)
    sim_engine.load_data(symbol, market_data)

    print("\n[5.1] Simulation Configuration:")
    print(f"  Initial Capital: ${sim_config.initial_capital:,.0f}")
    print(f"  Frequency: {sim_config.bar_frequency}")
    print(f"  Risk-Free Rate: {sim_config.risk_free_rate:.1%}")

    # Trading logic
    signals_generated = 0
    signals_approved = 0
    signals_rejected = 0

    def trading_strategy(engine, sym, bar):
        """Simple mean reversion strategy with AI enhancement."""
        nonlocal signals_generated, signals_approved, signals_rejected

        # Need enough data for signal generation (RSI model requires 100+ bars)
        current_bar = len(sim_engine.portfolio.equity_curve)
        if current_bar < 100:
            return

        # Generate signal every 5 days
        if current_bar % 5 != 0:
            return

        # Get all data up to current bar (RSI needs full history for calculation)
        recent_data = market_data.iloc[:current_bar]

        # Generate signal
        signal = signal_model.generate(recent_data, bar.timestamp)
        signals_generated += 1

        if signal.signal_type == SignalType.ENTRY and signal.direction == Direction.LONG:
            # Create portfolio state
            portfolio_state = PortfolioState(
                equity=sim_engine.portfolio.equity,
                cash=sim_engine.portfolio.cash,
                peak_equity=max([eq for _, eq in sim_engine.portfolio.equity_curve]),
                daily_pnl=0.0,
                daily_trades=0,
                open_positions={},
                total_positions=len(sim_engine.portfolio.positions),
                total_exposure=sum(
                    pos.market_value for pos in sim_engine.portfolio.positions.values()
                ),
                sector_exposures={},
                correlated_exposure=0.0,
            )

            # Propose trade
            position_size = min(
                100, int(sim_engine.portfolio.cash / bar.close)
            )  # Buy 100 shares or max affordable

            if position_size > 0:
                proposed_trade = ProposedTrade(
                    symbol=sym,
                    direction="long",
                    quantity=position_size,
                    entry_price=bar.close,
                    stop_price=bar.close * 0.95,  # 5% stop loss
                )

                # Risk evaluation
                passed, results, adjusted_signal = riskguard.evaluate_signal(
                    signal, proposed_trade, portfolio_state
                )

                if passed:
                    # Place order
                    from engines.proofbench.core.execution import Order, OrderSide, OrderType

                    order = Order(
                        symbol=sym,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=position_size,
                        timestamp=bar.timestamp,
                    )
                    sim_engine.pending_orders.append(order)
                    signals_approved += 1
                else:
                    signals_rejected += 1

    sim_engine.on_bar = trading_strategy

    print("\n[5.2] Running Simulation...")
    print(f"  Processing {len(market_data)} bars...")

    results = sim_engine.run()

    print("\n[5.3] Simulation Complete!")
    print(f"  Signals Generated: {signals_generated}")
    print(f"  Signals Approved: {signals_approved}")
    print(f"  Signals Rejected: {signals_rejected}")

    # ========== Phase 6: Performance Analysis ==========

    print("\n" + "=" * 80)
    print("PHASE 6: AI-POWERED PERFORMANCE ANALYSIS")
    print("=" * 80)

    narrator = LLMPerformanceNarrator(nvidia_api_key=nvidia_api_key)

    print("\n[6.1] Performance Metrics:")
    print(f"  Total Return: {results.metrics.total_return:.2%}")
    print(f"  Annualized Return: {results.metrics.annualized_return:.2%}")
    print(f"  Sharpe Ratio: {results.metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {results.metrics.sortino_ratio:.2f}")
    print(f"  Max Drawdown: {results.metrics.max_drawdown:.2%}")
    print(f"  Win Rate: {results.metrics.win_rate:.1%}")
    print(f"  Profit Factor: {results.metrics.profit_factor:.2f}")
    print(f"  Total Trades: {results.metrics.num_trades}")

    # AI Narration
    print("\n[6.2] Generating AI Performance Narration...")
    narration = narrator.narrate_results(results)

    print(f"\n{'='*80}")
    print("AI PERFORMANCE NARRATION")
    print(f"{'='*80}")
    print(f"\nModel: {narration['llm_model']}")
    print(f"\n{narration['narration']}")

    # Trade Pattern Analysis
    print("\n[6.3] AI Trade Pattern Analysis...")
    pattern_analysis = narrator.analyze_trade_patterns(results)

    print(f"\n{'='*80}")
    print("TRADE PATTERN ANALYSIS")
    print(f"{'='*80}")
    print(f"\n{pattern_analysis['analysis']}")

    # Optimization Suggestions
    print("\n[6.4] AI Optimization Suggestions...")
    suggestions = narrator.suggest_optimizations(results, focus="returns")

    print(f"\n{'='*80}")
    print("OPTIMIZATION SUGGESTIONS (Focus: Returns)")
    print(f"{'='*80}")
    for i, suggestion in enumerate(suggestions[:5], 1):
        print(f"{i}. {suggestion}")

    # ========== Summary ==========

    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)

    print(f"\n[OK] Strategy: {hypothesis.name}")
    print(f"[OK] Period: {start_date} to {end_date}")
    print(f"[OK] Initial Capital: ${sim_config.initial_capital:,.0f}")
    print(f"[OK] Final Equity: ${results.metrics.equity_final:,.0f}")
    print(f"[OK] Total Return: {results.metrics.total_return:.2%}")
    print(f"[OK] Sharpe Ratio: {results.metrics.sharpe_ratio:.2f}")
    print(f"[OK] Max Drawdown: {results.metrics.max_drawdown:.2%}")
    print(f"[OK] Win Rate: {results.metrics.win_rate:.1%}")
    print(f"[OK] Total Trades: {results.metrics.num_trades}")

    print(f"\n{'='*80}")
    print("DEMO COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " INTELLIGENT INVESTOR - COMPLETE BACKTEST DEMO ".center(78) + "|")
    print("|" + " Powered by NVIDIA AI Models ".center(78) + "|")
    print("+" + "=" * 78 + "+")

    # Get API key from environment
    nvidia_key = os.getenv("NVIDIA_API_KEY")

    if not nvidia_key:
        print("\nNote: NVIDIA_API_KEY not set. Running with rule-based fallbacks.")
        print("To enable AI: export NVIDIA_API_KEY='nvapi-...'")
        print("Get your key: https://build.nvidia.com/\n")

    try:
        # Run the complete backtest
        results = run_complete_backtest(nvidia_api_key=nvidia_key)

        print("\nNext Steps:")
        print("1. Try with different market conditions (edit generate_sample_market_data)")
        print("2. Modify strategy parameters in model_config")
        print("3. Add your NVIDIA API key for full AI insights")
        print("4. Experiment with different risk rules")
        print("5. Run with real market data from data providers\n")

    except Exception as e:
        print(f"\n[ERROR] Error during backtest: {e}")
        import traceback

        traceback.print_exc()
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install -e .[ai]")
