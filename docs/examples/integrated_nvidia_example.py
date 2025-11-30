"""
Complete Integrated Workflow with NVIDIA AI.

Demonstrates all engines working together with AI enhancement:
1. Cortex: Strategy hypothesis generation
2. SignalCore: AI-enhanced signal generation
3. RiskGuard: AI-explained risk evaluation
4. ProofBench: AI-narrated backtest analysis

Get API key: https://build.nvidia.com/
"""

from datetime import datetime
import os

import pandas as pd

# Import all NVIDIA-enhanced engines
from engines.cortex import CortexEngine, OutputType
from engines.proofbench import (
    LLMPerformanceNarrator,
    PerformanceMetrics,
    SimulationConfig,
    SimulationResults,
)
from engines.proofbench.core.portfolio import Portfolio
from engines.riskguard import (
    STANDARD_RISK_RULES,
    LLMEnhancedRiskGuard,
    LLMRiskAnalyzer,
    RiskGuardEngine,
)
from engines.riskguard.core.engine import PortfolioState, ProposedTrade
from engines.signalcore.core.signal import Direction
from engines.signalcore.models import LLMEnhancedModel, RSIMeanReversionModel
from engines.signalcore.models.llm_enhanced import LLMFeatureEngineer

# ==================== Configuration ====================

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")


# ==================== Example: Complete AI-Powered Trading Workflow ====================


def example_complete_workflow():  # noqa: PLR0915
    """Complete workflow with all NVIDIA-enhanced engines."""
    print("=" * 80)
    print("COMPLETE AI-POWERED TRADING WORKFLOW")
    print("=" * 80)

    if not NVIDIA_API_KEY:
        print("\nNote: NVIDIA_API_KEY not set. Will use rule-based fallbacks.")
        print("To enable full NVIDIA AI: export NVIDIA_API_KEY='nvapi-...'")
        print("Get your key: https://build.nvidia.com/\n")
    else:
        print("\n✓ NVIDIA AI Enabled - Full power mode!\n")

    # ==================== Phase 1: Strategy Generation (Cortex) ====================

    print("\n" + "=" * 80)
    print("PHASE 1: AI-POWERED STRATEGY GENERATION (Cortex)")
    print("=" * 80)

    cortex = CortexEngine(
        nvidia_api_key=NVIDIA_API_KEY,
        usd_code_enabled=True,
        embeddings_enabled=True,
    )

    # Define market conditions
    market_context = {
        "regime": "trending",
        "volatility": "low",
        "trend_strength": 0.75,
        "sector": "technology",
    }

    constraints = {
        "instrument_class": "equity",
        "max_position_pct": 0.10,
        "risk_tolerance": "moderate",
    }

    print("\n[1.1] Market Context Analysis")
    print(f"  Regime: {market_context['regime']}")
    print(f"  Volatility: {market_context['volatility']}")
    print(f"  Trend Strength: {market_context['trend_strength']}")

    # Generate hypothesis
    print("\n[1.2] Generating Strategy Hypothesis with AI...")
    hypothesis = cortex.generate_hypothesis(market_context, constraints)

    print("\n[1.3] Generated Strategy:")
    print(f"  Name: {hypothesis.name}")
    print(f"  Type: {hypothesis.strategy_type}")
    print(f"  Confidence: {hypothesis.confidence:.1%}")
    print(f"  Expected Sharpe: {hypothesis.expected_sharpe:.2f}")
    print(f"  Entry Conditions: {hypothesis.entry_conditions[:2]}")
    print("  Risk Parameters:")
    print(f"    - Max Position: {hypothesis.max_position_size_pct:.1%}")
    print(f"    - Stop Loss: {hypothesis.stop_loss_pct:.1%}")
    print(f"    - Take Profit: {hypothesis.take_profit_pct:.1%}")

    # Analyze strategy code
    strategy_code = """
def generate_signal(data, fast_ma, slow_ma):
    if data['sma_50'][-1] > data['sma_200'][-1]:
        if data['volume'][-1] > data['volume_avg']:
            return 'BUY'
    return 'HOLD'
"""

    print("\n[1.4] AI Code Analysis of Strategy...")
    code_analysis = cortex.analyze_code(strategy_code, "review")

    print(f"  Model: {code_analysis.model_used}")
    print(f"  Quality: {code_analysis.content['analysis']['code_quality']}")

    # ==================== Phase 2: Signal Generation (SignalCore) ====================

    print("\n" + "=" * 80)
    print("PHASE 2: AI-ENHANCED SIGNAL GENERATION (SignalCore)")
    print("=" * 80)

    # Create base RSI model
    from engines.signalcore.core.model import ModelConfig

    rsi_config = ModelConfig(
        model_id="rsi-ai-enhanced",
        model_type="mean_reversion",
        parameters={
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
        },
    )

    base_model = RSIMeanReversionModel(rsi_config)

    # Enhance with LLM
    print("\n[2.1] Creating AI-Enhanced Signal Model...")
    signalcore = LLMEnhancedModel(
        base_model=base_model,
        nvidia_api_key=NVIDIA_API_KEY,
        llm_enabled=True,
    )

    print(f"  Base Model: {base_model.config.model_id}")
    print(f"  AI Enhancement: {'Enabled' if signalcore.llm_enabled else 'Disabled'}")

    # Generate market data for signal
    print("\n[2.2] Generating Trading Signal...")

    # Create sample price data (oversold condition)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    prices = [100 - i * 0.5 for i in range(100)]  # Declining trend

    sample_data = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000000] * 100,
        }
    )

    signal = signalcore.generate(sample_data, datetime.utcnow())

    print("\n[2.3] Signal Generated:")
    print(f"  Symbol: {signal.symbol}")
    print(f"  Type: {signal.signal_type.value}")
    print(f"  Direction: {signal.direction.value}")
    print(f"  Probability: {signal.probability:.1%}")
    print(f"  Expected Return: {signal.expected_return:.2%}")
    print(f"  Score: {signal.score:.2f}")

    if "llm_interpretation" in signal.metadata:
        print("\n[2.4] AI Signal Interpretation:")
        print(f"  {signal.metadata['llm_interpretation']}")
        print(f"  Model: {signal.metadata.get('llm_model', 'N/A')}")

    # Feature engineering with AI
    print("\n[2.5] AI-Powered Feature Suggestions...")
    engineer = LLMFeatureEngineer(nvidia_api_key=NVIDIA_API_KEY)
    features = engineer.suggest_features(sample_data, "mean_reversion")

    print(f"  Suggested Features: {', '.join(features[:5])}")

    # ==================== Phase 3: Risk Evaluation (RiskGuard) ====================

    print("\n" + "=" * 80)
    print("PHASE 3: AI-EXPLAINED RISK EVALUATION (RiskGuard)")
    print("=" * 80)

    # Create AI-enhanced risk guard
    base_riskguard = RiskGuardEngine(rules=STANDARD_RISK_RULES.copy())

    print("\n[3.1] Initializing AI-Enhanced RiskGuard...")
    riskguard = LLMEnhancedRiskGuard(
        base_engine=base_riskguard,
        nvidia_api_key=NVIDIA_API_KEY,
        llm_enabled=True,
    )

    print(f"  Rules Loaded: {len(riskguard.list_rules())}")
    print(f"  AI Enhancement: {'Enabled' if riskguard.llm_enabled else 'Disabled'}")

    # Create portfolio state
    portfolio = PortfolioState(
        equity=100000.0,
        cash=60000.0,
        peak_equity=105000.0,
        daily_pnl=1200.0,
        daily_trades=5,
        open_positions={},
        total_positions=3,
        total_exposure=40000.0,
        sector_exposures={"Technology": 25000.0},
        correlated_exposure=5000.0,
    )

    # Propose trade based on signal
    proposed_trade = ProposedTrade(
        symbol=signal.symbol,
        direction="long" if signal.direction == Direction.LONG else "short",
        quantity=100,
        entry_price=50.0,
        stop_price=48.5,
        target_price=53.0,
        sector="Technology",
    )

    print("\n[3.2] Evaluating Trade with AI Risk Analysis...")
    print(f"  Symbol: {proposed_trade.symbol}")
    print(f"  Direction: {proposed_trade.direction}")
    print(f"  Quantity: {proposed_trade.quantity}")
    print(f"  Entry: ${proposed_trade.entry_price:.2f}")
    print(f"  Stop: ${proposed_trade.stop_price:.2f}")

    passed, results, adjusted_signal = riskguard.evaluate_signal(signal, proposed_trade, portfolio)

    print("\n[3.3] Risk Evaluation Results:")
    print(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}")
    print(f"  Checks Run: {len(results)}")
    print(f"  Checks Passed: {len([r for r in results if r.passed])}")

    if not passed:
        failed = [r for r in results if not r.passed]
        print("\n  Failed Checks:")
        for result in failed[:3]:
            print(f"    - {result.rule_name}: {result.action_taken}")

    if adjusted_signal and "risk_explanation" in adjusted_signal.metadata:
        print("\n[3.4] AI Risk Explanation:")
        print(f"  {adjusted_signal.metadata['risk_explanation']}")

    # Risk scenario analysis
    print("\n[3.5] AI Risk Scenario Analysis...")
    analyzer = LLMRiskAnalyzer(nvidia_api_key=NVIDIA_API_KEY)

    scenario_analysis = analyzer.analyze_risk_scenario(
        "Market volatility spike with 20% sector rotation", portfolio
    )

    print("  Scenario: Market volatility spike")
    print(f"  Model: {scenario_analysis.get('llm_model', 'rule-based')}")

    # ==================== Phase 4: Backtesting (ProofBench) ====================

    print("\n" + "=" * 80)
    print("PHASE 4: AI-NARRATED BACKTEST ANALYSIS (ProofBench)")
    print("=" * 80)

    # Create mock backtest results
    print("\n[4.1] Running Strategy Backtest...")

    metrics = PerformanceMetrics(
        total_return=0.32,
        annualized_return=0.24,
        volatility=0.14,
        downside_deviation=0.09,
        sharpe_ratio=1.7,
        sortino_ratio=2.7,
        calmar_ratio=2.4,
        max_drawdown=-0.10,
        avg_drawdown=-0.04,
        max_drawdown_duration=22.0,
        num_trades=65,
        win_rate=0.60,
        profit_factor=2.1,
        avg_win=680.0,
        avg_loss=-320.0,
        largest_win=2800.0,
        largest_loss=-700.0,
        avg_trade_duration=4.2,
        expectancy=216.0,
        recovery_factor=3.2,
        equity_final=132000.0,
    )

    config = SimulationConfig(initial_capital=100000.0, risk_free_rate=0.02)
    backtest_portfolio = Portfolio(100000.0)

    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    equity_curve = pd.DataFrame(
        {
            "timestamp": dates,
            "equity": [100000.0 + i * 87.7 for i in range(len(dates))],
        }
    )

    trades = pd.DataFrame({"pnl": [100.0] * 65})

    backtest_results = SimulationResults(
        config=config,
        metrics=metrics,
        portfolio=backtest_portfolio,
        equity_curve=equity_curve,
        trades=trades,
        orders=[],
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 12, 31),
    )

    print("  Period: 2024-01-01 to 2024-12-31")
    print(f"  Total Trades: {metrics.num_trades}")
    print(f"  Final Equity: ${metrics.equity_final:,.0f}")

    # AI-powered performance narration
    print("\n[4.2] Generating AI Performance Narration...")
    narrator = LLMPerformanceNarrator(nvidia_api_key=NVIDIA_API_KEY)

    narration = narrator.narrate_results(backtest_results)

    print("\n[4.3] Performance Summary:")
    print(f"  Model: {narration['llm_model']}")
    print(f"  Total Return: {narration['metrics_summary']['total_return']:.2%}")
    print(f"  Sharpe Ratio: {narration['metrics_summary']['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {narration['metrics_summary']['max_drawdown']:.2%}")
    print(f"  Win Rate: {narration['metrics_summary']['win_rate']:.1%}")

    print("\n[4.4] AI Narration:")
    print(f"{narration['narration']}")

    # Optimization suggestions
    print("\n[4.5] AI Optimization Suggestions...")
    suggestions = narrator.suggest_optimizations(backtest_results, focus="returns")

    print("  Focus: Maximizing Returns")
    for i, suggestion in enumerate(suggestions[:3], 1):
        print(f"  {i}. {suggestion}")

    # ==================== Phase 5: Summary & Next Steps ====================

    print("\n" + "=" * 80)
    print("PHASE 5: WORKFLOW SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    print("\n[5.1] Complete Workflow Executed:")
    print(f"  ✓ Strategy Generated: {hypothesis.name}")
    print(f"  ✓ Signal Generated: {signal.signal_type.value} {signal.direction.value}")
    print(f"  ✓ Risk Evaluated: {'PASSED' if passed else 'NEEDS REVIEW'}")
    print(f"  ✓ Backtest Analyzed: {metrics.total_return:.1%} return")

    print("\n[5.2] AI Models Utilized:")
    print("  - Cortex: Llama 3.1 405B (Strategy & Code Analysis)")
    print("  - Cortex: NV-Embed-QA (Semantic Understanding)")
    print("  - SignalCore: Llama 3.1 70B (Signal Interpretation)")
    print("  - RiskGuard: Llama 3.1 70B (Risk Explanation)")
    print("  - ProofBench: Llama 3.1 70B (Performance Narration)")

    print("\n[5.3] Key Insights:")
    print(f"  • Strategy Confidence: {hypothesis.confidence:.1%}")
    print(f"  • Signal Probability: {signal.probability:.1%}")
    print(f"  • Risk Checks: {len([r for r in results if r.passed])}/{len(results)} passed")
    print(f"  • Backtest Sharpe: {metrics.sharpe_ratio:.2f}")

    print("\n[5.4] Recommended Next Steps:")
    print("  1. Review AI insights and explanations")
    print("  2. Implement suggested optimizations")
    print("  3. Monitor strategy performance in paper trading")
    print("  4. Refine parameters based on AI recommendations")

    # Review all Cortex outputs
    cortex_outputs = cortex.get_outputs()
    print("\n[5.5] Cortex AI Activity:")
    print(f"  Total Outputs: {len(cortex_outputs)}")
    for output_type in [OutputType.HYPOTHESIS, OutputType.CODE_ANALYSIS]:
        count = len([o for o in cortex_outputs if o.output_type == output_type])
        print(f"  - {output_type.value}: {count}")


# ==================== Main ====================

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  INTELLIGENT INVESTOR - COMPLETE AI-POWERED WORKFLOW".center(78) + "║")
    print("║" + "  Powered by NVIDIA AI Models".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    try:
        example_complete_workflow()

        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
        print("Next Steps:")
        print("1. Get NVIDIA API key: https://build.nvidia.com/")
        print("2. Set environment: export NVIDIA_API_KEY='nvapi-...'")
        print("3. Run with real market data")
        print("4. Deploy to production with FlowRoute")
        print()
        print("All engines are AI-enhanced and ready for maximum POWER! 🚀")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(
            "\nNote: This example requires market data and NVIDIA API key for full functionality."
        )
        print("Run individual engine examples first to verify setup.")
