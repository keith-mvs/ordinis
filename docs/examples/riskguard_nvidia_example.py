"""
RiskGuard with NVIDIA AI Integration Example.

Demonstrates:
- LLM-enhanced trade evaluation with explanations
- Risk scenario analysis
- Rule optimization suggestions
- Integration with SignalCore

Get API key: https://build.nvidia.com/
"""

from datetime import datetime
import os

from engines.riskguard import (
    STANDARD_RISK_RULES,
    LLMEnhancedRiskGuard,
    LLMRiskAnalyzer,
    RiskGuardEngine,
)
from engines.riskguard.core.engine import PortfolioState, ProposedTrade
from engines.signalcore.core.signal import Direction, Signal, SignalType

# ==================== Configuration ====================

# Option 1: Set API key via environment variable
# os.environ["NVIDIA_API_KEY"] = "nvapi-..."

# Option 2: Pass API key directly (for demonstration)
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")  # None for rule-based fallback


# ==================== Example 1: Basic RiskGuard (Rule-Based) ====================


def example_basic_riskguard():
    """Basic RiskGuard usage without NVIDIA (rule-based)."""
    print("=" * 60)
    print("Example 1: Basic RiskGuard (Rule-Based)")
    print("=" * 60)

    # Create engine with standard rules
    riskguard = RiskGuardEngine(rules=STANDARD_RISK_RULES.copy())

    print(f"\nTotal Rules: {len(riskguard.list_rules())}")
    print(f"Enabled Rules: {len(riskguard.list_rules(enabled_only=True))}")

    # Create portfolio state
    portfolio = PortfolioState(
        equity=100000.0,
        cash=60000.0,
        peak_equity=105000.0,
        daily_pnl=1500.0,
        daily_trades=5,
        open_positions={},
        total_positions=3,
        total_exposure=40000.0,
        sector_exposures={"Technology": 20000.0, "Healthcare": 15000.0},
        correlated_exposure=5000.0,
    )

    # Create signal and proposed trade
    signal = Signal(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        signal_type=SignalType.ENTRY,
        direction=Direction.LONG,
        probability=0.75,
        expected_return=0.08,
        confidence_interval=(0.05, 0.12),
        score=0.8,
        model_id="sma_crossover",
        model_version="1.0.0",
    )

    trade = ProposedTrade(
        symbol="AAPL",
        direction="long",
        quantity=100,
        entry_price=150.0,
        stop_price=145.0,
        target_price=165.0,
        sector="Technology",
    )

    # Evaluate trade
    passed, results, adjusted_signal = riskguard.evaluate_signal(signal, trade, portfolio)

    print(f"\nTrade Evaluation: {'PASSED' if passed else 'FAILED'}")
    print(f"Checks Run: {len(results)}")
    print(f"Checks Passed: {len([r for r in results if r.passed])}")
    print(f"Checks Failed: {len([r for r in results if not r.passed])}")

    # Show failed checks
    failed = [r for r in results if not r.passed]
    if failed:
        print("\nFailed Checks:")
        for result in failed:
            print(f"  - {result.rule_name}: {result.message}")
            print(f"    Current: {result.current_value:.4f}, Threshold: {result.threshold:.4f}")
            print(f"    Action: {result.action_taken}")


# ==================== Example 2: RiskGuard with NVIDIA AI ====================


def example_nvidia_riskguard():
    """RiskGuard with NVIDIA AI integration."""
    print("\n" + "=" * 60)
    print("Example 2: RiskGuard with NVIDIA AI")
    print("=" * 60)

    # Check if API key is available
    if not NVIDIA_API_KEY:
        print("\nNote: NVIDIA_API_KEY not set. Will use rule-based fallback.")
        print("To enable NVIDIA: export NVIDIA_API_KEY='nvapi-...'")
        print("Get your key: https://build.nvidia.com/\n")

    # Create base engine
    base_engine = RiskGuardEngine(rules=STANDARD_RISK_RULES.copy())

    # Create LLM-enhanced engine
    riskguard = LLMEnhancedRiskGuard(
        base_engine=base_engine, nvidia_api_key=NVIDIA_API_KEY, llm_enabled=True
    )

    print(f"LLM Enabled: {riskguard.llm_enabled}")
    print(f"Total Rules: {len(riskguard.list_rules())}")

    # Create portfolio state
    portfolio = PortfolioState(
        equity=100000.0,
        cash=50000.0,
        peak_equity=100000.0,
        daily_pnl=-1200.0,  # Negative P&L
        daily_trades=8,
        open_positions={},
        total_positions=5,
        total_exposure=50000.0,
        sector_exposures={"Technology": 30000.0},
        correlated_exposure=8000.0,
    )

    # Create signal and proposed trade
    signal = Signal(
        symbol="NVDA",
        timestamp=datetime.utcnow(),
        signal_type=SignalType.ENTRY,
        direction=Direction.LONG,
        probability=0.82,
        expected_return=0.12,
        confidence_interval=(0.08, 0.16),
        score=0.85,
        model_id="rsi_mean_reversion",
        model_version="1.0.0",
    )

    trade = ProposedTrade(
        symbol="NVDA",
        direction="long",
        quantity=50,
        entry_price=450.0,
        stop_price=440.0,
        target_price=475.0,
        sector="Technology",
    )

    # Evaluate trade (with LLM explanation if API key provided)
    passed, results, adjusted_signal = riskguard.evaluate_signal(signal, trade, portfolio)

    print(f"\nTrade Evaluation: {'PASSED' if passed else 'FAILED'}")
    print(f"Checks Passed: {len([r for r in results if r.passed])}/{len(results)}")

    # Show LLM explanation if available
    if adjusted_signal and "risk_explanation" in adjusted_signal.metadata:
        print("\nLLM Risk Analysis:")
        print(f"Model: {adjusted_signal.metadata.get('risk_llm_model', 'N/A')}")
        print(f"\n{adjusted_signal.metadata['risk_explanation']}")
    else:
        print("\nNo LLM explanation (enable with NVIDIA_API_KEY)")

    # Show failed checks
    failed = [r for r in results if not r.passed]
    if failed:
        print("\nFailed Checks:")
        for result in failed:
            print(f"  - {result.rule_name}")
            print(f"    Current: {result.current_value:.4f}, Threshold: {result.threshold:.4f}")


# ==================== Example 3: Risk Scenario Analysis ====================


def example_risk_analysis():
    """Risk scenario analysis with NVIDIA AI."""
    print("\n" + "=" * 60)
    print("Example 3: Risk Scenario Analysis")
    print("=" * 60)

    # Create analyzer
    analyzer = LLMRiskAnalyzer(nvidia_api_key=NVIDIA_API_KEY)

    # Create portfolio state
    portfolio = PortfolioState(
        equity=100000.0,
        cash=40000.0,
        peak_equity=110000.0,
        daily_pnl=-2500.0,
        daily_trades=12,
        open_positions={},
        total_positions=8,
        total_exposure=60000.0,
        sector_exposures={"Technology": 35000.0, "Finance": 20000.0},
        correlated_exposure=15000.0,
    )

    # Analyze risk scenario
    print("\n[Scenario 1] Market Volatility Spike")
    print("-" * 40)

    scenario = "Sudden 30% increase in market volatility with sector rotation"
    analysis = analyzer.analyze_risk_scenario(scenario, portfolio)

    print(f"Model Used: {analysis.get('llm_model', 'N/A')}")
    print(f"\nAnalysis:\n{analysis['analysis']}")

    # Optimize rules based on performance
    print("\n[Scenario 2] Rule Optimization")
    print("-" * 40)

    performance = {"sharpe_ratio": 0.9, "max_drawdown": -0.18, "win_rate": 0.48}

    print("Current Performance:")
    print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {performance['max_drawdown']:.1%}")
    print(f"  Win Rate: {performance['win_rate']:.1%}")

    suggestions = analyzer.suggest_rule_optimization(portfolio, performance)

    print("\nOptimization Suggestions:")
    for i, suggestion in enumerate(suggestions[:5], 1):
        print(f"{i}. {suggestion}")

    # Explain a rule
    print("\n[Scenario 3] Rule Explanation")
    print("-" * 40)

    explanation = analyzer.explain_rule(
        "Daily Loss Limit", "Maximum daily loss of -3% before trading halts"
    )

    print("Rule: Daily Loss Limit")
    print(f"\nExplanation:\n{explanation}")


# ==================== Example 4: Integrated Workflow ====================


def example_integrated_workflow():
    """Complete workflow with SignalCore and RiskGuard."""
    print("\n" + "=" * 60)
    print("Example 4: Integrated Workflow")
    print("=" * 60)

    # Step 1: Initialize RiskGuard with NVIDIA
    riskguard = LLMEnhancedRiskGuard(
        base_engine=RiskGuardEngine(rules=STANDARD_RISK_RULES.copy()),
        nvidia_api_key=NVIDIA_API_KEY,
        llm_enabled=True,
    )

    analyzer = LLMRiskAnalyzer(nvidia_api_key=NVIDIA_API_KEY)

    print("\n[Step 1] Initialize RiskGuard")
    print(f"LLM Enabled: {riskguard.llm_enabled}")
    print(f"Rules Loaded: {len(riskguard.list_rules())}")

    # Step 2: Create portfolio
    portfolio = PortfolioState(
        equity=100000.0,
        cash=55000.0,
        peak_equity=105000.0,
        daily_pnl=800.0,
        daily_trades=4,
        open_positions={},
        total_positions=4,
        total_exposure=45000.0,
        sector_exposures={"Technology": 25000.0, "Consumer": 15000.0},
        correlated_exposure=7000.0,
    )

    print("\n[Step 2] Portfolio State")
    print(f"Equity: ${portfolio.equity:,.0f}")
    print(f"Positions: {portfolio.total_positions}")
    print(f"Daily P&L: ${portfolio.daily_pnl:,.0f}")

    # Step 3: Check available capacity
    capacity = riskguard.get_available_capacity("MSFT", portfolio)

    print("\n[Step 3] Available Capacity for MSFT")
    print(f"Max Value: ${capacity['max_value']:,.0f}")
    print(f"Limiting Rule: {capacity['limiting_rule']}")

    # Step 4: Evaluate trade with multiple signals
    trades = [
        ("MSFT", "long", 50, 380.0, 370.0, "Technology"),
        ("JPM", "long", 100, 150.0, 145.0, "Finance"),
        ("JNJ", "long", 80, 160.0, 155.0, "Healthcare"),
    ]

    print("\n[Step 4] Evaluate Multiple Trades")
    print("-" * 40)

    for symbol, direction, qty, entry, stop, sector in trades:
        signal = Signal(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            probability=0.70,
            expected_return=0.06,
            confidence_interval=(0.03, 0.09),
            score=0.72,
            model_id="test_model",
            model_version="1.0.0",
        )

        trade = ProposedTrade(
            symbol=symbol,
            direction=direction,
            quantity=qty,
            entry_price=entry,
            stop_price=stop,
            sector=sector,
        )

        passed, results, _ = riskguard.evaluate_signal(signal, trade, portfolio)

        status = " PASS" if passed else " FAIL"
        print(
            f"{symbol:6} {status}  Checks: {len([r for r in results if r.passed])}/{len(results)}"
        )

    # Step 5: Analyze risk scenarios
    print("\n[Step 5] Risk Scenario Analysis")
    print("-" * 40)

    scenario = "Federal Reserve announces unexpected rate hike"
    analysis = analyzer.analyze_risk_scenario(scenario, portfolio)

    print(f"Scenario: {scenario}")
    print(f"Model: {analysis.get('llm_model', 'rule-based')}")


# ==================== Main ====================

if __name__ == "__main__":
    print("\nRiskGuard with NVIDIA AI Integration\n")

    # Run examples
    example_basic_riskguard()
    example_nvidia_riskguard()
    example_risk_analysis()
    example_integrated_workflow()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Get NVIDIA API key: https://build.nvidia.com/")
    print("2. Set environment variable: export NVIDIA_API_KEY='nvapi-...'")
    print("3. Integrate RiskGuard with ProofBench for backtesting")
    print("4. Monitor risk rules and optimize based on performance")
    print("=" * 60)
