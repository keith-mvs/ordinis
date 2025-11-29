"""
Cortex Engine with NVIDIA AI Integration Example.

Demonstrates:
- Strategy hypothesis generation
- Code analysis using NVIDIA LLM
- Research synthesis
- Integration with SignalCore and RiskGuard

Get API key: https://build.nvidia.com/
"""

import os

from engines.cortex import CortexEngine, OutputType

# ==================== Configuration ====================

# Option 1: Set API key via environment variable
# os.environ["NVIDIA_API_KEY"] = "nvapi-..."

# Option 2: Pass API key directly (for demonstration)
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")  # None for rule-based fallback


# ==================== Example 1: Basic Cortex Usage (Rule-Based) ====================


def example_basic_cortex():
    """Basic Cortex usage without NVIDIA (rule-based fallback)."""
    print("=" * 60)
    print("Example 1: Basic Cortex (Rule-Based)")
    print("=" * 60)

    # Create engine without NVIDIA integration
    cortex = CortexEngine()

    # Generate strategy hypothesis based on market conditions
    market_context = {
        "regime": "trending",
        "volatility": "low",
        "trend_strength": 0.75,
    }

    hypothesis = cortex.generate_hypothesis(market_context)

    print(f"\nGenerated Hypothesis: {hypothesis.name}")
    print(f"Strategy Type: {hypothesis.strategy_type}")
    print(f"Confidence: {hypothesis.confidence:.2%}")
    print(f"Entry Conditions: {hypothesis.entry_conditions}")
    print(f"Expected Sharpe: {hypothesis.expected_sharpe}")

    # Analyze code
    code_sample = """
def calculate_rsi(prices, period=14):
    gains = []
    losses = []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
        else:
            losses.append(abs(change))
    return 100 - (100 / (1 + sum(gains) / sum(losses)))
"""

    analysis = cortex.analyze_code(code_sample, "review")

    print("\nCode Analysis:")
    print(f"Quality: {analysis.content['analysis']['code_quality']}")
    print(f"Suggestions: {analysis.content['analysis']['suggestions']}")
    print(f"Model Used: {analysis.model_used}")


# ==================== Example 2: Cortex with NVIDIA AI ====================


def example_nvidia_cortex():
    """Cortex with NVIDIA AI integration."""
    print("\n" + "=" * 60)
    print("Example 2: Cortex with NVIDIA AI")
    print("=" * 60)

    # Check if API key is available
    if not NVIDIA_API_KEY:
        print("\nNote: NVIDIA_API_KEY not set. Will use rule-based fallback.")
        print("To enable NVIDIA: export NVIDIA_API_KEY='nvapi-...'")
        print("Get your key: https://build.nvidia.com/\n")

    # Create engine with NVIDIA integration
    cortex = CortexEngine(
        nvidia_api_key=NVIDIA_API_KEY,
        usd_code_enabled=True,  # Enable chat model for code analysis
        embeddings_enabled=True,  # Enable embeddings for research
    )

    print(f"NVIDIA Chat Enabled: {cortex.usd_code_enabled}")
    print(f"NVIDIA Embeddings Enabled: {cortex.embeddings_enabled}")

    # Generate hypothesis with constraints
    market_context = {
        "regime": "mean_reverting",
        "volatility": "high",
    }

    constraints = {
        "instrument_class": "options",
        "max_position_pct": 0.05,
    }

    hypothesis = cortex.generate_hypothesis(market_context, constraints)

    print(f"\nGenerated Hypothesis: {hypothesis.name}")
    print(f"Strategy Type: {hypothesis.strategy_type}")
    print(f"Instrument Class: {hypothesis.instrument_class}")
    print(f"Max Position: {hypothesis.max_position_size_pct:.1%}")

    # Analyze code with NVIDIA model (if API key provided)
    trading_code = """
def backtest_strategy(data, entry_signal, exit_signal):
    positions = []
    equity = 10000

    for i in range(len(data)):
        if entry_signal[i] and not positions:
            shares = equity / data[i]['close']
            positions.append({'shares': shares, 'entry': data[i]['close']})
            equity = 0

        elif exit_signal[i] and positions:
            exit_price = data[i]['close']
            pnl = positions[0]['shares'] * (exit_price - positions[0]['entry'])
            equity += positions[0]['shares'] * exit_price
            positions = []

    return equity
"""

    analysis = cortex.analyze_code(trading_code, "review")

    print("\nCode Analysis Results:")
    print(f"Model Used: {analysis.model_used}")
    print(f"Confidence: {analysis.confidence:.1%}")

    if "llm_analysis" in analysis.content["analysis"]:
        print(f"\nAI Analysis:\n{analysis.content['analysis']['llm_analysis']}")
    else:
        print(f"Quality: {analysis.content['analysis']['code_quality']}")
        print(f"Suggestions: {analysis.content['analysis']['suggestions']}")

    # Synthesize research
    research = cortex.synthesize_research(
        query="What are the best practices for options trading?",
        sources=[
            "https://www.investopedia.com/options-basics-tutorial-4583012",
            "https://www.cboe.com/education/getting-started/",
        ],
        context={"focus": "risk_management"},
    )

    print("\nResearch Synthesis:")
    print(f"Query: {research.content['query']}")
    print(f"Sources Analyzed: {research.content['research']['sources_analyzed']}")
    print(f"Key Points: {research.content['research']['key_points']}")


# ==================== Example 3: Multi-Engine Integration ====================


def example_integrated_workflow():
    """Complete workflow with Cortex, SignalCore, and RiskGuard."""
    print("\n" + "=" * 60)
    print("Example 3: Integrated Workflow")
    print("=" * 60)

    # Initialize Cortex
    cortex = CortexEngine(
        nvidia_api_key=NVIDIA_API_KEY,
        usd_code_enabled=True,
    )

    # Step 1: Generate hypothesis
    market_context = {
        "regime": "trending",
        "volatility": "low",
        "sector": "technology",
    }

    hypothesis = cortex.generate_hypothesis(market_context)
    print(f"\n[Step 1] Generated: {hypothesis.name}")
    print(f"Confidence: {hypothesis.confidence:.1%}")

    # Step 2: Review SignalCore output (simulated)
    signalcore_output = {
        "signal_type": "entry",
        "symbol": "AAPL",
        "probability": 0.72,
        "expected_return": 0.08,
        "timeframe": "1d",
    }

    review = cortex.review_output("signalcore", signalcore_output)
    print("\n[Step 2] SignalCore Review:")
    print(f"Assessment: {review.content['assessment']}")
    print(f"Concerns: {review.content['concerns'] or 'None'}")
    print(f"Recommendations: {review.content['recommendations'] or 'None'}")

    # Step 3: Review RiskGuard output (simulated)
    riskguard_output = {
        "approved": True,
        "checks_passed": 8,
        "checks_failed": 0,
        "position_sized": True,
    }

    review = cortex.review_output("riskguard", riskguard_output)
    print("\n[Step 3] RiskGuard Review:")
    print(f"Assessment: {review.content['assessment']}")

    # Step 4: List all hypotheses
    all_hypotheses = cortex.list_hypotheses(min_confidence=0.60)
    print(f"\n[Step 4] Generated {len(all_hypotheses)} hypotheses (>60% confidence)")

    # Step 5: Get all outputs by type
    code_analyses = cortex.get_outputs(OutputType.CODE_ANALYSIS)
    reviews = cortex.get_outputs(OutputType.REVIEW)
    hypotheses_outputs = cortex.get_outputs(OutputType.HYPOTHESIS)

    print("\n[Step 5] Output Summary:")
    print(f"  Code Analyses: {len(code_analyses)}")
    print(f"  Reviews: {len(reviews)}")
    print(f"  Hypotheses: {len(hypotheses_outputs)}")

    # Engine state
    state = cortex.to_dict()
    print("\n[Engine State]:")
    print(f"  Total Outputs: {state['total_outputs']}")
    print(f"  Total Hypotheses: {state['total_hypotheses']}")


# ==================== Main ====================

if __name__ == "__main__":
    print("\nCortex Engine with NVIDIA AI Integration\n")

    # Run examples
    example_basic_cortex()
    example_nvidia_cortex()
    example_integrated_workflow()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Get NVIDIA API key: https://build.nvidia.com/")
    print("2. Set environment variable: export NVIDIA_API_KEY='nvapi-...'")
    print("3. Explore NVIDIA models: https://build.nvidia.com/explore/discover")
    print("4. Integrate Cortex with ProofBench for strategy validation")
    print("=" * 60)
