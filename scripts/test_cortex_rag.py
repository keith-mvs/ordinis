#!/usr/bin/env python
"""
Test Cortex RAG integration.

Verifies that CortexEngine can use RAG context to enhance
hypothesis generation and code analysis.

Usage:
    python scripts/test_cortex_rag.py
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from loguru import logger  # noqa: E402

from engines.cortex.core.engine import CortexEngine  # noqa: E402


def test_hypothesis_generation() -> None:
    """Test hypothesis generation with RAG context."""
    logger.info("=" * 60)
    logger.info("Testing Hypothesis Generation with RAG")
    logger.info("=" * 60)

    # Create Cortex engine with RAG enabled
    logger.info("Creating CortexEngine with rag_enabled=True...")
    engine = CortexEngine(rag_enabled=True)

    # Test market context
    market_context = {
        "regime": "mean_reverting",
        "volatility": "high",
        "trend_strength": 0.3,
    }

    logger.info(f"Market context: {market_context}")
    logger.info("Generating hypothesis...")

    # Generate hypothesis
    hypothesis = engine.generate_hypothesis(market_context)

    logger.info("\nHypothesis Generated:")
    logger.info(f"  ID: {hypothesis.hypothesis_id}")
    logger.info(f"  Name: {hypothesis.name}")
    logger.info(f"  Type: {hypothesis.strategy_type}")
    logger.info(f"  Confidence: {hypothesis.confidence:.2%}")
    logger.info(f"  Description: {hypothesis.description}")
    logger.info(f"  Rationale: {hypothesis.rationale}")

    # Get the output
    outputs = engine.get_outputs()
    if outputs:
        latest_output = outputs[-1]
        logger.info(f"\nOutput Reasoning: {latest_output.reasoning}")
        logger.info(
            f"RAG Context Available: {latest_output.metadata.get('rag_context_available', False)}"
        )

        if latest_output.metadata.get("rag_context_available"):
            logger.success("✓ RAG context was successfully used in hypothesis generation!")
        else:
            logger.warning("✗ RAG context was not available (ChromaDB may be empty)")


def test_code_analysis() -> None:
    """Test code analysis with RAG context."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Code Analysis with RAG")
    logger.info("=" * 60)

    # Create Cortex engine with RAG enabled
    logger.info("Creating CortexEngine with rag_enabled=True...")
    engine = CortexEngine(rag_enabled=True)

    # Sample code to analyze
    code = """
def calculate_rsi(prices, period=14):
    deltas = prices.diff()
    gain = deltas.where(deltas > 0, 0)
    loss = -deltas.where(deltas < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
"""

    logger.info("Analyzing code...")
    logger.info(f"Code snippet:\n{code}")

    # Analyze code
    output = engine.analyze_code(code, analysis_type="review")

    logger.info("\nCode Analysis Result:")
    logger.info(f"  Confidence: {output.confidence:.2%}")
    logger.info(f"  Reasoning: {output.reasoning}")
    logger.info(f"  Model Used: {output.model_used}")

    analysis = output.content.get("analysis", {})
    logger.info(f"  Code Quality: {analysis.get('code_quality')}")
    logger.info(f"  Suggestions: {analysis.get('suggestions')}")

    if "enhanced with RAG context" in output.reasoning:
        logger.success("✓ RAG context was successfully used in code analysis!")
    else:
        logger.warning("✗ RAG context was not available (ChromaDB may be empty)")


def test_engine_state() -> None:
    """Test engine state with RAG."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Engine State")
    logger.info("=" * 60)

    engine = CortexEngine(rag_enabled=True)
    state = engine.to_dict()

    logger.info("Engine State:")
    logger.info(f"  RAG Enabled: {state.get('rag_enabled')}")
    logger.info(f"  USD Code Enabled: {state.get('usd_code_enabled')}")
    logger.info(f"  Embeddings Enabled: {state.get('embeddings_enabled')}")

    if state.get("rag_enabled"):
        logger.success("✓ RAG is properly enabled in Cortex engine!")
    else:
        logger.error("✗ RAG is not enabled!")


def main() -> None:
    """Run all Cortex RAG integration tests."""
    logger.info("Starting Cortex RAG Integration Tests\n")

    try:
        # Test 1: Engine state
        test_engine_state()

        # Test 2: Hypothesis generation with RAG
        test_hypothesis_generation()

        # Test 3: Code analysis with RAG
        test_code_analysis()

        logger.info("\n" + "=" * 60)
        logger.success("All Cortex RAG Integration Tests Complete!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
