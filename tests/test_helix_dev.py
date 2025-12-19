"""Test Helix for real development tasks."""

import asyncio

from ordinis.ai.helix.config import HelixConfig
from ordinis.ai.helix.engine import Helix


async def test_code_review():
    """Test code review on actual Ordinis code."""
    helix = Helix(HelixConfig(default_temperature=0.1))

    # Read actual code from Ordinis
    from pathlib import Path

    code = Path("src/ordinis/ai/helix/models.py").read_text(encoding="utf-8")

    print("Reviewing: src/ordinis/ai/helix/models.py\n")

    response = await helix.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Python code reviewer for production systems. "
                    "Focus on: architecture, type safety, error handling, performance. "
                    "Be specific and actionable."
                ),
            },
            {
                "role": "user",
                "content": f"Review this models.py file:\n\n```python\n{code[:3000]}\n```",
            },
        ],
        max_tokens=1500,
    )

    print(response.content)
    print(f"\n--- Used {response.usage.total_tokens} tokens in {response.latency_ms:.0f}ms ---")


async def test_strategy_design():
    """Test strategy design generation."""
    helix = Helix(HelixConfig(default_temperature=0.2))

    print("\nDesigning: Iron Condor Strategy\n")

    response = await helix.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert quantitative trading strategist. "
                    "Design Python implementations for Ordinis trading platform. "
                    "Include: entry/exit logic, risk management, position sizing."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Design an iron condor strategy with:\n"
                    "- 10 delta wings\n"
                    "- 45 DTE entry\n"
                    "- 21 DTE exit\n"
                    "- 50% profit target\n"
                    "- Dynamic strike selection based on IV rank"
                ),
            },
        ],
        max_tokens=2500,
    )

    print(response.content)
    print(f"\n--- Used {response.usage.total_tokens} tokens in {response.latency_ms:.0f}ms ---")


async def test_architecture_question():
    """Test architectural guidance."""
    helix = Helix(HelixConfig(default_temperature=0.1))

    print("\nArchitecture Question: Engine Communication\n")

    response = await helix.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior software architect for the Ordinis trading platform. "
                    "Provide concrete, actionable architectural guidance."
                ),
            },
            {
                "role": "user",
                "content": (
                    "In Ordinis, we have multiple engines (Cortex, Synapse, Helix, Learning). "
                    "What's the best pattern for inter-engine communication? "
                    "Consider: async messaging, event bus, direct calls, shared state. "
                    "We need low latency for trading decisions."
                ),
            },
        ],
        max_tokens=1500,
    )

    print(response.content)
    print(f"\n--- Used {response.usage.total_tokens} tokens in {response.latency_ms:.0f}ms ---")


async def main():
    """Run all tests."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_helix_dev.py [review|strategy|architecture]")
        sys.exit(1)

    test_type = sys.argv[1].lower()

    if test_type == "review":
        await test_code_review()
    elif test_type == "strategy":
        await test_strategy_design()
    elif test_type == "architecture":
        await test_architecture_question()
    else:
        print(f"Unknown test: {test_type}")
        print("Options: review, strategy, architecture")


if __name__ == "__main__":
    asyncio.run(main())
