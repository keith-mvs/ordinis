"""Ask Helix about testing the refactored CortexEngine."""

import asyncio

from ordinis.ai.helix.config import HelixConfig
from ordinis.ai.helix.engine import Helix


async def main():
    """Test refactoring consultation."""
    helix = Helix(HelixConfig(default_temperature=0.1))

    response = await helix.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Python testing engineer. "
                    "Provide concrete, runnable test code for pytest."
                ),
            },
            {
                "role": "user",
                "content": """I've refactored CortexEngine with these changes:

1. Created NVIDIAAdapter class (src/ordinis/engines/cortex/core/nvidia_adapter.py)
2. Added TypedDict types: MarketContext, StrategyConstraints (types.py)
3. Added config validation at __init__ - raises ValueError if features enabled without API key
4. Added _enforce_history_limit() method - trims to max_history_size
5. Updated generate_hypothesis() to use typed parameters

Write 5 critical pytest test cases to verify these changes work correctly.
Include:
- Config validation test
- History limit enforcement test
- Type checking test
- NVIDIAAdapter integration test
- Mocking examples

Provide complete, runnable code for tests/test_engines/test_cortex/test_engine_refactor.py
""",
            },
        ],
        max_tokens=4000,
    )

    print(response.content)
    print(f"\n---\nTokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
