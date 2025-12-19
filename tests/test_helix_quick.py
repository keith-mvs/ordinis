"""Quick test of Helix for development tasks."""

import asyncio

from ordinis.ai.helix.config import HelixConfig
from ordinis.ai.helix.engine import Helix


async def test_basic():
    """Test basic Helix functionality."""
    config = HelixConfig(default_temperature=0.1)
    helix = Helix(config)

    # Health check
    print("Checking health...")
    health = await helix.health_check()
    print(f"Health: {health}")

    # Simple code review
    print("\nTesting code review...")
    response = await helix.generate(
        messages=[
            {"role": "system", "content": "You are a Python code reviewer. Be concise."},
            {"role": "user", "content": "Review this: def add(a, b): return a + b"},
        ],
        max_tokens=500,
    )

    print(f"\nResponse: {response.content}")
    print(f"Tokens used: {response.usage.total_tokens}")
    print(f"Latency: {response.latency_ms:.0f}ms")

    # Metrics
    metrics = helix.get_metrics()
    print("\nMetrics:")
    print(f"  Total requests: {metrics['total_requests']}")
    print(f"  Total tokens: {metrics['total_tokens']}")
    print(f"  Providers: {metrics['providers']}")


if __name__ == "__main__":
    asyncio.run(test_basic())
