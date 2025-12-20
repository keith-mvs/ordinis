"""
Run a Cortex Strategy Session to generate SOTA traces.
"""

import asyncio
import logging
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ordinis.ai.helix import Helix, HelixConfig
from ordinis.engines.cortex.core.engine import CortexEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


async def main():
    _logger.info("Initializing SOTA Stack (Helix + Cortex)...")

    # 1. Initialize Helix with DeepSeek R1
    helix_config = HelixConfig()
    helix_config.default_chat_model = "deepseek-r1"  # Force R1
    helix = Helix(helix_config)
    await helix.initialize()

    # Inject Mock if needed (AFTER initialization)
    if not os.getenv("NVIDIA_API_KEY"):
        _logger.warning("Injecting Mock Provider...")
        from ordinis.ai.helix.models import ChatResponse, ProviderType, UsageInfo
        from ordinis.ai.helix.providers.base import BaseProvider

        class MockDeepSeek(BaseProvider):
            @property
            def provider_type(self):
                return ProviderType.MOCK

            @property
            def is_available(self):
                return True

            async def chat(self, messages, model, **kwargs):
                return ChatResponse(
                    content="Based on the high volatility regime, I recommend a Momentum Breakout strategy. <think>The market is volatile, suggesting strong directional moves. Breakout strategies work well here.</think>",
                    model=model.model_id,
                    provider=ProviderType.MOCK,
                    usage=UsageInfo(100, 50, 150),
                )

            async def chat_stream(self, *args, **kwargs):
                yield "Mock"

            async def embed(self, *args, **kwargs):
                return []

        helix._providers[ProviderType.NVIDIA_API] = MockDeepSeek()

    # 2. Initialize Cortex
    cortex = CortexEngine(helix=helix)
    await cortex.initialize()

    # 3. Define a Strategy Request
    market_context = {
        "regime": "high_volatility",
        "trend": "bullish",
        "assets": ["BTC-USD", "ETH-USD"],
        "volatility": "high",
    }

    constraints = {"max_drawdown": 0.15, "target_sharpe": 2.0}

    _logger.info("Requesting Strategy Hypothesis from Cortex (DeepSeek R1)...")

    try:
        # 4. Generate Hypothesis
        hypothesis = await cortex.generate_hypothesis(
            market_context=market_context, constraints=constraints
        )

        _logger.info("Hypothesis Generated!")
        # Access fields directly from the StrategyHypothesis object, not .content
        _logger.info(f"ID: {hypothesis.hypothesis_id}")
        _logger.info(f"Type: {hypothesis.strategy_type}")
        _logger.info(f"Confidence: {hypothesis.confidence}")
        _logger.info(f"Rationale: {hypothesis.rationale[:200]}...")

        _logger.info("Session Complete. Traces should be in artifacts/traces/")

    except Exception as e:
        _logger.error(f"Session failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
