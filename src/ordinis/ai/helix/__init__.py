"""
Helix - Unified LLM Provider for Ordinis.

Abstracts LLM interactions across NVIDIA API and local models.
Provides consistent interface for:
- Chat completions (strategy generation, code analysis, research)
- Embeddings (semantic search, similarity)
- Model routing (primary/fallback)
- Rate limiting and retry logic
- Response caching

Example:
    from ordinis.ai.helix import Helix, HelixConfig

    config = HelixConfig(nvidia_api_key="nvapi-...")
    helix = Helix(config)

    # Chat completion
    response = await helix.generate(
        messages=[{"role": "user", "content": "Analyze market conditions"}],
        model="nemotron-super",  # Or use default
    )
    print(response.content)

    # Embeddings
    vectors = await helix.embed(["trading strategy", "risk management"])
    print(vectors.shape)
"""

from ordinis.ai.helix.config import HelixConfig
from ordinis.ai.helix.engine import Helix
from ordinis.ai.helix.models import (
    ChatMessage,
    ChatResponse,
    EmbeddingResponse,
    HelixError,
    ModelInfo,
    ModelType,
    ProviderType,
    UsageInfo,
)

__all__ = [
    "ChatMessage",
    "ChatResponse",
    "EmbeddingResponse",
    "Helix",
    "HelixConfig",
    "HelixError",
    "ModelInfo",
    "ModelType",
    "ProviderType",
    "UsageInfo",
]
