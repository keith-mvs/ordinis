"""
NVIDIA API provider implementation.

Integrates with NVIDIA AI Endpoints for LLM inference.
"""

from collections.abc import AsyncIterator
import logging
import time
from typing import Any

from ordinis.ai.helix.models import (
    ChatMessage,
    ChatResponse,
    EmbeddingResponse,
    HelixError,
    ModelInfo,
    ModelType,
    ProviderError,
    ProviderType,
    RateLimitError,
    UsageInfo,
)
from ordinis.ai.helix.providers.base import BaseProvider

_logger = logging.getLogger(__name__)


class NVIDIAProvider(BaseProvider):
    """NVIDIA AI Endpoints provider."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize NVIDIA provider.

        Args:
            api_key: NVIDIA API key. If None, uses NVIDIA_API_KEY env var.
        """
        import os

        self._api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self._chat_client: Any = None
        self._embed_client: Any = None
        self._initialized = False

    @property
    def provider_type(self) -> ProviderType:
        """Return provider type."""
        return ProviderType.NVIDIA_API

    @property
    def is_available(self) -> bool:
        """Check if NVIDIA API is available."""
        if not self._api_key:
            return False
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA  # noqa: F401

            return True
        except ImportError:
            return False

    def _ensure_chat_client(self, model: ModelInfo) -> Any:
        """Get or create chat client for model."""
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            # Create client for this specific model
            return ChatNVIDIA(
                model=model.model_id,
                nvidia_api_key=self._api_key,
                temperature=model.default_temperature,
                max_completion_tokens=model.max_output_tokens,
            )
        except ImportError as e:
            msg = "NVIDIA SDK not installed. Run: pip install langchain-nvidia-ai-endpoints"
            raise HelixError(msg, provider=self.provider_type, retriable=False) from e
        except Exception as e:
            msg = f"Failed to initialize NVIDIA chat client: {e}"
            raise ProviderError(msg, provider=self.provider_type) from e

    def _ensure_embed_client(self, model: ModelInfo) -> Any:
        """Get or create embedding client for model."""
        try:
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

            return NVIDIAEmbeddings(
                model=model.model_id,
                nvidia_api_key=self._api_key,
                truncate="END",
            )
        except ImportError as e:
            msg = "NVIDIA SDK not installed. Run: pip install langchain-nvidia-ai-endpoints"
            raise HelixError(msg, provider=self.provider_type, retriable=False) from e
        except Exception as e:
            msg = f"Failed to initialize NVIDIA embedding client: {e}"
            raise ProviderError(msg, provider=self.provider_type) from e

    async def chat(
        self,
        messages: list[ChatMessage],
        model: ModelInfo,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: object,
    ) -> ChatResponse:
        """Generate chat completion via NVIDIA API."""
        if model.model_type not in (ModelType.CHAT, ModelType.CODE):
            msg = f"Model {model.model_id} is not a chat model"
            raise HelixError(msg, model=model.model_id)

        start_time = time.perf_counter()

        try:
            client = self._ensure_chat_client(model)

            # Override temperature/max_completion_tokens if specified
            if temperature is not None:
                client.temperature = temperature
            if max_tokens is not None:
                client.max_completion_tokens = max_tokens

            # Convert messages to langchain format
            langchain_messages = [msg.to_dict() for msg in messages]

            # Invoke model
            response = client.invoke(langchain_messages)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract content
            content = response.content if hasattr(response, "content") else str(response)

            # Extract usage if available
            usage = UsageInfo.empty()
            if hasattr(response, "response_metadata"):
                metadata = response.response_metadata
                if "token_usage" in metadata:
                    token_usage = metadata["token_usage"]
                    usage = UsageInfo(
                        prompt_tokens=token_usage.get("prompt_tokens", 0),
                        completion_tokens=token_usage.get("completion_tokens", 0),
                        total_tokens=token_usage.get("total_tokens", 0),
                    )

            return ChatResponse(
                content=content,
                model=model.model_id,
                provider=self.provider_type,
                usage=usage,
                latency_ms=latency_ms,
                finish_reason="stop",
            )

        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError(
                    str(e), provider=self.provider_type, model=model.model_id
                ) from e
            raise ProviderError(str(e), provider=self.provider_type, model=model.model_id) from e

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: ModelInfo,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: object,
    ) -> AsyncIterator[str]:
        """Generate streaming chat completion."""
        if not model.supports_streaming:
            # Fall back to non-streaming
            response = await self.chat(
                messages, model, temperature, max_tokens, stop=None, **kwargs
            )
            yield response.content
            return

        try:
            client = self._ensure_chat_client(model)

            if temperature is not None:
                client.temperature = temperature
            if max_tokens is not None:
                client.max_tokens = max_tokens

            langchain_messages = [msg.to_dict() for msg in messages]

            # Stream response
            for chunk in client.stream(langchain_messages):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content

        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError(
                    str(e), provider=self.provider_type, model=model.model_id
                ) from e
            raise ProviderError(str(e), provider=self.provider_type, model=model.model_id) from e

    async def embed(
        self,
        texts: list[str],
        model: ModelInfo,
        **kwargs: object,
    ) -> EmbeddingResponse:
        """Generate embeddings via NVIDIA API."""
        if model.model_type != ModelType.EMBEDDING:
            msg = f"Model {model.model_id} is not an embedding model"
            raise HelixError(msg, model=model.model_id)

        start_time = time.perf_counter()

        try:
            client = self._ensure_embed_client(model)

            # Generate embeddings
            embeddings = client.embed_documents(texts)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Determine dimension from first embedding
            dimension = len(embeddings[0]) if embeddings else model.embedding_dim or 0

            # Estimate tokens (rough approximation)
            total_chars = sum(len(t) for t in texts)
            estimated_tokens = total_chars // 4

            return EmbeddingResponse(
                embeddings=embeddings,
                model=model.model_id,
                provider=self.provider_type,
                usage=UsageInfo(
                    prompt_tokens=estimated_tokens,
                    completion_tokens=0,
                    total_tokens=estimated_tokens,
                ),
                dimension=dimension,
                latency_ms=latency_ms,
            )

        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError(
                    str(e), provider=self.provider_type, model=model.model_id
                ) from e
            raise ProviderError(str(e), provider=self.provider_type, model=model.model_id) from e

    async def health_check(self) -> bool:
        """Verify NVIDIA API connectivity."""
        if not self.is_available:
            return False

        try:
            # Try a minimal embedding call to verify connectivity
            from ordinis.ai.helix.config import NVIDIA_MODELS

            model = NVIDIA_MODELS.get("nv-embedqa")
            if model:
                client = self._ensure_embed_client(model)
                client.embed_documents(["test"])
                return True
        except Exception as e:
            _logger.warning("NVIDIA health check failed: %s", e)
            return False

        return False
