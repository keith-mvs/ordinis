"""
Azure OpenAI provider implementation.

Integrates with Azure OpenAI Service for enterprise deployments.
"""

from collections.abc import AsyncIterator
import logging
import os
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


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI Service provider."""

    def __init__(
        self,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str = "2024-02-01",
    ):
        """
        Initialize Azure OpenAI provider.

        Args:
            api_key: Azure OpenAI API key. If None, uses AZURE_OPENAI_API_KEY env var.
            azure_endpoint: Azure endpoint URL. If None, uses AZURE_OPENAI_ENDPOINT env var.
            api_version: Azure OpenAI API version.
        """
        self._api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self._azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self._api_version = api_version
        self._client: Any = None

    @property
    def provider_type(self) -> ProviderType:
        """Return provider type."""
        return ProviderType.AZURE_OPENAI_API

    @property
    def is_available(self) -> bool:
        """Check if Azure OpenAI is available."""
        if not self._api_key or not self._azure_endpoint:
            return False
        try:
            from openai import AsyncAzureOpenAI  # noqa: F401

            return True
        except ImportError:
            return False

    def _ensure_client(self) -> Any:
        """Get or create async Azure OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncAzureOpenAI

                self._client = AsyncAzureOpenAI(
                    api_key=self._api_key,
                    azure_endpoint=self._azure_endpoint,
                    api_version=self._api_version,
                )
            except ImportError as e:
                msg = "OpenAI SDK not installed. Run: pip install openai>=1.0"
                raise HelixError(msg, provider=self.provider_type, retriable=False) from e
            except Exception as e:
                msg = f"Failed to initialize Azure OpenAI client: {e}"
                raise ProviderError(msg, provider=self.provider_type) from e
        return self._client

    async def chat(
        self,
        messages: list[ChatMessage],
        model: ModelInfo,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: object,
    ) -> ChatResponse:
        """Generate chat completion via Azure OpenAI."""
        if model.model_type not in (ModelType.CHAT, ModelType.CODE):
            msg = f"Model {model.model_id} is not a chat model"
            raise HelixError(msg, model=model.model_id)

        start_time = time.perf_counter()

        try:
            client = self._ensure_client()

            # Convert messages to Azure OpenAI format
            azure_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            # For Azure, model.model_id is the deployment name
            response = await client.chat.completions.create(
                model=model.model_id,  # This should be the Azure deployment name
                messages=azure_messages,
                temperature=temperature or model.default_temperature,
                max_tokens=max_tokens or model.max_output_tokens,
                stop=stop,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract response
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"

            # Extract usage
            usage = UsageInfo(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return ChatResponse(
                content=content,
                model=model.model_id,
                provider=self.provider_type,
                usage=usage,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
            )

        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str or "429" in error_str or "rate limit" in error_str:
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
            response = await self.chat(messages, model, temperature, max_tokens, **kwargs)
            yield response.content
            return

        try:
            client = self._ensure_client()

            azure_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            # Stream response
            stream = await client.chat.completions.create(
                model=model.model_id,  # Azure deployment name
                messages=azure_messages,
                temperature=temperature or model.default_temperature,
                max_tokens=max_tokens or model.max_output_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str or "429" in error_str:
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
        """Generate embeddings via Azure OpenAI."""
        if model.model_type != ModelType.EMBEDDING:
            msg = f"Model {model.model_id} is not an embedding model"
            raise HelixError(msg, model=model.model_id)

        start_time = time.perf_counter()

        try:
            client = self._ensure_client()

            # Generate embeddings (model is Azure deployment name)
            response = await client.embeddings.create(model=model.model_id, input=texts)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            dimension = len(embeddings[0]) if embeddings else model.embedding_dim or 0

            # Extract usage
            usage = UsageInfo(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return EmbeddingResponse(
                embeddings=embeddings,
                model=model.model_id,
                provider=self.provider_type,
                usage=usage,
                dimension=dimension,
                latency_ms=latency_ms,
            )

        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str or "429" in error_str:
                raise RateLimitError(
                    str(e), provider=self.provider_type, model=model.model_id
                ) from e
            raise ProviderError(str(e), provider=self.provider_type, model=model.model_id) from e

    async def health_check(self) -> bool:
        """Verify Azure OpenAI connectivity."""
        if not self.is_available:
            return False

        try:
            # Try minimal API call
            client = self._ensure_client()
            # Azure doesn't have models.list(), use a simple completion
            test_messages = [{"role": "user", "content": "test"}]
            await client.chat.completions.create(
                model="gpt-35-turbo", messages=test_messages, max_tokens=1
            )  # type: ignore[arg-type]
            return True
        except Exception as e:
            _logger.warning("Azure OpenAI health check failed: %s", e)
            return False
