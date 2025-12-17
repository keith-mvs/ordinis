"""
Helix - Unified LLM Provider Engine.

Orchestrates LLM providers with caching, rate limiting, and fallback.
"""

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
import hashlib
import logging
import time
from typing import Any

from ordinis.ai.helix.config import HelixConfig
from ordinis.ai.helix.models import (
    ChatMessage,
    ChatResponse,
    EmbeddingResponse,
    HelixError,
    ModelInfo,
    ModelNotFoundError,
    ModelType,
    ProviderType,
    RateLimitError,
)
from ordinis.ai.helix.providers.azure import AzureOpenAIProvider
from ordinis.ai.helix.providers.base import BaseProvider
from ordinis.ai.helix.providers.mistral import MistralProvider
from ordinis.ai.helix.providers.nvidia import NVIDIAProvider
from ordinis.ai.helix.providers.openai import OpenAIProvider
from ordinis.dashboard import TraceType, get_trace_logger
from ordinis.engines.base import (
    BaseEngine,
    GovernanceHook,
    HealthLevel,
    HealthStatus,
    PreflightContext,
)

_logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cached response entry."""

    response: ChatResponse | EmbeddingResponse
    timestamp: float
    hits: int = 0


@dataclass
class RateLimitState:
    """Rate limiter state."""

    request_times: list[float] = field(default_factory=list)
    token_counts: list[tuple[float, int]] = field(default_factory=list)
    semaphore: asyncio.Semaphore | None = None


class Helix(BaseEngine[HelixConfig]):
    """
    Unified LLM provider for Ordinis.

    Provides:
    - Chat completions (sync and async)
    - Embeddings generation
    - Model routing with fallback
    - Response caching
    - Rate limiting
    - Retry with exponential backoff
    - Governance integration (preflight checks and auditing)
    """

    def __init__(
        self,
        config: HelixConfig | None = None,
        governance_hook: GovernanceHook | None = None,
    ):
        """
        Initialize Helix.

        Args:
            config: Configuration options. Uses defaults if None.
            governance_hook: Optional governance hook.
        """
        super().__init__(config or HelixConfig(), governance_hook)

        # Validate configuration
        errors = self.config.validate()
        if errors:
            msg = f"Invalid Helix configuration: {'; '.join(errors)}"
            raise ValueError(msg)

        # Initialize providers
        self._providers: dict[ProviderType, BaseProvider] = {}
        # Providers initialized in _do_initialize

        # Initialize Trace Logger
        self._trace_logger = get_trace_logger()

        # Cache
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Rate limiting state
        self._rate_limit = RateLimitState(
            semaphore=asyncio.Semaphore(self.config.rate_limit.concurrent_requests)
            if self.config.rate_limit.enabled
            else None
        )

        # Metrics
        self._total_requests = 0
        self._total_tokens = 0
        self._cache_hits = 0
        self._cache_misses = 0

    async def _do_initialize(self) -> None:
        """Initialize engine resources."""
        self._init_providers()
        _logger.info("Helix engine initialized")

    async def _do_shutdown(self) -> None:
        """Shutdown engine resources."""
        self._providers.clear()
        _logger.info("Helix engine shutdown")

    async def _do_health_check(self) -> HealthStatus:
        """Check engine health."""
        provider_health = {}
        all_healthy = True

        for p_type, provider in self._providers.items():
            try:
                is_healthy = await provider.health_check()
                provider_health[p_type.value] = is_healthy
                if not is_healthy:
                    all_healthy = False
            except Exception:
                provider_health[p_type.value] = False
                all_healthy = False

        if not self._providers:
            return HealthStatus(
                level=HealthLevel.DEGRADED,
                message="No providers configured",
                details={"providers": {}},
            )

        return HealthStatus(
            level=HealthLevel.HEALTHY if all_healthy else HealthLevel.DEGRADED,
            message="All providers healthy" if all_healthy else "Some providers unhealthy",
            details={"providers": provider_health},
        )

    def _init_providers(self) -> None:
        """Initialize available providers."""
        # NVIDIA API provider
        if self.config.nvidia_api_key:
            try:
                nvidia = NVIDIAProvider(self.config.nvidia_api_key)
                if nvidia.is_available:
                    self._providers[ProviderType.NVIDIA_API] = nvidia
                    _logger.info("NVIDIA API provider initialized")
            except Exception as e:
                _logger.warning("Failed to initialize NVIDIA provider: %s", e)

        # Mistral API provider
        if self.config.mistral_api_key:
            try:
                mistral = MistralProvider(self.config.mistral_api_key)
                if mistral.is_available:
                    self._providers[ProviderType.MISTRAL_API] = mistral
                    _logger.info("Mistral API provider initialized")
            except Exception as e:
                _logger.warning("Failed to initialize Mistral provider: %s", e)

        # OpenAI API provider
        if self.config.openai_api_key:
            try:
                openai = OpenAIProvider(self.config.openai_api_key)
                if openai.is_available:
                    self._providers[ProviderType.OPENAI_API] = openai
                    _logger.info("OpenAI API provider initialized")
            except Exception as e:
                _logger.warning("Failed to initialize OpenAI provider: %s", e)

        # Azure OpenAI provider
        if self.config.azure_openai_api_key and self.config.azure_openai_endpoint:
            try:
                azure = AzureOpenAIProvider(
                    self.config.azure_openai_api_key, self.config.azure_openai_endpoint
                )
                if azure.is_available:
                    self._providers[ProviderType.AZURE_OPENAI_API] = azure
                    _logger.info("Azure OpenAI provider initialized")
            except Exception as e:
                _logger.warning("Failed to initialize Azure provider: %s", e)

        if not self._providers:
            _logger.warning("No LLM providers available")

    def _get_provider(self, model: ModelInfo) -> BaseProvider:
        """Get provider for model."""
        provider = self._providers.get(model.provider)
        if not provider:
            msg = f"No provider available for {model.provider.value}"
            raise HelixError(msg, provider=model.provider)
        return provider

    def _get_model(self, name: str | None, model_type: ModelType) -> ModelInfo:
        """Resolve model by name or get default."""
        if name:
            model = self.config.get_model(name)
            if not model:
                available = [m.model_id for m in self.config.list_models(model_type)]
                raise ModelNotFoundError(name, available)
            return model

        # Get default for type
        if model_type == ModelType.EMBEDDING:
            return self.config.models[self.config.default_embedding_model]
        return self.config.models[self.config.default_chat_model]

    def _get_fallback_model(self, model_type: ModelType) -> ModelInfo | None:
        """Get fallback model for type."""
        if not self.config.allow_fallback:
            return None

        if model_type == ModelType.EMBEDDING:
            return self.config.models.get(self.config.fallback_embedding_model)
        return self.config.models.get(self.config.fallback_chat_model)

    def _cache_key(self, messages: list[ChatMessage], model: str, **kwargs: Any) -> str:
        """Generate cache key for request."""
        content = f"{model}:{[m.to_dict() for m in messages]}:{sorted(kwargs.items())}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _embed_cache_key(self, texts: list[str], model: str) -> str:
        """Generate cache key for embeddings."""
        content = f"{model}:{sorted(texts)}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _get_cached(self, key: str) -> ChatResponse | EmbeddingResponse | None:
        """Get cached response if valid."""
        if not self.config.cache.enabled:
            return None

        entry = self._cache.get(key)
        if not entry:
            self._cache_misses += 1
            return None

        # Check TTL
        age = time.time() - entry.timestamp
        if age > self.config.cache.ttl_seconds:
            del self._cache[key]
            self._cache_misses += 1
            return None

        entry.hits += 1
        self._cache_hits += 1

        # Mark as cached
        response = entry.response
        response.cached = True
        return response

    def _set_cached(self, key: str, response: ChatResponse | EmbeddingResponse) -> None:
        """Cache response."""
        if not self.config.cache.enabled:
            return

        # Check type-specific caching settings
        if isinstance(response, ChatResponse) and not self.config.cache.cache_chat:
            return
        if isinstance(response, EmbeddingResponse) and not self.config.cache.cache_embeddings:
            return

        # Evict oldest if at capacity
        while len(self._cache) >= self.config.cache.max_entries:
            self._cache.popitem(last=False)

        self._cache[key] = CacheEntry(response=response, timestamp=time.time())

    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limits."""
        if not self.config.rate_limit.enabled:
            return

        now = time.time()
        window = 60.0  # 1 minute window

        # Clean old entries
        self._rate_limit.request_times = [
            t for t in self._rate_limit.request_times if now - t < window
        ]
        self._rate_limit.token_counts = [
            (t, c) for t, c in self._rate_limit.token_counts if now - t < window
        ]

        # Check request limit
        if len(self._rate_limit.request_times) >= self.config.rate_limit.requests_per_minute:
            wait_time = self._rate_limit.request_times[0] + window - now
            raise RateLimitError(
                f"Request rate limit exceeded. Retry in {wait_time:.1f}s",
                retry_after=wait_time,
            )

        # Check token limit
        total_tokens = sum(c for _, c in self._rate_limit.token_counts)
        if total_tokens >= self.config.rate_limit.tokens_per_minute:
            wait_time = self._rate_limit.token_counts[0][0] + window - now
            raise RateLimitError(
                f"Token rate limit exceeded. Retry in {wait_time:.1f}s",
                retry_after=wait_time,
            )

    def _record_usage(self, tokens: int) -> None:
        """Record usage for rate limiting."""
        if self.config.rate_limit.enabled:
            now = time.time()
            self._rate_limit.request_times.append(now)
            self._rate_limit.token_counts.append((now, tokens))

        self._total_requests += 1
        self._total_tokens += tokens

    async def _retry_with_backoff(
        self,
        func: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with retry and exponential backoff."""
        retry_cfg = self.config.retry
        last_error: Exception | None = None

        for attempt in range(retry_cfg.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except RateLimitError as e:
                last_error = e
                if e.retry_after:
                    await asyncio.sleep(e.retry_after)
                    continue
            except HelixError as e:
                if not e.retriable or attempt >= retry_cfg.max_retries:
                    raise
                last_error = e
            except Exception as e:
                if attempt >= retry_cfg.max_retries:
                    raise
                last_error = e

            # Calculate backoff delay
            delay = min(
                retry_cfg.initial_delay_ms * (retry_cfg.exponential_base**attempt),
                retry_cfg.max_delay_ms,
            )
            delay_seconds = delay / 1000

            if retry_cfg.jitter:
                import random

                delay_seconds *= 0.5 + random.random()

            _logger.warning(
                "Request failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1,
                retry_cfg.max_retries + 1,
                delay_seconds,
                last_error,
            )
            await asyncio.sleep(delay_seconds)

        raise last_error or HelixError("Request failed after retries")

    async def generate(
        self,
        messages: list[dict[str, str]] | list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Generate chat completion.

        Args:
            messages: Conversation messages
            model: Model name/alias (uses default if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            use_cache: Override cache setting
            **kwargs: Additional provider options

        Returns:
            Chat completion response
        """
        # Convert dict messages to ChatMessage
        if messages and isinstance(messages[0], dict):
            messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]

        # Resolve model
        model_info = self._get_model(model, ModelType.CHAT)

        # Governance Preflight
        context = PreflightContext(
            engine=self.name,
            action="generate",
            inputs={
                "model": model_info.model_id,
                "messages_count": len(messages),
                "max_tokens": max_tokens,
            },
        )

        async with self.track_operation("generate", context.inputs) as ctx:
            # Start Trace
            trace_id = self._trace_logger.start_trace()

            self._trace_logger.log(
                trace_type=TraceType.LLM_REQUEST,
                component="Helix",
                content={
                    "model": model_info.model_id,
                    "messages": [m.to_dict() for m in messages],
                    "parameters": {"temperature": temperature, "max_tokens": max_tokens, **kwargs},
                },
                trace_id=trace_id,
            )

            # Check cache
            cache_enabled = use_cache if use_cache is not None else self.config.cache.cache_chat
            if cache_enabled:
                cache_key = self._cache_key(messages, model_info.model_id, **kwargs)
                cached = self._get_cached(cache_key)
                if cached and isinstance(cached, ChatResponse):
                    ctx["model_used"] = cached.model
                    ctx["cached"] = True

                    # Log Cache Hit
                    self._trace_logger.log(
                        trace_type=TraceType.LLM_RESPONSE,
                        component="Helix",
                        content=asdict(cached),
                        trace_id=trace_id,
                        metadata={"cached": True},
                    )
                    return cached

            # Rate limit check
            await self._check_rate_limit()

            # Acquire semaphore for concurrent request limiting
            if self._rate_limit.semaphore:
                async with self._rate_limit.semaphore:
                    response = await self._do_generate(
                        messages, model_info, temperature, max_tokens, stop, cache_enabled, **kwargs
                    )
            else:
                response = await self._do_generate(
                    messages, model_info, temperature, max_tokens, stop, cache_enabled, **kwargs
                )

            ctx["model_used"] = response.model
            ctx["usage"] = response.usage

            # Log Response
            self._trace_logger.log(
                trace_type=TraceType.LLM_RESPONSE,
                component="Helix",
                content=asdict(response),
                trace_id=trace_id,
                metadata={"cached": False},
            )

            return response

    async def _do_generate(
        self,
        messages: list[ChatMessage],
        model_info: ModelInfo,
        temperature: float | None,
        max_tokens: int | None,
        stop: list[str] | None,
        cache_enabled: bool,
        **kwargs: Any,
    ) -> ChatResponse:
        """Execute chat generation with fallback."""
        try:
            provider = self._get_provider(model_info)
            response = await self._retry_with_backoff(
                provider.chat,
                messages,
                model_info,
                temperature or self.config.default_temperature,
                max_tokens or self.config.default_max_tokens,
                stop,
                **kwargs,
            )

            # Record usage
            self._record_usage(response.usage.total_tokens)

            # Cache response
            if cache_enabled:
                cache_key = self._cache_key(messages, model_info.model_id, **kwargs)
                self._set_cached(cache_key, response)

            return response

        except HelixError:
            # Try fallback model
            fallback = self._get_fallback_model(ModelType.CHAT)
            if fallback and fallback.model_id != model_info.model_id:
                _logger.info("Falling back to %s", fallback.display_name)
                return await self._do_generate(
                    messages, fallback, temperature, max_tokens, stop, cache_enabled, **kwargs
                )
            raise

    async def generate_stream(
        self,
        messages: list[dict[str, str]] | list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate streaming chat completion.

        Args:
            messages: Conversation messages
            model: Model name/alias
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider options

        Yields:
            Content chunks as they're generated
        """
        # Convert dict messages to ChatMessage
        if messages and isinstance(messages[0], dict):
            messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]

        model_info = self._get_model(model, ModelType.CHAT)

        # Governance Preflight
        context = PreflightContext(
            engine=self.name,
            action="generate_stream",
            inputs={
                "model": model_info.model_id,
                "messages_count": len(messages),
                "max_tokens": max_tokens,
            },
        )

        async with self.track_operation("generate_stream", context.inputs) as ctx:
            provider = self._get_provider(model_info)

            await self._check_rate_limit()

            # We can't easily track token usage for streams without a tokenizer or provider support
            # For now, we just mark it as streaming
            ctx["streaming"] = True

            async for chunk in provider.chat_stream(
                messages,
                model_info,
                temperature or self.config.default_temperature,
                max_tokens or self.config.default_max_tokens,
                **kwargs,
            ):
                yield chunk

    async def embed(
        self,
        texts: str | list[str],
        model: str | None = None,
        use_cache: bool | None = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings.

        Args:
            texts: Text(s) to embed
            model: Embedding model name/alias
            use_cache: Override cache setting

        Returns:
            Embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]

        model_info = self._get_model(model, ModelType.EMBEDDING)

        # Governance Preflight
        context = PreflightContext(
            engine=self.name,
            action="embed",
            inputs={
                "model": model_info.model_id,
                "text_count": len(texts),
            },
        )

        async with self.track_operation("embed", context.inputs) as ctx:
            # Check cache
            cache_enabled = (
                use_cache if use_cache is not None else self.config.cache.cache_embeddings
            )
            if cache_enabled:
                cache_key = self._embed_cache_key(texts, model_info.model_id)
                cached = self._get_cached(cache_key)
                if cached and isinstance(cached, EmbeddingResponse):
                    ctx["cached"] = True
                    return cached

            await self._check_rate_limit()

            try:
                provider = self._get_provider(model_info)
                response = await self._retry_with_backoff(provider.embed, texts, model_info)

                self._record_usage(response.usage.total_tokens)

                if cache_enabled:
                    cache_key = self._embed_cache_key(texts, model_info.model_id)
                    self._set_cached(cache_key, response)

                ctx["usage"] = response.usage
                return response

            except HelixError:
                fallback = self._get_fallback_model(ModelType.EMBEDDING)
                if fallback and fallback.model_id != model_info.model_id:
                    _logger.info("Falling back to %s", fallback.display_name)
                    provider = self._get_provider(fallback)
                    response = await self._retry_with_backoff(provider.embed, texts, fallback)
                    ctx["usage"] = response.usage
                    ctx["fallback_used"] = True
                    return response
                raise

    def list_models(self, model_type: ModelType | None = None) -> list[ModelInfo]:
        """List available models."""
        return self.config.list_models(model_type)

    def get_metrics(self) -> dict[str, Any]:
        """Get usage metrics."""
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            "cache_size": len(self._cache),
            "providers": list(self._providers.keys()),
        }

    async def health_check(self) -> dict[str, bool]:
        """Check health of all providers."""
        results: dict[str, bool] = {}
        for provider_type, provider in self._providers.items():
            try:
                results[provider_type.value] = await provider.health_check()
            except Exception:
                results[provider_type.value] = False
        return results

    # Synchronous convenience methods
    def generate_sync(
        self,
        messages: list[dict[str, str]] | list[ChatMessage],
        model: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Synchronous chat completion."""
        return asyncio.get_event_loop().run_until_complete(self.generate(messages, model, **kwargs))

    def embed_sync(
        self,
        texts: str | list[str],
        model: str | None = None,
    ) -> EmbeddingResponse:
        """Synchronous embedding generation."""
        return asyncio.get_event_loop().run_until_complete(self.embed(texts, model))
