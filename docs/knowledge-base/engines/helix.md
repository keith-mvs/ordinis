# ordinis.ai.helix.engine

Helix - Unified LLM Provider Engine.

Orchestrates LLM providers with caching, rate limiting, and fallback.

## CacheEntry

Cached response entry.

### Methods

#### `__init__(self, response: ordinis.ai.helix.models.ChatResponse | ordinis.ai.helix.models.EmbeddingResponse, timestamp: float, hits: int = 0) -> None`


---

## Helix

Unified LLM provider for Ordinis.

Provides:
- Chat completions (sync and async)
- Embeddings generation
- Model routing with fallback
- Response caching
- Rate limiting
- Retry with exponential backoff
- Governance integration (preflight checks and auditing)

### Methods

#### `__init__(self, config: ordinis.ai.helix.config.HelixConfig | None = None, governance_hook: ordinis.engines.base.hooks.GovernanceHook | None = None)`

Initialize Helix.

Args:
    config: Configuration options. Uses defaults if None.
    governance_hook: Optional governance hook.

#### `audit(self, action: str, inputs: dict[str, typing.Any] | None = None, outputs: dict[str, typing.Any] | None = None, model_used: str | None = None, latency_ms: float | None = None, **metadata: Any) -> None`

Record an audit event.

Args:
    action: The action performed.
    inputs: Operation inputs (sanitized).
    outputs: Operation outputs (sanitized).
    model_used: AI model used (if applicable).
    latency_ms: Operation latency.
    **metadata: Additional context.

#### `embed(self, texts: str | list[str], model: str | None = None, use_cache: bool | None = None) -> ordinis.ai.helix.models.EmbeddingResponse`

Generate embeddings.

Args:
    texts: Text(s) to embed
    model: Embedding model name/alias
    use_cache: Override cache setting

Returns:
    Embedding vectors

#### `embed_sync(self, texts: str | list[str], model: str | None = None) -> ordinis.ai.helix.models.EmbeddingResponse`

Synchronous embedding generation.

#### `generate(self, messages: list[dict[str, str]] | list[ordinis.ai.helix.models.ChatMessage], model: str | None = None, temperature: float | None = None, max_tokens: int | None = None, stop: list[str] | None = None, use_cache: bool | None = None, **kwargs: Any) -> ordinis.ai.helix.models.ChatResponse`

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

#### `generate_stream(self, messages: list[dict[str, str]] | list[ordinis.ai.helix.models.ChatMessage], model: str | None = None, temperature: float | None = None, max_tokens: int | None = None, **kwargs: Any) -> collections.abc.AsyncIterator[str]`

Generate streaming chat completion.

Args:
    messages: Conversation messages
    model: Model name/alias
    temperature: Sampling temperature
    max_tokens: Maximum tokens to generate
    **kwargs: Additional provider options

Yields:
    Content chunks as they're generated

#### `generate_sync(self, messages: list[dict[str, str]] | list[ordinis.ai.helix.models.ChatMessage], model: str | None = None, **kwargs: Any) -> ordinis.ai.helix.models.ChatResponse`

Synchronous chat completion.

#### `get_metrics(self) -> dict[str, typing.Any]`

Get usage metrics.

#### `health_check(self) -> dict[str, bool]`

Check health of all providers.

#### `initialize(self) -> None`

Initialize the engine.

Sets up resources and transitions to READY state.

Raises:
    RuntimeError: If engine is already initialized.
    Exception: If initialization fails.

#### `list_models(self, model_type: ordinis.ai.helix.models.ModelType | None = None) -> list[ordinis.ai.helix.models.ModelInfo]`

List available models.

#### `preflight(self, action: str, inputs: dict[str, typing.Any] | None = None, **metadata: Any) -> ordinis.engines.base.hooks.PreflightResult`

Run governance preflight check.

Args:
    action: The action being attempted.
    inputs: Operation inputs (sanitized).
    **metadata: Additional context.

Returns:
    PreflightResult with decision.

#### `shutdown(self) -> None`

Shutdown the engine.

Cleans up resources and transitions to STOPPED state.

#### `track_operation(self, action: str, inputs: dict[str, typing.Any] | None = None) -> collections.abc.AsyncIterator[dict[str, typing.Any]]`

Context manager for tracking operations.

Handles preflight, metrics, and audit automatically.

Args:
    action: The action being performed.
    inputs: Operation inputs.

Yields:
    Context dict for storing outputs.

Raises:
    PermissionError: If preflight denies the operation.

Example:
    async with self.track_operation("generate_signal", {"symbol": "AAPL"}) as ctx:
        result = await self._generate_signal("AAPL")
        ctx["outputs"] = {"signal": result}


---

## RateLimitState

Rate limiter state.

### Methods

#### `__init__(self, request_times: list[float] = <factory>, token_counts: list[tuple[float, int]] = <factory>, semaphore: asyncio.locks.Semaphore | None = None) -> None`


---
