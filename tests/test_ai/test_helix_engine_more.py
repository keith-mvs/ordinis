import time
import pytest

from ordinis.ai.helix.engine import Helix
from ordinis.ai.helix.models import (
    ModelInfo,
    UsageInfo,
    ChatResponse,
    EmbeddingResponse,
    ModelType,
    ProviderType,
    RateLimitError,
)


class DummyProvider2:
    @property
    def provider_type(self):
        return ProviderType.MOCK

    @property
    def is_available(self):
        return True

    async def health_check(self):
        return True

    async def chat(self, messages: list[dict[str, str]], model, temperature=None, max_tokens=None, stop=None, **kwargs):
        # reference messages to avoid "unused variable" warnings in linters
        _ = messages
        return ChatResponse(
            content="ok2",
            model=model.model_id,
            provider=model.provider,
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def chat_stream(self, messages: list[dict[str, str]], model, temperature=None, max_tokens=None, **kwargs):
        # reference messages to avoid "unused variable" warnings in linters
        _ = messages
        yield "c1"

    async def embed(self, texts, model, **kwargs):
        # reference texts to avoid "unused variable" warnings in linters
        _ = texts
        return EmbeddingResponse(
            embeddings=[[0.3, 0.4]],
            model=model.model_id,
            provider=model.provider,
            usage=UsageInfo.empty(),
            dimension=2,
        )


@pytest.mark.asyncio
async def test_cache_set_get_and_embed_cache(monkeypatch):
    engine = Helix(config=None)
    engine.config.prefer_local = True

    # Use dummy provider
    dummy = DummyProvider2()
    engine._providers[ProviderType.MOCK] = dummy

    # embedding cache
    engine.config.cache.cache_embeddings = True
    model = ModelInfo(
        model_id="dummy-embed",
        display_name="DummyEmbed",
        model_type=ModelType.EMBEDDING,
        provider=ProviderType.MOCK,
        context_length=512,
        embedding_dim=2,
    )

    # Call embed; should set cache
    monkeypatch.setattr(engine, "_get_model", lambda name, mt: model)
    res = await engine.embed(["a"], model="dummy-embed")
    assert isinstance(res, EmbeddingResponse)
    # now get cached entry by constructing key
    key = engine._embed_cache_key(["a"], model.model_id)
    cached = engine._get_cached(key)
    assert cached is not None
    assert cached.cached is True


@pytest.mark.asyncio
async def test_check_rate_limit_triggers_and_retry_with_backoff(monkeypatch):
    engine = Helix(config=None)
    engine.config.prefer_local = True

    # Force request times to exceed limit
    engine._rate_limit.request_times = [time.time()] * engine.config.rate_limit.requests_per_minute

    with pytest.raises(RateLimitError):
        await engine._check_rate_limit()

    # Test retry behaviour: function that fails once with RateLimitError then succeeds
    calls = {"n": 0}

    async def flaky():
        if calls["n"] == 0:
            calls["n"] += 1
            raise RateLimitError("tmp", retry_after=0.01)
        return "ok"

    res = await engine._retry_with_backoff(flaky)
    assert res == "ok"


def test_get_fallback_and_model_resolution():
    engine = Helix(config=None)
    engine.config.prefer_local = True
    fb = engine._get_fallback_model(ModelType.CHAT)
    assert fb is not None
    assert fb.model_type == ModelType.CHAT


@pytest.mark.asyncio
async def test_chat_cache_hits(monkeypatch):
    engine = Helix(config=None)
    engine.config.prefer_local = True
    engine.config.cache.cache_chat = True

    dummy = DummyProvider2()
    engine._providers[ProviderType.MOCK] = dummy

    model = ModelInfo(
        model_id="dummy-chat",
        display_name="DummyChat",
        model_type=ModelType.CHAT,
        provider=ProviderType.MOCK,
        context_length=1024,
    )

    monkeypatch.setattr(engine, "_get_model", lambda name, mt: model)

    r1 = await engine.generate([{"role": "user", "content": "x"}], model="dummy-chat")
    r2 = await engine.generate([{"role": "user", "content": "x"}], model="dummy-chat")

    assert isinstance(r1, ChatResponse)
    assert isinstance(r2, ChatResponse)
    # second call should be cached
    assert engine.get_metrics()["cache_hits"] >= 1
